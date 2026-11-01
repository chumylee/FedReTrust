"""Viceroy: Algorithm 1 of Lewis et al. (JMLR 24(30), 2023).

The pinned author implementation supplies initialization, first-round handling
and its modified FoolsGold FedScale. The common simulator supplies local SGD
deltas, rather than the authors' sums of raw gradients. The published raw
weighted SUM is preserved: coefficients are not normalized to sum to one.
"""
from dataclasses import asdict, dataclass
from pathlib import Path
import math
import sys

import torch

from experiment_core import AggregationResult, file_sha256, state_sha256


@dataclass(frozen=True)
class ViceroyConfig:
    omega: float = sys.float_info.epsilon ** (1 / 56)
    eta: float = 0.2
    numerical_epsilon: float = 1e-12

    def __post_init__(self):
        if not math.isfinite(self.omega) or not 0 <= self.omega <= 1:
            raise ValueError("History decay must be finite and in [0, 1].")
        if not math.isfinite(self.eta) or not 0 < self.eta <= 1:
            raise ValueError("Reputation factor must be finite and in (0, 1].")
        if not math.isfinite(self.numerical_epsilon) or not 0 < self.numerical_epsilon < 1:
            raise ValueError("Numerical epsilon must be finite and in (0, 1).")


@torch.inference_mode()
def fedscale(vectors, numerical_epsilon=1e-12):
    """Author Viceroy's stateless, full-vector modified FoolsGold scale.

    Includes cosine pardoning, division by the largest raw weight, and the
    logit transform with +0.5. All-identical nonzero directions make the
    author's division undefined. If the largest raw weight <= epsilon we
    return zeros explicitly. This numerical policy is not a new detector.
    """
    if vectors.device.type != "cpu" or not vectors.is_floating_point() or vectors.ndim != 2 or 0 in vectors.shape:
        raise ValueError("Expected a nonempty floating-point CPU matrix.")
    if not torch.isfinite(vectors).all():
        raise FloatingPointError("Non-finite FedScale input.")
    x = vectors.detach().double()
    norms = torch.linalg.vector_norm(x, dim=1, keepdim=True)
    normalized = x / torch.where(norms > 0, norms, torch.ones_like(norms))
    cosine = (normalized @ normalized.T).clamp(-1, 1)
    cosine -= torch.eye(len(x), dtype=torch.float64)
    maximum = cosine.max(dim=1).values
    # Pardoning uses maxima computed before any pair is adjusted.
    for i in range(len(x)):
        for j in range(len(x)):
            if i != j and maximum[i] < maximum[j]:
                cosine[i, j] *= maximum[i] / maximum[j]
    weights = (1 - cosine.max(dim=1).values).clamp(0, 1)
    peak = float(weights.max())
    if peak <= numerical_epsilon:
        return torch.zeros(len(x), dtype=torch.float64)
    weights /= peak
    weights[weights == 1] = 0.99
    active = weights != 0
    weights[active] = torch.log(weights[active] / (1 - weights[active])) + 0.5
    weights.clamp_(0, 1)
    if not torch.isfinite(weights).all():
        raise FloatingPointError("Non-finite FedScale weights.")
    return weights


def _vectors(submissions, broadcast):
    if not submissions or len({s.client_id for s in submissions}) != len(submissions):
        raise ValueError("Provide distinct client submissions.")
    if any(type(s.client_id) is not int or type(s.num_examples) is not int or s.num_examples <= 0 for s in submissions):
        raise ValueError("Client IDs must be integers and sample counts positive integers.")
    if not broadcast:
        raise ValueError("The broadcast must be nonempty.")
    for value in broadcast.values():
        if value.device.type != "cpu" or not value.is_floating_point() or value.numel() == 0:
            raise ValueError("Expected nonempty floating-point CPU parameters.")
        if not torch.isfinite(value).all():
            raise FloatingPointError("Non-finite broadcast.")
    ordered = sorted(submissions, key=lambda s: s.client_id)
    rows = []
    for submission in ordered:
        if set(submission.state) != set(broadcast):
            raise ValueError("Submitted model keys differ from broadcast.")
        pieces = []
        for name, reference in broadcast.items():
            value = submission.state[name]
            if value.shape != reference.shape or value.dtype != reference.dtype or value.device.type != "cpu":
                raise ValueError(f"Incompatible submitted parameter: {name}")
            if not torch.isfinite(value).all():
                raise FloatingPointError("Non-finite submitted model.")
            pieces.append((value.detach().double() - reference.detach().double()).flatten())
        rows.append(torch.cat(pieces))
    layout = tuple((name, tuple(value.shape), str(value.dtype)) for name, value in broadcast.items())
    return ordered, torch.stack(rows), layout


class ViceroyAggregator:
    name = "Viceroy"

    def __init__(self, options=None):
        self.options = ViceroyConfig() if options is None else options
        if not isinstance(self.options, ViceroyConfig):
            raise TypeError("Use ViceroyConfig for settings.")
        resources = Path(__file__).parents[1] / "resources"
        self.settings = {
            **asdict(self.options), "initial_reputation": 1.0, "initial_history": "zeros",
            "first_round_reputation_update": "skipped_as_in_author_code",
            "fedscale": "author_modified_FoolsGold; full_vector; pardoning; logit_plus_0.5",
            "weighting": "reputation_blend_of_new_history_and_current_FedScale",
            "aggregation": "raw_weighted_sum; no_weight_sum_or_client_count_division",
            "delta_convention": "trained_minus_broadcast; add_aggregate; server_multiplier_1",
            "sample_counts_used": False, "trusted_root_used": False,
            "participation": "fixed_client_set; contiguous_rounds_starting_at_1",
            "accumulation_dtype": "float64",
            "degenerate_fedscale_policy": "maximum_raw_weight_le_epsilon_returns_zeros",
            "zero_total_weight_policy": "keep_broadcast; still_update_history_and_reputation",
            "implementation_sha256": file_sha256(Path(__file__)),
            "paper_sha256": file_sha256(resources / "Viceroy_Lewis_2023.pdf"),
            "reference_manifest_sha256": file_sha256(resources / "viceroy_reference/source_manifest.json"),
        }
        self.reset()

    def reset(self):
        self.client_ids = None
        self.histories = None
        self.reputations = None
        self.layout = None
        self.last_round = 0

    def snapshot(self):
        """Save the final defence state separately from the inference checkpoint."""
        if self.histories is None:
            raise ValueError("No observations yet.")
        return {"client_ids": list(self.client_ids), "last_round": self.last_round,
                "layout": self.layout, "options": asdict(self.options),
                "histories": self.histories.clone(), "reputations": self.reputations.clone()}

    @torch.inference_mode()
    def aggregate(self, submissions, global_state, round_number):
        if type(round_number) is not int or round_number != self.last_round + 1:
            raise ValueError("Rounds must be contiguous and start at 1; reset between runs.")
        ordered, current, layout = _vectors(submissions, global_state)
        ids = tuple(s.client_id for s in ordered)
        if self.client_ids is not None and (ids != self.client_ids or layout != self.layout):
            raise ValueError("Client membership and model layout must stay fixed within a run.")
        old_history = torch.zeros_like(current) if self.histories is None else self.histories
        old_reps = torch.ones(len(ids), dtype=torch.float64) if self.reputations is None else self.reputations
        current_scale = fedscale(current, self.options.numerical_epsilon)
        old_history_scale = fedscale(old_history, self.options.numerical_epsilon)
        updated = round_number > 1
        if updated:
            reputations = (old_reps + self.options.eta / 2 * (1 - 2 * abs(old_history_scale - current_scale))).clamp(0, 1)
        else:
            reputations = old_reps.clone()
        histories = self.options.omega * old_history + current
        history_scale = fedscale(histories, self.options.numerical_epsilon)
        coefficients = reputations * history_scale + (1 - reputations) * current_scale
        delta = torch.zeros_like(current[0])
        for point, weight in zip(current, coefficients):
            delta.add_(point, alpha=float(weight))
        result, offset = {}, 0
        for name, reference in global_state.items():
            result[name] = (reference.detach().double() + delta[offset:offset + reference.numel()].reshape(reference.shape)).to(reference.dtype)
            offset += reference.numel()
        if not all(torch.isfinite(value).all() for value in result.values()):
            raise FloatingPointError("Non-finite aggregate.")
        mass = float(coefficients.sum())
        metrics = {str(client): {
            "previous_reputation": float(old_reps[i]), "reputation": float(reputations[i]),
            "reputation_change": float(reputations[i] - old_reps[i]),
            "previous_history_scale": float(old_history_scale[i]), "current_scale": float(current_scale[i]),
            "new_history_scale": float(history_scale[i]), "aggregation_weight": float(coefficients[i]),
            "normalized_weight_share": float(coefficients[i]) / mass if mass > 0 else 0.0,
            "raw_update_l2_norm": float(torch.linalg.vector_norm(current[i])),
            "previous_history_l2_norm": float(torch.linalg.vector_norm(old_history[i])),
            "new_history_l2_norm": float(torch.linalg.vector_norm(histories[i])),
        } for i, client in enumerate(ids)}
        diagnostics = {"reputation_update_applied": updated, "coefficient_mass": mass,
                       "zero_total_weight": mass == 0, "aggregate_update_l2_norm": float(torch.linalg.vector_norm(delta)),
                       "history_sha256": state_sha256({"histories": histories}),
                       "client_metrics": metrics}
        # Commit only after validation and calculation succeed. Never mutate input models.
        self.client_ids, self.layout = ids, layout
        self.histories, self.reputations, self.last_round = histories, reputations, round_number
        return AggregationResult(result, {client: float(w) for client, w in zip(ids, coefficients)}, diagnostics)
