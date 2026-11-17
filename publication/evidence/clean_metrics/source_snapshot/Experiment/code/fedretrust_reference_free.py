"""Exploratory FedReTrust v0.2: peer evidence, bounded influence, restoration.

This is a proposed combination, not a reproduction of RFA, FLTrust or centered
clipping, and no inherited robustness theorem is claimed. No data/reference,
loss, labels, scenario or evaluator enters the aggregation interface.
"""
from dataclasses import asdict, dataclass
from pathlib import Path
import math

import torch

from experiment_core import AggregationResult, file_sha256
from rfa import _flatten_submissions, smoothed_weiszfeld, RFAConfig


@dataclass(frozen=True)
class ReferenceFreeConfig:
    norm_cap: float = 6.0
    distance_onset: float = 2.0
    alignment_low: float = .25
    alignment_full: float = .75
    restoration_step: float = .5
    use_restoration: bool = True
    epsilon: float = 1e-12

    def __post_init__(self):
        for key, value in asdict(self).items():
            if key != "use_restoration" and (isinstance(value, bool) or not math.isfinite(value)):
                raise ValueError(f"Invalid finite setting: {key}")
        if (type(self.use_restoration) is not bool or self.norm_cap <= 0
                or self.distance_onset <= 0 or self.epsilon <= 0
                or not 0 <= self.alignment_low < self.alignment_full
                or not 0 < self.restoration_step <= 1):
            raise ValueError("Invalid reference-free configuration.")


def restored_weight(previous, current, options):
    """Immediate restriction; at most two supportive participations from 0 to 1.

    An absent client retains its last weight: absence is not evidence of repair.
    A fresh update is always assessed even if its previous weight was zero.
    """
    if not (math.isfinite(previous) and math.isfinite(current) and
            0 <= previous <= 1 and 0 <= current <= 1):
        raise ValueError("Weights must be finite and in [0, 1].")
    return min(current, previous + options.restoration_step) if options.use_restoration else current


class ReferenceFreeFedReTrust:
    def __init__(self, options=None):
        self.options = options or ReferenceFreeConfig()
        self.name = "FedReTrust-RF-recovery" if self.options.use_restoration else "FedReTrust-RF-static"
        self.settings = {
            **asdict(self.options), "version": "0.2-exploratory",
            "trusted_reference_examples": 0, "server_learning_rate": 1.,
            "anchor": "three_step_RFA_on_norm_capped_updates",
            "sample_count_assumption": "authentic_counts; no_sybil_defence",
            "secure_aggregation_implemented": False,
            "implementation_sha256": file_sha256(Path(__file__)),
            "rfa_implementation_sha256": file_sha256(Path(__file__).with_name("rfa.py")),
        }
        self.reset()

    def reset(self):
        self.weights = {}
        self.last_round = 0

    @torch.inference_mode()
    def aggregate(self, submissions, broadcast, round_number):
        if type(round_number) is not int or round_number <= self.last_round:
            raise ValueError("Round numbers must increase; reset between runs.")
        points = _flatten_submissions(submissions, broadcast)
        cfg = self.options
        base = torch.tensor([s.num_examples for s in submissions], dtype=torch.float64)
        base /= base.sum()
        norms = torch.linalg.vector_norm(points, dim=1)
        # torch.median uses the lower median for even cohorts. Zero majority
        # therefore bounds all nonzero updates to zero, rather than admitting
        # a minority solely because their update is nonzero.
        cap = cfg.norm_cap * float(norms.median())
        scales = (cap / norms.clamp_min(cfg.epsilon)).clamp(max=1.)
        scales[norms <= cfg.epsilon] = 1.
        bounded = points * scales[:, None]
        anchor, anchor_weights, _ = smoothed_weiszfeld(bounded, base, RFAConfig())
        anchor_norm = float(torch.linalg.vector_norm(anchor))
        distances = torch.linalg.vector_norm(points - anchor, dim=1)
        median_distance = float(distances.median())
        cosines = ((points @ anchor) / (norms * anchor_norm).clamp_min(cfg.epsilon)).clamp(-1., 1.)
        median_cosine = float(cosines.median())
        # Distance alone must not label a useful but different update harmful.
        # The distance gate grows from zero at 2x to one at 4x median distance.
        outlier = (distances / max(cfg.distance_onset * median_distance, cfg.epsilon) - 1.).clamp(0., 1.)
        alignment = ((cosines - cfg.alignment_low * median_cosine) /
                     max((cfg.alignment_full - cfg.alignment_low) * median_cosine, cfg.epsilon)).clamp(0., 1.)
        evidence = 1. - outlier * (1. - alignment)
        fallback = anchor_norm <= cfg.epsilon or median_cosine <= cfg.epsilon
        if fallback:
            # No meaningful shared direction: use the bounded RFA anchor,
            # and do not update per-client memory from undefined evidence.
            q = torch.ones_like(base)
            updated = dict(self.weights)
            coefficients = anchor_weights * scales
        else:
            q = torch.tensor([restored_weight(self.weights.get(s.client_id, 1.), float(evidence[i]), cfg)
                              for i, s in enumerate(submissions)], dtype=torch.float64)
            updated = {**self.weights, **{s.client_id: float(q[i]) for i, s in enumerate(submissions)}}
            mass = float(torch.dot(base, q))
            if mass <= cfg.epsilon:
                fallback = True
                coefficients = anchor_weights * scales
            else:
                # Normalize trust BEFORE magnitude clipping. Dividing by the
                # clipped coefficient mass would undo the magnitude bound.
                coefficients = base * q / mass * scales
        delta = (points * coefficients[:, None]).sum(0)
        if not torch.isfinite(delta).all():
            raise FloatingPointError("Non-finite reference-free aggregate.")
        state, offset = {}, 0
        for name, reference in broadcast.items():
            piece = delta[offset:offset + reference.numel()].reshape(reference.shape)
            state[name] = (reference.double() + piece).to(reference.dtype)
            if not torch.isfinite(state[name]).all():
                raise FloatingPointError("Non-finite reconstructed model.")
            offset += reference.numel()
        metrics = {str(s.client_id): {
            "anchor_cosine": float(cosines[i]), "distance_to_anchor": float(distances[i]),
            "outlier_score": float(outlier[i]), "alignment_score": float(alignment[i]),
            "evidence_weight": float(evidence[i]) if not fallback else None,
            "restored_weight": float(q[i]) if not fallback else None,
            "clip_factor": float(scales[i]), "sample_share": float(base[i]),
        } for i, s in enumerate(submissions)}
        info = {"coefficient_convention": "effective_coefficients_on_raw_client_deltas",
                "client_metrics": metrics, "reference_forward_examples": 0,
                "anchor_l2_norm": anchor_norm, "norm_cap": cap,
                "median_distance": median_distance, "median_cosine": median_cosine,
                "bounded_rfa_fallback": fallback}
        # Commit only after validating the complete output.
        self.weights, self.last_round = updated, round_number
        return AggregationResult(state, {s.client_id: float(coefficients[i]) for i, s in enumerate(submissions)}, info)
