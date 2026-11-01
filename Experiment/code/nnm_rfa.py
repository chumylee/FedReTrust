"""Reference-free NNM pre-aggregation followed by the existing approximate RFA.

NNM uses the arithmetic mean of n-f nearest updates, including the anchor:
https://byzfl.epfl.ch/aggregators/classes/nnm.html
https://proceedings.mlr.press/v206/allouah23a/allouah23a.pdf

The experiment declares f=3 independently of the scenario. Original client
sample shares weight the *anchor mixtures* in RFA; they do not weight neighbors
inside NNM. This unequal-count adaptation is explicit, not a claimed theorem.
Neither the three-iteration solver nor multi-step local training inherits an
exact geometric-median convergence or Byzantine-learning guarantee here.
"""
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from experiment_core import AggregationResult, file_sha256
import rfa
from rfa import RFAConfig, _flatten_submissions, smoothed_weiszfeld


@dataclass(frozen=True)
class NNMRFAConfig:
    # A fixed declared maximum, never the actual number of active attackers.
    f: int = 3

    def __post_init__(self):
        if type(self.f) is not int or self.f < 0:
            raise ValueError("The declared Byzantine budget f must be a nonnegative integer.")


@torch.inference_mode()
def nearest_neighbor_mixing(points, client_ids, f):
    """Return arithmetic mixtures, their row-stochastic matrix A, and neighbors.

    Self is always included. Other neighbors are ordered by Euclidean distance,
    then integer client ID, making ties deterministic under input permutation.
    Rows and columns of A follow client_ids. No sample weighting occurs here.
    """
    NNMRFAConfig(f=f)
    if (points.device.type != "cpu" or not points.is_floating_point()
            or points.ndim != 2 or 0 in points.shape):
        raise ValueError("Expected a nonempty matrix of floating-point CPU vectors.")
    n = len(points)
    if (len(client_ids) != n or any(type(cid) is not int for cid in client_ids)
            or len(set(client_ids)) != n):
        raise ValueError("Provide one distinct integer client ID per vector.")
    if 2 * f >= n:
        raise ValueError("NNM-RFA requires 0 <= f < n/2 for the participating clients.")
    points = points.detach().double()
    if not torch.isfinite(points).all():
        raise FloatingPointError("Non-finite input update.")
    distances = torch.cdist(points, points, p=2, compute_mode="donot_use_mm_for_euclid_dist")
    if not torch.isfinite(distances).all():
        raise FloatingPointError("Non-finite pairwise update distance.")
    neighborhood_size = n - f
    matrix = torch.zeros((n, n), dtype=torch.float64)
    neighbors = {}
    for i, cid in enumerate(client_ids):
        others = sorted((j for j in range(n) if j != i),
                        key=lambda j: (float(distances[i, j]), client_ids[j]))
        selected = [i, *others[:neighborhood_size - 1]]
        matrix[i, selected] = 1.0 / neighborhood_size
        neighbors[str(cid)] = [client_ids[j] for j in selected]
    mixtures = matrix @ points
    if not torch.isfinite(mixtures).all():
        raise FloatingPointError("Non-finite neighbor mixture.")
    return mixtures, matrix, neighbors


@torch.inference_mode()
def nnm_rfa_rule(submissions, broadcast, options=None):
    options = NNMRFAConfig() if options is None else options
    if not isinstance(options, NNMRFAConfig):
        raise TypeError("Use NNMRFAConfig for NNM-RFA settings.")
    if any(type(s.client_id) is not int for s in submissions):
        raise ValueError("Client IDs must be integers.")
    ordered = sorted(submissions, key=lambda s: s.client_id)
    points = _flatten_submissions(ordered, broadcast)
    client_ids = [s.client_id for s in ordered]
    mixtures, matrix, neighbors = nearest_neighbor_mixing(points, client_ids, options.f)
    # Reuse the original RFA solver unchanged, including its three-step budget.
    mixture_delta, mixture_weights, solver = smoothed_weiszfeld(
        mixtures, [s.num_examples for s in ordered], RFAConfig())
    coefficients = matrix.T @ mixture_weights
    # Expand A^T w onto original signed updates before constructing the model.
    # ContributionRecorder independently checks this same coefficient contract.
    delta = (points * coefficients[:, None]).sum(dim=0)
    if not torch.isfinite(delta).all():
        raise FloatingPointError("Non-finite expanded aggregate update.")
    state, offset = {}, 0
    for name, reference in broadcast.items():
        piece = delta[offset:offset + reference.numel()].reshape(reference.shape)
        state[name] = (reference.detach().double() + piece).to(reference.dtype)
        if not torch.isfinite(state[name]).all():
            raise FloatingPointError("Non-finite aggregated model parameter.")
        offset += reference.numel()
    client_metrics = {}
    for i, cid in enumerate(client_ids):
        client_metrics[str(cid)] = {
            "base_sample_weight": solver["base_weights"][i],
            "anchor_mixture_weight": float(mixture_weights[i]),
            "effective_update_coefficient": float(coefficients[i]),
            "aggregation_weight": float(coefficients[i]),
            "relative_weight": float(coefficients[i]) / solver["base_weights"][i],
            "neighbor_ids": neighbors[str(cid)],
            "mixture_membership_count": int((matrix[:, i] > 0).sum()),
            "raw_update_l2_norm": float(torch.linalg.vector_norm(points[i])),
        }
    diagnostics = {
        "coefficient_convention": "effective_coefficients_on_raw_client_deltas",
        "declared_f": options.f, "neighborhood_size": len(ordered) - options.f,
        "client_order": client_ids, "mixing_matrix": matrix.tolist(),
        "neighbor_ids": neighbors, "mixture_weights": mixture_weights.tolist(),
        "original_client_coefficients": coefficients.tolist(),
        "coefficient_expansion": "A_transpose_times_final_RFA_mixture_weights",
        "mixture_reconstruction_l2_error": float(torch.linalg.vector_norm(delta - mixture_delta)),
        "aggregate_update_l2_norm": float(torch.linalg.vector_norm(delta)),
        "reference_forward_examples": 0,
        "solver": solver, "client_metrics": client_metrics,
    }
    return AggregationResult(state, dict(zip(client_ids, coefficients.tolist())), diagnostics)


class NNMRFAAggregator:
    name = "NNM-RFA"

    def __init__(self, options=None):
        self.options = NNMRFAConfig() if options is None else options
        if not isinstance(self.options, NNMRFAConfig):
            raise TypeError("Use NNMRFAConfig for NNM-RFA settings.")
        self.settings = {
            **asdict(self.options), "rfa": asdict(RFAConfig()),
            "budget_interpretation": "fixed_declared_maximum; not_round_or_scenario_truth",
            "neighborhood_size": "participating_n_minus_f; requires_n_greater_than_2f",
            "mixing": "equal_neighbor_arithmetic_mean_including_self",
            "neighbor_distance": "joint_l2_on_raw_trained_minus_broadcast_updates",
            "tie_breaking": "self_always_included; other_ties_by_ascending_client_id",
            "mixture_base_weighting": "original_sample_shares_over_anchor_mixtures",
            "unequal_count_adaptation_has_claimed_theorem": False,
            "coefficient_convention": "effective_coefficients_on_raw_client_deltas",
            "coefficient_expansion": "A_transpose_times_final_RFA_mixture_weights",
            "accumulation_dtype": "float64",
            "delta_convention": "trained_minus_broadcast; add_aggregate",
            "reference_forward_examples": 0,
            "secure_aggregation_implemented": False,
            "implementation_sha256": file_sha256(Path(__file__)),
            "rfa_implementation_sha256": file_sha256(Path(rfa.__file__)),
            "nnm_reference_url": "https://byzfl.epfl.ch/aggregators/classes/nnm.html",
            "nnm_paper_url": "https://proceedings.mlr.press/v206/allouah23a/allouah23a.pdf",
        }

    def aggregate(self, submissions, global_state, round_number):
        return nnm_rfa_rule(submissions, global_state, self.options)
