"""RFA via the joint weighted geometric median (Pillutla et al., 2022).

Algorithms 1-2 and Appendix D.2: smoothed Weiszfeld reweighting, with sample
counts as base weights. We compute on trained-minus-broadcast deltas and add
the result; translation equivariance makes this equivalent to aggregating
the submitted models. This simulator does not implement secure aggregation.
"""
from dataclasses import asdict, dataclass
from pathlib import Path
import math

import torch

from experiment_core import AggregationResult, file_sha256


@dataclass(frozen=True)
class RFAConfig:
    max_iterations: int = 3
    smoothing: float = 1e-6
    relative_tolerance: float = 1e-6

    def __post_init__(self):
        if type(self.max_iterations) is not int or self.max_iterations < 1:
            raise ValueError("The Weiszfeld iteration budget must be a positive integer.")
        if not math.isfinite(self.smoothing) or self.smoothing <= 0:
            raise ValueError("Smoothing must be finite and positive.")
        if not math.isfinite(self.relative_tolerance) or self.relative_tolerance < 0:
            raise ValueError("Relative tolerance must be finite and nonnegative; zero disables early stopping.")


def _average(points, weights):
    result = torch.zeros_like(points[0])
    for point, weight in zip(points, weights):
        result.add_(point, alpha=float(weight))
    return result


def _objectives(distances, alphas, smoothing):
    # The paper's smoothed norm is r for r >= nu, and r^2/(2*nu)+nu/2 below nu.
    smoothed = distances.clone()
    small = distances < smoothing
    smoothed[small] = 0.5 * smoothing * ((distances[small] / smoothing).square() + 1.0)
    return float(torch.dot(alphas, distances)), float(torch.dot(alphas, smoothed))


@torch.inference_mode()
def smoothed_weiszfeld(points, sample_weights, options=None):
    """Return an approximate joint median, its final convex weights and diagnostics.

    The initial sample-weighted mean costs one averaging operation. Each of up
    to max_iterations reweighting steps costs one more. The stopping criterion
    uses the paper's smoothed objective; reaching the budget is not a certificate
    of exact geometric-median convergence.
    """
    options = RFAConfig() if options is None else options
    if not isinstance(options, RFAConfig):
        raise TypeError("Use RFAConfig for solver settings.")
    if points.device.type != "cpu" or not points.is_floating_point() or points.ndim != 2 or 0 in points.shape:
        raise ValueError("Expected a nonempty matrix of floating-point CPU vectors.")
    points = points.detach().double()
    alphas = torch.as_tensor(sample_weights, dtype=torch.float64, device="cpu")
    if alphas.shape != (len(points),) or not torch.isfinite(alphas).all() or not (alphas > 0).all():
        raise ValueError("Provide one finite positive base weight per point.")
    if not torch.isfinite(points).all():
        raise FloatingPointError("Non-finite input vector.")
    total = alphas.sum()
    if not torch.isfinite(total):
        raise FloatingPointError("Base weight sum overflow.")
    alphas = alphas / total
    estimate = _average(points, alphas)
    distances = torch.linalg.vector_norm(points - estimate, dim=1)
    objective, smoothed_objective = _objectives(distances, alphas, options.smoothing)
    if not math.isfinite(smoothed_objective):
        raise FloatingPointError("Non-finite initial aggregation objective.")
    trace = [{"iteration": 0, "objective": objective, "smoothed_objective": smoothed_objective,
              "step_l2_norm": 0.0, "relative_objective_change": None}]
    termination = "iteration_budget"
    for iteration in range(1, options.max_iterations + 1):
        weighting_distances = distances
        beta = alphas / weighting_distances.clamp_min(options.smoothing)
        weights = beta / beta.sum()
        if not torch.isfinite(weights).all() or not (weights > 0).all():
            raise FloatingPointError("Invalid inverse-distance weights.")
        updated = _average(points, weights)
        distances = torch.linalg.vector_norm(points - updated, dim=1)
        objective, new_smoothed_objective = _objectives(distances, alphas, options.smoothing)
        if not math.isfinite(new_smoothed_objective):
            raise FloatingPointError("Non-finite aggregation objective.")
        relative_change = abs(smoothed_objective - new_smoothed_objective) / smoothed_objective
        trace.append({"iteration": iteration, "objective": objective, "smoothed_objective": new_smoothed_objective,
                      "step_l2_norm": float(torch.linalg.vector_norm(updated - estimate)),
                      "relative_objective_change": relative_change})
        estimate, smoothed_objective = updated, new_smoothed_objective
        if options.relative_tolerance > 0 and relative_change <= options.relative_tolerance:
            termination = "relative_objective_tolerance"
            break
    return estimate, weights, {
        "iterations_used": iteration, "averaging_operations": iteration + 1,
        "termination": termination, "objective_trace": trace,
        "initial_objective": trace[0]["objective"], "final_objective": objective,
        "initial_smoothed_objective": trace[0]["smoothed_objective"],
        "final_smoothed_objective": smoothed_objective,
        "base_weights": alphas.tolist(),
        "weighting_distances": weighting_distances.tolist(),
        "final_distances": distances.tolist(),
    }


def _flatten_submissions(submissions, broadcast):
    if not submissions or len({s.client_id for s in submissions}) != len(submissions):
        raise ValueError("Provide distinct client submissions.")
    if any(type(s.num_examples) is not int or s.num_examples <= 0 for s in submissions):
        raise ValueError("Client sample counts must be positive integers.")
    if not broadcast:
        raise ValueError("The broadcast model must be nonempty.")
    for name, tensor in broadcast.items():
        if tensor.device.type != "cpu" or not tensor.is_floating_point() or tensor.numel() == 0:
            raise ValueError("Expected nonempty floating-point CPU parameters.")
        if not torch.isfinite(tensor).all():
            raise FloatingPointError("Non-finite broadcast parameter.")
    points = []
    for submission in submissions:
        if set(submission.state) != set(broadcast):
            raise ValueError("Submitted model keys differ from the broadcast.")
        pieces = []
        for name, reference in broadcast.items():
            tensor = submission.state[name]
            if tensor.shape != reference.shape or tensor.dtype != reference.dtype or tensor.device.type != "cpu":
                raise ValueError(f"Incompatible submitted parameter: {name}")
            if not torch.isfinite(tensor).all():
                raise FloatingPointError("Non-finite submitted parameter.")
            pieces.append((tensor.detach().double() - reference.detach().double()).flatten())
        points.append(torch.cat(pieces))
    return torch.stack(points)


@torch.inference_mode()
def rfa_rule(submissions, broadcast, options=None):
    options = RFAConfig() if options is None else options
    points = _flatten_submissions(submissions, broadcast)
    delta, weights, solver = smoothed_weiszfeld(points, [s.num_examples for s in submissions], options)
    result, offset = {}, 0
    for name, reference in broadcast.items():
        piece = delta[offset:offset + reference.numel()].reshape(reference.shape)
        result[name] = (reference.detach().double() + piece).to(reference.dtype)
        offset += reference.numel()
    base_weights = solver.pop("base_weights")
    weighting_distances = solver.pop("weighting_distances")
    final_distances = solver.pop("final_distances")
    metrics = {}
    for i, submission in enumerate(submissions):
        metrics[str(submission.client_id)] = {
            "base_sample_weight": base_weights[i], "aggregation_weight": float(weights[i]),
            "relative_weight": float(weights[i]) / base_weights[i],
            "weighting_distance": weighting_distances[i], "distance_to_returned_estimate": final_distances[i],
            "smoothing_active": weighting_distances[i] <= options.smoothing,
            "raw_update_l2_norm": float(torch.linalg.vector_norm(points[i])),
        }
    diagnostics = {**solver, "aggregate_update_l2_norm": float(torch.linalg.vector_norm(delta)),
                   "client_metrics": metrics}
    return AggregationResult(result, {s.client_id: float(w) for s, w in zip(submissions, weights)}, diagnostics)


class RFAAggregator:
    name = "RFA"

    def __init__(self, options=None):
        self.options = RFAConfig() if options is None else options
        resources = Path(__file__).parents[1] / "resources"
        self.settings = {
            **asdict(self.options), "weighting": "sample_count_then_inverse_joint_l2_distance",
            "initialization": "sample_weighted_mean", "accumulation_dtype": "float64",
            "delta_convention": "trained_minus_broadcast; add_aggregate",
            "stopping_objective": "smoothed_geometric_median; relative_change_over_previous_objective",
            "budget_accounting": "one_initial_mean_plus_iterations_used_reweighted_means",
            "secure_aggregation_implemented": False,
            "implementation_sha256": file_sha256(Path(__file__)),
            "paper_sha256": file_sha256(resources / "RFA_Pillutla_2022.pdf"),
            "reference_manifest_sha256": file_sha256(resources / "rfa_reference/source_manifest.json"),
        }

    def aggregate(self, submissions, global_state, round_number):
        return rfa_rule(submissions, global_state, self.options)
