"""Client-local utility of including each update in a fixed-weight aggregate.

The full and leave-one-out states are counterfactual evaluation models only.
Removing a client never renormalizes the remaining weights. Report and audit
views stay in the existing local evaluator; no training or global advancement
occurs here. This is the same simulated boundary, not a privacy guarantee.
"""
import copy
import math

import torch

from client_validation_local import _validate_models, evaluate_local_candidates


EVALUATION_SEMANTICS = {
    "kind": "fixed_weight_leave_one_out_client_contribution",
    "loss_gain": "loss(without_client) - loss(full_aggregate)",
    "accuracy_gain": "accuracy(full_aggregate) - accuracy(without_client)",
    "positive_loss_gain": "Including the client lowers the evaluated loss in this aggregate context.",
    "class_and_flipped_gains": "Same contribution sign; missing-class values remain null.",
    "weights": "Fixed across the full and all leave-one-out models; no removal renormalization.",
    "scope": "Conditional contribution to this fixed aggregate; not standalone-update utility, Shapley value, or future model recovery.",
}


@torch.inference_mode()
def build_aggregate_counterfactuals(broadcast, submitted_states, weights=None):
    """Return (full_state, {client_id: without_client_state}) for ten clients.

    With the default weights, full = w + sum_i 0.1*(submitted_i-w), and
    without_i = w + sum_{j!=i} 0.1*(submitted_j-w). An optional explicit weight
    mapping must be positive and sum to one; it is never inferred or normalized.
    The pilot uses only the default equal weights for its equal-size clients.

    Both models are built from the unrounded float64 total. In particular, a
    leave-one-out model is NOT computed by subtracting from an already-rounded
    full model. Each output is cast once to the broadcast dtype (pilot float32).
    """
    _validate_models(broadcast, submitted_states, validator_id=0)
    ids = sorted(submitted_states)
    weights = {cid: .1 for cid in ids} if weights is None else dict(weights)
    if (set(weights) != set(ids) or any(type(cid) is not int for cid in weights)
            or any(isinstance(value, bool) or not isinstance(value, (int, float))
                   or not math.isfinite(value) or value <= 0 for value in weights.values())
            or not math.isclose(sum(weights.values()), 1., rel_tol=0, abs_tol=1e-12)):
        raise ValueError("Provide one explicit finite positive weight per client, summing to one.")
    full, without = {}, {cid: {} for cid in ids}
    for name, reference in broadcast.items():
        origin = reference.detach().double()
        contributions = torch.stack([(submitted_states[cid][name].detach().double() - origin) * weights[cid]
                                     for cid in ids])
        total = contributions.sum(dim=0)
        full[name] = (origin + total).to(reference.dtype)
        if not torch.isfinite(full[name]).all():
            raise FloatingPointError("Non-finite full aggregate parameter.")
        for position, cid in enumerate(ids):
            without[cid][name] = (origin + (total - contributions[position])).to(reference.dtype)
            if not torch.isfinite(without[cid][name]).all():
                raise FloatingPointError("Non-finite leave-one-out aggregate parameter.")
    return full, without


def _reverse_gain(value):
    if value is None:
        return None
    if isinstance(value, list):
        return [_reverse_gain(item) for item in value]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("Local gain summaries must be finite scalars or lists with missing-class nulls.")
    return -value


def _contribution_row(row):
    result = {}
    for name, value in row.items():
        prefix = "flipped_" if name.startswith("flipped_") else ""
        field = name[len(prefix):]
        if field.endswith("_gain") or field.endswith("_gains"):
            result[name] = _reverse_gain(value)
        elif field.startswith("baseline_"):
            result[f"{prefix}full_aggregate_{field[len('baseline_'):]}"] = copy.deepcopy(value)
        elif field.startswith("candidate_") and field != "candidate_id":
            result[f"{prefix}without_client_{field[len('candidate_'):]}"] = copy.deepcopy(value)
        else:
            result[name] = copy.deepcopy(value)
    result["evaluation_semantics"] = EVALUATION_SEMANTICS["kind"]
    return result


def evaluate_aggregate_contributions(full_state, without_states, report_view, audit_view, config, validator_id):
    """Return positive gains when INCLUDING the client improves aggregate utility.

    The reused evaluator initially measures full-minus-without loss. Reversing
    every mean/class/accuracy gain (including the flipped-label counterfactual)
    gives without-minus-full loss and full-minus-without accuracy. Absolute
    metrics are renamed to full_aggregate_* and without_client_* to prevent
    the earlier marginal-candidate interpretation from carrying over silently.
    Report/audit separation, cached inputs, forward counts and RNG preservation
    remain those of evaluate_local_candidates; no extra model forward is added.
    """
    measured = evaluate_local_candidates(full_state, without_states, report_view, audit_view, config, validator_id)
    cost = copy.deepcopy(measured["cost"])
    cost["full_aggregate_forward_examples"] = cost.pop("broadcast_forward_examples")
    cost["without_client_forward_examples"] = cost.pop("candidate_forward_examples")
    cost["full_aggregate_models_per_role"] = 1
    cost["without_client_models_per_role"] = 10
    return {
        "report_rows": [_contribution_row(row) for row in measured["report_rows"]],
        "audit_rows": [_contribution_row(row) for row in measured["audit_rows"]],
        "cost": cost, "evaluation_semantics": copy.deepcopy(EVALUATION_SEMANTICS),
    }
