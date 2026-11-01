"""Draft 24-round schedules and evaluation-only measurements for notebook 11.

These are experimental design choices, not FedReTrust rules. Simulator intent,
event schedules and class distributions NEVER enter ContributionRecorder.
"""
import copy
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from experiment_core import AggregationResult, file_sha256
from fltrust import flatten_delta
from scenario_utils import Event, Scenario


PRIMARY_SCENARIOS = (
    "clean", "abrupt_shot_noise", "gradual_stripe", "label_flip",
    "gaussian_update", "attack_stops", "recurring_attack", "mixed_drift_and_attack",
)
SECONDARY_CORE_SCENARIOS = ("honest_change_stops",)
MATCHED_CONTROL_SCENARIOS = ("mixed_drift_only_control",)
OBSERVABILITY_CHECK_SCENARIOS = ("ambiguous_malicious_input",)
PERFORMANCE_SCENARIOS = PRIMARY_SCENARIOS + SECONDARY_CORE_SCENARIOS + MATCHED_CONTROL_SCENARIOS
TASK_NAMES = ("clean", "shot_noise_s3", "stripe")


def longer_scenarios():
    """Proposal-length draft; all windows are inclusive, observed post-update."""
    affected = (0, 1, 2)
    drift = Event("input_change", (3, 4, 5), ((5, 24),), corruption="shot_noise")
    scenarios = {
        "clean": Scenario("clean"),
        "abrupt_shot_noise": Scenario("abrupt_shot_noise", (
            Event("input_change", affected, ((5, 24),), corruption="shot_noise"),)),
        "gradual_stripe": Scenario("gradual_stripe", (
            Event("input_change", affected, ((5, 24),), corruption="stripe", ramp_rounds=8),)),
        "label_flip": Scenario("label_flip", (
            Event("label_flip", affected, ((5, 24),), intent="malicious"),)),
        "gaussian_update": Scenario("gaussian_update", (
            Event("gaussian_update", affected, ((5, 24),), intent="malicious"),)),
        "attack_stops": Scenario("attack_stops", (
            Event("label_flip", affected, ((5, 10),), intent="malicious"),)),
        "recurring_attack": Scenario("recurring_attack", (
            Event("label_flip", affected, ((5, 8), (15, 18)), intent="malicious"),)),
        "mixed_drift_and_attack": Scenario("mixed_drift_and_attack", (
            drift, Event("label_flip", affected, ((5, 10),), intent="malicious"))),
        "honest_change_stops": Scenario("honest_change_stops", (
            Event("input_change", affected, ((5, 10),), corruption="shot_noise"),)),
        "mixed_drift_only_control": Scenario("mixed_drift_only_control", (drift,)),
        "ambiguous_malicious_input": Scenario("ambiguous_malicious_input", (
            Event("input_change", affected, ((5, 24),), corruption="shot_noise", intent="malicious"),)),
    }
    for scenario in scenarios.values():
        scenario.validate(tuple(range(10)), 24)
    return scenarios


def matched_reference_name(scenario_name):
    if scenario_name not in longer_scenarios():
        raise ValueError("Unknown draft scenario.")
    return "mixed_drift_only_control" if scenario_name == "mixed_drift_and_attack" else "clean"


def schedule_rows(scenarios=None):
    scenarios = longer_scenarios() if scenarios is None else scenarios
    rows = []
    for name, scenario in scenarios.items():
        for round_number in range(1, 25):
            for client in range(10):
                active = scenario.active(client, round_number)
                rows.append({"scenario": name, "round": round_number, "client": client,
                             "input_change_fraction": sum(e.strength(round_number) for e in active if e.kind == "input_change"),
                             "honest_input_change": any(e.kind == "input_change" and e.intent == "honest" for e in active),
                             "attack_active": any(e.intent == "malicious" for e in active),
                             "event_kinds": "+".join(e.kind for e in active) or "clean"})
    return rows


class ClasswiseEvaluator:
    """Record sufficient statistics on already reserved evaluation views.

    Empty classes have a count of zero, not an invented accuracy. No predictions
    or evaluation statistics are passed back to the aggregation interface.
    """
    name = "classwise_10_digit_evaluator_v1"

    def __init__(self):
        self.settings = {"classes": 10, "implementation_sha256": file_sha256(__file__),
                         "access": "evaluation_only", "empty_class_accuracy": "undefined"}

    @torch.inference_mode()
    def evaluate(self, model, view, config):
        model.eval()
        counts = np.zeros(10, dtype=np.int64)
        correct = np.zeros(10, dtype=np.int64)
        losses = np.zeros(10, dtype=np.float64)
        for images, targets in DataLoader(view, batch_size=config.evaluation_batch_size,
                                          shuffle=False, num_workers=0):
            scores = model(images)
            if scores.ndim != 2 or scores.shape[1] != 10 or targets.min() < 0 or targets.max() > 9:
                raise ValueError("Expected ten-class digit predictions and labels.")
            batch_losses = nn.functional.cross_entropy(scores, targets, reduction="none")
            if not torch.isfinite(scores).all() or not torch.isfinite(batch_losses).all():
                raise FloatingPointError("Non-finite evaluation output.")
            prediction = scores.argmax(1)
            for digit in range(10):
                mask = targets == digit
                counts[digit] += int(mask.sum())
                correct[digit] += int((prediction[mask] == targets[mask]).sum())
                losses[digit] += float(batch_losses[mask].double().sum())
        if counts.sum() == 0:
            raise ValueError("Cannot evaluate an empty view.")
        result = {"accuracy": float(correct.sum() / counts.sum()),
                  "loss": float(losses.sum() / counts.sum()), "examples": int(counts.sum())}
        for digit in range(10):
            result.update({f"class_examples_{digit}": int(counts[digit]),
                           f"class_correct_{digit}": int(correct[digit]),
                           f"class_loss_sum_{digit}": float(losses[digit])})
        return result


def legitimate_task_weights(scenario, round_number, client_class_counts, client_ids=None):
    """Standardized legitimate task, equal weight per chosen client.

    Full saved client label frequencies define q_i. Honest input changes mix
    clean and transformed task accuracies by their scheduled fraction, including
    gradual fractions. Poisoned labels/updates never redefine the target labels.
    This population-standardized mixture is not the empirical changed-mask
    class mix of the 512-image sample; both sample and full counts are recorded.
    """
    counts = np.asarray(client_class_counts, dtype=np.float64)
    if counts.shape != (10, 10) or not np.isfinite(counts).all() or (counts < 0).any() or (counts.sum(1) <= 0).any():
        raise ValueError("Supply positive-population counts for ten clients and digits.")
    if type(round_number) is not int or not 0 <= round_number <= 24:
        raise ValueError("Round must be in 0..24.")
    clients = tuple(range(10)) if client_ids is None else tuple(client_ids)
    if not clients or len(set(clients)) != len(clients) or any(type(c) is not int or not 0 <= c < 10 for c in clients):
        raise ValueError("Choose distinct client IDs in 0..9.")
    weights = {name: np.zeros(10) for name in TASK_NAMES}
    for client in clients:
        q = counts[client] / counts[client].sum() / len(clients)
        changes = [e for e in scenario.active(client, round_number)
                   if e.kind == "input_change" and e.intent == "honest"]
        if len(changes) > 1:
            raise ValueError("Ambiguous task mixture.")
        fraction = changes[0].strength(round_number) if changes else 0.0
        weights["clean"] += q * (1 - fraction)
        if changes:
            task = "shot_noise_s3" if changes[0].corruption == "shot_noise" else "stripe"
            if changes[0].severity != 3:
                raise ValueError("The recorded shot-noise task uses severity three.")
            weights[task] += q * fraction
    return weights


def weighted_task_score(task_metrics, weights):
    """Combine per-class sufficient statistics; fail on missing weighted cells."""
    accuracy, loss, mass = 0.0, 0.0, 0.0
    for task, vector in weights.items():
        vector = np.asarray(vector, dtype=float)
        if vector.shape != (10,) or not np.isfinite(vector).all() or (vector < 0).any():
            raise ValueError("Task weights must be nonnegative finite ten-vectors.")
        if vector.sum() and task not in task_metrics:
            raise ValueError("Missing evaluation task.")
        for digit, weight in enumerate(vector):
            if weight == 0:
                continue
            row = task_metrics[task]
            count = row[f"class_examples_{digit}"]
            right, loss_sum = row[f"class_correct_{digit}"], row[f"class_loss_sum_{digit}"]
            if count <= 0 or not 0 <= right <= count or not math.isfinite(loss_sum) or loss_sum < 0:
                raise ValueError("Invalid or missing weighted class statistics.")
            accuracy += weight * right / count
            loss += weight * loss_sum / count
        mass += float(vector.sum())
    if not math.isclose(mass, 1.0, rel_tol=0, abs_tol=1e-12):
        raise ValueError("Task weights must sum to one.")
    return {"accuracy": float(accuracy), "loss": float(loss)}


def primary_development_score(scenario_round_scores):
    """Draft equal-scenario/equal-round accuracy objective, rounds 5..24.

    Diagnostic controls are excluded; require complete coverage, never silently
    reward a failed/shortened candidate by averaging fewer observations.
    """
    if set(scenario_round_scores) != set(PRIMARY_SCENARIOS):
        raise ValueError("Primary objective requires exactly eight primary scenarios.")
    per_scenario = {}
    for name, values in scenario_round_scores.items():
        if set(values) != set(range(5, 25)):
            raise ValueError("Require every round from 5 through 24.")
        series = np.asarray([values[r] for r in range(5, 25)], dtype=float)
        if not np.isfinite(series).all() or (series < 0).any() or (series > 1).any():
            raise ValueError("Accuracy must be finite and in [0,1].")
        per_scenario[name] = float(series.mean())
    return {"accuracy": float(np.mean(list(per_scenario.values()))), "per_scenario": per_scenario}


class ContributionRecorder:
    """Measure a baseline without changing its resulting model or state.

    Effective coefficients act on raw signed deltas. Gram entries allow offline
    group contribution norms without feeding simulator groups to the defence.
    Projection shares can be negative or exceed one; they are not trust scores.
    """
    def __init__(self, base):
        self.base, self.name = base, base.name
        self.settings = {**copy.deepcopy(base.settings), "measurement": {
            "name": "contribution_recorder_v1", "implementation_sha256": file_sha256(__file__),
            "coefficient_convention": "effective_coefficients_on_raw_client_deltas",
            "gram_matrix": "applied_contributions", "timing": "base_and_recorder_separate"}}

    def reset(self):
        if hasattr(self.base, "reset"):
            self.base.reset()

    def aggregate(self, submissions, broadcast, round_number):
        started = time.perf_counter()
        result = self.base.aggregate(submissions, broadcast, round_number)
        base_seconds = time.perf_counter() - started
        return self._measure(submissions, broadcast, result, base_seconds)

    @torch.inference_mode()
    def _measure(self, submissions, broadcast, result, base_seconds):
        started = time.perf_counter()
        ordered = sorted(submissions, key=lambda s: s.client_id)
        if not ordered or len({s.client_id for s in ordered}) != len(ordered) or set(result.coefficients) != {s.client_id for s in ordered}:
            raise ValueError("Coefficients must cover distinct submissions exactly.")
        metrics = result.diagnostics.get("client_metrics", {})
        convention = result.diagnostics.get("coefficient_convention")
        if convention == "effective_coefficients_on_raw_client_deltas":
            coefficients = [result.coefficients[s.client_id] for s in ordered]
        elif self.base.name == "FLTrust":
            coefficients = [metrics[str(s.client_id)]["effective_update_coefficient"] for s in ordered]
        elif self.base.name in {"FedAvg", "RFA", "Viceroy"}:
            coefficients = [result.coefficients[s.client_id] for s in ordered]
        else:
            raise ValueError("Unspecified effective-coefficient contract.")
        coeff = torch.tensor(coefficients, dtype=torch.float64)
        if not torch.isfinite(coeff).all() or (coeff < 0).any():
            raise ValueError("Expected finite nonnegative applied coefficients.")
        points = torch.stack([flatten_delta(s.state, broadcast) for s in ordered])
        contributions = points * coeff[:, None]
        gram = contributions @ contributions.T
        raw_norms = torch.linalg.vector_norm(points, dim=1)
        norms = torch.linalg.vector_norm(contributions, dim=1)
        total = contributions.sum(0)
        total_norm = float(torch.linalg.vector_norm(total))
        norm_sum, coefficient_mass = float(norms.sum()), float(coeff.sum())
        # Prove the coefficients describe the actual update, including casting.
        max_error, offset = 0.0, 0
        for name, reference in broadcast.items():
            size = reference.numel()
            expected = (reference.double() + total[offset:offset + size].reshape(reference.shape)).to(reference.dtype)
            torch.testing.assert_close(expected, result.state[name], rtol=2e-6, atol=2e-8)
            max_error = max(max_error, float((expected.double() - result.state[name].double()).abs().max()))
            offset += size
        rows = {}
        for i, s in enumerate(ordered):
            rows[str(s.client_id)] = {**metrics.get(str(s.client_id), {}),
                "recorder_native_coefficient": float(result.coefficients[s.client_id]),
                "applied_delta_coefficient": float(coeff[i]), "raw_delta_l2_norm": float(raw_norms[i]),
                "applied_contribution_l2_norm": float(norms[i]),
                "coefficient_share": float(coeff[i]) / coefficient_mass if coefficient_mass else None,
                "contribution_norm_share": float(norms[i]) / norm_sum if norm_sum else None,
                "aggregate_projection_share": float(torch.dot(contributions[i], total)) / total_norm**2 if total_norm else None}
        diagnostics = {**result.diagnostics,
            "coefficient_convention": "effective_coefficients_on_raw_client_deltas", "client_metrics": rows,
            "contribution_measurements": {"client_order": [s.client_id for s in ordered],
                "gram_matrix": gram.tolist(), "sum_individual_norms": norm_sum,
                "aggregate_l2_norm": total_norm, "effective_coefficient_mass": coefficient_mass,
                "cancellation_ratio": total_norm / norm_sum if norm_sum else None,
                "zero_contribution_mass": norm_sum == 0.0,
                "zero_aggregate_from_cancellation": norm_sum > 0.0 and total_norm == 0.0,
                "reconstruction_max_abs_error": max_error,
                "defence_aggregation_seconds": base_seconds,
                "metric_recording_seconds": time.perf_counter() - started}}
        return AggregationResult(result.state, {s.client_id: float(coeff[i]) for i, s in enumerate(ordered)}, diagnostics)


def group_contribution_norm(measurements, clients):
    order = measurements["client_order"]
    clients = tuple(clients)
    if len(set(clients)) != len(clients) or any(c not in order for c in clients):
        raise ValueError("Unknown or duplicate group member.")
    gram = np.asarray(measurements["gram_matrix"], dtype=float)
    if gram.shape != (len(order), len(order)) or not np.isfinite(gram).all():
        raise ValueError("Invalid contribution Gram matrix.")
    positions = [order.index(c) for c in clients]
    squared = float(gram[np.ix_(positions, positions)].sum())
    tolerance = 1e-10 * max(1.0, float(np.abs(gram).sum()))
    if squared < -tolerance:
        raise ValueError("Negative squared group norm.")
    return math.sqrt(max(0.0, squared))


def validate_matched_runs(observed, reference):
    import json
    expected_observed = longer_scenarios().get(observed.get("scenario", {}).get("name"))
    if expected_observed is None or json.dumps(observed["scenario"], sort_keys=True) != json.dumps(expected_observed.to_dict(), sort_keys=True):
        raise ValueError("Unknown or altered observed scenario.")
    keys = ("config", "method", "partition_sha256", "selection_sha256", "initial_state_sha256",
            "code_sha256", "versions", "validation_tasks", "evaluation_recorder")
    for key in keys:
        if key not in observed or key not in reference or observed[key] != reference[key]:
            raise ValueError(f"Unmatched or missing run field: {key}")
    for key in ("official_test_used", "evaluation_data"):
        if observed.get(key) != reference.get(key):
            raise ValueError(f"Unmatched evaluation provenance: {key}")
    expected = matched_reference_name(observed["scenario"]["name"])
    if reference["scenario"] != longer_scenarios()[expected].to_dict():
        # Normalize tuples/lists from JSON round-trips before comparison.
        if json.dumps(reference["scenario"], sort_keys=True) != json.dumps(longer_scenarios()[expected].to_dict(), sort_keys=True):
            raise ValueError("Wrong no-attack control scenario.")
    return True


def _complete_series(values, horizon, unit_interval=True):
    if type(horizon) is not int or horizon < 1 or set(values) != set(range(1, horizon + 1)):
        raise ValueError("Require complete round-indexed observations, 1..horizon.")
    array = np.asarray([values[r] for r in range(1, horizon + 1)], dtype=float)
    if not np.isfinite(array).all() or (array < 0).any() or (unit_interval and (array > 1).any()):
        raise ValueError("Invalid finite nonnegative measurement series.")
    return array


def paired_damage(reference, observed, windows, horizon=24):
    """Signed pp damage and positive area on predeclared inclusive windows."""
    ref, obs = _complete_series(reference, horizon), _complete_series(observed, horizon)
    chosen = []
    previous = 0
    for start, end in windows:
        if type(start) is not int or type(end) is not int or start <= previous or end < start or end > horizon:
            raise ValueError("Windows must be ordered, non-overlapping and within the horizon.")
        chosen.extend(range(start, end + 1))
        previous = end
    if not chosen:
        raise ValueError("Choose a nonempty measurement window.")
    damage = (ref - obs) * 100
    values = damage[np.asarray(chosen) - 1]
    return {"rounds": chosen, "damage_pp": values.tolist(), "mean_signed_damage_pp": float(values.mean()),
            "positive_damage_area_pp_rounds": float(np.maximum(values, 0).sum()),
            "maximum_signed_damage_pp": float(values.max()), "last_window_damage_pp": float(values[-1])}


def coefficient_ratios(reference, observed, horizon=24, epsilon=1e-12):
    """Diagnostic ratios on applied coefficients, not delivered influence."""
    return _measurement_ratios(reference, observed, horizon, epsilon)


def contribution_norm_ratios(reference, observed, horizon=24, epsilon=1e-12):
    """Matched ratios of ||applied coefficient * raw delta|| for each client."""
    return _measurement_ratios(reference, observed, horizon, epsilon)


def _measurement_ratios(reference, observed, horizon, epsilon):
    ref = _complete_series(reference, horizon, False)
    obs = _complete_series(observed, horizon, False)
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("Epsilon must be positive and finite.")
    ratios = {r: float(obs[r-1] / ref[r-1]) if ref[r-1] > epsilon else None for r in range(1, horizon + 1)}
    if any(value is not None and not math.isfinite(value) for value in ratios.values()):
        raise FloatingPointError("Non-finite matched measurement ratio.")
    return ratios


def sustained_event(flags, start, end, consecutive=3, censor_reason="horizon"):
    """Earliest streak start and its later confirmation; undefined breaks streaks."""
    if type(start) is not int or type(end) is not int or start < 1 or end < start - 1:
        raise ValueError("Invalid eligible interval.")
    if type(consecutive) is not int or consecutive < 1 or any(r not in flags for r in range(start, end + 1)):
        raise ValueError("Incomplete flags or invalid streak length.")
    if any(flags[r] is not None and type(flags[r]) is not bool for r in range(start, end + 1)):
        raise ValueError("Flags must be booleans or undefined.")
    streak = 0
    for r in range(start, end + 1):
        value = flags[r]
        streak = streak + 1 if value is True else 0
        if streak == consecutive:
            return {"streak_start_round": r - consecutive + 1, "confirmation_round": r,
                    "right_censored": False, "censor_reason": None,
                    "eligible_rounds": end - start + 1,
                    "undefined_rounds": sum(flags[k] is None for k in range(start, end + 1))}
    return {"streak_start_round": None, "confirmation_round": None,
            "right_censored": True, "censor_reason": censor_reason,
            "eligible_rounds": end - start + 1,
            "undefined_rounds": sum(flags[k] is None for k in range(start, end + 1))}


def _recovery(flags, attack_end, next_attack, horizon, consecutive):
    if type(attack_end) is not int or not 1 <= attack_end <= horizon:
        raise ValueError("Attack end is outside the run.")
    if next_attack is not None and (type(next_attack) is not int or not attack_end < next_attack <= horizon):
        raise ValueError("Next attack must occur after this attack ends.")
    end = next_attack - 1 if next_attack is not None else horizon
    result = sustained_event(flags, attack_end + 1, end, consecutive,
                             "next_attack" if next_attack is not None else "horizon")
    result.update({"attack_end_round": attack_end, "last_eligible_round": end,
                   "rounds_to_streak_start": result["streak_start_round"] - attack_end if not result["right_censored"] else None,
                   "rounds_to_confirmation": result["confirmation_round"] - attack_end if not result["right_censored"] else None,
                   "criterion_met_at_attack_end": flags[attack_end]})
    return result


def model_recovery(reference, observed, attack_end, next_attack=None, horizon=24, tolerance_pp=1.0, consecutive=3):
    ref, obs = _complete_series(reference, horizon), _complete_series(observed, horizon)
    if not math.isfinite(tolerance_pp) or not 0 <= tolerance_pp <= 100:
        raise ValueError("Tolerance must be in percentage points, 0..100.")
    flags = {r: bool(obs[r-1] >= ref[r-1] - tolerance_pp / 100 - 1e-12) for r in range(1, horizon + 1)}
    return _recovery(flags, attack_end, next_attack, horizon, consecutive)


def influence_recovery(reference, observed, attack_end, next_attack=None, horizon=24, minimum_ratio=.8, consecutive=3):
    """Historical coefficient-restoration diagnostic; see contribution_restoration_event."""
    if not math.isfinite(minimum_ratio) or minimum_ratio <= 0:
        raise ValueError("Influence recovery ratio must be positive.")
    ratios = coefficient_ratios(reference, observed, horizon)
    flags = {r: None if v is None else bool(v >= minimum_ratio - 1e-12) for r, v in ratios.items()}
    return _recovery(flags, attack_end, next_attack, horizon, consecutive)


def suppression_event(reference, observed, start, end, horizon=24, maximum_ratio=.5, consecutive=3):
    """Historical coefficient-reduction diagnostic; see contribution_reduction_event."""
    if not math.isfinite(maximum_ratio) or maximum_ratio <= 0 or end > horizon:
        raise ValueError("Invalid suppression threshold or interval.")
    ratios = coefficient_ratios(reference, observed, horizon)
    flags = {r: None if v is None else bool(v < maximum_ratio) for r, v in ratios.items()}
    result = sustained_event(flags, start, end, consecutive, "window_end")
    result["rounds_from_onset_to_confirmation"] = result["confirmation_round"] - start + 1 if not result["right_censored"] else None
    result["suppressed_client_rounds"] = sum(flags[r] is True for r in range(start, end + 1))
    result["defined_client_rounds"] = sum(flags[r] is not None for r in range(start, end + 1))
    return result


def contribution_reduction_event(reference_norms, observed_norms, start, end,
                                 horizon=24, maximum_ratio=.5, consecutive=3):
    """Descriptive drop in the actual applied update norm; not proof of error."""
    if not math.isfinite(maximum_ratio) or maximum_ratio <= 0 or end > horizon:
        raise ValueError("Invalid reduction threshold or interval.")
    ratios = contribution_norm_ratios(reference_norms, observed_norms, horizon)
    flags = {r: None if v is None else bool(v < maximum_ratio) for r, v in ratios.items()}
    result = sustained_event(flags, start, end, consecutive, "window_end")
    result["rounds_from_onset_to_confirmation"] = result["confirmation_round"] - start + 1 if not result["right_censored"] else None
    result["reduced_client_rounds"] = sum(flags[r] is True for r in range(start, end + 1))
    result["defined_client_rounds"] = sum(flags[r] is not None for r in range(start, end + 1))
    return result


def contribution_restoration_event(reference_norms, observed_norms, event_end,
                                   next_event=None, horizon=24, minimum_ratio=.8, consecutive=3):
    """Descriptive norm restoration, with the proposal's three-round streak."""
    if not math.isfinite(minimum_ratio) or minimum_ratio <= 0:
        raise ValueError("Restoration threshold must be positive.")
    ratios = contribution_norm_ratios(reference_norms, observed_norms, horizon)
    flags = {r: None if v is None else bool(v >= minimum_ratio - 1e-12) for r, v in ratios.items()}
    return _recovery(flags, event_end, next_event, horizon, consecutive)


def _classify_episode(flags, event_start, event_end, next_event, horizon, consecutive, measure):
    if type(event_start) is not int or not 1 <= event_start <= event_end:
        raise ValueError("Event window must start within the run.")
    streak = _recovery(flags, event_end, next_event, horizon, consecutive)
    event_flags = [flags[r] for r in range(event_start, event_end + 1)]
    impairment = any(v is False for v in event_flags)
    undefined_event = sum(v is None for v in event_flags)
    post_flags = [flags[r] for r in range(event_end + 1, streak["last_eligible_round"] + 1)]
    delayed_impairment = any(v is False for v in post_flags)
    if not impairment and not undefined_event:
        if delayed_impairment:
            status = "delayed_post_event_impairment"
        elif post_flags and all(v is None for v in post_flags):
            status = "unassessable"
        else:
            status = "no_qualifying_impairment_observed"
    elif not impairment and undefined_event:
        status = "unassessable"
    elif flags[event_end] is True:
        status = "within_criterion_at_event_end"
    elif not streak["right_censored"]:
        status = "post_event_restored"
    elif streak["eligible_rounds"] == streak["undefined_rounds"]:
        status = "unassessable"
    else:
        status = "restoration_not_observed_before_censoring"
    return {"measurement": measure, "event_start_round": event_start, "event_end_round": event_end,
            "status": status, "qualifying_impairment_observed_during_event": impairment,
            "delayed_post_event_impairment_observed": delayed_impairment,
            "criterion_met_at_event_end": flags[event_end], "undefined_event_rounds": undefined_event,
            "observed_restoration_delay_rounds": streak["rounds_to_confirmation"] if status == "post_event_restored" else None,
            "post_event_qualifying_streak": streak}


def classify_model_episode(reference, observed, event_start, event_end, next_event=None,
                           horizon=24, tolerance_pp=1.0, consecutive=3):
    """Separate an observed return from already satisfying the proposal criterion."""
    ref, obs = _complete_series(reference, horizon), _complete_series(observed, horizon)
    if not math.isfinite(tolerance_pp) or not 0 <= tolerance_pp <= 100:
        raise ValueError("Tolerance must be 0..100 percentage points.")
    flags = {r: bool(obs[r-1] >= ref[r-1] - tolerance_pp / 100 - 1e-12) for r in range(1, horizon + 1)}
    return _classify_episode(flags, event_start, event_end, next_event, horizon, consecutive, "model_accuracy")


def classify_contribution_episode(reference_norms, observed_norms, event_start, event_end,
                                  next_event=None, horizon=24, minimum_ratio=.8, consecutive=3):
    """Use the same episode statuses for a client's delivered contribution norm."""
    if not math.isfinite(minimum_ratio) or minimum_ratio <= 0:
        raise ValueError("Restoration threshold must be positive.")
    ratios = contribution_norm_ratios(reference_norms, observed_norms, horizon)
    flags = {r: None if v is None else bool(v >= minimum_ratio - 1e-12) for r, v in ratios.items()}
    return _classify_episode(flags, event_start, event_end, next_event, horizon, consecutive,
                             "applied_contribution_l2_norm")
