"""Predeclared, offline exploratory screening of reference-free observations.

Simulator events are read only here to label already-recorded updates. This
module is not imported by an aggregator and never changes a training decision.
Only the standard library is needed; the statistical unit is not a client-round.
"""
from collections import defaultdict
import copy
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REGIMES = ("IID", "non-IID")
METHODS = ("FedAvg", "NNM-RFA")
SCENARIOS = ("clean", "abrupt_shot_noise", "gradual_stripe", "label_flip")
SEEDS = (20261014, 20261015)
HONEST_GROUPS = ("clean", "noise_changed", "noise_unchanged", "stripe_changed",
                 "stripe_unchanged", "label_honest")
# Order is also the predeclared tie-breaker. No direction is fitted to labels.
SCORE_SPECS = (
    ("whole_cosine", "whole_cosine", True),
    ("penultimate_cosine", "penultimate_cosine", True),
    ("head_cosine", "head_cosine", True),
    ("head_class_cosine_mean", "head_class_cosine_mean", True),
    ("head_class_cosine_min", "head_class_cosine_min", True),
    ("layer_disagreement", "layer_disagreement", False),
    ("temporal_cosine", "temporal_cosine", True),
    ("temporal_relative_change", "temporal_relative_change", False),
)
BASELINE = SCORE_SPECS[0][0]
CHECK_SLICES = (("FedAvg", 20261015), ("NNM-RFA", 20261014), ("NNM-RFA", 20261015))

DIAGNOSTIC_PROTOCOL = {
    "version": "reference_free_observation_screen_v1",
    "status": "exploratory_predeclared_mechanism_screen",
    "expected_runs": 32,
    "methods": list(METHODS), "regimes": list(REGIMES),
    "scenarios": list(SCENARIOS), "seeds": list(SEEDS),
    "client_ids": list(range(10)), "recorded_rounds": [1, 24], "analysis_rounds": [5, 24],
    "scores_in_fixed_order": [{"name": name, "feature": feature,
                               "formula": "1 - feature" if invert else "feature"}
                              for name, feature, invert in SCORE_SPECS],
    "unused_observation": "head_relative_norm is recorded but is not a candidate score",
    "calibration": {
        "method": "FedAvg", "seed": 20261014,
        "honest_groups_per_regime": list(HONEST_GROUPS),
        "threshold": "maximum of honest regime/subgroup order-statistic thresholds",
        "order_statistic": "sorted_scores[n - floor(n / 20) - 1] in each nonempty group",
        "decision": "flag if score > threshold (equality is not flagged)",
        "maximum_empirical_group_fpr": 0.05,
        "selection": "one alternative with maximum minimum attack recall over the two regimes",
        "ties": "first alternative in scores_in_fixed_order; no test-slice tie-breaking",
    },
    "checks": [{"method": method, "seed": seed} for method, seed in CHECK_SLICES],
    "gate": {
        "all_check_slices_must_pass": True, "maximum_honest_group_fpr": 0.05,
        "minimum_attack_recall_in_each_regime": 0.80,
        "minimum_worst_regime_recall_improvement": 0.10,
        "improvement_definition": "min(candidate recall over regimes) - min(baseline recall over regimes)",
        "baseline_threshold": "whole_cosine calibrated separately using the identical honest-FPR rule",
        "no_recalibration": True,
        "meaning": "go/no-go for further research; not proof of an accurate deployable defence",
    },
    "undefined_features": "Missing, nonnumeric, nonfinite or out-of-domain features are undefined. Any undefined calibration score disqualifies that candidate; any undefined selected/baseline check score fails that slice. No imputation or best-subset reporting.",
    "zero_norms": "A mathematically undefined cosine must be recorded as null, never fabricated as zero or one. Temporal warmup is outside rounds 5-24.",
    "labels": "Active events in each saved run.json, used exclusively offline; no hard-coded client identities or assumed clean warmup in a defence.",
    "inference": "Descriptive correlated client-round counts only; no binomial intervals, IID significance tests or independent-replication claims. Seeds and data were used in earlier development. NNM-RFA seed14 is trajectory transfer on the calibration seed, not independent validation.",
    "scope": "Only clean, noise, stripe and persistent label flips in two existing partitions; no recovery, recurring/adaptive attack, unseen-domain or general-security conclusion.",
    "common_broadcast_probe": {
        "purpose": "Descriptive evaluation of already-selected score and frozen thresholds only; cannot rescue a failed gate. Contradictions preclude a strong separability claim.",
        "rounds": [5, 8, 12, 20],
        "temporal_warning": "Independent probes have no client history. A selected temporal score is unevaluable; do not substitute a non-temporal score.",
        "descriptive_contradiction": "Selected-score FPR above 5% in any honest subgroup or attack recall below 80% in any seed/regime slice requires investigation; not an additional gate or significance test.",
    },
}


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def score_value(features, spec):
    """Return None for undefined observations, preserving missingness explicitly."""
    _, feature, invert = spec
    value = features.get(feature)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    value = float(value)
    if invert:
        if value < -1.000001 or value > 1.000001:
            return None
        return 1.0 - min(1.0, max(-1.0, value))
    return value if value >= 0 else None


def conservative_threshold(values):
    """Smallest observed cutoff allowing at most floor(n/20) strict exceedances."""
    values = list(values)
    if not values or any(v is None or not math.isfinite(v) for v in values):
        raise ValueError("Threshold calibration requires nonempty finite scores.")
    values.sort()
    return float(values[len(values) - len(values) // 20 - 1])


def event_group(scenario, client, round_number):
    """Labels depend on active recorded events, not on client IDs or a catalogue."""
    active = [event for event in scenario["events"] if client in event["clients"]
              and any(start <= round_number <= end for start, end in event["windows"])]
    if any(event["intent"] == "malicious" for event in active):
        return "attack", True
    changed = any(event["kind"] == "input_change" and event["intent"] == "honest" for event in active)
    name = scenario["name"]
    if name == "clean":
        return "clean", False
    if name == "abrupt_shot_noise":
        return "noise_changed" if changed else "noise_unchanged", False
    if name == "gradual_stripe":
        return "stripe_changed" if changed else "stripe_unchanged", False
    if name == "label_flip":
        return "label_honest", False
    raise ValueError(f"Unregistered diagnostic scenario: {name}")


def load_observations(records):
    """Require the full planned grid and every client/round; never silently drop runs."""
    expected = set(itertools.product(METHODS, REGIMES, SEEDS, SCENARIOS))
    seen, observations, inputs = set(), [], []
    for record in records:
        job = record["job"]
        key = (job["method"], job["regime"], job["seed"], job["scenario"])
        if key not in expected or key in seen or record.get("status") != "complete":
            raise ValueError(f"Unexpected, duplicate or incomplete diagnostic run: {key}")
        seen.add(key)
        run_path = ROOT / record["run_path"]
        metadata = json.loads((run_path / "run.json").read_text())
        scenario = metadata["scenario"]
        if (scenario["name"] != job["scenario"]
                or metadata["config"]["development_seed"] != job["seed"]
                or metadata["config"]["rounds"] != 24 or metadata.get("official_test_used") is not False):
            raise ValueError(f"Diagnostic metadata mismatch or official-test access: {key}")
        diagnostics = json.loads((run_path / "aggregation_diagnostics.json").read_text())
        rounds_seen = set()
        for item in diagnostics:
            round_number = item["round"]
            if type(round_number) is not int or round_number not in range(1, 25) or round_number in rounds_seen:
                raise ValueError(f"Missing, duplicate or invalid round: {key}, {round_number}")
            rounds_seen.add(round_number)
            features = item["values"]["update_observation"]["feature_rows"]
            if set(features) != {str(i) for i in range(10)}:
                raise ValueError(f"Diagnostic client coverage mismatch: {key}, {round_number}")
            for client in range(10):
                group, attack = event_group(scenario, client, round_number)
                observations.append({"method": key[0], "regime": key[1], "seed": key[2],
                                     "scenario": key[3], "round": round_number, "client": client,
                                     "group": group, "attack": attack,
                                     "scores": {spec[0]: score_value(features[str(client)], spec)
                                                for spec in SCORE_SPECS}})
        if rounds_seen != set(range(1, 25)):
            raise ValueError(f"Incomplete rounds in diagnostic run: {key}")
        inputs.append({"job": copy.deepcopy(job), "run_path": str(run_path),
                       "run_sha256": _hash(run_path / "run.json"),
                       "diagnostics_sha256": _hash(run_path / "aggregation_diagnostics.json")})
    if seen != expected:
        raise ValueError(f"Incomplete diagnostic grid: {len(seen)} of {len(expected)} runs")
    return observations, inputs


def calibrate_scores(observations):
    calibration = [row for row in observations if row["method"] == "FedAvg"
                   and row["seed"] == SEEDS[0] and 5 <= row["round"] <= 24]
    honest = defaultdict(list)
    attacks = defaultdict(list)
    for row in calibration:
        if row["attack"]:
            attacks[row["regime"]].append(row)
        else:
            honest[(row["regime"], row["group"])].append(row)
    if set(honest) != set(itertools.product(REGIMES, HONEST_GROUPS)) or set(attacks) != set(REGIMES):
        raise ValueError("Calibration lacks an expected honest subgroup or attack regime.")
    result = {}
    for spec in SCORE_SPECS:
        name = spec[0]
        missing = sum(row["scores"][name] is None for row in calibration)
        if missing:
            result[name] = {"eligible": False, "undefined_calibration_rows": missing,
                            "threshold": None, "minimum_attack_recall": None}
            continue
        group_thresholds = [{"regime": regime, "group": group, "n": len(rows),
                             "threshold": conservative_threshold([r["scores"][name] for r in rows])}
                            for (regime, group), rows in sorted(honest.items())]
        threshold = max(group["threshold"] for group in group_thresholds)
        recall = {regime: sum(row["scores"][name] > threshold for row in rows) / len(rows)
                  for regime, rows in attacks.items()}
        result[name] = {"eligible": True, "undefined_calibration_rows": 0,
                        "threshold": threshold, "honest_group_cutoffs": group_thresholds,
                        "attack_recall_by_regime": recall, "minimum_attack_recall": min(recall.values())}
    alternatives = [spec[0] for spec in SCORE_SPECS[1:] if result[spec[0]]["eligible"]]
    selected = max(alternatives, key=lambda name: result[name]["minimum_attack_recall"]) if alternatives else None
    return result, selected


def group_metrics(observations, calibration):
    groups = defaultdict(list)
    for row in observations:
        if 5 <= row["round"] <= 24:
            groups[(row["method"], row["seed"], row["regime"], row["scenario"], row["group"])].append(row)
    result = []
    for key, rows in sorted(groups.items()):
        for spec in SCORE_SPECS:
            name = spec[0]
            threshold = calibration[name]["threshold"]
            valid = [row for row in rows if row["scores"][name] is not None]
            flagged = sum(row["scores"][name] > threshold for row in valid) if threshold is not None else None
            attack = rows[0]["attack"]
            result.append(dict(zip(("method", "seed", "regime", "scenario", "group"), key),
                               score=name, threshold=threshold, attack=attack,
                               phase="calibration" if key[:2] == ("FedAvg", SEEDS[0]) else "check",
                               n_rows=len(rows), n_valid=len(valid), n_undefined=len(rows)-len(valid),
                               n_clients=len({row["client"] for row in rows}), n_rounds=len({row["round"] for row in rows}),
                               flagged=flagged,
                               rate=flagged/len(valid) if valid and flagged is not None else None,
                               rate_name="attack_recall" if attack else "honest_false_positive_rate"))
    return result


def evaluate_gate(metrics, calibration, selected):
    checks = []
    for method, seed in CHECK_SLICES:
        reasons, details = [], {}
        if selected is None or not calibration[BASELINE]["eligible"]:
            reasons.append("No eligible selected score or whole-cosine baseline from calibration")
        else:
            for name in (selected, BASELINE):
                rows = [r for r in metrics if r["method"] == method and r["seed"] == seed and r["score"] == name]
                honest = [r for r in rows if not r["attack"]]
                attacks = {r["regime"]: r for r in rows if r["attack"]}
                complete = (len(honest) == 12 and set(attacks) == set(REGIMES)
                            and all(r["n_undefined"] == 0 and r["rate"] is not None for r in rows))
                details[name] = {"complete": complete,
                                 "worst_honest_group_fpr": max((r["rate"] for r in honest if r["rate"] is not None), default=None),
                                 "attack_recall_by_regime": {regime: row["rate"] for regime, row in attacks.items()}}
                if not complete:
                    reasons.append(f"Undefined scores or incomplete groups for {name}")
            candidate, baseline = details[selected], details[BASELINE]
            if candidate["complete"] and baseline["complete"]:
                # Integer comparisons give the exact predeclared 5% and 80% bounds.
                selected_rows = [r for r in metrics if r["method"] == method and r["seed"] == seed and r["score"] == selected]
                if any(r["flagged"] * 20 > r["n_rows"] for r in selected_rows if not r["attack"]):
                    reasons.append("Selected score exceeds 5% FPR in an honest subgroup")
                if any(r["flagged"] * 5 < r["n_rows"] * 4 for r in selected_rows if r["attack"]):
                    reasons.append("Selected score has attack recall below 80% in a regime")
                improvement = min(candidate["attack_recall_by_regime"].values()) - min(baseline["attack_recall_by_regime"].values())
                details["worst_regime_recall_improvement"] = improvement
                details["paired_regime_recall_improvement"] = {regime: candidate["attack_recall_by_regime"][regime] - baseline["attack_recall_by_regime"][regime] for regime in REGIMES}
                if improvement < .10 - 1e-12:
                    reasons.append("Worst-regime recall improvement over baseline is below 10 percentage points")
        checks.append({"method": method, "seed": seed, "pass": not reasons, "reasons": reasons, "metrics": details})
    passed = all(item["pass"] for item in checks)
    return {"pass": passed, "decision": "go_to_further_research" if passed else "no_go_for_this_fixed_score_screen",
            "checks": checks, "interpretation": "Exploratory gate only; failure does not rule out all mechanisms, and passing does not establish general separation or defence performance."}


def analyse_diagnostics(records, output_dir):
    """Read all 32 completed runs; emit a frozen-rule summary and every score/group."""
    observations, inputs = load_observations(records)
    calibration, selected = calibrate_scores(observations)
    metrics = group_metrics(observations, calibration)
    summary = {"protocol": copy.deepcopy(DIAGNOSTIC_PROTOCOL),
               "analysis_source_sha256": _hash(__file__), "inputs": inputs,
               "n_runs": len(inputs), "n_recorded_client_rounds": len(observations),
               "n_analysis_client_rounds": sum(5 <= r["round"] <= 24 for r in observations),
               "calibration": calibration, "selected_score": selected,
               "baseline_score": BASELINE, "gate": evaluate_gate(metrics, calibration, selected),
               "pre_event": {"recorded_rounds": [1, 4], "used_for_selection_or_gate": False},
               "evidence_limits": DIAGNOSTIC_PROTOCOL["inference"]}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "diagnostic_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n")
    with (output_dir / "diagnostic_groups.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    return summary


def _verify_artifacts(directory, manifest, required):
    directory = Path(directory).resolve()
    hashes = manifest.get("artifact_sha256", {})
    if not set(required).issubset(hashes):
        raise ValueError("Probe manifest lacks required artifact fingerprints.")
    for name, digest in hashes.items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory) or _hash(path) != digest:
            raise ValueError(f"Probe artifact fingerprint changed: {name}")


def _probe_key(row):
    return tuple(row[name] for name in ("regime", "seed", "round", "scenario", "client"))


def _probe_index(rows, expected, label):
    indexed = {}
    for row in rows:
        key = _probe_key(row)
        if key in indexed or key not in expected:
            raise ValueError(f"Duplicate or unexpected {label} probe key: {key}")
        indexed[key] = row
    if set(indexed) != expected:
        raise ValueError(f"Incomplete {label} probe coverage: {len(indexed)} of {len(expected)}")
    return indexed


def analyse_probe(probe_dir, diagnostic_summary, output_dir):
    """Score matched-broadcast probes without fitting, substituting or gating anew.

    Joins the separately saved simulator truth only after reading observations.
    Probe artifacts and their constituent blocks must all match saved hashes.
    ``diagnostic_summary`` is an analysis dictionary or its saved JSON path.
    """
    probe_dir = Path(probe_dir).resolve()
    if isinstance(diagnostic_summary, (str, Path)):
        diagnostic_summary = json.loads(Path(diagnostic_summary).read_text())
    frozen = copy.deepcopy(diagnostic_summary)
    if frozen["protocol"]["version"] != DIAGNOSTIC_PROTOCOL["version"]:
        raise ValueError("Diagnostic summary uses an unrecognised protocol.")
    selected = frozen["selected_score"]
    specs = {spec[0]: spec for spec in SCORE_SPECS}
    if selected is not None and selected not in {spec[0] for spec in SCORE_SPECS[1:]}:
        raise ValueError("Unregistered selected probe score.")
    manifest = json.loads((probe_dir / "summary.json").read_text())
    if (manifest["status"] != "complete" or manifest["local_trainings"] != 640
            or manifest["feature_rows"] != 640 or manifest["official_test_used"] is not False
            or manifest["temporal_features_usable"] is not False
            or manifest["reference_forward_examples"] != 0 or manifest["evaluation_forward_examples"] != 0):
        raise ValueError("Probe summary differs from the declared completed observational study.")
    required = ("features.json", "features.csv", "offline_truth.json", "offline_truth.csv")
    _verify_artifacts(probe_dir, manifest, required)
    rounds = DIAGNOSTIC_PROTOCOL["common_broadcast_probe"]["rounds"]
    expected_blocks = set(itertools.product(REGIMES, SEEDS, rounds))
    seen_blocks, block_features, block_truth = set(), [], []
    for block in manifest["blocks"]:
        directory = (probe_dir / block["directory"]).resolve()
        if not directory.is_relative_to(probe_dir) or _hash(directory / "result.json") != block["result_sha256"]:
            raise ValueError("Probe block result fingerprint changed.")
        result = json.loads((directory / "result.json").read_text())
        provenance = result["provenance"]
        key = (provenance["regime"], provenance["seed"], provenance["round"])
        if (key in seen_blocks or key not in expected_blocks or result["status"] != "complete"
                or provenance != block["provenance"] or result["local_trainings"] != 40
                or result["official_test_used"] is not False or result["clean_replay_identical"] is not True
                or result["temporal_features_usable"] is not False
                or result["reference_forward_examples"] != 0 or result["evaluation_forward_examples"] != 0):
            raise ValueError("Probe block metadata or coverage differs.")
        seen_blocks.add(key)
        _verify_artifacts(directory, result, (*required, "training.json", "diagnostics.json"))
        block_features.extend(json.loads((directory / "features.json").read_text()))
        block_truth.extend(json.loads((directory / "offline_truth.json").read_text()))
    if seen_blocks != expected_blocks:
        raise ValueError("Probe needs all 16 declared checkpoint blocks.")
    expected = set(itertools.product(REGIMES, SEEDS, rounds, SCENARIOS, range(10)))
    features = _probe_index(json.loads((probe_dir / "features.json").read_text()), expected, "feature")
    truth = _probe_index(json.loads((probe_dir / "offline_truth.json").read_text()), expected, "truth")
    if features != _probe_index(block_features, expected, "block feature") or truth != _probe_index(block_truth, expected, "block truth"):
        raise ValueError("Combined probe artifacts differ from their constituent blocks.")
    groups = defaultdict(list)
    for key, row in features.items():
        label = truth[key]
        if row.get("broadcast_state_sha256") != label.get("broadcast_state_sha256"):
            raise ValueError("Probe observation and truth broadcast fingerprints differ.")
        attack, changed = label["is_attack"], label["is_honest_change"]
        if type(attack) is not bool or type(changed) is not bool or (attack and changed):
            raise ValueError("Invalid offline probe intent labels.")
        name = row["scenario"]
        if attack:
            group = "attack"
        elif name == "clean":
            group = "clean"
        elif name == "abrupt_shot_noise":
            group = "noise_changed" if changed else "noise_unchanged"
        elif name == "gradual_stripe":
            group = "stripe_changed" if changed else "stripe_unchanged"
        else:
            group = "label_honest"
        if any(row["features"].get(name) is not None for name in ("temporal_cosine", "temporal_relative_change")):
            raise ValueError("Independent probe acquired temporal history.")
        groups[(row["seed"], row["regime"], name, group, attack)].append(row)
    scored = []
    chosen = [BASELINE] + ([selected] if selected is not None else [])
    for name in chosen:
        threshold = frozen["calibration"][name]["threshold"]
        if threshold is not None and (isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not math.isfinite(threshold)):
            raise ValueError("Invalid frozen probe threshold.")
        for key, rows in sorted(groups.items()):
            values = [score_value(row["features"], specs[name]) for row in rows]
            valid = [value for value in values if value is not None]
            flagged = sum(value > threshold for value in valid) if threshold is not None else None
            scored.append({"seed": key[0], "regime": key[1], "scenario": key[2], "group": key[3],
                           "attack": key[4], "score": name, "role": "baseline" if name == BASELINE else "selected",
                           "threshold": threshold, "n_rows": len(values), "n_valid": len(valid),
                           "n_undefined": len(values)-len(valid), "n_clients": len({r["client"] for r in rows}),
                           "n_checkpoints": len({r["round"] for r in rows}), "flagged": flagged,
                           "rate": flagged/len(valid) if valid and flagged is not None else None,
                           "rate_name": "attack_recall" if key[4] else "honest_false_positive_rate"})
    temporal = selected in {"temporal_cosine", "temporal_relative_change"}
    selected_rows = [row for row in scored if row["role"] == "selected"]
    evaluable = bool(selected_rows) and all(row["n_undefined"] == 0 and row["rate"] is not None for row in selected_rows)
    slices = []
    for regime, seed in itertools.product(REGIMES, SEEDS):
        rows = [r for r in selected_rows if r["regime"] == regime and r["seed"] == seed]
        complete = (len(rows) == 7 and len([r for r in rows if r["attack"]]) == 1
                    and all(r["n_undefined"] == 0 and r["rate"] is not None for r in rows))
        contradictions = []
        if complete:
            if any(r["flagged"] * 20 > r["n_rows"] for r in rows if not r["attack"]):
                contradictions.append("Selected-score honest subgroup FPR exceeds 5%")
            if any(r["flagged"] * 5 < r["n_rows"] * 4 for r in rows if r["attack"]):
                contradictions.append("Selected-score attack recall is below 80%")
        slices.append({"regime": regime, "seed": seed, "evaluable": complete,
                       "descriptive_contradiction": bool(contradictions) if complete else None,
                       "reasons": contradictions})
    evaluable = evaluable and all(row["evaluable"] for row in slices)
    contradiction = any(row["descriptive_contradiction"] is True for row in slices) if evaluable else None
    summary = {"protocol_version": DIAGNOSTIC_PROTOCOL["version"],
               "analysis_source_sha256": _hash(__file__), "probe_manifest_sha256": _hash(probe_dir / "summary.json"),
               "probe_artifact_sha256": manifest["artifact_sha256"], "n_observations": len(features),
               "selected_score": selected, "selected_score_evaluable": evaluable,
               "selected_score_status": "temporal_unevaluable_no_history" if temporal else "evaluable" if evaluable else "unevaluable_missing_score_or_observations",
               "thresholds": {name: frozen["calibration"][name]["threshold"] for name in chosen},
               "trajectory_gate_pass": frozen["gate"]["pass"], "trajectory_gate_modified": False,
               "descriptive_probe_contradiction": contradiction, "selected_score_slices": slices,
               "groups": scored,
               "interpretation": "Frozen selected score and whole-cosine baseline only; no fitting, replacement, threshold change or gate rescue. Independent common broadcasts remove current global-model trajectory differences, not all uncertainty. Checkpoints and client observations are correlated, and seeds/data were used previously. A temporal selected score has no usable history here. A contradiction requires investigation before claiming reliable separation."}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "probe_score_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n")
    with (output_dir / "probe_score_groups.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(scored[0]))
        writer.writeheader()
        writer.writerows(scored)
    return summary
