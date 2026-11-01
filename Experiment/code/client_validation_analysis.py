"""Offline, fixed-rule study of client-local validation reports.

Only this analysis reads scenario intent and constructs adversarial reports.
The score_reports API receives bounded reports and IDs, never simulator truth,
audit data, source labels, or a trusted validation dataset.
"""
from collections import defaultdict
import copy
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[1]
RULES = ("peer_median", "peer_top3_mean", "peer_fourth_highest")
PROFILES = ("truthful", "directional_extreme", "sign_reversal", "corrupted_labels")
REGIMES = ("IID", "non-IID")
SEEDS = (20261014, 20261015)
ROUNDS = (5, 8, 12, 20)
SCENARIOS = ("clean", "abrupt_shot_noise", "gradual_stripe", "label_flip",
             "attack_stops", "mixed_stripe_attack")
GROUPS_BY_SCENARIO = {
    "clean": ("clean",),
    "abrupt_shot_noise": ("changed", "unchanged"),
    "gradual_stripe": ("changed", "unchanged"),
    "label_flip": ("poisoned", "unchanged"),
    "attack_stops": ("poisoned", "repaired", "unchanged"),
    "mixed_stripe_attack": ("poisoned", "changed", "unchanged"),
}

DIAGNOSTIC_PROTOCOL = {
    "version": "client_local_validation_fixed_controls_v1",
    "status": "exploratory_multiple_fixed_controls_no_heldout_validation_claim",
    "expected_cases": 96, "regimes": list(REGIMES), "seeds": list(SEEDS),
    "rounds": list(ROUNDS), "scenarios": list(SCENARIOS), "clients": list(range(10)),
    "candidate_model": "Common clean FedAvg broadcast w + 0.1 * submitted local delta; no aggregate is installed or advanced.",
    "reports": {
        "metric": "Mean baseline Brier loss minus candidate Brier loss; Brier = 0.5 * sum_class(probability - target_indicator)^2",
        "range": [-1.0, 1.0], "examples_per_validator": 128,
        "audit_examples_per_validator": 128,
        "holdout": "Client-assigned indices disjoint from both seeds' 512 training selections, global development/reference reserves, and the other local holdout role.",
        "self_exclusion": "Each candidate is scored using exactly the other nine validators.",
        "class_statistics": "Diagnostic only; no class reweighting, classifier fitting or class-statistic input to a rule. Empty classes remain undefined.",
    },
    "rules_in_fixed_order": list(RULES),
    "rule_definitions": {
        "peer_median": "Median of the nine peer gains",
        "peer_top3_mean": "Mean of the three highest peer gains; intentionally minority-sensitive and vulnerable to lying support",
        "peer_fourth_highest": "Fourth-highest peer gain, using fixed f=3; a positive score needs at least one honest supporting peer under this report budget",
    },
    "decision": "Accept iff score > 0; equality is not accepted. No fitted threshold or calibration selection.",
    "profiles": {
        "truthful": "Every peer reports the correct-target local gain",
        "directional_extreme": "Bad validators report +1 for an active poisoned candidate and -1 for every other candidate",
        "sign_reversal": "Bad validators negate their correct-target local gain",
        "corrupted_labels": "Bad validators report the gain computed with y mapped to 9-y on the same predictions",
    },
    "reporting_cohorts": {
        "clean_noise_stripe": [7, 8, 9],
        "label_flip_and_mixed": [0, 1, 2],
        "attack_stops": "Clients 0,1,2 at checkpoints 5 and 8; no malicious reporting at checkpoints 12 and 20",
        "intent_distinction": "Reporting-only malicious actors still submit honest-intent updates; report lies and update intent are separate labels.",
    },
    "primary_gate": {
        "minimum_honest_retention": .95, "minimum_poison_rejection": .80,
        "minimum_repaired_readmission": .80,
        "grouping": "Every regime, seed, scenario, relevant update group and fixed report profile separately; repaired updates have their own 80% criterion rather than being pooled with unchanged updates.",
        "all_conditions_required_for_each_rule": True,
        "selection": "Go only if at least one fixed rule passes the complete grid; first in rules_in_fixed_order if multiple pass. No post-hoc score, threshold or profile selection.",
        "interpretation": "A gate for a separately declared follow-up experiment, not deployment, novelty or statistical superiority.",
    },
    "audit": {
        "access": "Independent correct-target current-domain audit images are offline evaluation only and never enter a score or gate.",
        "self_useful": "Among honest-intent updates, report retention when the candidate's own audit Brier gain is strictly positive.",
        "changed_cohort_useful": "Among honestly changed candidates, separately report retention when mean audit Brier gain across the honestly changed domain cohort is strictly positive.",
        "other_metrics": "Report mean audit gains for all domains, the changed cohort and remaining domains; rejection of an audit-negative honest update is not automatically a utility error.",
    },
    "fault_budget_envelope": {
        "budget": "At most three malicious clients; a poisoned candidate leaves at most two lying peers, while an honest-intent candidate may face three. The scoring rule itself never receives this offline label.",
        "construction": "For each monotone symmetric rule, lower endpoint replaces k largest peer gains with -1; upper endpoint replaces k smallest with +1.",
        "meaning": "Per-candidate worst-cohort envelope, potentially different identities for different candidates. It is not a single simultaneous fixed-cohort report attack. Report its gate separately from the primary four-profile gate.",
    },
    "repair": "Both poisoned training and report misconduct stop after round10. At 12/20, stateless decisions should match the corresponding clean/TRUTHFUL common-broadcast case; other clean profiles deliberately still contain reporting-only attackers. This is decision-level readmission/equivalence, not learned rehabilitation, historical trust recovery or global-model recovery.",
    "inference": "Only two reused development seeds, fixed partitions and four correlated common checkpoints. Client-round decisions are descriptive, not independent replications; no confidence interval or held-out validation claim.",
    "scope": "This bounded Brier/control pilot is motivated by distributed validation such as FLShield, not a reproduction or a guaranteed defence. Individual scaled-update utility does not establish utility of their eventual combined aggregate.",
    "motivation_source": "https://arxiv.org/pdf/2308.05832",
}


def _bounded(value):
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
            or value < -1.000000000001 or value > 1.000000000001):
        raise ValueError("Validation gains must be finite numbers in [-1,1].")
    return min(1., max(-1., float(value)))


def _ids(candidate_ids, validator_ids):
    candidates, validators = list(candidate_ids), list(validator_ids)
    if (len(validators) != 10 or len(set(validators)) != 10 or not candidates
            or len(set(candidates)) != len(candidates) or not set(candidates).issubset(validators)
            or any(type(cid) is not int for cid in candidates + validators)):
        raise ValueError("Distinct integer IDs and ten validators including every candidate are required.")
    return candidates, validators


def _score_peers(values, rule):
    if rule not in RULES or len(values) != 9:
        raise ValueError("Use a declared fixed rule with exactly nine peers.")
    ordered = sorted(_bounded(value) for value in values)
    if rule == "peer_median":
        return ordered[4]
    if rule == "peer_top3_mean":
        return sum(ordered[-3:]) / 3
    return ordered[-4]


def score_reports(matrix, candidate_ids, validator_ids, rule):
    """Score candidate-by-validator gains, excluding self; receives no oracle data."""
    candidates, validators = _ids(candidate_ids, validator_ids)
    if len(matrix) != len(candidates) or any(len(row) != len(validators) for row in matrix):
        raise ValueError("Expected a candidate-by-validator report matrix.")
    result = {}
    for cid, row in zip(candidates, matrix):
        values = [_bounded(value) for value in row]
        result[cid] = _score_peers([value for vid, value in zip(validators, values) if vid != cid], rule)
    return result


def profile_reports(matrix, flipped_matrix, candidate_ids, validator_ids, profile,
                    bad_validator_ids=(), poisoned_candidate_ids=()):
    """Offline report attacker; truth is used here, never inside score_reports."""
    candidates, validators = _ids(candidate_ids, validator_ids)
    if profile not in PROFILES or len(matrix) != len(candidates) or len(flipped_matrix) != len(candidates):
        raise ValueError("Invalid report profile or matrix shape.")
    bad, poisoned = set(bad_validator_ids), set(poisoned_candidate_ids)
    if len(bad) > 3 or not bad.issubset(validators) or not poisoned.issubset(candidates):
        raise ValueError("Invalid offline reporting cohort or candidate intent.")
    result = []
    for cid, row, flipped in zip(candidates, matrix, flipped_matrix):
        if len(row) != 10 or len(flipped) != 10:
            raise ValueError("Expected ten reports per candidate.")
        row, flipped = [_bounded(v) for v in row], [_bounded(v) for v in flipped]
        values = []
        for vid, value, corrupt in zip(validators, row, flipped):
            if vid in bad:
                if profile == "directional_extreme":
                    value = 1. if cid in poisoned else -1.
                elif profile == "sign_reversal":
                    value = -value
                elif profile == "corrupted_labels":
                    value = corrupt
            values.append(value)
        result.append(values)
    return result


def fault_envelope(peer_values, rule, lying_peers):
    """Exact extrema for these monotone symmetric rules under <=k bounded lies."""
    if type(lying_peers) is not int or lying_peers not in (2, 3):
        raise ValueError("The declared offline envelope uses two or three lying peers.")
    values = sorted(_bounded(v) for v in peer_values)
    if len(values) != 9:
        raise ValueError("An envelope requires nine self-excluded peer reports.")
    lower = values[:-lying_peers] + [-1.] * lying_peers
    upper = [1.] * lying_peers + values[lying_peers:]
    return {"minimum_score": _score_peers(lower, rule), "maximum_score": _score_peers(upper, rule),
            "lying_peer_budget": lying_peers}


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _matrix_rows(rows, field):
    indexed = {}
    for row in rows:
        key = (row["candidate_id"], row["validator_id"])
        if key in indexed or any(type(cid) is not int or cid not in range(10) for cid in key):
            raise ValueError("Duplicate or invalid candidate/validator pair.")
        indexed[key] = row
    if set(indexed) != set(itertools.product(range(10), repeat=2)):
        raise ValueError("Each validation matrix requires all 100 candidate/validator pairs.")
    return [[_bounded(indexed[cid, vid][field]) for vid in range(10)] for cid in range(10)], indexed


def _labels(scenario, round_number):
    poisoned, changed, repaired = set(), set(), set()
    for event in scenario["events"]:
        active = any(start <= round_number <= end for start, end in event["windows"])
        clients = set(event["clients"])
        if not clients.issubset(range(10)):
            raise ValueError("Scenario references an unknown client.")
        if event["intent"] == "malicious":
            if event["kind"] != "label_flip":
                raise ValueError("This pilot only declares label-poisoned updates.")
            if active:
                poisoned.update(clients)
            elif any(end < round_number for _, end in event["windows"]):
                repaired.update(clients)
        elif active and event["kind"] == "input_change":
            changed.update(clients)
    repaired -= poisoned
    if poisoned & changed or repaired & changed:
        raise ValueError("The pilot declares distinct poisoned, changed and repaired groups.")
    groups = {}
    for cid in range(10):
        groups[cid] = ("poisoned" if cid in poisoned else "repaired" if cid in repaired else
                       "changed" if cid in changed else "clean" if scenario["name"] == "clean" else "unchanged")
    return poisoned, changed, repaired, groups


def _reporting_cohort(scenario, round_number):
    if scenario in ("clean", "abrupt_shot_noise", "gradual_stripe"):
        return (7, 8, 9)
    return () if scenario == "attack_stops" and round_number > 10 else (0, 1, 2)


def _load_cases(study):
    manifest = json.loads((study / "result_manifest.json").read_text())
    expected = set(itertools.product(REGIMES, SEEDS, SCENARIOS, ROUNDS))
    seen, cases, provenance, coverage = set(), [], [], []
    for record in manifest["records"]:
        job = record["job"]
        key = tuple(job[name] for name in ("regime", "seed", "scenario", "round"))
        if key in seen or key not in expected or record.get("status", "complete") != "complete":
            raise ValueError("Duplicate, undeclared or incomplete validation case.")
        seen.add(key)
        directory = (ROOT / record["run_path"]).resolve()
        fingerprints = record["artifact_sha256"]
        if not {"metadata.json", "reports.json", "audit.json"}.issubset(fingerprints):
            raise ValueError("Validation case lacks required artifact fingerprints.")
        for name, digest in fingerprints.items():
            path = (directory / name).resolve()
            if not path.is_relative_to(directory) or _hash(path) != digest:
                raise ValueError(f"Validation artifact fingerprint changed: {name}")
        metadata = json.loads((directory / "metadata.json").read_text())
        scenario = metadata["scenario"]
        if scenario["name"] != job["scenario"] or ("job" in metadata and metadata["job"] != job):
            raise ValueError("Saved validation scenario/job mismatch.")
        reports = json.loads((directory / "reports.json").read_text())
        audits = json.loads((directory / "audit.json").read_text())
        report_matrix, report_index = _matrix_rows(reports, "mean_brier_gain")
        flipped_matrix, _ = _matrix_rows(reports, "flipped_mean_brier_gain")
        audit_matrix, audit_index = _matrix_rows(audits, "mean_brier_gain")
        for role, index in (("report", report_index), ("audit", audit_index)):
            for vid in range(10):
                row = index[0, vid]
                counts = row["class_counts"]
                if (row["examples"] != 128 or len(counts) != 10 or sum(counts) != 128
                        or any(type(n) is not int or n < 0 for n in counts)
                        or any(index[cid, vid]["class_counts"] != counts or index[cid, vid]["examples"] != 128 for cid in range(10))):
                    raise ValueError("Validation class coverage is invalid or changes across candidate models.")
                coverage.append({**job, "validator_id": vid, "role": role, "examples": 128,
                                 "observed_classes": sum(n > 0 for n in counts),
                                 **{f"class_{digit}_count": n for digit, n in enumerate(counts)}})
        poisoned, changed, repaired, groups = _labels(scenario, job["round"])
        cases.append({"job": job, "metadata": metadata, "reports": report_matrix, "flipped": flipped_matrix,
                      "audit": audit_matrix, "poisoned": poisoned, "changed": changed,
                      "repaired": repaired, "groups": groups})
        provenance.append({"job": copy.deepcopy(job), "run_path": str(directory),
                           "artifact_sha256": copy.deepcopy(fingerprints)})
    if seen != expected:
        raise ValueError(f"Incomplete client-validation grid: {len(seen)} of 96 cases.")
    return cases, provenance, coverage


def _audit_measures(case, cid):
    values, changed = case["audit"][cid], case["changed"]
    changed_gain = statistics.mean(values[vid] for vid in changed) if changed else None
    remaining = [values[vid] for vid in range(10) if vid not in changed]
    return {"self_audit_gain": values[cid], "all_domains_audit_gain": statistics.mean(values),
            "changed_cohort_audit_gain": changed_gain,
            "remaining_domains_audit_gain": statistics.mean(remaining) if remaining else None,
            "self_audit_useful": values[cid] > 0,
            "changed_cohort_audit_useful": changed_gain > 0 if changed_gain is not None else None}


def decision_rows(cases):
    decisions, envelopes = [], []
    ids = list(range(10))
    for case in cases:
        job = case["job"]
        bad = _reporting_cohort(job["scenario"], job["round"])
        audit = {cid: _audit_measures(case, cid) for cid in ids}
        for profile in PROFILES:
            reports = profile_reports(case["reports"], case["flipped"], ids, ids, profile, bad, case["poisoned"])
            for rule in RULES:
                scores = score_reports(reports, ids, ids, rule)
                for cid in ids:
                    decisions.append({**job, "candidate_id": cid, "rule": rule, "profile": profile,
                                      "group": case["groups"][cid], "poisoned_update": cid in case["poisoned"],
                                      "reporter_cohort": ",".join(map(str, bad)) if profile != "truthful" else "",
                                      "lying_peers": len([vid for vid in bad if vid != cid]) if profile != "truthful" else 0,
                                      "score": scores[cid], "accepted": scores[cid] > 0, **audit[cid]})
        for rule in RULES:
            for cid in ids:
                poison = cid in case["poisoned"]
                envelope = fault_envelope([value for vid, value in enumerate(case["reports"][cid]) if vid != cid], rule, 2 if poison else 3)
                envelopes.append({**job, "candidate_id": cid, "rule": rule,
                                  "profile": "any_peer_fault_envelope", "group": case["groups"][cid],
                                  "poisoned_update": poison, **envelope,
                                  "accepted": envelope["maximum_score"] > 0 if poison else envelope["minimum_score"] > 0,
                                  "guaranteed_accept": envelope["minimum_score"] > 0,
                                  "guaranteed_reject": envelope["maximum_score"] <= 0, **audit[cid]})
    return decisions, envelopes


def grouped_rates(rows):
    keys = ("rule", "profile", "regime", "seed", "scenario", "group")
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    result = []
    for values, members in sorted(groups.items()):
        accepted = sum(row["accepted"] for row in members)
        poisoned, repaired = values[-1] == "poisoned", values[-1] == "repaired"
        successes = len(members) - accepted if poisoned else accepted
        denominator, numerator = (5, 4) if poisoned or repaired else (20, 19)
        result.append({**dict(zip(keys, values)), "n_decisions": len(members),
                       "n_clients": len({row["candidate_id"] for row in members}),
                       "n_checkpoints": len({row["round"] for row in members}),
                       "accepted": accepted, "rejected": len(members)-accepted,
                       "acceptance_rate": accepted/len(members), "successes": successes,
                       "success_rate": successes/len(members),
                       "metric": "poison_rejection" if poisoned else "repaired_readmission" if repaired else "honest_update_retention",
                       "required_rate": numerator/denominator,
                       "pass": successes * denominator >= len(members) * numerator})
    return result


def gate_summary(rates, profiles=PROFILES):
    rules = {}
    expected = {(profile, regime, seed, scenario, group) for profile in profiles
                for regime in REGIMES for seed in SEEDS for scenario, groups in GROUPS_BY_SCENARIO.items() for group in groups}
    for rule in RULES:
        rows = [row for row in rates if row["rule"] == rule]
        actual = {(r["profile"], r["regime"], r["seed"], r["scenario"], r["group"]) for r in rows}
        if actual != expected or len(rows) != len(expected):
            raise ValueError(f"Gate lacks the complete declared group grid for {rule}.")
        failures = [row for row in rows if not row["pass"]]
        rules[rule] = {"pass": not failures, "groups": len(rows), "failed_groups": len(failures), "failures": failures}
    passing = [rule for rule in RULES if rules[rule]["pass"]]
    return {"pass": bool(passing), "selected_rule": passing[0] if passing else None,
            "passing_rules": passing, "rules": rules,
            "decision": "go_to_separately_declared_followup" if passing else "no_go_for_these_fixed_controls"}


def audit_utility_rows(decisions):
    keys = ("rule", "profile", "regime", "seed", "scenario", "group")
    groups = defaultdict(list)
    for row in decisions:
        if not row["poisoned_update"]:
            groups[tuple(row[key] for key in keys)].append(row)
    result = []
    for values, rows in sorted(groups.items()):
        for criterion, useful in (("positive_self_audit_gain", [r for r in rows if r["self_audit_useful"]]),
                                  ("positive_changed_cohort_audit_gain", [r for r in rows if r["group"] == "changed" and r["changed_cohort_audit_useful"]])):
            accepted = sum(row["accepted"] for row in useful)
            result.append({**dict(zip(keys, values)), "audit_criterion": criterion,
                           "n_honest_intent_decisions": len(rows), "n_audit_useful": len(useful),
                           "useful_accepted": accepted, "useful_rejected": len(useful)-accepted,
                           "useful_retention": accepted/len(useful) if useful else None,
                           "mean_all_domains_audit_gain_if_accepted": statistics.mean(r["all_domains_audit_gain"] for r in rows if r["accepted"]) if any(r["accepted"] for r in rows) else None,
                           "used_in_primary_gate": False})
    return result


def repair_equivalence(cases, decisions):
    indexed = {(c["job"]["regime"], c["job"]["seed"], c["job"]["scenario"], c["job"]["round"]): c for c in cases}
    scores = {(r["regime"], r["seed"], r["scenario"], r["round"], r["candidate_id"], r["rule"], r["profile"]): r["score"] for r in decisions}
    results = []
    for case in cases:
        if not case["repaired"]:
            continue
        j = case["job"]
        reference = indexed[j["regime"], j["seed"], "clean", j["round"]]
        for cid in sorted(case["repaired"]):
            equal_scores = all(scores[j["regime"], j["seed"], j["scenario"], j["round"], cid, rule, profile]
                               == scores[j["regime"], j["seed"], "clean", j["round"], cid, rule, "truthful"]
                               for rule in RULES for profile in PROFILES)
            results.append({**j, "candidate_id": cid,
                            "correct_report_gains_exactly_equal_clean": case["reports"][cid] == reference["reports"][cid],
                            "audit_gains_exactly_equal_clean": case["audit"][cid] == reference["audit"][cid],
                            "all_rules_and_profiles_equal_clean_truthful_score": equal_scores})
    return {"n_repaired_observations": len(results),
            "all_equal": bool(results) and all(r["correct_report_gains_exactly_equal_clean"] and r["audit_gains_exactly_equal_clean"] and r["all_rules_and_profiles_equal_clean_truthful_score"] for r in results),
            "comparisons": results, "interpretation": DIAGNOSTIC_PROTOCOL["repair"]}


def _write_csv(path, rows):
    if not rows:
        raise ValueError("Refusing to write an empty scientific result table.")
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(key for row in rows for key in row)))
        writer.writeheader()
        writer.writerows(rows)


def analyse_study(study_path):
    study = Path(study_path)
    cases, inputs, coverage = _load_cases(study)
    decisions, envelopes = decision_rows(cases)
    rates, envelope_rates = grouped_rates(decisions), grouped_rates(envelopes)
    audit_rows = audit_utility_rows(decisions)
    gate, envelope_gate = gate_summary(rates), gate_summary(envelope_rates, ("any_peer_fault_envelope",))
    summary = {"protocol": copy.deepcopy(DIAGNOSTIC_PROTOCOL), "analysis_source_sha256": _hash(__file__),
               "inputs": inputs, "counts": {"cases": len(cases), "candidate_updates": len(cases)*10,
                                            "candidate_rule_profile_decisions": len(decisions),
                                            "fault_envelopes": len(envelopes)},
               "gate": gate, "envelope_gate": envelope_gate,
               "repair_equivalence": repair_equivalence(cases, decisions),
               "audit_definitions": copy.deepcopy(DIAGNOSTIC_PROTOCOL["audit"]),
               "cost": {"declared_no_cache_total_forward_examples": 96*10*11*256,
                        "declared_report_forward_examples": 96*10*11*128,
                        "declared_audit_forward_examples": 96*10*11*128,
                        "declared_local_candidate_trainings": 960,
                        "case_cost_records": [{"job": c["job"],
                                               "cost": c["metadata"].get("cost", c["metadata"].get("costs")),
                                               "local_validation_costs": c["metadata"].get("local_validation_costs"),
                                               "parameter_count": c["metadata"].get("parameter_count"),
                                               "float32_model_payload_bytes": c["metadata"].get("float32_model_payload_bytes")}
                                              for c in cases],
                        "interpretation": "Declared no-cache evaluation workload, not measured timing or traffic. Audit evaluation is research overhead. Per-case measured costs are retained only when recorded in metadata. Model/report payload sizes are not measured network traffic."},
               "interpretation": "Every fixed control/profile is reported. No thresholds or rules fitted to these results; any passing rule is an exploratory screen on reused development evidence. Primary fixed-cohort gate and candidate-specific worst-cohort envelope are distinct."}
    output = study / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in (("decision_rows", decisions), ("grouped_rates", rates),
                       ("audit_utility", audit_rows), ("fault_envelopes", envelopes),
                       ("envelope_rates", envelope_rates), ("class_coverage", coverage)):
        _write_csv(output / f"{name}.csv", rows)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return summary
