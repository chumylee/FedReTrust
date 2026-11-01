"""Fixed-rule leave-one-out aggregate validation and matched-context comparison.

The prior isolated-update analysis is imported read-only for identical controls,
profiles and gates. This module owns the new estimand, provenance, outputs and
paired analysis; it never modifies or overwrites the prior study.
"""
from collections import defaultdict
import copy
import json
import math
from pathlib import Path
import statistics

import client_validation_analysis as isolated
from client_validation_analysis import (ROOT, RULES, PROFILES, REGIMES, SEEDS, ROUNDS,
    SCENARIOS, score_reports, profile_reports, fault_envelope, grouped_rates,
    gate_summary, decision_rows, audit_utility_rows)


PRIOR_STUDY = ROOT / "outputs/client_validation/20261030_v1"
DIAGNOSTIC_PROTOCOL = copy.deepcopy(isolated.DIAGNOSTIC_PROTOCOL)
DIAGNOSTIC_PROTOCOL.update({
    "version": "aggregate_leave_one_out_validation_fixed_controls_v1",
    "candidate_model": "Full = w + sum_j 0.1 * delta_j; without_i = w + sum_{j != i} 0.1 * delta_j. Retained weights are not renormalised. The absent update has weight zero.",
    "gain_context": "Each candidate is assessed by its marginal contribution within the same all-ten-update aggregate, not by its update applied alone. Other clients' submitted updates remain present, including poisoned ones.",
    "scope": "A paired aggregate-context utility pilot, not a FLShield reproduction or a deployed defence. Full-model outcomes are one-step counterfactuals from saved clean broadcasts, not multi-round accuracy or outcomes of aggregating only accepted updates.",
    "repair": "Both poisoned training and reporting stop after round10. At12/20 the full and without-client models, gains and stateless decisions should match the clean/TRUTHFUL aggregate counterfactual. This is decision-level clean-equivalence/readmission, not learned trust rehabilitation or global-model recovery.",
    "paired_comparison": {
        "reference": "outputs/client_validation/20261030_v1",
        "controls": "Recompute prior fixed-rule decisions from verified case artifacts; same96 jobs, local report/audit views, broadcasts, optimiser configuration and960 submitted model hashes.",
        "report": "All paired candidate/rule/profile decisions and per-seed group contrasts; no selection by relative improvement and no post-hoc changes to gates.",
        "full_audit": "Compare the new full-aggregate audit metrics with prior recorded broadcast audit metrics on the identical private audit views. This requires no new forwards and never enters scoring or a gate.",
    },
})
DIAGNOSTIC_PROTOCOL["reports"]["metric"] = "Mean Brier loss(without_i) minus mean Brier loss(full); positive means the submitted update improves the full aggregate relative to omitting it at fixed remaining weights. Brier =0.5*sum_class(probability-target_indicator)^2."
DIAGNOSTIC_PROTOCOL["audit"]["self_useful"] = "Among honest-intent updates, report retention when their own-domain independent audit marginal contribution loss(without_i)-loss(full) is strictly positive."
DIAGNOSTIC_PROTOCOL["audit"]["changed_cohort_useful"] = "Among honestly changed candidates, separately report retention when their mean marginal audit contribution across the honestly changed domain cohort is strictly positive."
DIAGNOSTIC_PROTOCOL["audit"]["other_metrics"] = "Report marginal audit contributions on all, changed and remaining domains. Separately report full-aggregate audit outcomes relative to the same original broadcast; do not sum nonlinear leave-one-out contributions as if they were the full improvement."


def _job_key(job):
    return tuple(job[name] for name in ("regime", "seed", "scenario", "round"))


def _read_case_rows(inputs):
    result = {}
    for item in inputs:
        path = Path(item["run_path"])
        if "training.json" not in item["artifact_sha256"]:
            raise ValueError("Paired utility analysis requires fingerprinted local-training records.")
        data = {role: json.loads((path / filename).read_text())
                for role, filename in (("report_rows", "reports.json"), ("audit_rows", "audit.json"), ("training", "training.json"))}
        result[_job_key(item["job"])] = data
    return result


def _training_index(rows):
    result = {}
    for row in rows:
        cid = row["client_id"]
        if type(cid) is not int or cid not in range(10) or cid in result:
            raise ValueError("Local-training provenance must contain ten distinct clients.")
        result[cid] = row["submitted_state_sha256"]
    if set(result) != set(range(10)):
        raise ValueError("Missing submitted-update fingerprint for a client.")
    return result


def validate_pairing(cases, prior_cases, raw, prior_raw):
    """Pair exact inputs before attributing observed differences to utility context."""
    current = {_job_key(c["job"]): c for c in cases}
    previous = {_job_key(c["job"]): c for c in prior_cases}
    if set(current) != set(previous):
        raise ValueError("Aggregate and isolated studies do not cover identical jobs.")
    audits = []
    for key, case in current.items():
        old = previous[key]
        for field in ("config", "scenario", "source_broadcast_sha256", "private_plan_sha256"):
            if field not in case["metadata"] or field not in old["metadata"] or case["metadata"][field] != old["metadata"][field]:
                raise ValueError(f"Paired utility input differs: {field}, {key}")
        if case["metadata"].get("marginal_rate") != .1 or old["metadata"].get("marginal_rate") != .1:
            raise ValueError("Both utility contexts require the same fixed0.1 client share.")
        if not case["metadata"].get("full_aggregate_state_sha256"):
            raise ValueError("Missing full-aggregate model fingerprint.")
        if case["metadata"].get("evaluation_semantics", {}).get("kind") != "fixed_weight_leave_one_out_client_contribution":
            raise ValueError("Validation metadata does not declare fixed-weight leave-one-out contribution.")
        trained, old_trained = _training_index(raw[key]["training"]), _training_index(prior_raw[key]["training"])
        if trained != old_trained:
            raise ValueError(f"Submitted local models differ between utility contexts: {key}")
        for role in ("report_rows", "audit_rows"):
            rows = {(r["candidate_id"], r["validator_id"]): r for r in raw[key][role]}
            old_rows = {(r["candidate_id"], r["validator_id"]): r for r in prior_raw[key][role]}
            for pair, row in rows.items():
                if row["class_counts"] != old_rows[pair]["class_counts"] or row["examples"] != old_rows[pair]["examples"]:
                    raise ValueError("Paired local validation examples or class coverage differ.")
                prefixes = ("", "flipped_") if role == "report_rows" else ("",)
                for prefix in prefixes:
                    gain = row[prefix + "mean_brier_gain"]
                    difference = row[prefix + "without_client_mean_brier"] - row[prefix + "full_aggregate_mean_brier"]
                    if not math.isfinite(difference) or not math.isclose(gain, difference, abs_tol=1e-12, rel_tol=0):
                        raise ValueError("Marginal Brier gain has an inconsistent sign or absolute metrics.")
        audits.append({"job": copy.deepcopy(case["job"]), "paired_client_updates": 10,
                       "broadcast_plan_config_scenario_identical": True,
                       "submitted_model_fingerprints_identical": True,
                       "local_class_coverage_identical": True,
                       "marginal_brier_gain_sign_verified": True})
    return {"cases": len(audits), "candidate_updates": len(audits)*10, "all_paired": True, "checks": audits}


def paired_decision_rows(current, previous):
    keys = ("regime", "seed", "scenario", "round", "candidate_id", "rule", "profile")
    before = {tuple(r[key] for key in keys): r for r in previous}
    after = {tuple(r[key] for key in keys): r for r in current}
    if len(before) != len(previous) or len(after) != len(current) or set(before) != set(after):
        raise ValueError("Paired decisions are incomplete or duplicated.")
    rows = []
    for key, row in sorted(after.items()):
        old = before[key]
        if row["group"] != old["group"] or row["poisoned_update"] != old["poisoned_update"]:
            raise ValueError("Paired update-intent groups differ.")
        poisoned = row["poisoned_update"]
        old_success, success = (not old["accepted"], not row["accepted"]) if poisoned else (old["accepted"], row["accepted"])
        result = {**dict(zip(keys, key)), "group": row["group"], "poisoned_update": poisoned,
                  "isolated_score": old["score"], "aggregate_score": row["score"],
                  "isolated_accepted": old["accepted"], "aggregate_accepted": row["accepted"],
                  "acceptance_change": int(row["accepted"]) - int(old["accepted"]),
                  "isolated_success": old_success, "aggregate_success": success}
        for metric in ("self_audit_gain", "all_domains_audit_gain", "changed_cohort_audit_gain", "remaining_domains_audit_gain"):
            result["isolated_" + metric], result["aggregate_" + metric] = old[metric], row[metric]
        rows.append(result)
    return rows


def paired_grouped_rates(rows):
    fields = ("rule", "profile", "regime", "seed", "scenario", "group")
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in fields)].append(row)
    results = []
    for key, members in sorted(groups.items()):
        n = len(members)
        old_accept, new_accept = sum(r["isolated_accepted"] for r in members), sum(r["aggregate_accepted"] for r in members)
        old_success, new_success = sum(r["isolated_success"] for r in members), sum(r["aggregate_success"] for r in members)
        results.append({**dict(zip(fields, key)), "n_paired_decisions": n,
                        "isolated_accepted": old_accept, "aggregate_accepted": new_accept,
                        "isolated_acceptance_rate": old_accept/n, "aggregate_acceptance_rate": new_accept/n,
                        "acceptance_change_pp": 100*(new_accept-old_accept)/n,
                        "isolated_success_rate": old_success/n, "aggregate_success_rate": new_success/n,
                        "success_change_pp": 100*(new_success-old_success)/n,
                        "newly_accepted": sum(r["acceptance_change"] == 1 for r in members),
                        "newly_rejected": sum(r["acceptance_change"] == -1 for r in members),
                        "interpretation": "Descriptive matched context change, not independent observations or an additional gate"})
    return results


def paired_audit_rows(rows):
    """One row per candidate/checkpoint, avoiding twelvefold profile/rule duplication."""
    selected = [r for r in rows if r["rule"] == RULES[0] and r["profile"] == "truthful"]
    fields = ("regime", "seed", "scenario", "round", "candidate_id", "group", "poisoned_update")
    results = []
    for row in selected:
        result = {field: row[field] for field in fields}
        for metric in ("self_audit_gain", "all_domains_audit_gain", "changed_cohort_audit_gain", "remaining_domains_audit_gain"):
            before, after = row["isolated_"+metric], row["aggregate_"+metric]
            result["isolated_"+metric], result["aggregate_"+metric] = before, after
            result[metric+"_change"] = after-before if before is not None and after is not None else None
        results.append(result)
    return results


def full_aggregate_audit(cases, raw, prior_raw):
    """Use matched existing broadcast audits; no extra evaluation or server access."""
    rows, grouped = [], []
    for case in cases:
        key = _job_key(case["job"])
        current = {(r["candidate_id"], r["validator_id"]): r for r in raw[key]["audit_rows"]}
        previous = {(r["candidate_id"], r["validator_id"]): r for r in prior_raw[key]["audit_rows"]}
        local = []
        for vid in range(10):
            current_row, previous_row = current[0, vid], previous[0, vid]
            for cid in range(10):
                for metric in ("mean_brier", "mean_cross_entropy", "accuracy"):
                    if (current[cid, vid]["full_aggregate_"+metric] != current_row["full_aggregate_"+metric]
                            or previous[cid, vid]["baseline_"+metric] != previous_row["baseline_"+metric]):
                        raise ValueError("Repeated full/broadcast audit metrics differ across candidate IDs.")
            brier = previous_row["baseline_mean_brier"] - current_row["full_aggregate_mean_brier"]
            accuracy = current_row["full_aggregate_accuracy"] - previous_row["baseline_accuracy"]
            ce = previous_row["baseline_mean_cross_entropy"] - current_row["full_aggregate_mean_cross_entropy"]
            row = {**case["job"], "validator_id": vid,
                   "domain": "changed" if vid in case["changed"] else "remaining",
                   "audit_examples": current_row["examples"],
                   "broadcast_mean_brier": previous_row["baseline_mean_brier"],
                   "full_aggregate_mean_brier": current_row["full_aggregate_mean_brier"],
                   "full_aggregate_brier_gain": brier,
                   "broadcast_accuracy": previous_row["baseline_accuracy"],
                   "full_aggregate_accuracy": current_row["full_aggregate_accuracy"],
                   "full_aggregate_accuracy_gain_pp": 100*accuracy,
                   "full_aggregate_cross_entropy_gain": ce}
            local.append(row)
        rows.extend(local)
        cohorts = {"all_domains": local}
        if case["changed"]:
            cohorts.update({"changed_domains": [r for r in local if r["domain"] == "changed"],
                            "remaining_domains": [r for r in local if r["domain"] == "remaining"]})
        for name, cohort in cohorts.items():
            grouped.append({**case["job"], "domain_group": name, "n_validators": len(cohort),
                            "audit_examples": sum(r["audit_examples"] for r in cohort),
                            **{metric: statistics.mean(r[metric] for r in cohort) for metric in
                               ("broadcast_mean_brier", "full_aggregate_mean_brier", "full_aggregate_brier_gain",
                                "broadcast_accuracy", "full_aggregate_accuracy", "full_aggregate_accuracy_gain_pp",
                                "full_aggregate_cross_entropy_gain")}})
    return rows, grouped


def analyse_study(study_path, prior_study_path=PRIOR_STUDY):
    study, prior = Path(study_path), Path(prior_study_path)
    if study.resolve() == prior.resolve():
        raise ValueError("Aggregate analysis cannot overwrite the isolated study.")
    cases, inputs, coverage = isolated._load_cases(study)
    old_cases, old_inputs, _ = isolated._load_cases(prior)
    raw, old_raw = _read_case_rows(inputs), _read_case_rows(old_inputs)
    pairing = validate_pairing(cases, old_cases, raw, old_raw)
    decisions, envelopes = decision_rows(cases)
    old_decisions, _ = decision_rows(old_cases)
    rates, envelope_rates = grouped_rates(decisions), grouped_rates(envelopes)
    gate = gate_summary(rates)
    envelope_gate = gate_summary(envelope_rates, ("any_peer_fault_envelope",))
    paired = paired_decision_rows(decisions, old_decisions)
    full_audit, full_groups = full_aggregate_audit(cases, raw, old_raw)
    repair = isolated.repair_equivalence(cases, decisions)
    repair["interpretation"] = DIAGNOSTIC_PROTOCOL["repair"]
    summary = {
        "protocol": copy.deepcopy(DIAGNOSTIC_PROTOCOL), "analysis_source_sha256": isolated._hash(__file__),
        "reused_scoring_analysis_source_sha256": isolated._hash(isolated.__file__), "inputs": inputs,
        "counts": {"cases": len(cases), "candidate_updates": len(cases)*10,
                   "candidate_rule_profile_decisions": len(decisions), "fault_envelopes": len(envelopes)},
        "gate": gate, "envelope_gate": envelope_gate, "repair_equivalence": repair,
        "audit_definitions": copy.deepcopy(DIAGNOSTIC_PROTOCOL["audit"]),
        "paired_comparison": {"reference_study": str(prior.resolve()), "reference_inputs": old_inputs,
                              "pairing": pairing, "paired_rule_profile_decisions": len(paired),
                              "interpretation": DIAGNOSTIC_PROTOCOL["paired_comparison"]},
        "cost": {"declared_no_cache_total_forward_examples": 96*10*11*256,
                 "declared_report_forward_examples": 96*10*11*128,
                 "declared_audit_forward_examples": 96*10*11*128,
                 "declared_local_candidate_trainings": 960,
                 "naive_peer_model_transmissions_per_case": 100,
                 "naive_peer_scalar_reports_per_case": 90,
                 "case_cost_records": [{"job": c["job"], "cost": c["metadata"].get("cost"),
                                        "local_validation_costs": c["metadata"].get("local_validation_costs"),
                                        "parameter_count": c["metadata"].get("parameter_count"),
                                        "float32_model_payload_bytes": c["metadata"].get("float32_model_payload_bytes")}
                                       for c in cases],
                 "interpretation": "11 models per validator/role (full plus ten without-client models); self scoring is excluded but its evaluation supports offline audit. A naive deployment sends each validator the full model plus nine peer-omission models:100 payload transmissions/case, versus90 isolated candidates previously. Declared payload counts are not measured traffic; audit forwards are research overhead."},
        "interpretation": "All fixed rules/profiles are retained and the original gate is unchanged. Paired improvement alone does not imply gate success. Full aggregate audit outcomes are single-step outcomes including all submitted updates, not an accepted-only or multi-round defence."}
    output = study / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in (("decision_rows", decisions), ("grouped_rates", rates),
                       ("audit_utility", audit_utility_rows(decisions)), ("fault_envelopes", envelopes),
                       ("envelope_rates", envelope_rates), ("class_coverage", coverage),
                       ("paired_decisions", paired), ("paired_grouped_rates", paired_grouped_rates(paired)),
                       ("paired_audit_context", paired_audit_rows(paired)),
                       ("full_aggregate_audit", full_audit), ("full_aggregate_audit_groups", full_groups)):
        isolated._write_csv(output / f"{name}.csv", rows)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n")
    (output / "pairing_audit.json").write_text(json.dumps(pairing, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return summary
