"""Known-answer tests for local-validation scores, lies, audits and study gates."""
import copy
import hashlib
import itertools
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from client_validation_analysis import (DIAGNOSTIC_PROTOCOL, PROFILES, REGIMES, ROUNDS,
    RULES, SCENARIOS, SEEDS, _labels, _load_cases, _reporting_cohort, _score_peers,
    analyse_study, audit_utility_rows, decision_rows, fault_envelope, gate_summary,
    grouped_rates, profile_reports, score_reports)


def scenario(name):
    events = []
    if name in ("abrupt_shot_noise", "gradual_stripe"):
        events.append({"kind": "input_change", "intent": "honest", "clients": [0, 1, 2],
                       "windows": [[5, 24]]})
    if name in ("label_flip", "attack_stops", "mixed_stripe_attack"):
        events.append({"kind": "label_flip", "intent": "malicious", "clients": [0, 1, 2],
                       "windows": [[5, 10 if name == "attack_stops" else 24]]})
    if name == "mixed_stripe_attack":
        events.append({"kind": "input_change", "intent": "honest", "clients": [3, 4, 5],
                       "windows": [[5, 24]]})
    return {"name": name, "events": events}


def cases():
    result = []
    for regime, seed, name, round_number in itertools.product(REGIMES, SEEDS, SCENARIOS, ROUNDS):
        job = {"regime": regime, "seed": seed, "scenario": name, "round": round_number}
        poisoned, changed, repaired, groups = _labels(scenario(name), round_number)
        report = [[-.1 if cid in poisoned else .1 for _ in range(10)] for cid in range(10)]
        result.append({"job": job, "metadata": {"job": job, "scenario": scenario(name)},
                       "reports": report, "flipped": [[-value for value in row] for row in report],
                       "audit": copy.deepcopy(report), "poisoned": poisoned, "changed": changed,
                       "repaired": repaired, "groups": groups})
    return result


def write_study(root):
    records = []
    for index, case in enumerate(cases()):
        directory = root / f"case_{index}"
        directory.mkdir()
        report, audit = [], []
        for cid, vid in itertools.product(range(10), repeat=2):
            common = {"candidate_id": cid, "validator_id": vid, "examples": 128,
                      "class_counts": [13]*8 + [12]*2}
            report.append({**common, "mean_brier_gain": case["reports"][cid][vid],
                           "flipped_mean_brier_gain": case["flipped"][cid][vid]})
            audit.append({**common, "mean_brier_gain": case["audit"][cid][vid]})
        for name, value in (("metadata.json", case["metadata"]), ("reports.json", report), ("audit.json", audit)):
            (directory / name).write_text(json.dumps(value))
        records.append({"job": case["job"], "run_path": str(directory), "status": "complete",
                        "artifact_sha256": {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                                            for name in ("metadata.json", "reports.json", "audit.json")}})
    (root / "result_manifest.json").write_text(json.dumps({"records": records}))
    return records


class ClientValidationAnalysisTests(unittest.TestCase):
    def test_peer_rules_and_self_exclusion_are_exact_and_id_based(self):
        ids = list(range(10))
        for own in (-1., 1.):
            matrix = [[own if cid == vid else .2 for vid in ids] for cid in ids]
            for rule in RULES:
                result = score_reports(matrix, ids, ids, rule)
                self.assertTrue(all(abs(value-.2) < 1e-12 for value in result.values()))
        peers = [-.8, -.7, -.6, -.5, -.4, -.3, .1, .2, .9]
        self.assertEqual(_score_peers(peers, "peer_median"), -.4)
        self.assertAlmostEqual(_score_peers(peers, "peer_top3_mean"), .4)
        self.assertEqual(_score_peers(peers, "peer_fourth_highest"), -.3)
        matrix = [[1. if vid == 7 else .2 for vid in reversed(ids)]]
        self.assertEqual(score_reports(matrix, [7], list(reversed(ids)), "peer_median"), {7: .2})
        for value in (float("nan"), float("inf"), 1.1, True):
            with self.assertRaises(ValueError):
                score_reports([[value]*10], [0], ids, RULES[0])

    def test_profiles_are_bounded_and_truth_only_affects_offline_attacker(self):
        ids = list(range(10))
        truthful, flipped = [[.1]*10 for _ in ids], [[.4]*10 for _ in ids]
        before = copy.deepcopy(truthful)
        extreme = profile_reports(truthful, flipped, ids, ids, "directional_extreme", (0, 1, 2), (0, 1, 2))
        self.assertEqual(extreme[0][:4], [1., 1., 1., .1])
        self.assertEqual(extreme[3][:4], [-1., -1., -1., .1])
        self.assertEqual(profile_reports(truthful, flipped, ids, ids, "truthful", (0, 1, 2), (0,))[0], [.1]*10)
        self.assertEqual(profile_reports(truthful, flipped, ids, ids, "sign_reversal", (0,), ())[0][:2], [-.1, .1])
        self.assertEqual(profile_reports(truthful, flipped, ids, ids, "corrupted_labels", (0,), ())[0][:2], [.4, .1])
        self.assertEqual(truthful, before)
        self.assertEqual(_reporting_cohort("clean", 20), (7, 8, 9))
        self.assertEqual(_reporting_cohort("attack_stops", 8), (0, 1, 2))
        self.assertEqual(_reporting_cohort("attack_stops", 12), ())

    def test_envelope_matches_exhaustive_corrupted_cohort_and_endpoint_search(self):
        values = [-.9, -.5, -.3, -.1, 0., .2, .4, .6, .8]
        for rule in RULES:
            for count in (2, 3):
                brute = []
                for cohort in itertools.combinations(range(9), count):
                    for reports in itertools.product((-1., 1.), repeat=count):
                        attacked = values.copy()
                        for index, value in zip(cohort, reports):
                            attacked[index] = value
                        brute.append(_score_peers(attacked, rule))
                result = fault_envelope(values, rule, count)
                self.assertAlmostEqual(result["minimum_score"], min(brute))
                self.assertAlmostEqual(result["maximum_score"], max(brute))

    def test_total_fault_budget_and_reporting_only_actor_intent_are_separate(self):
        rows, envelopes = decision_rows(cases())
        poison = [row for row in rows if row["scenario"] == "label_flip" and row["candidate_id"] == 0 and row["profile"] == "directional_extreme"]
        self.assertTrue(all(row["lying_peers"] == 2 and row["poisoned_update"] for row in poison))
        reporter = [row for row in rows if row["scenario"] == "clean" and row["candidate_id"] == 7 and row["profile"] == "directional_extreme"]
        self.assertTrue(all(row["lying_peers"] == 2 and not row["poisoned_update"] and row["group"] == "clean" for row in reporter))
        self.assertTrue(all(row["lying_peer_budget"] == (2 if row["poisoned_update"] else 3) for row in envelopes))
        repaired = [row for row in rows if row["group"] == "repaired"]
        self.assertTrue(all(row["round"] in (12, 20) and row["lying_peers"] == 0 for row in repaired))

    def test_full_grid_gate_all_profiles_both_seeds_and_fixed_rule_order(self):
        observations, envelopes = decision_rows(cases())
        gate = gate_summary(grouped_rates(observations))
        self.assertTrue(gate["pass"])
        self.assertEqual(gate["selected_rule"], "peer_median")
        self.assertEqual(gate["passing_rules"], ["peer_median", "peer_fourth_highest"])
        self.assertFalse(gate["rules"]["peer_top3_mean"]["pass"])
        self.assertEqual(gate["rules"]["peer_median"]["groups"], 208)
        # One false rejection among twelve minority decisions is above the5% limit.
        for row in observations:
            if (row["rule"], row["profile"], row["regime"], row["seed"], row["scenario"], row["round"], row["candidate_id"]) == ("peer_median", "corrupted_labels", "non-IID", SEEDS[1], "gradual_stripe", 20, 0):
                row["accepted"] = False
        changed = gate_summary(grouped_rates(observations))
        self.assertFalse(changed["rules"]["peer_median"]["pass"])
        self.assertEqual(changed["selected_rule"], "peer_fourth_highest")
        with self.assertRaisesRegex(ValueError, "complete declared group"):
            gate_summary(grouped_rates([r for r in observations if r["seed"] == SEEDS[0]]))
        self.assertTrue(gate_summary(grouped_rates(envelopes), ("any_peer_fault_envelope",))["pass"])

    def test_audit_usefulness_is_independent_of_intent_gate_and_score(self):
        original = cases()
        observed, _ = decision_rows(original)
        before = [(r["score"], r["accepted"]) for r in observed]
        for case in original:
            case["audit"] = [[-.4]*10 for _ in range(10)]
        changed, _ = decision_rows(original)
        self.assertEqual(before, [(r["score"], r["accepted"]) for r in changed])
        self.assertEqual(gate_summary(grouped_rates(observed)), gate_summary(grouped_rates(changed)))
        utility = audit_utility_rows(changed)
        self.assertTrue(all(row["n_audit_useful"] == 0 and row["useful_retention"] is None for row in utility))
        # A zero gain is not accepted and is not deemed audit-useful.
        one = cases()[0]
        one["reports"] = [[0.]*10 for _ in range(10)]
        one["audit"] = [[0.]*10 for _ in range(10)]
        zero, _ = decision_rows([one])
        self.assertTrue(all(not row["accepted"] and not row["self_audit_useful"] for row in zero if row["profile"] == "truthful"))

    def test_small_group_gate_uses_exact_integer_rates_and_separate_repair_target(self):
        rows = []
        for group, total, accepted in (("changed", 12, 11), ("repaired", 6, 5), ("poisoned", 6, 1)):
            for i in range(total):
                rows.append({"rule": RULES[0], "profile": PROFILES[0], "regime": REGIMES[0],
                             "seed": SEEDS[0], "scenario": "attack_stops", "group": group,
                             "candidate_id": i % 3, "round": i // 3, "accepted": i < accepted})
        result = {row["group"]: row for row in grouped_rates(rows)}
        self.assertFalse(result["changed"]["pass"])
        self.assertTrue(result["repaired"]["pass"])
        self.assertTrue(result["poisoned"]["pass"])
        self.assertEqual(result["repaired"]["required_rate"], .8)
        self.assertEqual(result["changed"]["required_rate"], .95)

    def test_complete_study_outputs_provenance_repair_and_cost_units(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = write_study(root)
            result = analyse_study(root)
            self.assertEqual(result["counts"], {"cases": 96, "candidate_updates": 960,
                                               "candidate_rule_profile_decisions": 11520, "fault_envelopes": 2880})
            self.assertTrue(result["gate"]["pass"])
            self.assertTrue(result["repair_equivalence"]["all_equal"])
            self.assertEqual(result["repair_equivalence"]["n_repaired_observations"], 24)
            self.assertEqual(result["cost"]["declared_no_cache_total_forward_examples"], 2703360)
            self.assertEqual(result["cost"]["declared_report_forward_examples"], 1351680)
            for name in ("summary.json", "decision_rows.csv", "grouped_rates.csv", "audit_utility.csv",
                         "fault_envelopes.csv", "envelope_rates.csv", "class_coverage.csv"):
                self.assertTrue((root / "analysis" / name).exists())
            self.assertEqual(json.loads((root / "analysis/summary.json").read_text())["protocol"], DIAGNOSTIC_PROTOCOL)
            # Completeness and artifact checks happen before analysis writes results.
            (root / "result_manifest.json").write_text(json.dumps({"records": records[:-1]}))
            with self.assertRaisesRegex(ValueError, "Incomplete client-validation grid"):
                _load_cases(root)
            (root / "result_manifest.json").write_text(json.dumps({"records": records}))
            (Path(records[0]["run_path"]) / "reports.json").write_text("[]")
            with self.assertRaisesRegex(ValueError, "artifact fingerprint"):
                _load_cases(root)


if __name__ == "__main__":
    unittest.main()
