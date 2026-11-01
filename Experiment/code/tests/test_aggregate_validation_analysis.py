"""Matched-input, sign, audit-reuse and paired-contrast tests for aggregate utility."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import client_validation_analysis as prior
from aggregate_validation_analysis import (DIAGNOSTIC_PROTOCOL, _read_case_rows,
    analyse_study, full_aggregate_audit, paired_decision_rows, paired_grouped_rates,
    validate_pairing)
from test_client_validation_analysis import write_study


def make_pair(root):
    old, new = root / "isolated", root / "aggregate"
    old.mkdir(); new.mkdir()
    old_records, new_records = write_study(old), write_study(new)
    for previous, current in zip(old_records, new_records):
        job = current["job"]
        key = (job["regime"], job["seed"], job["scenario"], job["round"])
        source = hashlib.sha256(str((job["regime"], job["seed"], job["round"])).encode()).hexdigest()
        for is_new, record in ((False, previous), (True, current)):
            directory = Path(record["run_path"])
            metadata = json.loads((directory / "metadata.json").read_text())
            metadata.update({"config": {"rounds": 24, "development_seed": job["seed"]},
                             "source_broadcast_sha256": source, "private_plan_sha256": "d"*64,
                             "marginal_rate": .1})
            if is_new:
                metadata.update({"full_aggregate_state_sha256": "f"*64,
                                 "evaluation_semantics": {"kind": "fixed_weight_leave_one_out_client_contribution"}})
            (directory / "metadata.json").write_text(json.dumps(metadata))
            for role in ("reports.json", "audit.json"):
                rows = json.loads((directory / role).read_text())
                for row in rows:
                    # One known context reversal, with every other observation paired.
                    if not is_new and key == ("IID", 20261014, "gradual_stripe", 5) and row["candidate_id"] == 0:
                        row["mean_brier_gain"] = -.1
                        if role == "reports.json":
                            row["flipped_mean_brier_gain"] = .1
                    gain = row["mean_brier_gain"]
                    if is_new:
                        row.update({"full_aggregate_mean_brier": .4,
                                    "without_client_mean_brier": .4+gain,
                                    "full_aggregate_mean_cross_entropy": .8,
                                    "without_client_mean_cross_entropy": .8+gain,
                                    "full_aggregate_accuracy": .75,
                                    "without_client_accuracy": .75-gain/2})
                        if role == "reports.json":
                            row.update({"flipped_full_aggregate_mean_brier": .5,
                                        "flipped_without_client_mean_brier": .5+row["flipped_mean_brier_gain"]})
                    else:
                        row.update({"baseline_mean_brier": .5, "candidate_mean_brier": .5-gain,
                                    "baseline_mean_cross_entropy": 1., "baseline_accuracy": .7})
                (directory / role).write_text(json.dumps(rows))
            training = [{"client_id": cid, "submitted_state_sha256": hashlib.sha256(str((key, cid)).encode()).hexdigest()}
                        for cid in range(10)]
            (directory / "training.json").write_text(json.dumps(training))
            record["artifact_sha256"] = {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                                         for name in ("metadata.json", "reports.json", "audit.json", "training.json")}
    (old / "result_manifest.json").write_text(json.dumps({"records": old_records}))
    (new / "result_manifest.json").write_text(json.dumps({"records": new_records}))
    return old, new


class AggregateValidationAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.old, cls.new = make_pair(Path(cls.temporary.name))
        cls.cases, cls.inputs, _ = prior._load_cases(cls.new)
        cls.old_cases, cls.old_inputs, _ = prior._load_cases(cls.old)
        cls.raw, cls.old_raw = _read_case_rows(cls.inputs), _read_case_rows(cls.old_inputs)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_new_protocol_preserves_fixed_gates_without_mutating_prior(self):
        self.assertEqual(DIAGNOSTIC_PROTOCOL["primary_gate"], prior.DIAGNOSTIC_PROTOCOL["primary_gate"])
        self.assertEqual(DIAGNOSTIC_PROTOCOL["profiles"], prior.DIAGNOSTIC_PROTOCOL["profiles"])
        self.assertNotEqual(DIAGNOSTIC_PROTOCOL["reports"]["metric"], prior.DIAGNOSTIC_PROTOCOL["reports"]["metric"])
        self.assertIn("Retained weights are not renormalised", DIAGNOSTIC_PROTOCOL["candidate_model"])
        self.assertIn("Common clean FedAvg broadcast", prior.DIAGNOSTIC_PROTOCOL["candidate_model"])
        self.assertNotEqual(DIAGNOSTIC_PROTOCOL["version"], prior.DIAGNOSTIC_PROTOCOL["version"])

    def test_exact_pairing_verifies_all_960_updates(self):
        result = validate_pairing(self.cases, self.old_cases, self.raw, self.old_raw)
        self.assertEqual(result["cases"], 96)
        self.assertEqual(result["candidate_updates"], 960)
        self.assertTrue(result["all_paired"])
        changed = copy.deepcopy(self.cases)
        changed[0]["metadata"]["private_plan_sha256"] = "different"
        with self.assertRaisesRegex(ValueError, "private_plan_sha256"):
            validate_pairing(changed, self.old_cases, self.raw, self.old_raw)
        modified = copy.deepcopy(self.raw)
        key = next(iter(modified))
        modified[key]["training"][0]["submitted_state_sha256"] = "different"
        with self.assertRaisesRegex(ValueError, "Submitted local models differ"):
            validate_pairing(self.cases, self.old_cases, modified, self.old_raw)

    def test_brier_contribution_sign_and_declared_context_are_checked(self):
        modified = copy.deepcopy(self.raw)
        key = next(iter(modified))
        modified[key]["report_rows"][0]["mean_brier_gain"] *= -1
        with self.assertRaisesRegex(ValueError, "inconsistent sign"):
            validate_pairing(self.cases, self.old_cases, modified, self.old_raw)
        changed = copy.deepcopy(self.cases)
        changed[0]["metadata"]["evaluation_semantics"] = {"kind": "isolated_update"}
        with self.assertRaisesRegex(ValueError, "leave-one-out contribution"):
            validate_pairing(changed, self.old_cases, self.raw, self.old_raw)

    def test_paired_decision_transition_and_denominator_are_known(self):
        current, _ = prior.decision_rows(self.cases)
        previous, _ = prior.decision_rows(self.old_cases)
        paired = paired_decision_rows(current, previous)
        self.assertEqual(len(paired), 11520)
        row = next(r for r in paired if (r["regime"], r["seed"], r["scenario"], r["round"], r["candidate_id"], r["rule"], r["profile"]) == ("IID", 20261014, "gradual_stripe", 5, 0, "peer_median", "truthful"))
        self.assertFalse(row["isolated_accepted"])
        self.assertTrue(row["aggregate_accepted"])
        self.assertEqual(row["acceptance_change"], 1)
        grouped = paired_grouped_rates(paired)
        cell = next(r for r in grouped if (r["regime"], r["seed"], r["scenario"], r["group"], r["rule"], r["profile"]) == ("IID", 20261014, "gradual_stripe", "changed", "peer_median", "truthful"))
        self.assertEqual((cell["isolated_accepted"], cell["aggregate_accepted"], cell["n_paired_decisions"]), (11, 12, 12))
        self.assertAlmostEqual(cell["success_change_pp"], 100/12)
        with self.assertRaisesRegex(ValueError, "incomplete or duplicated"):
            paired_decision_rows(current[:-1], previous)

    def test_full_aggregate_audit_reuses_broadcast_once_per_validator(self):
        rows, groups = full_aggregate_audit(self.cases, self.raw, self.old_raw)
        self.assertEqual(len(rows), 960)
        self.assertEqual(len(groups), 192)
        self.assertTrue(all(abs(r["full_aggregate_brier_gain"]-.1) < 1e-12 for r in rows))
        self.assertTrue(all(abs(r["full_aggregate_accuracy_gain_pp"]-5.) < 1e-12 for r in rows))
        self.assertEqual({r["audit_examples"] for r in groups if r["domain_group"] == "all_domains"}, {1280})
        changed = copy.deepcopy(self.raw)
        key = next(iter(changed))
        changed[key]["audit_rows"][10]["full_aggregate_accuracy"] = .6
        with self.assertRaisesRegex(ValueError, "Repeated full/broadcast audit"):
            full_aggregate_audit(self.cases, changed, self.old_raw)

    def test_complete_output_schema_and_prior_artifacts_unchanged(self):
        prior_digest = hashlib.sha256((self.old / "result_manifest.json").read_bytes()).hexdigest()
        result = analyse_study(self.new, self.old)
        self.assertEqual(result["counts"]["candidate_rule_profile_decisions"], 11520)
        self.assertTrue(result["gate"]["pass"])
        self.assertEqual(result["gate"]["selected_rule"], "peer_median")
        self.assertEqual(result["paired_comparison"]["pairing"]["candidate_updates"], 960)
        self.assertTrue(result["repair_equivalence"]["all_equal"])
        self.assertEqual(result["cost"]["declared_no_cache_total_forward_examples"], 2703360)
        self.assertEqual(result["cost"]["naive_peer_model_transmissions_per_case"], 100)
        for name in ("summary.json", "pairing_audit.json", "paired_decisions.csv", "paired_grouped_rates.csv",
                     "paired_audit_context.csv", "full_aggregate_audit.csv", "full_aggregate_audit_groups.csv"):
            self.assertTrue((self.new / "analysis" / name).exists())
        self.assertFalse((self.old / "analysis").exists())
        self.assertEqual(hashlib.sha256((self.old / "result_manifest.json").read_bytes()).hexdigest(), prior_digest)
        with self.assertRaisesRegex(ValueError, "cannot overwrite"):
            analyse_study(self.old, self.old)


if __name__ == "__main__":
    unittest.main()
