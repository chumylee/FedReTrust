"""Known-answer checks for the offline diagnostic selection and leakage boundary."""
import copy
import hashlib
import itertools
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mechanism_diagnostics import (BASELINE, DIAGNOSTIC_PROTOCOL, METHODS, REGIMES, SCENARIOS, SEEDS,
    SCORE_SPECS, analyse_diagnostics, analyse_probe, calibrate_scores, conservative_threshold,
    evaluate_gate, event_group, group_metrics, load_observations, score_value)


def scenario(name):
    events = []
    if name != "clean":
        attack = name == "label_flip"
        events = [{"clients": [0, 1, 2], "windows": [[5, 24]],
                   "kind": "label_flip" if attack else "input_change",
                   "intent": "malicious" if attack else "honest"}]
    return {"name": name, "events": events}


def feature_row(attack):
    return {"whole_cosine": 1., "penultimate_cosine": 0. if attack else 1.,
            "head_cosine": 0. if attack else 1., "head_class_cosine_mean": 1.,
            "head_class_cosine_min": 1., "layer_disagreement": 0.,
            "temporal_cosine": 1., "temporal_relative_change": 0., "head_relative_norm": 1.}


def synthetic_observations():
    rows = []
    for method, regime, seed, name in itertools.product(METHODS, REGIMES, SEEDS, SCENARIOS):
        for round_number in range(5, 25):
            for client in range(10):
                group, attack = event_group(scenario(name), client, round_number)
                features = feature_row(attack)
                rows.append({"method": method, "regime": regime, "seed": seed,
                             "scenario": name, "round": round_number, "client": client,
                             "group": group, "attack": attack,
                             "scores": {spec[0]: score_value(features, spec) for spec in SCORE_SPECS}})
    return rows


def write_grid(root):
    records = []
    for i, (method, regime, seed, name) in enumerate(itertools.product(METHODS, REGIMES, SEEDS, SCENARIOS)):
        path = root / str(i)
        path.mkdir()
        job = {"method": method, "regime": regime, "seed": seed, "scenario": name}
        metadata = {"scenario": scenario(name), "config": {"rounds": 24, "development_seed": seed},
                    "official_test_used": False}
        diagnostics = []
        for round_number in range(1, 25):
            features = {str(client): feature_row(event_group(metadata["scenario"], client, round_number)[1])
                        for client in range(10)}
            if round_number == 1:
                for row in features.values():
                    row["temporal_cosine"] = None
                    row["temporal_relative_change"] = None
            diagnostics.append({"round": round_number,
                                "values": {"update_observation": {"feature_rows": features}}})
        (path / "run.json").write_text(json.dumps(metadata))
        (path / "aggregation_diagnostics.json").write_text(json.dumps(diagnostics))
        records.append({"status": "complete", "job": job, "run_path": str(path)})
    return records


def write_probe(root, contradiction=False):
    blocks, all_features, all_truth = [], [], []
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    for i, (regime, seed, round_number) in enumerate(itertools.product(REGIMES, SEEDS, (5, 8, 12, 20))):
        directory = root / f"block_{i}"
        directory.mkdir()
        provenance = {"regime": regime, "seed": seed, "round": round_number,
                      "broadcast_state_sha256": f"common_broadcast_{i}"}
        features, truth = [], []
        for name in SCENARIOS:
            for client in range(10):
                group, attack = event_group(scenario(name), client, round_number)
                identity = {**provenance, "scenario": name, "client": client}
                values = feature_row(attack)
                values["temporal_cosine"] = values["temporal_relative_change"] = None
                if contradiction and name == "gradual_stripe" and client < 3 and regime == "non-IID" and seed == SEEDS[1]:
                    values["penultimate_cosine"] = 0.
                features.append({**identity, "features": values})
                truth.append({**identity, "is_attack": attack,
                              "is_honest_change": group in {"noise_changed", "stripe_changed"}})
        for name, value in (("features.json", features), ("offline_truth.json", truth),
                            ("training.json", []), ("diagnostics.json", [])):
            (directory / name).write_text(json.dumps(value))
        for name in ("features.csv", "offline_truth.csv"):
            (directory / name).write_text("synthetic_fixture\n")
        names = ("features.json", "offline_truth.json", "training.json", "diagnostics.json",
                 "features.csv", "offline_truth.csv")
        result = {"status": "complete", "provenance": provenance, "local_trainings": 40,
                  "official_test_used": False, "clean_replay_identical": True,
                  "temporal_features_usable": False, "reference_forward_examples": 0,
                  "evaluation_forward_examples": 0,
                  "artifact_sha256": {name: digest(directory / name) for name in names}}
        (directory / "result.json").write_text(json.dumps(result))
        blocks.append({"directory": directory.name, "provenance": provenance,
                       "result_sha256": digest(directory / "result.json")})
        all_features.extend(features)
        all_truth.extend(truth)
    for name, value in (("features.json", all_features), ("offline_truth.json", all_truth)):
        (root / name).write_text(json.dumps(value))
    for name in ("features.csv", "offline_truth.csv"):
        (root / name).write_text("synthetic_fixture\n")
    summary = {"status": "complete", "blocks": blocks, "local_trainings": 640,
               "feature_rows": 640, "official_test_used": False, "temporal_features_usable": False,
               "reference_forward_examples": 0, "evaluation_forward_examples": 0,
               "artifact_sha256": {name: digest(root / name) for name in
                                   ("features.json", "features.csv", "offline_truth.json", "offline_truth.csv")}}
    (root / "summary.json").write_text(json.dumps(summary))


def frozen_diagnostic_summary():
    observations = synthetic_observations()
    calibration, selected = calibrate_scores(observations)
    return {"protocol": copy.deepcopy(DIAGNOSTIC_PROTOCOL), "selected_score": selected,
            "calibration": calibration,
            "gate": evaluate_gate(group_metrics(observations, calibration), calibration, selected)}


class MechanismDiagnosticTests(unittest.TestCase):
    def test_conservative_threshold_exact_counts_and_ties(self):
        for n in (1, 19, 20, 39, 40, 60, 140, 200):
            values = list(range(n))
            threshold = conservative_threshold(values)
            self.assertEqual(sum(value > threshold for value in values), n // 20)
        self.assertEqual(conservative_threshold([3.] * 60), 3.)
        self.assertEqual(sum(value > 3. for value in [3.] * 60), 0)
        with self.assertRaises(ValueError):
            conservative_threshold([None, 1.])

    def test_missing_and_invalid_scores_never_become_honest_zeros(self):
        spec = SCORE_SPECS[0]
        for value in (None, float("nan"), float("inf"), True, -1.1, 1.1):
            self.assertIsNone(score_value({"whole_cosine": value}, spec))
        self.assertEqual(score_value({"whole_cosine": -1.}, spec), 2.)
        self.assertEqual(score_value({"whole_cosine": 1.00000001}, spec), 0.)

    def test_truth_comes_from_recorded_event_and_active_window(self):
        changed = {"name": "label_flip", "events": [{"clients": [7, 8, 9],
                   "windows": [[9, 12]], "kind": "label_flip", "intent": "malicious"}]}
        self.assertEqual(event_group(changed, 7, 8), ("label_honest", False))
        self.assertEqual(event_group(changed, 7, 9), ("attack", True))
        self.assertEqual(event_group(changed, 0, 9), ("label_honest", False))
        self.assertEqual(event_group(changed, 7, 13), ("label_honest", False))

    def test_calibration_worst_group_and_fixed_tie_break_without_check_leakage(self):
        observations = synthetic_observations()
        calibration, selected = calibrate_scores(observations)
        self.assertEqual(selected, "penultimate_cosine")  # head_cosine ties exactly.
        self.assertEqual(calibration[selected]["minimum_attack_recall"], 1.)
        self.assertEqual(calibration[BASELINE]["minimum_attack_recall"], 0.)
        # Arbitrarily change every check slice. Selection and cutoffs must not move.
        for row in observations:
            if (row["method"], row["seed"]) != ("FedAvg", SEEDS[0]):
                row["scores"][selected] = None
                row["scores"]["head_cosine"] = 100.
        repeated, repeated_selected = calibrate_scores(observations)
        self.assertEqual((repeated, repeated_selected), (calibration, selected))
        gate = evaluate_gate(group_metrics(observations, calibration), calibration, selected)
        self.assertFalse(gate["pass"])
        self.assertTrue(all(not row["pass"] for row in gate["checks"]))
        # A high honest score in just one subgroup fixes the global cutoff.
        original = synthetic_observations()
        for row in original:
            if (row["method"], row["seed"], row["regime"], row["group"]) == ("FedAvg", SEEDS[0], "non-IID", "stripe_changed"):
                row["scores"]["penultimate_cosine"] = .8
        recalibrated, _ = calibrate_scores(original)
        self.assertEqual(recalibrated["penultimate_cosine"]["threshold"], .8)

    def test_gate_requires_each_trajectory_and_honest_minority(self):
        observations = synthetic_observations()
        calibration, selected = calibrate_scores(observations)
        self.assertTrue(evaluate_gate(group_metrics(observations, calibration), calibration, selected)["pass"])
        for row in observations:
            if (row["method"], row["seed"], row["regime"], row["group"]) == ("NNM-RFA", SEEDS[0], "non-IID", "noise_changed"):
                row["scores"][selected] = 1.
        gate = evaluate_gate(group_metrics(observations, calibration), calibration, selected)
        self.assertFalse(gate["pass"])
        self.assertEqual([r["pass"] for r in gate["checks"]], [True, False, True])
        self.assertIn("5% FPR", gate["checks"][1]["reasons"][0])

    def test_gate_fails_without_improvement_even_if_recall_is_high(self):
        observations = synthetic_observations()
        for row in observations:
            if row["attack"]:
                row["scores"][BASELINE] = 1.
        calibration, selected = calibrate_scores(observations)
        gate = evaluate_gate(group_metrics(observations, calibration), calibration, selected)
        self.assertFalse(gate["pass"])
        self.assertTrue(all("10 percentage points" in r["reasons"][0] for r in gate["checks"]))

    def test_missing_calibration_disqualifies_candidate_without_imputation(self):
        observations = synthetic_observations()
        observations[0]["scores"]["penultimate_cosine"] = None
        calibration, selected = calibrate_scores(observations)
        self.assertFalse(calibration["penultimate_cosine"]["eligible"])
        self.assertEqual(selected, "head_cosine")
        for row in observations:
            if (row["method"], row["seed"]) == ("FedAvg", SEEDS[0]):
                for name, _, _ in SCORE_SPECS[1:]:
                    row["scores"][name] = None
        calibration, selected = calibrate_scores(observations)
        self.assertIsNone(selected)
        self.assertFalse(evaluate_gate(group_metrics(observations, calibration), calibration, selected)["pass"])

    def test_all_32_records_rounds_and_clients_are_consumed_and_audited(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = write_grid(root)
            result = analyse_diagnostics(records, root / "analysis")
            self.assertEqual(result["n_runs"], 32)
            self.assertEqual(result["n_recorded_client_rounds"], 7680)
            self.assertEqual(result["n_analysis_client_rounds"], 6400)
            self.assertEqual(result["selected_score"], "penultimate_cosine")
            self.assertTrue(result["gate"]["pass"])
            self.assertEqual(len(result["inputs"]), 32)
            self.assertTrue((root / "analysis/diagnostic_groups.csv").exists())
            json.loads((root / "analysis/diagnostic_summary.json").read_text())
            with self.assertRaisesRegex(ValueError, "Incomplete diagnostic grid"):
                load_observations(records[:-1])
            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_observations(records + [records[0]])
            invalid = copy.deepcopy(records)
            invalid[0]["status"] = "failed"
            with self.assertRaisesRegex(ValueError, "incomplete"):
                load_observations(invalid)
            path = Path(records[0]["run_path"]) / "aggregation_diagnostics.json"
            diagnostics = json.loads(path.read_text())
            diagnostics[-1]["values"]["update_observation"]["feature_rows"].pop("9")
            path.write_text(json.dumps(diagnostics))
            with self.assertRaisesRegex(ValueError, "client coverage"):
                load_observations(records)

    def test_probe_uses_only_frozen_scores_and_never_rescues_a_failed_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_probe(root)
            frozen = frozen_diagnostic_summary()
            frozen["gate"]["pass"] = False
            before = copy.deepcopy(frozen)
            result = analyse_probe(root, frozen, root / "analysis")
            self.assertEqual(frozen, before)
            self.assertEqual(result["n_observations"], 640)
            self.assertEqual(set(row["score"] for row in result["groups"]), {BASELINE, "penultimate_cosine"})
            self.assertTrue(result["selected_score_evaluable"])
            self.assertFalse(result["trajectory_gate_pass"])
            self.assertFalse(result["trajectory_gate_modified"])
            self.assertFalse(result["descriptive_probe_contradiction"])
            self.assertEqual(len(result["groups"]), 56)
            self.assertEqual({row["n_rows"] for row in result["groups"] if row["attack"]}, {12})
            self.assertTrue((root / "analysis/probe_score_groups.csv").exists())
            # Temporal selection remains selected and unevaluable, with no substitute.
            frozen["selected_score"] = "temporal_cosine"
            temporal = analyse_probe(root, frozen, root / "temporal_analysis")
            self.assertEqual(temporal["selected_score_status"], "temporal_unevaluable_no_history")
            self.assertFalse(temporal["selected_score_evaluable"])
            self.assertIsNone(temporal["descriptive_probe_contradiction"])
            self.assertEqual(set(row["score"] for row in temporal["groups"]), {BASELINE, "temporal_cosine"})

    def test_probe_reports_a_minority_contradiction_without_changing_trajectory_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_probe(root, contradiction=True)
            result = analyse_probe(root, frozen_diagnostic_summary(), root / "analysis")
            self.assertTrue(result["trajectory_gate_pass"])
            self.assertTrue(result["descriptive_probe_contradiction"])
            contradicted = [row for row in result["selected_score_slices"] if row["descriptive_contradiction"]]
            self.assertEqual([(row["regime"], row["seed"]) for row in contradicted], [("non-IID", SEEDS[1])])

    def test_probe_refuses_tampered_combined_or_block_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_probe(root)
            combined = root / "features.json"
            original = combined.read_text()
            combined.write_text(original + " ")
            with self.assertRaisesRegex(ValueError, "artifact fingerprint"):
                analyse_probe(root, frozen_diagnostic_summary(), root / "analysis")
            combined.write_text(original)
            (root / "block_0/training.json").write_text("[1]")
            with self.assertRaisesRegex(ValueError, "artifact fingerprint"):
                analyse_probe(root, frozen_diagnostic_summary(), root / "analysis")


if __name__ == "__main__":
    unittest.main()
