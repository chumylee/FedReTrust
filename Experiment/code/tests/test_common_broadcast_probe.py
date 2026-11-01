"""Probe isolation, atomic resumption and provenance checks; no real training."""
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import copy
import csv
import io
import json
import sys
import tempfile
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common_broadcast_probe as probe
from experiment_core import TrainConfig, file_sha256, state_sha256, write_json
from research_runs import canonical_hash, make_aggregator
from scenario_metrics import longer_scenarios


def small_state():
    shapes = {"features.0.weight": (2, 1, 1, 1), "features.0.bias": (2,),
              "features.3.weight": (2, 2, 1, 1), "features.3.bias": (2,),
              "classifier.1.weight": (3, 2), "classifier.1.bias": (3,),
              "classifier.3.weight": (10, 3), "classifier.3.bias": (10,)}
    return {key: torch.zeros(shape, dtype=torch.float64) for key, shape in shapes.items()}


class TinyModel:
    loads = 0

    def __init__(self): self.state = small_state()

    def load_state_dict(self, state, strict=True):
        type(self).loads += 1
        self.state = copy.deepcopy(state)

    def state_dict(self): return self.state


def fixture_data(seed=20261014, partition="partition"):
    selection = {"clients": {str(i): list(range(512 * i, 512 * (i + 1))) for i in range(10)},
                 "development_validation": [], "index_space": "official_train", "selection_seed": seed}
    pool = SimpleNamespace(data=torch.zeros((5120, 1, 1), dtype=torch.uint8), targets=torch.arange(5120) % 10)
    return SimpleNamespace(selection=selection, pool=pool, manifest_sha256=partition,
                           manifest={"normalization": {"mean": [.1], "std": [.5]}})


class CommonBroadcastProbeTests(unittest.TestCase):
    def test_cases_keep_same_broadcast_and_samples_with_fresh_temporal_state(self):
        data, broadcast = fixture_data(), small_state()
        cfg = TrainConfig(rounds=24)
        digest, calls = state_sha256(broadcast), []
        TinyModel.loads = 0

        def fake_training(model, view, config, cid, round_number, given_broadcast):
            self.assertEqual(state_sha256(given_broadcast), digest)
            self.assertEqual(state_sha256(model.state_dict()), digest)
            calls.append((cid, round_number, view.indices, int(view.changed.sum()), view.flip_labels))
            sign = -1 if view.flip_labels else 1
            delta = sign * (cid + 1) * .01 + int(view.changed.sum()) * .00001
            return {key: value + delta for key, value in given_broadcast.items()}, {
                "online_train_loss": 1., "shuffle_seed": 7 + cid, "optimizer_steps": 8}

        with patch.object(probe, "MnistCNN", TinyModel), patch.object(probe, "train_client", fake_training):
            for name in probe.CONDITIONS:
                observation, trained, truth = probe.probe_case(data, cfg, broadcast, longer_scenarios()[name], 8)
                self.assertEqual(len(trained), 10)
                self.assertEqual(observation["client_order"], list(range(10)))
                for values in observation["feature_rows"].values():
                    self.assertIsNone(values["temporal_cosine"])
                    self.assertIsNone(values["temporal_relative_change"])
                    self.assertNotIn("is_attack", values)
                self.assertEqual(sum(row["is_attack"] for row in truth), 3 if name == "label_flip" else 0)
                self.assertEqual(sum(row["is_honest_change"] for row in truth),
                                 3 if name in {"abrupt_shot_noise", "gradual_stripe"} else 0)
        self.assertEqual(len(calls), 40)
        self.assertEqual(TinyModel.loads, 4)  # Initial load only; no returned aggregate is installed.
        self.assertEqual(state_sha256(broadcast), digest)
        for cid, round_number, indices, _, _ in calls:
            self.assertEqual(round_number, 8)
            self.assertEqual(indices, tuple(data.selection["clients"][str(cid)]))
        self.assertEqual(calls[10][3], 512)  # Abrupt noise.
        self.assertEqual(calls[20][3], 256)  # Stripe ramp at round 8, not full-strength replay.
        self.assertTrue(calls[30][4])

    def test_record_selection_requires_exact_declared_pairs(self):
        records = [{"job": {"method": "FedAvg", "scenario": "clean", "regime": regime, "seed": seed}}
                   for regime in probe.PARTITIONS for seed in probe.SEEDS]
        self.assertEqual(len(probe._select_records(records)), 4)
        for invalid in (records[:-1], records + [records[0]]):
            with self.assertRaises(ValueError): probe._select_records(invalid)

    def test_baseline_validation_rejects_selection_and_checkpoint_identity_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "code").mkdir()
            (root / "code/experiment_core.py").write_text("saved source\n")
            run = root / "outputs/baseline"
            run.mkdir(parents=True)
            cfg = TrainConfig(rounds=24)
            selection = fixture_data().selection
            code_digest = file_sha256(root / "code/experiment_core.py")
            # Use the REAL factory to catch adapter-name/settings integration mismatches.
            observed = probe.UpdateObserver(make_aggregator("FedAvg", fixture_data(), cfg))
            metadata = {"config": asdict(cfg), "official_test_used": False,
                        "method": {"name": observed.name, "settings": observed.settings},
                        "scenario": {"name": "clean"}, "code_sha256": {"experiment_core.py": code_digest},
                        "selection_sha256": canonical_hash(selection)}
            write_json(run / "run.json", metadata)
            write_json(run / "status.json", {"status": "complete"})
            write_json(run / "selection.json", selection)
            checkpoints = {r: small_state() for r in probe.CHECKPOINT_ROUNDS}
            torch.save(checkpoints, run / "observation_checkpoints.pt")
            rows = [{"round": r, "client": cid, "broadcast_state_sha256": state_sha256(checkpoints[r]),
                     "submitted_state_sha256": "client-state", "shuffle_seed": cid}
                    for r in probe.CHECKPOINT_ROUNDS for cid in range(10)]
            probe._write_csv(run / "client_metrics.csv", rows)
            names = [path.name for path in run.iterdir()]
            record = {"job": {"method": "FedAvg", "scenario": "clean", "regime": "IID", "seed": 20261014},
                      "status": "complete", "run_path": "outputs/baseline",
                      "source_sha256": {"experiment_core.py": code_digest},
                      "artifact_sha256": {name: file_sha256(run / name) for name in names}}
            with patch.object(probe, "ROOT", root):
                self.assertEqual(probe._validate_baseline(record)[1], cfg)
                for invalid_method in ({"name": "FedAvg", "settings": probe.FedAvgAggregator.settings},
                                       {**metadata["method"], "settings": {**observed.settings, "server_learning_rate": .5}},
                                       {**metadata["method"], "settings": {**observed.settings, "base_method": "RFA"}},
                                       {**metadata["method"], "settings": {**observed.settings, "base_settings": {"weighting": "uniform"}}}):
                    write_json(run / "run.json", {**metadata, "method": invalid_method})
                    record["artifact_sha256"]["run.json"] = file_sha256(run / "run.json")
                    with self.subTest(method=invalid_method), self.assertRaisesRegex(ValueError, "rate-one FedAvg adapter"):
                        probe._validate_baseline(record)
                write_json(run / "run.json", metadata)
                record["artifact_sha256"]["run.json"] = file_sha256(run / "run.json")
                # Even if an artifact manifest is refreshed, the semantic selection must match the run.
                changed = copy.deepcopy(selection)
                changed["clients"]["0"][0] = 99999
                write_json(run / "selection.json", changed)
                record["artifact_sha256"]["selection.json"] = file_sha256(run / "selection.json")
                with self.assertRaisesRegex(ValueError, "selection fingerprint"):
                    probe._validate_baseline(record)
                write_json(run / "selection.json", selection)
                record["artifact_sha256"]["selection.json"] = file_sha256(run / "selection.json")
                rows[0]["broadcast_state_sha256"] = "wrong-broadcast"
                probe._write_csv(run / "client_metrics.csv", rows)
                record["artifact_sha256"]["client_metrics.csv"] = file_sha256(run / "client_metrics.csv")
                with self.assertRaisesRegex(ValueError, "pre-update broadcast"):
                    probe._validate_baseline(record)

    def test_atomic_resume_does_no_extra_local_training_and_rejects_tampered_block(self):
        records = [{"job": {"method": "FedAvg", "scenario": "clean", "regime": regime, "seed": seed},
                    "artifact_sha256": {"observation_checkpoints.pt": "fixed"}}
                   for regime in probe.PARTITIONS for seed in probe.SEEDS]
        case_calls = []

        def validated(record):
            job = record["job"]
            cfg = TrainConfig(rounds=24, partition_id=probe.PARTITIONS[job["regime"]], development_seed=job["seed"])
            data = fixture_data(job["seed"])
            metadata = {"config": asdict(cfg), "selection_sha256": canonical_hash(data.selection), "partition_sha256": "partition"}
            checkpoints = {r: small_state() for r in probe.CHECKPOINT_ROUNDS}
            rows = {r: {cid: {"submitted_state_sha256": f"{job['seed']}/{r}/{cid}", "shuffle_seed": str(cid)}
                        for cid in range(10)} for r in probe.CHECKPOINT_ROUNDS}
            return probe.ROOT / "outputs/mock", cfg, metadata, data.selection, checkpoints, rows

        def fake_case(data, cfg, broadcast, scenario, round_number):
            case_calls.append((cfg.development_seed, cfg.partition_id, scenario.name, round_number))
            rows = {str(cid): {"whole_cosine": .5, "temporal_cosine": None, "temporal_relative_change": None,
                              "layer_cosines": [None, .5, .5, .5], "head_class_cosines": [.5] * 10} for cid in range(10)}
            trained = [{"client": cid, "shuffle_seed": cid,
                        "submitted_state_sha256": f"{cfg.development_seed}/{round_number}/{cid}"} for cid in range(10)]
            truth = [{"client": cid, "is_attack": scenario.name == "label_flip" and cid < 3} for cid in range(10)]
            return {"client_order": list(range(10)), "feature_rows": rows, "diagnostics_seconds": 0.}, trained, truth

        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(probe, "_validate_baseline", validated), \
                patch.object(probe, "_source_hashes", return_value={"source": "fixed"}), \
                patch.object(probe, "load_data", side_effect=lambda root, cfg, task_names: fixture_data(cfg.development_seed)), \
                patch.object(probe, "probe_case", fake_case), patch("sys.stdout", new_callable=io.StringIO):
            output = Path(temporary)
            result = probe.run_probe(records, output)
            self.assertEqual(len(case_calls), 64)
            self.assertEqual(result["feature_rows"], 640)
            self.assertEqual(result["local_trainings"], 640)
            self.assertFalse(result["official_test_used"])
            self.assertEqual(result["reference_forward_examples"], 0)
            resumed = probe.run_probe(records, output)
            self.assertEqual(len(case_calls), 64)
            self.assertEqual(resumed, result)
            features = json.loads((output / "features.json").read_text())
            self.assertTrue(all("is_attack" not in row and "is_attack" not in row["features"] for row in features))
            with (output / "features.csv").open(newline="") as handle:
                csv_rows = list(csv.DictReader(handle))
            self.assertEqual(csv_rows[0]["temporal_cosine"], "")
            self.assertEqual(csv_rows[0]["layer_cosine_features.0"], "")
            block = output / result["blocks"][0]["directory"]
            (block / "features.json").write_text("[]\n")
            with self.assertRaisesRegex(ValueError, "artifact changed"):
                probe.run_probe(records, output)
            self.assertEqual(len(case_calls), 64)


if __name__ == "__main__":
    unittest.main()
