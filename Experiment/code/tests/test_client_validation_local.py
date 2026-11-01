"""Local holdout isolation and paired-loss checks; no client training."""
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import copy
import json
import math
import random
import struct
import sys
import tempfile
import unittest

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import client_validation_local as local
from experiment_core import TrainConfig, file_sha256, state_sha256, write_json
from research_runs import canonical_hash


class IndexedView(Dataset):
    def __init__(self, indices, labels):
        self.indices, self.labels = tuple(indices), labels
        self.pool, self.flip_labels = SimpleNamespace(train=True), False

    def __len__(self): return len(self.indices)

    def __getitem__(self, position):
        index = self.indices[position]
        return torch.tensor([index]), int(self.labels[index])


class LookupModel(nn.Module):
    calls = 0

    def __init__(self):
        super().__init__()
        # Intentionally consumes RNG; the public evaluator must restore it.
        self.table = nn.Parameter(torch.randn((4, 10), dtype=torch.float64))

    def forward(self, images):
        type(self).calls += 1
        return self.table[images[:, 0].long()]


def models():
    baseline = {"table": torch.zeros((4, 10), dtype=torch.float64)}
    candidates = {cid: copy.deepcopy(baseline) for cid in range(10)}
    probabilities = torch.full((4, 10), .2/9, dtype=torch.float64)
    for row in range(4): probabilities[row, row] = .8
    candidates[0] = {"table": probabilities.log()}
    return baseline, candidates


def data_fixture(root):
    label_path = root / "data/MNIST/raw/train-labels-idx1-ubyte"
    label_path.parent.mkdir(parents=True)
    label_path.write_bytes(struct.pack(">II", 2049, 60000) + (np.arange(60000) % 10).astype(np.uint8).tobytes())
    label_digest = file_sha256(label_path)
    records, cases = [], {}
    for regime, partition_id in local.PARTITIONS.items():
        pools = {str(cid): list(range(6300 + cid * 5370, 6300 + (cid + 1) * 5370)) for cid in range(10)}
        if regime == "non-IID": pools = {cid: list(reversed(values)) for cid, values in pools.items()}
        partition = {"official_train_size": 60000,
                     "index_spaces": {"clients": "official_train", "server_reference": "official_train",
                                      "development_validation": "official_train", "reserved_final_test": "official_test"},
                     "indices": {"clients": pools, "server_reference": list(range(300)),
                                 "development_validation": list(range(300, 6300))},
                     "dataset_files": {label_path.name: {"raw_sha256": label_digest}}}
        path = root / "data/partitions" / partition_id / "indices.json"
        path.parent.mkdir(parents=True)
        write_json(path, partition)
        for seed in local.SEEDS:
            offset = 0 if seed == local.SEEDS[0] else 256
            selection = {"clients": {cid: values[offset:offset + 512] for cid, values in pools.items()},
                         "development_validation": list(range(300, 1300)),
                         "selection_seed": seed, "index_space": "official_train"}
            cfg = TrainConfig(partition_id=partition_id, development_seed=seed, rounds=24)
            run = root / "outputs" / f"{regime}_{seed}"
            run.mkdir(parents=True)
            record = {"job": {"method": "FedAvg", "scenario": "clean", "regime": regime, "seed": seed},
                      "artifact_sha256": {"observation_checkpoints.pt": "verified_by_stub"}}
            metadata = {"partition_sha256": file_sha256(path), "selection_sha256": canonical_hash(selection)}
            cases[(regime, seed)] = [run, cfg, metadata, selection, {}, {}]
            records.append(record)

    def validated(record):
        return cases[(record["job"]["regime"], record["job"]["seed"])]
    return records, cases, validated


class ClientValidationLocalTests(unittest.TestCase):
    def test_unstratified_plan_deterministic_globally_disjoint_and_excludes_every_training_pool(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records, cases, validated = data_fixture(root)
            torch_rng, numpy_rng, python_rng = torch.random.get_rng_state().clone(), np.random.get_state(), random.getstate()
            with patch.object(local, "_validate_baseline", side_effect=validated) as verifier:
                plan = local.build_local_validation_plan(root, records)
                repeated = local.build_local_validation_plan(root, list(reversed(records)))
                self.assertEqual(verifier.call_count, 8)
            self.assertEqual(plan, repeated)
            self.assertTrue(torch.equal(torch_rng, torch.random.get_rng_state()))
            np.testing.assert_equal(numpy_rng, np.random.get_state())
            self.assertEqual(python_rng, random.getstate())
            forbidden = set(plan["exclusions"]["training_indices"]) | set(range(6300))
            expected_training = {index for case in cases.values() for values in case[3]["clients"].values() for index in values}
            self.assertEqual(set(plan["exclusions"]["training_indices"]), expected_training)
            self.assertEqual(plan["exclusions"]["training_union_size"], 15360)
            report_all, audit_all = set(), set()
            for regime, seeds in plan["clients"].items():
                for seed, clients in seeds.items():
                    partition = json.loads((root / "data/partitions" / local.PARTITIONS[regime] / "indices.json").read_text())
                    for cid, split in clients.items():
                        for role, accumulator in (("report", report_all), ("audit", audit_all)):
                            indices = split[f"{role}_indices"]
                            self.assertEqual(len(indices), 128)
                            self.assertEqual(indices, sorted(set(indices)))
                            self.assertFalse(set(indices) & forbidden)
                            self.assertTrue(set(indices) <= set(partition["indices"]["clients"][cid]))
                            self.assertEqual(sum(split[f"{role}_class_counts"]), 128)
                            accumulator.update(indices)
            self.assertFalse(report_all & audit_all)
            self.assertFalse(set(plan["role_banks"]["report_indices"]) & set(plan["role_banks"]["audit_indices"]))
            self.assertEqual(plan["storage_boundary"], "client_side_only")
            self.assertFalse(plan["official_test_loaded"])
            json.dumps(plan, allow_nan=False)

    def test_plan_rejects_reserved_training_indices_missing_records_and_invalid_sizes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records, cases, validated = data_fixture(root)
            with patch.object(local, "_validate_baseline", side_effect=validated):
                for invalid in (records[:-1], records + [records[0]]):
                    with self.assertRaises(ValueError): local.build_local_validation_plan(root, invalid)
                for count in (0, -1, True, 1.5):
                    with self.assertRaises(ValueError): local.build_local_validation_plan(root, records, report_size=count)
                with self.assertRaisesRegex(ValueError, "Insufficient"):
                    local.build_local_validation_plan(root, records, report_size=5000)
                case = cases[("IID", local.SEEDS[0])]
                case[3]["clients"]["0"][0] = 300  # A globally reserved development example.
                case[2]["selection_sha256"] = canonical_hash(case[3])
                with self.assertRaisesRegex(ValueError, "own client pool"):
                    local.build_local_validation_plan(root, records)

    def test_brier_known_probabilities_paired_gain_missing_classes_and_flipped_targets(self):
        broadcast, candidates = models()
        report, audit = IndexedView([0, 1], [0, 1, 2, 3]), IndexedView([2, 3], [0, 1, 2, 3])
        cfg = TrainConfig(evaluation_batch_size=128)
        LookupModel.calls = 0
        with patch.object(local, "MnistCNN", LookupModel):
            result = local.evaluate_local_candidates(broadcast, candidates, report, audit, cfg, 4)
        row = result["report_rows"][0]
        self.assertAlmostEqual(row["baseline_mean_brier"], .45)
        self.assertAlmostEqual(row["candidate_mean_brier"], 1/45)
        self.assertAlmostEqual(row["mean_brier_gain"], 77/180)
        self.assertAlmostEqual(row["mean_cross_entropy_gain"], math.log(8))
        self.assertEqual(row["accuracy_gain"], .5)
        self.assertEqual(row["class_counts"], [1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
        self.assertAlmostEqual(row["class_brier_gains"][0], 77/180)
        self.assertTrue(all(value is None for value in row["class_brier_gains"][2:]))
        self.assertAlmostEqual(row["flipped_candidate_mean_brier"], .8)
        self.assertAlmostEqual(row["flipped_mean_brier_gain"], -.35)
        self.assertEqual(row["flipped_class_counts"], [0, 0, 0, 0, 0, 0, 0, 0, 1, 1])
        self.assertEqual(result["report_rows"][1]["mean_brier_gain"], 0.)
        self.assertEqual(result["audit_rows"][0]["accuracy_gain"], 1.)
        self.assertTrue(all(not key.startswith("flipped_") for key in result["audit_rows"][0]))
        self.assertEqual(LookupModel.calls, 22)  # 11 models per role, no flipped-label re-evaluation.
        self.assertEqual(result["cost"]["total_forward_examples"], 44)
        self.assertEqual(result["cost"]["broadcast_forward_examples"], 4)
        self.assertEqual(result["cost"]["optimizer_steps"], 0)
        self.assertEqual(len(result["report_rows"]), 10)
        for split in ("report_rows", "audit_rows"):
            for item in result[split]:
                self.assertEqual(item["validator_id"], 4)
                self.assertFalse({"images", "indices", "labels", "logits", "predictions"} & set(item))
                self.assertGreaterEqual(item["mean_brier_gain"], -1.)
                self.assertLessEqual(item["mean_brier_gain"], 1.)
        json.dumps(result, allow_nan=False)

    def test_evaluator_preserves_input_states_rng_and_chunked_results(self):
        broadcast, candidates = models()
        views = (IndexedView([0, 1], [0, 1, 2, 3]), IndexedView([2, 3], [0, 1, 2, 3]))
        hashes = [state_sha256(broadcast), *[state_sha256(state) for state in candidates.values()]]
        rng, numpy_rng, python_rng = torch.random.get_rng_state().clone(), np.random.get_state(), random.getstate()
        with patch.object(local, "MnistCNN", LookupModel):
            a = local.evaluate_local_candidates(broadcast, candidates, *views, TrainConfig(evaluation_batch_size=1), 0)
            b = local.evaluate_local_candidates(broadcast, candidates, *views, TrainConfig(evaluation_batch_size=128), 0)
        self.assertEqual(a["report_rows"], b["report_rows"])
        self.assertEqual(a["audit_rows"], b["audit_rows"])
        self.assertEqual(a["cost"]["forward_batches"], 44)
        self.assertEqual(b["cost"]["forward_batches"], 22)
        self.assertEqual(hashes, [state_sha256(broadcast), *[state_sha256(state) for state in candidates.values()]])
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        np.testing.assert_equal(numpy_rng, np.random.get_state())
        self.assertEqual(python_rng, random.getstate())

    def test_overlapping_audit_flipped_views_official_test_and_missing_candidates_rejected(self):
        broadcast, candidates = models()
        report = IndexedView([0, 1], [0, 1, 2, 3])
        for invalid in (IndexedView([1, 2], [0, 1, 2, 3]), IndexedView([], [0, 1, 2, 3])):
            with self.assertRaises(ValueError):
                local.evaluate_local_candidates(broadcast, candidates, report, invalid, TrainConfig(), 0)
        audit = IndexedView([2, 3], [0, 1, 2, 3])
        audit.flip_labels = True
        with self.assertRaisesRegex(ValueError, "unflipped"):
            local.evaluate_local_candidates(broadcast, candidates, report, audit, TrainConfig(), 0)
        audit.flip_labels, audit.pool.train = False, False
        with self.assertRaisesRegex(ValueError, "Official-test"):
            local.evaluate_local_candidates(broadcast, candidates, report, audit, TrainConfig(), 0)
        audit.pool.train = True
        with self.assertRaises(ValueError):
            local.evaluate_local_candidates(broadcast, {0: candidates[0]}, report, audit, TrainConfig(), 0)
        broken = copy.deepcopy(candidates)
        broken[0]["table"][0, 0] = float("nan")
        with self.assertRaises(FloatingPointError):
            local.evaluate_local_candidates(broadcast, broken, report, audit, TrainConfig(), 0)


if __name__ == "__main__":
    unittest.main()
