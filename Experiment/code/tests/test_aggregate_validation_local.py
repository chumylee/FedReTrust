"""Fixed-weight counterfactual arithmetic and local contribution-sign checks."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import copy
import json
import math
import random
import sys
import unittest

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import aggregate_validation_local as aggregate
import client_validation_local as local
from experiment_core import TrainConfig, state_sha256


class IndexedView(Dataset):
    def __init__(self, indices):
        self.indices = tuple(indices)
        self.pool, self.flip_labels = SimpleNamespace(train=True), False

    def __len__(self): return len(self.indices)

    def __getitem__(self, position):
        index = self.indices[position]
        return torch.tensor([index]), index


class LookupModel(nn.Module):
    calls = 0

    def __init__(self):
        super().__init__()
        self.table = nn.Parameter(torch.randn((4, 10), dtype=torch.float64))

    def forward(self, images):
        type(self).calls += 1
        return self.table[images[:, 0].long()]


def known_models():
    probabilities = torch.full((4, 10), .2/9, dtype=torch.float64)
    for row in range(4): probabilities[row, row] = .8
    full = {"table": probabilities.log()}
    without = {cid: copy.deepcopy(full) for cid in range(10)}
    without[0] = {"table": torch.zeros((4, 10), dtype=torch.float64)}
    return full, without


class AggregateValidationLocalTests(unittest.TestCase):
    def test_fixed_weights_remove_one_contribution_without_renormalizing(self):
        broadcast = {"weight": torch.tensor([[10., -20.]], dtype=torch.float32),
                     "bias": torch.tensor([5.], dtype=torch.float32)}
        submitted = {cid: {name: value + cid + 1 for name, value in broadcast.items()} for cid in range(10)}
        full, without = aggregate.build_aggregate_counterfactuals(broadcast, submitted)
        for name, value in broadcast.items():
            torch.testing.assert_close(full[name], value + 5.5, rtol=0, atol=0)
            for cid in range(10):
                expected = (value.double() + 5.5 - .1*(cid+1)).float()
                torch.testing.assert_close(without[cid][name], expected, rtol=0, atol=0)
        self.assertEqual(float(without[9]["weight"][0, 0]), 14.5)
        self.assertNotEqual(float(without[9]["weight"][0, 0]), 15.)  # Mean of the other nine submitted models.

    def test_leave_one_out_uses_unrounded_float64_total(self):
        broadcast = {"x": torch.tensor([100000000.], dtype=torch.float32)}
        submitted = {cid: copy.deepcopy(broadcast) for cid in range(10)}
        submitted[0]["x"] += 16.
        submitted[1]["x"] += 32.
        full, without = aggregate.build_aggregate_counterfactuals(broadcast, submitted)
        self.assertEqual(float(full["x"]), 100000008.)
        self.assertEqual(float(without[0]["x"]), 100000000.)
        naive = (full["x"].double() - .1*(submitted[0]["x"].double()-broadcast["x"].double())).float()
        self.assertFalse(torch.equal(without[0]["x"], naive))

    def test_optional_weights_are_fixed_and_input_order_does_not_change_models(self):
        broadcast = {"x": torch.tensor([10.], dtype=torch.float64)}
        submitted = {cid: {"x": torch.tensor([11.+cid], dtype=torch.float64)} for cid in range(10)}
        weights = {cid: .55 if cid == 0 else .05 for cid in range(10)}
        full, without = aggregate.build_aggregate_counterfactuals(broadcast, submitted, weights)
        shuffled, excluded = aggregate.build_aggregate_counterfactuals(broadcast, dict(reversed(list(submitted.items()))), weights)
        self.assertAlmostEqual(float(full["x"]), 13.25)
        self.assertAlmostEqual(float(without[0]["x"]), 12.7)
        torch.testing.assert_close(full["x"], shuffled["x"], rtol=0, atol=0)
        for cid in without:
            torch.testing.assert_close(without[cid]["x"], excluded[cid]["x"], rtol=0, atol=0)

    def test_builder_rejects_invalid_inputs_and_preserves_states_and_rng(self):
        broadcast = {"x": torch.tensor([2.], dtype=torch.float32)}
        submitted = {cid: {"x": torch.tensor([float(cid)], dtype=torch.float32)} for cid in range(10)}
        before = [state_sha256(broadcast), *map(state_sha256, submitted.values())]
        rng = torch.random.get_rng_state().clone()
        aggregate.build_aggregate_counterfactuals(broadcast, submitted)
        self.assertEqual(before, [state_sha256(broadcast), *map(state_sha256, submitted.values())])
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        for weights in ({cid: .2 for cid in range(10)}, {cid: .1 for cid in range(9)},
                        {cid: (True if cid == 0 else .1) for cid in range(10)},
                        {cid: (float("nan") if cid == 0 else .1) for cid in range(10)},
                        {cid: (-.1 if cid == 0 else 1.1/9) for cid in range(10)}):
            with self.subTest(weights=weights), self.assertRaises(ValueError):
                aggregate.build_aggregate_counterfactuals(broadcast, submitted, weights)
        with self.assertRaises(ValueError):
            aggregate.build_aggregate_counterfactuals(broadcast, {0: submitted[0]})
        invalid = copy.deepcopy(submitted)
        invalid[0]["x"][0] = float("inf")
        with self.assertRaises(FloatingPointError): aggregate.build_aggregate_counterfactuals(broadcast, invalid)

    def test_known_positive_contribution_all_gain_signs_and_absolute_names(self):
        full, without = known_models()
        report, audit = IndexedView([0, 1]), IndexedView([2, 3])
        LookupModel.calls = 0
        with patch.object(local, "MnistCNN", LookupModel):
            result = aggregate.evaluate_aggregate_contributions(full, without, report, audit, TrainConfig(), 2)
        row = result["report_rows"][0]
        self.assertEqual(row["candidate_id"], 0)
        self.assertEqual(row["validator_id"], 2)
        self.assertAlmostEqual(row["mean_brier_gain"], 77/180)
        self.assertAlmostEqual(row["mean_cross_entropy_gain"], math.log(8))
        self.assertEqual(row["accuracy_gain"], .5)
        self.assertAlmostEqual(row["class_brier_gains"][0], 77/180)
        self.assertAlmostEqual(row["class_cross_entropy_gains"][1], math.log(8))
        self.assertTrue(all(value is None for value in row["class_brier_gains"][2:]))
        self.assertAlmostEqual(row["full_aggregate_mean_brier"], 1/45)
        self.assertAlmostEqual(row["without_client_mean_brier"], .45)
        self.assertAlmostEqual(row["flipped_mean_brier_gain"], -.35)
        self.assertAlmostEqual(row["flipped_class_brier_gains"][9], -.35)
        self.assertAlmostEqual(row["flipped_mean_cross_entropy_gain"], math.log(2/9))
        self.assertAlmostEqual(row["flipped_class_cross_entropy_gains"][8], math.log(2/9))
        self.assertEqual(row["flipped_accuracy_gain"], 0.)
        self.assertAlmostEqual(row["flipped_full_aggregate_mean_brier"], .8)
        self.assertAlmostEqual(row["flipped_without_client_mean_brier"], .45)
        self.assertEqual(result["audit_rows"][0]["accuracy_gain"], 1.)
        self.assertEqual(result["report_rows"][1]["mean_brier_gain"], 0.)
        for split in ("report_rows", "audit_rows"):
            for item in result[split]:
                self.assertFalse(any(key.startswith(("baseline_", "flipped_baseline_")) for key in item))
                self.assertFalse(any(key.startswith(("candidate_", "flipped_candidate_")) and key != "candidate_id" for key in item))
                self.assertFalse({"images", "indices", "labels", "logits", "predictions"} & set(item))
        self.assertEqual(LookupModel.calls, 22)
        self.assertEqual(result["cost"]["report_forward_examples"], 22)
        self.assertEqual(result["cost"]["audit_forward_examples"], 22)
        self.assertEqual(result["cost"]["full_aggregate_forward_examples"], 4)
        self.assertEqual(result["cost"]["without_client_forward_examples"], 40)
        self.assertEqual(result["cost"]["optimizer_steps"], 0)
        json.dumps(result, allow_nan=False)

    def test_removal_that_improves_loss_has_negative_contribution_with_rng_and_inputs_unchanged(self):
        good, alternatives = known_models()
        full = alternatives[0]
        without = {cid: copy.deepcopy(full) for cid in range(10)}
        without[0] = good
        hashes = [state_sha256(full), *map(state_sha256, without.values())]
        torch_rng, numpy_rng, python_rng = torch.random.get_rng_state().clone(), np.random.get_state(), random.getstate()
        report, audit = IndexedView([0, 1]), IndexedView([2, 3])
        with patch.object(local, "MnistCNN", LookupModel):
            result = aggregate.evaluate_aggregate_contributions(full, without, report, audit, TrainConfig(), 0)
        self.assertAlmostEqual(result["report_rows"][0]["mean_brier_gain"], -77/180)
        self.assertEqual(hashes, [state_sha256(full), *map(state_sha256, without.values())])
        self.assertTrue(torch.equal(torch_rng, torch.random.get_rng_state()))
        np.testing.assert_equal(numpy_rng, np.random.get_state())
        self.assertEqual(python_rng, random.getstate())
        with self.assertRaisesRegex(ValueError, "disjoint"):
            aggregate.evaluate_aggregate_contributions(full, without, report, report, TrainConfig(), 0)


if __name__ == "__main__":
    unittest.main()
