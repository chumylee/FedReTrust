"""Passive-observer invariance, geometry, and causal-history checks."""
from pathlib import Path
import copy
import json
import math
import random
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiment_core import AggregationResult, ClientSubmission, FedAvgAggregator, state_sha256
from rfa import RFAAggregator
from update_observer import LAYER_NAMES, PARAMETER_NAMES, SCALAR_FEATURES, UpdateObserver, observe_updates


def tiny_state(fill=0.0):
    shapes = ((1, 1, 1, 1), (1,), (1, 1, 1, 1), (1,), (2, 1), (2,), (10, 2), (10,))
    return {name: torch.full(shape, fill, dtype=torch.float64) for name, shape in zip(PARAMETER_NAMES, shapes)}


def client(client_id, fill=1.0, count=10, head=None):
    state = tiny_state(fill)
    if head is not None:
        state["classifier.3.weight"].fill_(head)
        state["classifier.3.bias"].fill_(head)
    return ClientSubmission(client_id, count, state)


class SpyAggregator:
    name = "spy"

    def __init__(self):
        self.settings = {"nested": {"option": 1}}
        self.reset_count = 0
        self.calls = []
        self.result = AggregationResult(tiny_state(7), {13: 0.7, 99: 0.3}, {"existing": {"value": 4}})

    def aggregate(self, submissions, broadcast, round_number):
        self.calls.append((submissions, broadcast, round_number, torch.is_inference_mode_enabled()))
        return self.result

    def reset(self):
        self.reset_count += 1


class UpdateObserverTests(unittest.TestCase):
    def test_exact_delegation_result_identity_and_preserved_existing_diagnostics(self):
        base = SpyAggregator()
        observer = UpdateObserver(base)
        clients, broadcast = [client(13), client(99, 2)], tiny_state()
        state_object, coefficients_object = base.result.state, base.result.coefficients
        got = observer.aggregate(clients, broadcast, 5)
        self.assertIs(got, base.result)
        self.assertIs(got.state, state_object)
        self.assertIs(got.coefficients, coefficients_object)
        self.assertEqual(got.diagnostics["existing"], {"value": 4})
        self.assertIs(base.calls[0][0], clients)
        self.assertIs(base.calls[0][1], broadcast)
        self.assertEqual(base.calls[0][2:], (5, False))
        self.assertEqual(observer.name, base.name)
        self.assertEqual(got.diagnostics["update_observation"]["client_order"], [13, 99])

    def test_fedavg_and_rfa_results_are_bitwise_unchanged(self):
        clients = [client(8, 1, 40), client(3, 2, 7, head=-1), client(20, -3, 11)]
        broadcast = tiny_state(.25)
        for factory in (FedAvgAggregator, RFAAggregator):
            with self.subTest(aggregator=factory.__name__):
                expected = factory().aggregate(clients, broadcast, 8)
                got = UpdateObserver(factory()).aggregate(clients, broadcast, 8)
                self.assertEqual(state_sha256(got.state), state_sha256(expected.state))
                self.assertEqual(got.coefficients, expected.coefficients)
                original_diagnostics = {k: v for k, v in got.diagnostics.items() if k != "update_observation"}
                self.assertEqual(original_diagnostics, expected.diagnostics)

    def test_no_input_rng_or_thread_changes_and_observations_repeat(self):
        clients, broadcast = [client(6), client(1, 2, head=-1)], tiny_state()
        before = [state_sha256(broadcast), *[state_sha256(c.state) for c in clients]]
        torch_rng, numpy_rng, python_rng = torch.random.get_rng_state().clone(), np.random.get_state(), random.getstate()
        threads = torch.get_num_threads()
        first, _ = observe_updates(clients, broadcast)
        second, _ = observe_updates(clients, broadcast)
        self.assertEqual(before, [state_sha256(broadcast), *[state_sha256(c.state) for c in clients]])
        self.assertTrue(torch.equal(torch_rng, torch.random.get_rng_state()))
        after_numpy = np.random.get_state()
        self.assertEqual(numpy_rng[0], after_numpy[0])
        np.testing.assert_array_equal(numpy_rng[1], after_numpy[1])
        self.assertEqual(numpy_rng[2:], after_numpy[2:])
        self.assertEqual(python_rng, random.getstate())
        self.assertEqual(threads, torch.get_num_threads())
        self.assertGreaterEqual(first.pop("diagnostics_seconds"), 0)
        second.pop("diagnostics_seconds")
        self.assertEqual(first, second)

    def test_layer_and_all_class_signs_use_sample_weighted_current_reference(self):
        clients, broadcast = [client(0, count=100), client(1, count=1, head=-1)], tiny_state()
        observed, _ = observe_updates(clients, broadcast)
        positive, negative = observed["feature_rows"]["0"], observed["feature_rows"]["1"]
        self.assertAlmostEqual(positive["head_cosine"], 1)
        self.assertAlmostEqual(negative["head_cosine"], -1)
        self.assertAlmostEqual(negative["penultimate_cosine"], 1)
        for value in negative["layer_cosines"][:3]: self.assertAlmostEqual(value, 1)
        self.assertAlmostEqual(negative["layer_cosines"][3], -1)
        self.assertEqual(len(negative["head_class_cosines"]), 10)
        for value in negative["head_class_cosines"]: self.assertAlmostEqual(value, -1)
        self.assertAlmostEqual(negative["head_class_cosine_mean"], -1)
        self.assertAlmostEqual(negative["head_class_cosine_min"], -1)
        self.assertAlmostEqual(negative["layer_disagreement"], 2)
        self.assertAlmostEqual(negative["head_relative_norm"], math.sqrt(30) / (math.sqrt(38) + 1e-12))
        # With equal counts the two head deltas cancel in the RFA reference.
        clients[0].num_examples = 1
        equal, _ = observe_updates(clients, broadcast)
        self.assertIsNone(equal["feature_rows"]["0"]["head_cosine"])
        self.assertIsNone(equal["feature_rows"]["1"]["head_class_cosine_min"])

    def test_each_output_row_includes_its_bias_and_covers_class_nine(self):
        good, changed = client(0, fill=0, count=100), client(1, fill=0, count=1)
        good.state["classifier.3.bias"].fill_(1)
        changed.state["classifier.3.bias"].fill_(1)
        changed.state["classifier.3.bias"][9] = -1
        observation, _ = observe_updates([good, changed], tiny_state())
        row = observation["feature_rows"]["1"]
        self.assertEqual(row["head_class_cosines"], [1.] * 9 + [-1.])
        self.assertEqual(row["head_class_cosine_min"], -1)
        self.assertAlmostEqual(row["head_class_cosine_mean"], .8)

    def test_zero_norms_and_first_temporal_values_are_null(self):
        observer = UpdateObserver(FedAvgAggregator())
        first = observer.aggregate([client(0, 0)], tiny_state(), 1).diagnostics["update_observation"]
        row = first["feature_rows"]["0"]
        self.assertEqual(set(row), set(SCALAR_FEATURES) | {"layer_cosines", "head_class_cosines"})
        for key in SCALAR_FEATURES:
            self.assertIsNone(row[key])
        self.assertEqual(row["layer_cosines"], [None] * 4)
        self.assertEqual(row["head_class_cosines"], [None] * 10)
        second = observer.aggregate([client(0, 0)], tiny_state(), 2).diagnostics["update_observation"]
        self.assertIsNone(second["feature_rows"]["0"]["temporal_cosine"])
        self.assertIsNone(second["feature_rows"]["0"]["temporal_relative_change"])
        json.dumps([first, second], allow_nan=False)

    def test_mixed_zero_layers_and_class_rows_do_not_fabricate_summary_scores(self):
        only_head = client(0, 0)
        only_head.state["classifier.3.bias"][:9] = 1
        observation, _ = observe_updates([only_head], tiny_state())
        row = observation["feature_rows"]["0"]
        self.assertAlmostEqual(row["whole_cosine"], 1)
        self.assertAlmostEqual(row["head_cosine"], 1)
        self.assertEqual(row["layer_cosines"][:3], [None] * 3)
        self.assertEqual(row["head_class_cosines"], [1.] * 9 + [None])
        self.assertIsNone(row["head_class_cosine_mean"])
        self.assertIsNone(row["head_class_cosine_min"])
        self.assertIsNone(row["layer_disagreement"])
        self.assertAlmostEqual(row["head_relative_norm"], 1)
        json.dumps(observation, allow_nan=False)

    def test_temporal_history_is_by_client_id_causal_and_snapshot_not_alias(self):
        observer = UpdateObserver(FedAvgAggregator(), checkpoint_rounds=())
        first_clients = [client(9, 1), client(3, -2)]
        first = observer.aggregate(first_clients, tiny_state(), 5).diagnostics["update_observation"]
        saved = copy.deepcopy(first)
        for tensor in first_clients[0].state.values(): tensor.fill_(1000)
        # New broadcast changes absolute models, while client deltas stay 2 and -4.
        next_clients = [client(3, 6), client(9, 12), client(5, 11)]
        next_broadcast = dict(reversed(list(tiny_state(10).items())))
        second = observer.aggregate(next_clients, next_broadcast, 8).diagnostics["update_observation"]
        for cid in ("9", "3"):
            self.assertAlmostEqual(second["feature_rows"][cid]["temporal_cosine"], 1)
            self.assertAlmostEqual(second["feature_rows"][cid]["temporal_relative_change"], 1)
        self.assertIsNone(second["feature_rows"]["5"]["temporal_cosine"])
        self.assertEqual(first, saved)
        # A previously absent client is compared with its own last observation.
        observer.aggregate([client(9, 2)], tiny_state(), 9)
        returned = observer.aggregate([client(3, -4)], tiny_state(), 12).diagnostics["update_observation"]
        self.assertAlmostEqual(returned["feature_rows"]["3"]["temporal_relative_change"], 0)
        with self.assertRaisesRegex(ValueError, "rounds must increase"):
            observer.aggregate([client(3)], tiny_state(), 8)

    def test_checkpoints_are_preaggregation_clones_and_reset_delegates(self):
        base = SpyAggregator()
        observer = UpdateObserver(base, checkpoint_rounds=(5, 8))
        broadcast = tiny_state(2)
        observer.aggregate([client(0, 3)], broadcast, 5)
        expected = state_sha256(broadcast)
        self.assertEqual(state_sha256(observer.checkpoints[5]), expected)
        self.assertNotEqual(state_sha256(observer.checkpoints[5]), state_sha256(base.result.state))
        for tensor in broadcast.values(): tensor.add_(10)
        self.assertEqual(state_sha256(observer.checkpoints[5]), expected)
        observer.reset()
        self.assertEqual(base.reset_count, 1)
        self.assertEqual(observer.checkpoints, {})
        base.result.diagnostics.pop("update_observation")
        observation = observer.aggregate([client(0)], tiny_state(), 1).diagnostics["update_observation"]
        self.assertIsNone(observation["feature_rows"]["0"]["temporal_cosine"])
        self.assertEqual(observer.checkpoints, {})

    def test_settings_are_independent_and_record_reference_semantics(self):
        base = SpyAggregator()
        observer = UpdateObserver(base)
        self.assertEqual(base.settings, {"nested": {"option": 1}})
        observer.settings["nested"]["option"] = 2
        self.assertEqual(base.settings["nested"]["option"], 1)
        metadata = observer.settings["update_observation"]
        self.assertEqual(metadata["layer_order"], list(LAYER_NAMES))
        self.assertEqual(metadata["reference_rfa"], {"max_iterations": 3, "smoothing": 1e-6, "relative_tolerance": 1e-6})
        self.assertEqual(len(metadata["implementation_sha256"]), 64)

    def test_failed_delegate_does_not_commit_observation_history_or_checkpoint(self):
        class FailedAggregator(FedAvgAggregator):
            def aggregate(self, submissions, broadcast, round_number):
                raise RuntimeError("synthetic delegate failure")
        observer = UpdateObserver(FailedAggregator())
        with self.assertRaisesRegex(RuntimeError, "synthetic"):
            observer.aggregate([client(0)], tiny_state(), 5)
        self.assertEqual(observer.checkpoints, {})
        observer.base = FedAvgAggregator()
        observation = observer.aggregate([client(0, 2)], tiny_state(), 5).diagnostics["update_observation"]
        self.assertIsNone(observation["feature_rows"]["0"]["temporal_cosine"])

    def test_invalid_model_schema_and_inputs_fail_before_delegation(self):
        for problem in ("missing", "extra", "head_classes", "head_width", "conv_width", "bias_rank", "nonfinite"):
            with self.subTest(problem=problem):
                broadcast = tiny_state()
                if problem == "missing": broadcast.pop("classifier.3.bias")
                elif problem == "extra": broadcast["other"] = torch.ones(1)
                elif problem == "head_classes":
                    broadcast["classifier.3.weight"] = torch.zeros((9, 2), dtype=torch.float64)
                    broadcast["classifier.3.bias"] = torch.zeros(9, dtype=torch.float64)
                elif problem == "head_width": broadcast["classifier.3.weight"] = torch.zeros((10, 3), dtype=torch.float64)
                elif problem == "conv_width": broadcast["features.3.weight"] = torch.zeros((1, 2, 1, 1), dtype=torch.float64)
                elif problem == "bias_rank": broadcast["features.0.bias"] = torch.zeros((1, 1), dtype=torch.float64)
                else: broadcast["features.0.bias"][0] = float("nan")
                base = SpyAggregator()
                with self.assertRaises((ValueError, FloatingPointError)):
                    UpdateObserver(base).aggregate([client(0)], broadcast, 1)
                self.assertEqual(base.calls, [])
        with self.assertRaises(ValueError): observe_updates([client(True)], tiny_state())
        with self.assertRaises(ValueError): observe_updates([client(0), client(0)], tiny_state())
        with self.assertRaises(ValueError): UpdateObserver(FedAvgAggregator(), checkpoint_rounds=(False,))


if __name__ == "__main__":
    unittest.main()
