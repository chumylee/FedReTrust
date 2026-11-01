"""Independent arithmetic checks for FLTrust and its trusted-reference boundary."""
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

import numpy as np
import torch
from torch.utils.data import TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiment_core import ClientSubmission, MnistCNN, TrainConfig, clone_state, evaluate, state_sha256
from fltrust import FLTrustConfig, RootReference, build_root_reference, flatten_delta, fltrust_rule, train_root


class FLTrustTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)

    def setUp(self):
        self.broadcast = {"w": torch.tensor([10., -4.], dtype=torch.float64)}
        self.root = torch.tensor([2., 0.], dtype=torch.float64)

    def submission(self, client, delta, count=512):
        return ClientSubmission(client, count, {"w": self.broadcast["w"] + torch.tensor(delta, dtype=torch.float64)})

    def test_independent_hand_calculation(self):
        # cosines 1 and .6; normalized updates [2,0] and [1.2,1.6].
        submissions = [self.submission(0, [5, 0]), self.submission(1, [3, 4])]
        result = fltrust_rule(submissions, self.broadcast, self.root)
        expected = self.broadcast["w"].numpy() + (np.array([2., 0.]) + .6 * np.array([1.2, 1.6])) / 1.6
        np.testing.assert_allclose(result.state["w"].numpy(), expected, rtol=0, atol=1e-12)
        self.assertAlmostEqual(result.coefficients[0], .625)
        self.assertAlmostEqual(result.coefficients[1], .375)
        self.assertAlmostEqual(result.diagnostics["client_metrics"]["1"]["trust_score"], .6)

    def test_negative_and_orthogonal_updates_have_zero_weight(self):
        submissions = [self.submission(0, [5, 0]), self.submission(1, [-9, 0]), self.submission(2, [0, 7])]
        result = fltrust_rule(submissions, self.broadcast, self.root)
        torch.testing.assert_close(result.state["w"], self.broadcast["w"] + self.root, rtol=0, atol=0)
        self.assertEqual(result.coefficients, {0: 1., 1: 0., 2: 0.})
        self.assertEqual(result.diagnostics["client_metrics"]["1"]["cosine_similarity"], -1.)

    def test_positive_magnitude_scaling_does_not_change_result(self):
        a = fltrust_rule([self.submission(0, [1, 0]), self.submission(1, [3, 4])], self.broadcast, self.root)
        b = fltrust_rule([self.submission(0, [1000, 0]), self.submission(1, [.3, .4])], self.broadcast, self.root)
        torch.testing.assert_close(a.state["w"], b.state["w"], rtol=0, atol=1e-12)
        self.assertAlmostEqual(b.diagnostics["client_metrics"]["1"]["normalized_update_l2_norm"], 2.)

    def test_sample_counts_are_not_an_extra_weight(self):
        a = fltrust_rule([self.submission(0, [1, 0], 1), self.submission(1, [3, 4], 1000)], self.broadcast, self.root)
        b = fltrust_rule([self.submission(0, [1, 0], 1000), self.submission(1, [3, 4], 1)], self.broadcast, self.root)
        torch.testing.assert_close(a.state["w"], b.state["w"], rtol=0, atol=0)

    def test_zero_client_update_is_excluded(self):
        result = fltrust_rule([self.submission(0, [0, 0]), self.submission(1, [1, 0])], self.broadcast, self.root)
        self.assertEqual(result.coefficients[0], 0)
        self.assertEqual(result.diagnostics["client_metrics"]["0"]["norm_scale"], 0)

    def test_zero_reference_keeps_global(self):
        result = fltrust_rule([self.submission(0, [1, 1])], self.broadcast, self.root * 0)
        self.assertEqual(state_sha256(result.state), state_sha256(self.broadcast))
        self.assertEqual(result.diagnostics["fallback_reason"], "zero_reference_update")

    def test_no_positive_trust_keeps_global(self):
        result = fltrust_rule([self.submission(0, [-1, 0]), self.submission(1, [0, 2])], self.broadcast, self.root)
        self.assertEqual(state_sha256(result.state), state_sha256(self.broadcast))
        self.assertEqual(result.diagnostics["fallback_reason"], "no_positive_trust")
        self.assertEqual(sum(result.coefficients.values()), 0)

    def test_global_learning_rate_scales_descent_delta(self):
        result = fltrust_rule([self.submission(0, [9, 0])], self.broadcast, self.root, global_learning_rate=.25)
        torch.testing.assert_close(result.state["w"], self.broadcast["w"] + .25 * self.root, rtol=0, atol=0)

    def test_invalid_inputs_are_rejected(self):
        item = self.submission(0, [1, 0])
        with self.assertRaises(ValueError):
            fltrust_rule([item, item], self.broadcast, self.root)
        with self.assertRaises(ValueError):
            fltrust_rule([item], self.broadcast, torch.ones(3))
        with self.assertRaises(FloatingPointError):
            fltrust_rule([self.submission(0, [float("nan"), 0])], self.broadcast, self.root)
        with self.assertRaises(ValueError):
            FLTrustConfig(root_steps=0)

    def test_inputs_are_not_modified_and_parameter_order_is_consistent(self):
        broadcast = {"a": torch.tensor([2.]), "b": torch.tensor([-1.])}
        local = {"b": torch.tensor([3.]), "a": torch.tensor([5.])}
        before = state_sha256(broadcast), state_sha256(local)
        delta = flatten_delta(local, broadcast)
        torch.testing.assert_close(delta, torch.tensor([3., 4.], dtype=torch.float64), rtol=0, atol=0)
        fltrust_rule([ClientSubmission(0, 1, local)], broadcast, delta)
        self.assertEqual(before, (state_sha256(broadcast), state_sha256(local)))

    def test_root_builder_copies_only_reserved_images(self):
        pool = SimpleNamespace(data=torch.arange(12 * 28 * 28).reshape(12, 28, 28).remainder(256).byte(),
                               targets=torch.arange(12).remainder(10))
        data = SimpleNamespace(pool=pool, manifest_sha256="test", manifest={
            "normalization": {"mean": [.1307], "std": [.3081]},
            "indices": {"server_reference": [0, 1], "clients": {"0": list(range(2, 10))}, "development_validation": [10, 11]}})
        reference = build_root_reference(data)
        self.assertEqual(len(reference.dataset), 2)
        self.assertFalse(hasattr(reference.dataset, "pool"))
        self.assertEqual(reference.metadata["original_indices"], [0, 1])
        copied = reference.dataset.tensors[0].clone()
        pool.data.zero_()
        self.assertTrue(torch.equal(copied, reference.dataset.tensors[0]))
        data.manifest["indices"]["server_reference"] = [2]
        with self.assertRaises(ValueError):
            build_root_reference(data)

    def test_root_training_is_reproducible_isolated_and_moves_downhill(self):
        torch.manual_seed(10)
        model = MnistCNN()
        broadcast = clone_state(model)
        images, targets = torch.zeros(4, 1, 28, 28), torch.zeros(4, dtype=torch.long)
        reference = RootReference(TensorDataset(images, targets), {"examples": 4})
        options = FLTrustConfig(root_steps=1, root_batch_size=4, root_momentum=0)
        before_rng = torch.get_rng_state().clone()
        delta, metrics = train_root(broadcast, reference, options, 20261014, 1)
        self.assertTrue(torch.equal(before_rng, torch.get_rng_state()))
        self.assertEqual(metrics["root_broadcast_state_sha256"], state_sha256(broadcast))
        repeated, _ = train_root(broadcast, reference, options, 20261014, 1)
        torch.testing.assert_close(delta, repeated, rtol=0, atol=0)
        local, offset = {}, 0
        for name, value in broadcast.items():
            local[name] = (value.double() + delta[offset:offset + value.numel()].reshape(value.shape)).to(value.dtype)
            offset += value.numel()
        result = fltrust_rule([ClientSubmission(0, 4, local)], broadcast, delta)
        config = TrainConfig(rounds=1)
        before_loss = evaluate(model, reference.dataset, config)["loss"]
        model.load_state_dict(result.state)
        after_loss = evaluate(model, reference.dataset, config)["loss"]
        self.assertLess(after_loss, before_loss)
        self.assertEqual(metrics["root_optimizer_steps"], 1)
        self.assertEqual(metrics["root_example_visits"], 4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
