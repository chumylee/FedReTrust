"""Check server-scale fidelity, particularly FLTrust's rescaled vectors."""
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

import numpy as np
import torch
from torch.utils.data import TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baseline_calibration import ServerLearningRate, server_rate_candidates
from experiment_core import ClientSubmission, FedAvgAggregator, MnistCNN, clone_state, state_sha256
from fltrust import FLTrustAggregator, FLTrustConfig, RootReference, fltrust_rule
from rfa import RFAAggregator
from viceroy import ViceroyAggregator
from viceroy_diagnostics import ServerStepControl
from test_viceroy import flat, submissions


class FixedRootFLTrust:
    name = "FLTrust"
    options = SimpleNamespace(global_learning_rate=1.0)
    settings = {"global_learning_rate": 1.0}

    def __init__(self, root):
        self.root = torch.tensor(root, dtype=torch.float64)

    def aggregate(self, clients, broadcast, round_number):
        return fltrust_rule(clients, broadcast, self.root)


class BaselineCalibrationTests(unittest.TestCase):
    def test_equal_grid_has_declared_viceroy_anchor(self):
        for name in ("FedAvg", "FLTrust", "RFA", "Viceroy"):
            grid = server_rate_candidates(name, 10)
            np.testing.assert_allclose([c["server_learning_rate"] for c in grid],
                                      [.05, .1, .2] if name == "Viceroy" else [.5, 1, 2])
        self.assertEqual(server_rate_candidates("Viceroy", 20)[1]["server_learning_rate"], .05)

    def test_identity_rate_preserves_each_base_state_and_native_scores(self):
        clients, broadcast = submissions([[.3, .2], [.1, .5], [2., -1.]], origin=[4., -2.])
        for factory in (FedAvgAggregator, RFAAggregator, ViceroyAggregator, lambda: FixedRootFLTrust([2., 0.])):
            expected = factory().aggregate(clients, broadcast, 1)
            actual = ServerLearningRate(factory(), 1).aggregate(clients, broadcast, 1)
            self.assertEqual(state_sha256(expected.state), state_sha256(actual.state))
            for client, weight in expected.coefficients.items():
                self.assertEqual(actual.diagnostics["client_metrics"][str(client)]["base_native_coefficient"], weight)

    def test_fedavg_extrapolation_preserves_unequal_sample_weights(self):
        clients, broadcast = submissions([[1., 0.], [0., 1.]], counts=[1, 3], origin=[4., -2.])
        result = ServerLearningRate(FedAvgAggregator(), 2).aggregate(clients, broadcast, 1)
        np.testing.assert_allclose(flat(result.state), [4.5, -.5], atol=1e-14)
        self.assertEqual(result.coefficients, {0: .5, 1: 1.5})

    def test_fltrust_scales_normalized_vectors_not_raw_trust_shares(self):
        clients, broadcast = submissions([[5., 0.], [3., 4.]], origin=[10., -4.])
        result = ServerLearningRate(FixedRootFLTrust([2., 0.]), .25).aggregate(clients, broadcast, 1)
        # Trust shares .625/.375, both norm scales .4; unscaled update [1.7,.6].
        np.testing.assert_allclose(flat(result.state), [10.425, -3.85], atol=1e-14)
        self.assertAlmostEqual(result.coefficients[0], .0625)
        self.assertAlmostEqual(result.coefficients[1], .0375)
        self.assertAlmostEqual(result.diagnostics["base_native_coefficient_mass"], 1.)
        self.assertAlmostEqual(result.diagnostics["base_effective_coefficient_mass"], .4)

    def test_fltrust_positive_magnitude_invariance_survives_server_scaling(self):
        first, broadcast = submissions([[5., 0.], [3., 4.]], origin=[10., -4.])
        second, _ = submissions([[500., 0.], [.3, .4]], origin=[10., -4.])
        a = ServerLearningRate(FixedRootFLTrust([2., 0.]), 2).aggregate(first, broadcast, 1)
        b = ServerLearningRate(FixedRootFLTrust([2., 0.]), 2).aggregate(second, broadcast, 1)
        torch.testing.assert_close(flat(a.state), flat(b.state), rtol=0, atol=1e-12)

    def test_actual_fltrust_can_train_root_before_inference_only_scaling(self):
        torch.manual_seed(31)
        broadcast = clone_state(MnistCNN())
        local = {k: v.clone() for k, v in broadcast.items()}
        local["classifier.3.bias"][0] += .1
        reference = RootReference(TensorDataset(torch.zeros(4, 1, 28, 28), torch.zeros(4, dtype=torch.long)),
                                  {"examples": 4})
        options = FLTrustConfig(root_steps=1, root_batch_size=4, root_momentum=0)
        base = FLTrustAggregator(reference, 20261014, options)
        result = ServerLearningRate(base, .5).aggregate([ClientSubmission(0, 4, local)], broadcast, 1)
        self.assertEqual(result.diagnostics["root_optimizer_steps"], 1)
        self.assertGreater(result.diagnostics["root_update_l2_norm"], 0)
        self.assertGreater(result.diagnostics["applied_update_l2_norm"], 0)
        self.assertTrue(all(torch.isfinite(v).all() for v in result.state.values()))

    def test_viceroy_matches_previous_fixed_scale_control_over_twenty_rounds(self):
        new = ServerLearningRate(ViceroyAggregator(), .1)
        old = ServerStepControl(ViceroyAggregator(), .1)
        rng = np.random.default_rng(327)
        for r in range(1, 21):
            clients, broadcast = submissions(rng.normal(size=(5, 9)), origin=np.full(9, .3))
            a, b = new.aggregate(clients, broadcast, r), old.aggregate(clients, broadcast, r)
            self.assertEqual(state_sha256(a.state), state_sha256(b.state))
            self.assertEqual(a.coefficients, b.coefficients)
            torch.testing.assert_close(new.base.histories, old.base.histories, rtol=0, atol=0)
            torch.testing.assert_close(new.base.reputations, old.base.reputations, rtol=0, atol=0)
        new.reset()
        self.assertEqual(new.base.last_round, 0)
        self.assertIsNone(new.base.histories)

    def test_fixed_scale_does_not_renormalize_viceroy_mass(self):
        clients, broadcast = submissions([[1., 0.], [1., 0.], [0., 1.]], origin=[4., -2.])
        result = ServerLearningRate(ViceroyAggregator(), .1).aggregate(clients, broadcast, 1)
        self.assertEqual(result.coefficients, {0: 0., 1: 0., 2: .1})
        np.testing.assert_allclose(flat(result.state), [4., -1.9], atol=1e-14)
        self.assertAlmostEqual(result.diagnostics["applied_effective_coefficient_mass"], .1)

    def test_scaled_rfa_remains_on_its_median_direction(self):
        clients, broadcast = submissions([[0., 1.], [.1, 1.], [8., -9.]], origin=[3., -2.])
        base = RFAAggregator().aggregate(clients, broadcast, 1)
        result = ServerLearningRate(RFAAggregator(), .5).aggregate(clients, broadcast, 1)
        np.testing.assert_allclose(flat(result.state), flat(broadcast) + .5 * (flat(base.state) - flat(broadcast)), atol=1e-12)
        self.assertEqual(result.diagnostics["iterations_used"], base.diagnostics["iterations_used"])

    def test_zero_update_policies_and_input_immutability(self):
        for factory, points in ((ViceroyAggregator, [[1., 2.]] * 3),
                                (lambda: FixedRootFLTrust([0., 0.]), [[1., 0.]] * 3)):
            clients, broadcast = submissions(points, origin=[3., 2.])
            before = [state_sha256(broadcast), *[state_sha256(s.state) for s in clients]]
            result = ServerLearningRate(factory(), .5).aggregate(clients, broadcast, 1)
            self.assertEqual(state_sha256(result.state), state_sha256(broadcast))
            self.assertEqual(before, [state_sha256(broadcast), *[state_sha256(s.state) for s in clients]])

    def test_invalid_rates_and_double_fltrust_scaling_are_rejected(self):
        for value in [0, -1, float("inf"), float("nan"), True]:
            with self.assertRaises(ValueError): ServerLearningRate(FedAvgAggregator(), value)
        base = FixedRootFLTrust([2., 0.])
        base.options = SimpleNamespace(global_learning_rate=.5)
        with self.assertRaises(ValueError): ServerLearningRate(base, .5)
        with self.assertRaises(ValueError): server_rate_candidates("FedAvg", 10, [1., 1.])
        with self.assertRaises(ValueError): server_rate_candidates("Viceroy", 0)
        with self.assertRaises(ValueError): server_rate_candidates("Unknown", 10)


if __name__ == "__main__": unittest.main()
