"""Verify experimental controls change only the declared server-step rule."""
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiment_core import FedAvgAggregator
from viceroy import ViceroyAggregator
from viceroy_diagnostics import ServerStepControl
from test_viceroy import flat, submissions


class ServerStepControlTests(unittest.TestCase):
    def test_fixed_scale_and_mass_normalization_are_distinct(self):
        clients, broadcast = submissions([[1., 0.], [0., 1.]], origin=[4., -2.])
        fixed = ServerStepControl(ViceroyAggregator(), .1).aggregate(clients, broadcast, 1)
        normalized = ServerStepControl(ViceroyAggregator(), normalize_mass=True).aggregate(clients, broadcast, 1)
        np.testing.assert_allclose(flat(fixed.state), [4.1, -1.9], atol=1e-14)
        np.testing.assert_allclose(flat(normalized.state), [4.5, -1.5], atol=1e-14)
        self.assertEqual(fixed.coefficients, {0: .1, 1: .1})
        self.assertEqual(normalized.coefficients, {0: .5, 1: .5})

    def test_enlarged_fedavg_keeps_sample_count_weighting(self):
        clients, broadcast = submissions([[1., 0.], [0., 1.]], counts=[1, 3])
        result = ServerStepControl(FedAvgAggregator(), 10).aggregate(clients, broadcast, 1)
        np.testing.assert_allclose(flat(result.state), [2.5, 7.5], atol=1e-14)
        self.assertEqual(result.coefficients, {0: 2.5, 1: 7.5})
        self.assertAlmostEqual(result.diagnostics["applied_cosine_to_fedavg"], 1.)
        self.assertAlmostEqual(result.diagnostics["applied_to_fedavg_norm_ratio"], 10.)

    def test_control_does_not_scale_the_history_observations(self):
        reference = ViceroyAggregator()
        controlled = ServerStepControl(ViceroyAggregator(), .1)
        rng = np.random.default_rng(45)
        for r in range(1, 4):
            clients, broadcast = submissions(rng.normal(size=(4, 5)))
            reference.aggregate(clients, broadcast, r)
            controlled.aggregate(clients, broadcast, r)
            torch.testing.assert_close(controlled.base.histories, reference.histories, rtol=0, atol=0)
            torch.testing.assert_close(controlled.base.reputations, reference.reputations, rtol=0, atol=0)
        controlled.reset()
        self.assertEqual(controlled.base.last_round, 0)

    def test_zero_mass_and_zero_delta_are_finite(self):
        for points in [[[1., 2.]] * 3, [[0., 0.]] * 3]:
            clients, broadcast = submissions(points, origin=[3., 2.])
            result = ServerStepControl(ViceroyAggregator(), normalize_mass=True).aggregate(clients, broadcast, 1)
            torch.testing.assert_close(flat(result.state), flat(broadcast), rtol=0, atol=0)
            self.assertIsNone(result.diagnostics["applied_cosine_to_fedavg"])

    def test_identity_scale_preserves_base_output_exactly(self):
        clients, broadcast = submissions([[.3, .2], [.1, .5], [2., -1.]])
        base = ViceroyAggregator().aggregate(clients, broadcast, 1)
        control = ServerStepControl(ViceroyAggregator()).aggregate(clients, broadcast, 1)
        self.assertEqual(base.coefficients, control.coefficients)
        torch.testing.assert_close(flat(base.state), flat(control.state), rtol=0, atol=0)

    def test_invalid_settings_rejected(self):
        for value in [0, -1, float('inf'), float('nan'), True]:
            with self.assertRaises(ValueError): ServerStepControl(FedAvgAggregator(), value)
        with self.assertRaises(ValueError): ServerStepControl(FedAvgAggregator(), normalize_mass=1)


if __name__ == "__main__": unittest.main()
