import copy
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiment_core import ClientSubmission
from fedretrust_directional import CANDIDATE_CONFIGS, DirectionalConfig, DirectionalFedReTrust
from scenario_metrics import ContributionRecorder


def clients(points, offset=0):
    return [ClientSubmission(i, 10, {"x": torch.tensor(p, dtype=torch.float64) + offset})
            for i, p in enumerate(points)]


class DirectionalChecks(unittest.TestCase):
    def test_opposing_nearby_updates_rejected_without_distance_gate(self):
        b = {"x": torch.zeros(2, dtype=torch.float64)}
        for cfg in CANDIDATE_CONFIGS.values():
            r = ContributionRecorder(DirectionalFedReTrust(cfg)).aggregate(
                clients([[1., .1]] * 7 + [[-.25, -.025]] * 3), b, 1)
            self.assertEqual(r.coefficients[7], 0.)
            torch.testing.assert_close(r.state["x"], torch.tensor([1., .1], dtype=torch.float64))

    def test_aligned_large_minority_retained_but_norm_bounded(self):
        b = {"x": torch.zeros(2, dtype=torch.float64)}
        for cfg in CANDIDATE_CONFIGS.values():
            for size in (4., 1e3, 1e9):
                r = ContributionRecorder(DirectionalFedReTrust(cfg)).aggregate(
                    clients([[1., 0.]] * 7 + [[size, 0.]] * 3), b, 1)
                self.assertGreater(float(r.state["x"][0]), 1.)
                self.assertLessEqual(float(r.state["x"].norm()), 6. + 1e-10)
                if size == 4.:
                    self.assertAlmostEqual(r.coefficients[7], .1)

    def test_repair_restores_zero_client_and_absence_does_not(self):
        b = {"x": torch.zeros(2, dtype=torch.float64)}
        for cfg in CANDIDATE_CONFIGS.values():
            a = DirectionalFedReTrust(cfg)
            a.aggregate(clients([[1., 0.]] * 7 + [[-1., 0.]] * 3), b, 1)
            self.assertEqual(a.weights[9], 0.)
            a.aggregate(clients([[1., 0.]] * 7), b, 2)
            self.assertEqual(a.weights[9], 0.)
            a.aggregate(clients([[1., 0.]] * 10), b, 3)
            self.assertEqual(a.weights[9], .5)
            a.aggregate(clients([[1., 0.]] * 10), b, 4)
            self.assertEqual(a.weights[9], 1.)

    def test_uniform_updates_sample_weighted_and_zero_reference(self):
        b = {"x": torch.zeros(2, dtype=torch.float64)}
        for cfg in CANDIDATE_CONFIGS.values():
            s = clients([[1., 2.]] * 10)
            s[0].num_examples = 20
            r = ContributionRecorder(DirectionalFedReTrust(cfg)).aggregate(s, b, 1)
            torch.testing.assert_close(r.state["x"], torch.tensor([1., 2.], dtype=torch.float64))
            self.assertAlmostEqual(r.coefficients[0], 2/11)
            self.assertEqual(r.diagnostics["reference_forward_examples"], 0)

    def test_degenerate_geometry_falls_back_without_changing_memory(self):
        b = {"x": torch.zeros(2, dtype=torch.float64)}
        for cfg in CANDIDATE_CONFIGS.values():
            for points in ([[0., 0.]] * 10, [[1., 0.]] * 5 + [[-1., 0.]] * 5):
                a = DirectionalFedReTrust(cfg)
                a.weights = {0: .25}
                r = ContributionRecorder(a).aggregate(clients(points), b, 1)
                self.assertTrue(r.diagnostics["bounded_rfa_fallback"])
                self.assertEqual(a.weights, {0: .25})
                self.assertTrue(torch.isfinite(r.state["x"]).all())

    def test_permutation_and_broadcast_translation(self):
        b = {"x": torch.zeros(2, dtype=torch.float64)}
        shift = torch.tensor([20., -8.], dtype=torch.float64)
        points = [[1., .1], [1.2, .2], [.8, -.1], [9., 1.]]
        for cfg in CANDIDATE_CONFIGS.values():
            a = DirectionalFedReTrust(cfg).aggregate(clients(points), b, 1)
            p = DirectionalFedReTrust(cfg).aggregate(list(reversed(clients(points))), b, 1)
            t = DirectionalFedReTrust(cfg).aggregate(clients(points, shift), {"x": shift}, 1)
            torch.testing.assert_close(a.state["x"], p.state["x"])
            torch.testing.assert_close(a.state["x"] + shift, t.state["x"])

    def test_invalid_input_atomic_and_reset(self):
        b = {"x": torch.zeros(2, dtype=torch.float64)}
        for cfg in CANDIDATE_CONFIGS.values():
            a = DirectionalFedReTrust(cfg)
            s = clients([[1., 2.]] * 10)
            before = copy.deepcopy(s)
            a.aggregate(s, b, 1)
            for left, right in zip(s, before):
                torch.testing.assert_close(left.state["x"], right.state["x"])
            saved = copy.deepcopy(a.weights)
            s[0].state["x"][0] = float("nan")
            with self.assertRaises(FloatingPointError): a.aggregate(s, b, 2)
            self.assertEqual(saved, a.weights)
            self.assertEqual(a.last_round, 1)
            with self.assertRaises(ValueError): a.aggregate(before, b, 1)
            a.reset()
            self.assertEqual(a.weights, {})
            self.assertEqual(a.last_round, 0)

    def test_invalid_configuration(self):
        for kw in ({"alignment_full":0}, {"alignment_full":1.1}, {"unit_anchor":1},
                   {"norm_cap":float("nan")}, {"restoration_step":2}, {"epsilon":0}):
            with self.assertRaises(ValueError): DirectionalConfig(**kw)


if __name__ == "__main__":
    unittest.main()
