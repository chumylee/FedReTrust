import copy
from dataclasses import replace
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiment_core import ClientSubmission
from fedretrust_reference_free import ReferenceFreeConfig, ReferenceFreeFedReTrust, restored_weight
from scenario_metrics import ContributionRecorder


def clients(points, offset=None):
    return [ClientSubmission(i, 10, {"x": torch.tensor(p, dtype=torch.float64) +
                                  (0 if offset is None else offset)}) for i, p in enumerate(points)]


class ReferenceFreeChecks(unittest.TestCase):
    def test_uniform_updates_preserved_and_coefficients_reconstruct(self):
        s = clients([[1., 2.]] * 10)
        r = ContributionRecorder(ReferenceFreeFedReTrust()).aggregate(s, {"x": torch.zeros(2, dtype=torch.float64)}, 1)
        torch.testing.assert_close(r.state["x"], torch.tensor([1., 2.], dtype=torch.float64))
        self.assertAlmostEqual(sum(r.coefficients.values()), 1.)
        self.assertEqual(r.diagnostics["reference_forward_examples"], 0)

    def test_large_aligned_updates_retained_more_than_opposing_updates(self):
        b = {"x": torch.zeros(2, dtype=torch.float64)}
        honest = ReferenceFreeFedReTrust().aggregate(clients([[1., .1]] * 7 + [[8., 1.]] * 3), b, 1)
        bad = ReferenceFreeFedReTrust().aggregate(clients([[1., .1]] * 7 + [[-8., -1.]] * 3), b, 1)
        self.assertGreater(honest.coefficients[8], bad.coefficients[8])
        self.assertGreater(honest.state["x"][0], 1.)
        self.assertGreater(bad.state["x"][0], .8)

    def test_magnitude_bound_under_arbitrarily_large_minority(self):
        b = {"x": torch.zeros(2, dtype=torch.float64)}
        for size in (1e2, 1e6, 1e10):
            r = ReferenceFreeFedReTrust().aggregate(clients([[1., 0.]] * 7 + [[size, size]] * 3), b, 1)
            self.assertLessEqual(float(r.state["x"].norm()), 6. + 1e-10)

    def test_recovery_observes_fresh_updates_even_after_zero(self):
        options = ReferenceFreeConfig()
        self.assertEqual(restored_weight(1., 0., options), 0.)
        self.assertEqual(restored_weight(0., 1., options), .5)
        self.assertEqual(restored_weight(.5, 1., options), 1.)
        self.assertEqual(restored_weight(0., 1., replace(options, use_restoration=False)), 1.)

    def test_zero_and_cancelling_inputs_finite(self):
        b = {"x": torch.zeros(2, dtype=torch.float64)}
        for points in ([[0., 0.]] * 10, [[1., 0.]] * 5 + [[-1., 0.]] * 5):
            r = ContributionRecorder(ReferenceFreeFedReTrust()).aggregate(clients(points), b, 1)
            self.assertTrue(torch.isfinite(r.state["x"]).all())
            self.assertTrue(r.diagnostics["bounded_rfa_fallback"])

    def test_permutation_and_broadcast_translation(self):
        points = [[1., .1], [1.2, .2], [.8, -.1], [9., 1.]]
        b = {"x": torch.zeros(2, dtype=torch.float64)}
        first = ReferenceFreeFedReTrust().aggregate(clients(points), b, 1)
        permuted = ReferenceFreeFedReTrust().aggregate(list(reversed(clients(points))), b, 1)
        shift = torch.tensor([20., -8.], dtype=torch.float64)
        shifted = ReferenceFreeFedReTrust().aggregate(clients(points, shift), {"x": shift}, 1)
        torch.testing.assert_close(first.state["x"], permuted.state["x"])
        torch.testing.assert_close(first.state["x"] + shift, shifted.state["x"])

    def test_failed_call_does_not_commit_memory_or_mutate_inputs(self):
        agg = ReferenceFreeFedReTrust()
        b = {"x": torch.zeros(2, dtype=torch.float64)}
        s = clients([[1., 2.]] * 10)
        original = copy.deepcopy(s)
        agg.aggregate(s, b, 1)
        for left, right in zip(s, original):
            torch.testing.assert_close(left.state["x"], right.state["x"])
        before = copy.deepcopy(agg.weights)
        s[0].state["x"][0] = float("nan")
        with self.assertRaises(FloatingPointError): agg.aggregate(s, b, 2)
        self.assertEqual(agg.weights, before)
        self.assertEqual(agg.last_round, 1)
        with self.assertRaises(ValueError): agg.aggregate(original, b, 1)
        agg.reset()
        self.assertEqual(agg.weights, {})
        self.assertEqual(agg.last_round, 0)


if __name__ == "__main__":
    unittest.main()
