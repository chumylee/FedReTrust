import copy
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiment_core import ClientSubmission
from fedretrust_directional import DirectionalConfig, DirectionalFedReTrust
from fedretrust_v05 import FedReTrustV05
from rfa import RFAConfig, smoothed_weiszfeld
from scenario_metrics import ContributionRecorder


def clients(points, counts=None, offset=None):
    counts = counts or [10] * len(points)
    offset = torch.zeros(len(points[0]), dtype=torch.float64) if offset is None else offset
    return [ClientSubmission(i, count, {"x": torch.tensor(p, dtype=torch.float64) + offset})
            for i, (p, count) in enumerate(zip(points, counts))]


class FedReTrustV05Checks(unittest.TestCase):
    def setUp(self):
        self.broadcast = {"x": torch.zeros(2, dtype=torch.float64)}

    def test_fixed_v03_configuration_and_reference_free_metadata(self):
        rule = FedReTrustV05()
        self.assertEqual(rule.options, DirectionalConfig())
        self.assertEqual(rule.name, "FedReTrust-v05")
        self.assertEqual(rule.settings["trusted_reference_examples"], 0)
        self.assertFalse(rule.settings["secure_aggregation_implemented"])
        self.assertEqual(rule.settings["restoration_step"], .5)
        with self.assertRaises(TypeError):
            FedReTrustV05(DirectionalConfig(alignment_full=.5))

    def test_mass_complement_algebra_equal_and_heterogeneous_counts(self):
        points = [[1., .1]] * 7 + [[-.25, -.025]] * 3
        for counts in ([10] * 10, [13, 9, 11, 7, 17, 8, 12, 5, 6, 4]):
            with self.subTest(counts=counts):
                submissions = clients(points, counts)
                rule = FedReTrustV05()
                result = ContributionRecorder(rule).aggregate(submissions, self.broadcast, 1)
                old = DirectionalFedReTrust().aggregate(submissions, self.broadcast, 1)
                raw = torch.tensor(points, dtype=torch.float64)
                base = torch.tensor(counts, dtype=torch.float64)
                base /= base.sum()
                scales = torch.tensor([result.diagnostics["client_metrics"][str(i)]["clip_factor"]
                                       for i in range(len(points))], dtype=torch.float64)
                bounded = raw * scales[:, None]
                anchor, _, _ = smoothed_weiszfeld(bounded, base, RFAConfig())
                q = torch.tensor([rule.weights[i] for i in range(len(points))], dtype=torch.float64)
                m = float(torch.dot(base, q))
                direct = ((base * q)[:, None] * bounded).sum(0)
                torch.testing.assert_close(result.state["x"], direct + (1. - m) * anchor,
                                           rtol=1e-12, atol=1e-12)
                torch.testing.assert_close(result.state["x"], m * old.state["x"] + (1. - m) * anchor,
                                           rtol=1e-12, atol=1e-12)
                reconstructed = sum(result.coefficients[i] * raw[i] for i in range(len(points)))
                torch.testing.assert_close(result.state["x"], reconstructed, rtol=1e-12, atol=1e-12)
                for i in range(len(points)):
                    diag = result.diagnostics["client_metrics"][str(i)]
                    self.assertAlmostEqual(result.coefficients[i], diag["direct_effective_coefficient"]
                                           + diag["rfa_effective_coefficient"])

    def test_rejected_updates_have_zero_direct_but_positive_indirect_influence(self):
        rule = FedReTrustV05()
        result = rule.aggregate(clients([[1., .1]] * 7 + [[-.25, -.025]] * 3), self.broadcast, 1)
        for i in (7, 8, 9):
            self.assertEqual(rule.weights[i], 0.)
            self.assertEqual(result.diagnostics["client_metrics"][str(i)]["direct_effective_coefficient"], 0.)
            self.assertGreater(result.coefficients[i], 0.)
        self.assertAlmostEqual(result.diagnostics["trust_mass"], .7)
        self.assertAlmostEqual(result.diagnostics["robust_complement_mass"], .3)

    def test_all_q_one_exactly_matches_v03_including_clipping(self):
        for points in ([[1., .1], [.9, 0.], [1.1, .2], [1., -.1]],
                       [[1., 0.]] * 7 + [[1e6, 0.]] * 3):
            for counts in ([10] * len(points), list(range(1, len(points) + 1))):
                submissions = clients(points, counts)
                old = DirectionalFedReTrust().aggregate(submissions, self.broadcast, 1)
                rule = FedReTrustV05()
                new = rule.aggregate(submissions, self.broadcast, 1)
                self.assertTrue(all(q == 1. for q in rule.weights.values()))
                self.assertTrue(torch.equal(old.state["x"], new.state["x"]))
                self.assertEqual(old.coefficients, new.coefficients)
                self.assertEqual(new.diagnostics["robust_complement_mass"], 0.)

    def test_evidence_and_memory_match_v03_on_identical_inputs_and_state(self):
        # This compares shared inputs, not diverging closed-loop trajectories.
        old, new = DirectionalFedReTrust(), FedReTrustV05()
        cases = [clients([[1., .1]] * 7 + [[-.25, -.025]] * 3),
                 clients([[1., .1]] * 10), clients([[1., .1]] * 10)]
        for number, submissions in enumerate(cases, 1):
            a = old.aggregate(submissions, self.broadcast, number)
            b = new.aggregate(submissions, self.broadcast, number)
            self.assertEqual(old.weights, new.weights)
            for client in a.diagnostics["client_metrics"]:
                for key in ("anchor_cosine", "evidence_weight", "restored_weight", "clip_factor", "sample_share"):
                    self.assertEqual(a.diagnostics["client_metrics"][client][key],
                                     b.diagnostics["client_metrics"][client][key])
            self.assertEqual(b.diagnostics["reference_forward_examples"], 0)

    def test_restoration_absence_and_reset(self):
        rule = FedReTrustV05()
        rule.aggregate(clients([[1., 0.]] * 7 + [[-1., 0.]] * 3), self.broadcast, 1)
        self.assertEqual(rule.weights[9], 0.)
        rule.aggregate(clients([[1., 0.]] * 7), self.broadcast, 2)
        self.assertEqual(rule.weights[9], 0.)
        rule.aggregate(clients([[1., 0.]] * 10), self.broadcast, 3)
        self.assertEqual(rule.weights[9], .5)
        rule.aggregate(clients([[1., 0.]] * 10), self.broadcast, 4)
        self.assertEqual(rule.weights[9], 1.)
        rule.reset()
        self.assertEqual(rule.weights, {})
        self.assertEqual(rule.last_round, 0)

    def test_fallback_matches_v03_and_preserves_memory(self):
        for points in ([[0., 0.]] * 10, [[1., 0.]] * 5 + [[-1., 0.]] * 5):
            old, new = DirectionalFedReTrust(), FedReTrustV05()
            old.weights = {0: .25}
            new.weights = {0: .25}
            a = old.aggregate(clients(points), self.broadcast, 1)
            b = new.aggregate(clients(points), self.broadcast, 1)
            self.assertTrue(b.diagnostics["bounded_rfa_fallback"])
            self.assertEqual(new.weights, {0: .25})
            self.assertEqual(b.coefficients, a.coefficients)
            self.assertTrue(torch.equal(b.state["x"], a.state["x"]))
            self.assertEqual(b.diagnostics["robust_complement_mass"], 1.)

    def test_permutation_translation_and_nonmutation(self):
        points = [[1., .1], [.9, .05], [1.1, .2], [-.1, -.2]]
        submissions = clients(points, [10, 12, 8, 7])
        snapshot = copy.deepcopy(submissions)
        broadcast = copy.deepcopy(self.broadcast)
        first = FedReTrustV05().aggregate(submissions, broadcast, 1)
        permuted = FedReTrustV05().aggregate(list(reversed(submissions)), broadcast, 1)
        shift = torch.tensor([10., -8.], dtype=torch.float64)
        translated = FedReTrustV05().aggregate(clients(points, [10, 12, 8, 7], shift), {"x": shift}, 1)
        torch.testing.assert_close(first.state["x"], permuted.state["x"], rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(first.state["x"] + shift, translated.state["x"], rtol=1e-12, atol=1e-12)
        for before, after in zip(snapshot, submissions):
            self.assertEqual(before.num_examples, after.num_examples)
            self.assertTrue(torch.equal(before.state["x"], after.state["x"]))
        self.assertTrue(torch.equal(broadcast["x"], self.broadcast["x"]))

    def test_failed_input_does_not_commit_state(self):
        rule = FedReTrustV05()
        submissions = clients([[1., 0.]] * 7 + [[-1., 0.]] * 3)
        rule.aggregate(submissions, self.broadcast, 1)
        saved = copy.deepcopy(rule.weights)
        submissions[0].state["x"][0] = float("nan")
        with self.assertRaises(FloatingPointError):
            rule.aggregate(submissions, self.broadcast, 2)
        self.assertEqual(rule.weights, saved)
        self.assertEqual(rule.last_round, 1)
        with self.assertRaises(ValueError):
            rule.aggregate(clients([[1., 0.]] * 10), self.broadcast, 1)


if __name__ == "__main__":
    unittest.main()
