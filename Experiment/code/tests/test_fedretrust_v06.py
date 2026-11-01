import copy
import inspect
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiment_core import ClientSubmission
from fedretrust_directional import DirectionalConfig
from fedretrust_v05 import FedReTrustV05
from fedretrust_v06 import COEFFICIENT_CONVENTION, FedReTrustV06
from v06_recording import InnovationContributionRecorder


def clients(points, counts=None, broadcast=None):
    counts = counts or [10] * len(points)
    broadcast = {"x": torch.zeros(len(points[0]), dtype=torch.float64)} if broadcast is None else broadcast
    return [ClientSubmission(i, count, {"x": torch.as_tensor(point, dtype=broadcast["x"].dtype) + broadcast["x"]})
            for i, (point, count) in enumerate(zip(points, counts))]


class InnovationMethodChecks(unittest.TestCase):
    def setUp(self):
        self.broadcast = {"x": torch.zeros(2, dtype=torch.float64)}

    def test_fixed_configuration_and_no_reference_or_truth_interface(self):
        rule = FedReTrustV06()
        self.assertEqual(rule.options, DirectionalConfig())
        self.assertEqual(rule.name, "FedReTrust-v06")
        self.assertEqual(rule.settings["innovation_multiplier"], 2.)
        self.assertEqual(rule.settings["trusted_reference_examples"], 0)
        self.assertFalse(rule.settings["theorem_or_novelty_claim"])
        self.assertEqual(list(inspect.signature(rule.aggregate).parameters), ["submissions", "broadcast", "round_number"])
        with self.assertRaises(TypeError):
            FedReTrustV06(innovation_multiplier=1.)

    def test_innovation_bound_limits_fully_trusted_aligned_burst(self):
        submissions = clients([[1., 0.]] * 7 + [[100., 0.]] * 3)
        rule = FedReTrustV06()
        result = InnovationContributionRecorder(rule).aggregate(submissions, self.broadcast, 1)
        old = FedReTrustV05().aggregate(submissions, self.broadcast, 1)
        self.assertTrue(all(q == 1. for q in rule.weights.values()))
        self.assertEqual(result.diagnostics["innovation_radius"], 2.)
        self.assertFalse(result.diagnostics["zero_median_innovation_bypass"])
        self.assertLess(float(result.state["x"][0]), float(old.state["x"][0]))
        self.assertAlmostEqual(float(result.state["x"][0]), 1.3)
        for i in range(10):
            self.assertLessEqual(float(rule.history[i].norm()), 2.)
        self.assertAlmostEqual(result.coefficients[7], .002)
        self.assertAlmostEqual(result.diagnostics["client_metrics"]["7"]["filtered_effective_coefficient"], .1)

    def test_no_clip_matches_v05_exactly_with_identical_prior_trust(self):
        points = [[1., .1], [.9, .05], [1.1, .2], [1., -.1]]
        old, rule = FedReTrustV05(), FedReTrustV06()
        old.weights = {0: .2, 1: .7, 2: 1., 3: .4}
        rule.weights = dict(old.weights)
        submissions = clients(points, [7, 9, 11, 13])
        a, b = old.aggregate(submissions, self.broadcast, 1), rule.aggregate(submissions, self.broadcast, 1)
        self.assertTrue(all(v["innovation_factor"] == 1. for v in b.diagnostics["client_metrics"].values()))
        self.assertTrue(torch.equal(a.state["x"], b.state["x"]))
        self.assertEqual(a.coefficients, b.coefficients)
        self.assertEqual(old.weights, rule.weights)

    def test_clipped_updates_receive_exact_v05_scoring_on_filtered_values(self):
        points = [[1., .1]] * 7 + [[-9., -3.], [50., 10.], [.4, .1]]
        rule = FedReTrustV06()
        rule.weights = {8: .1, 9: .25}
        previous_trust = dict(rule.weights)
        result = rule.aggregate(clients(points), self.broadcast, 1)
        filtered_submissions = clients([rule.history[i].tolist() for i in range(10)])
        old = FedReTrustV05()
        old.weights = previous_trust
        expected = old.aggregate(filtered_submissions, self.broadcast, 1)
        self.assertTrue(torch.equal(result.state["x"], expected.state["x"]))
        self.assertEqual(rule.weights, old.weights)
        for i in range(10):
            observed = result.diagnostics["client_metrics"][str(i)]
            self.assertEqual(observed["filtered_effective_coefficient"], expected.coefficients[i])
            for key in ("anchor_cosine", "evidence_weight", "restored_weight", "clip_factor", "sample_share"):
                self.assertEqual(observed[key], expected.diagnostics["client_metrics"][str(i)][key])

    def test_stable_client_specific_offsets_are_retained(self):
        rule = FedReTrustV06()
        history = [[1., .2]] * 7 + [[4., .8]] * 3
        rule.history = {i: torch.tensor(p, dtype=torch.float64) for i, p in enumerate(history)}
        points = [[p[0] + .05, p[1] + .01] for p in history]
        result = rule.aggregate(clients(points), self.broadcast, 1)
        self.assertFalse(result.diagnostics["zero_median_innovation_bypass"])
        for i, point in enumerate(points):
            self.assertEqual(result.diagnostics["client_metrics"][str(i)]["innovation_factor"], 1.)
            torch.testing.assert_close(rule.history[i], torch.tensor(point, dtype=torch.float64), rtol=0., atol=0.)
        self.assertGreater(float(rule.history[7].norm()), 3. * float(rule.history[0].norm()))

    def test_legitimate_abrupt_change_can_lag_and_history_can_remain_poisoned(self):
        rule = FedReTrustV06()
        rule.history = {i: torch.tensor([1., 0.], dtype=torch.float64) for i in range(10)}
        result = rule.aggregate(clients([[1.1, 0.]] * 7 + [[3., 0.]] * 3), self.broadcast, 1)
        self.assertAlmostEqual(result.diagnostics["innovation_radius"], .2)
        self.assertAlmostEqual(float(rule.history[7][0]), 1.2)
        self.assertLess(float(rule.history[7][0]), 3.)  # Same limit applies to honest drift.
        rule = FedReTrustV06()
        rule.history = {i: torch.tensor([1., 0.] if i < 7 else [-3., 0.], dtype=torch.float64)
                        for i in range(10)}
        rule.aggregate(clients([[1.1, 0.]] * 7 + [[1., 0.]] * 3), self.broadcast, 1)
        self.assertLess(float(rule.history[7][0]), 0.)  # Repaired raw update does not erase stale poison immediately.

    def test_zero_median_bypass_prevents_stable_majority_freezing_minority(self):
        rule = FedReTrustV06()
        rule.history = {i: torch.tensor([1., 0.], dtype=torch.float64) for i in range(10)}
        result = rule.aggregate(clients([[1., 0.]] * 7 + [[5., 0.]] * 3), self.broadcast, 1)
        self.assertEqual(result.diagnostics["median_innovation_l2_norm"], 0.)
        self.assertTrue(result.diagnostics["zero_median_innovation_bypass"])
        self.assertEqual(result.diagnostics["client_metrics"]["7"]["innovation_factor"], 1.)
        self.assertTrue(torch.equal(rule.history[7], torch.tensor([5., 0.], dtype=torch.float64)))

    def test_lower_median_and_double_precision_history(self):
        b = {"x": torch.zeros(2, dtype=torch.float32)}
        rule = FedReTrustV06()
        result = rule.aggregate(clients([[1., 1.], [2., 2.], [3., 3.], [11., 0.]], broadcast=b), b, 1)
        self.assertAlmostEqual(result.diagnostics["median_innovation_l2_norm"], (8.) ** .5)
        self.assertEqual(rule.history[3].dtype, torch.float64)
        self.assertFalse(torch.equal(rule.history[3], rule.history[3].float().double()))
        self.assertEqual(result.diagnostics["history_tensor_bytes"], 4 * 2 * 8)

    def test_absence_does_not_change_history_and_reset_clears_all_state(self):
        rule = FedReTrustV06()
        rule.aggregate(clients([[1., .1]] * 10), self.broadcast, 1)
        previous = rule.history[9].clone()
        rule.aggregate(clients([[2., .2]] * 7), self.broadcast, 2)
        self.assertTrue(torch.equal(rule.history[9], previous))
        self.assertEqual(rule.last_round, 2)
        rule.reset()
        self.assertEqual(rule.history, {})
        self.assertEqual(rule.weights, {})
        self.assertEqual(rule.last_round, 0)
        self.assertIsNone(rule.layout_signature)

    def test_invalid_input_layout_and_computation_are_atomic(self):
        rule = FedReTrustV06()
        rule.aggregate(clients([[1., .1]] * 10), self.broadcast, 1)
        old_history = {i: v.clone() for i, v in rule.history.items()}
        old_weights, signature = dict(rule.weights), rule.layout_signature
        bad = clients([[1., .1]] * 10)
        bad[0].state["x"][0] = float("nan")
        with self.assertRaises(FloatingPointError):
            rule.aggregate(bad, self.broadcast, 2)
        with patch("fedretrust_v06.smoothed_weiszfeld", side_effect=RuntimeError("failed solver")):
            with self.assertRaises(RuntimeError):
                rule.aggregate(clients([[2., .2]] * 10), self.broadcast, 2)
        with self.assertRaises(ValueError):
            rule.aggregate(clients([[1., .1, 0.]] * 10), {"x": torch.zeros(3, dtype=torch.float64)}, 2)
        self.assertEqual(rule.weights, old_weights)
        self.assertEqual(rule.last_round, 1)
        self.assertEqual(rule.layout_signature, signature)
        for i in old_history:
            self.assertTrue(torch.equal(rule.history[i], old_history[i]))

    def test_nonmutation_and_permutation(self):
        points = [[1., .2], [1.2, .2], [.8, .3], [8., 2.]]
        submissions = clients(points, [7, 13, 9, 11])
        snapshot = copy.deepcopy(submissions)
        a, b = FedReTrustV06(), FedReTrustV06()
        a.history = {i: torch.tensor([.3, .05], dtype=torch.float64) for i in range(4)}
        b.history = copy.deepcopy(a.history)
        old_map = a.history
        first = a.aggregate(submissions, self.broadcast, 1)
        reversed_result = b.aggregate(list(reversed(submissions)), self.broadcast, 1)
        torch.testing.assert_close(first.state["x"], reversed_result.state["x"], rtol=1e-12, atol=1e-12)
        for i, (left, right) in enumerate(zip(snapshot, submissions)):
            self.assertTrue(torch.equal(left.state["x"], right.state["x"]))
            self.assertTrue(torch.equal(old_map[i], torch.tensor([.3, .05], dtype=torch.float64)))


class InnovationRecorderChecks(unittest.TestCase):
    def test_affine_reconstruction_and_grams_include_history(self):
        for counts in ([10] * 10, [7, 9, 11, 13, 8, 12, 10, 8, 9, 11]):
            with self.subTest(counts=counts):
                b = {"x": torch.tensor([8., -3.], dtype=torch.float64)}
                rule = FedReTrustV06()
                rule.history = {i: torch.tensor([1., .2], dtype=torch.float64) for i in range(10)}
                previous = copy.deepcopy(rule.history)
                raw = torch.tensor([[1.1, .22]] * 7 + [[4., 2.]] * 3, dtype=torch.float64)
                result = InnovationContributionRecorder(rule).aggregate(clients(raw.tolist(), counts, b), b, 1)
                diag = result.diagnostics["client_metrics"]
                current = torch.stack([result.coefficients[i] * raw[i] for i in range(10)])
                history = torch.stack([diag[str(i)]["historical_coefficient"] * previous[i] for i in range(10)])
                full = torch.stack([diag[str(i)]["filtered_effective_coefficient"] * rule.history[i] for i in range(10)])
                torch.testing.assert_close(b["x"] + current.sum(0) + history.sum(0), result.state["x"], rtol=1e-12, atol=1e-12)
                self.assertFalse(torch.allclose(b["x"] + current.sum(0), result.state["x"]))
                measurements = result.diagnostics["contribution_measurements"]
                torch.testing.assert_close(torch.tensor(measurements["gram_matrix"], dtype=torch.float64), full @ full.T)
                torch.testing.assert_close(torch.tensor(measurements["current_gram_matrix"], dtype=torch.float64), current @ current.T)
                torch.testing.assert_close(torch.tensor(measurements["historical_gram_matrix"], dtype=torch.float64), history @ history.T)
                torch.testing.assert_close(torch.tensor(measurements["current_history_cross_gram_matrix"], dtype=torch.float64), current @ history.T)
                self.assertTrue(measurements["historical_term_present"])
                self.assertEqual(result.diagnostics["coefficient_convention"], COEFFICIENT_CONVENTION)
                self.assertEqual(measurements["effective_coefficient_mass_scope"], "current_raw_delta_part_only")
                self.assertGreaterEqual(measurements["defence_aggregation_seconds"], 0.)
                self.assertGreaterEqual(measurements["metric_recording_seconds"], 0.)
                self.assertLess(measurements["reconstruction_max_abs_error"], 1e-12)
                self.assertLess(measurements["affine_reconstruction_max_abs_error"], 1e-12)
                json.dumps(result.diagnostics, allow_nan=False)

    def test_recording_is_observational_and_supports_multiple_parameters(self):
        b = {"weight": torch.zeros((2, 2), dtype=torch.float32), "bias": torch.ones(2, dtype=torch.float32)}
        submissions = [ClientSubmission(i, 10, {"weight": b["weight"] + (1. if i < 7 else 20.),
                                               "bias": b["bias"] + .1}) for i in range(10)]
        plain, recorded = FedReTrustV06(), FedReTrustV06()
        a = plain.aggregate(submissions, b, 1)
        z = InnovationContributionRecorder(recorded).aggregate(submissions, b, 1)
        for name in b:
            self.assertTrue(torch.equal(a.state[name], z.state[name]))
        self.assertEqual(a.coefficients, z.coefficients)
        self.assertEqual(plain.weights, recorded.weights)
        for i in plain.history:
            self.assertTrue(torch.equal(plain.history[i], recorded.history[i]))

    def test_recorder_failure_rolls_back_candidate_state(self):
        base = FedReTrustV06()
        wrapper = InnovationContributionRecorder(base)
        b = {"x": torch.zeros(2, dtype=torch.float64)}
        wrapper.aggregate(clients([[1., 0.]] * 10), b, 1)
        saved_history = copy.deepcopy(base.history)
        saved_weights = dict(base.weights)
        with patch.object(wrapper, "_measure", side_effect=AssertionError("audit failed")):
            with self.assertRaises(AssertionError):
                wrapper.aggregate(clients([[2., 0.]] * 10), b, 2)
        self.assertEqual(base.last_round, 1)
        self.assertEqual(base.weights, saved_weights)
        for i in saved_history:
            self.assertTrue(torch.equal(base.history[i], saved_history[i]))

    def test_degenerate_zero_and_opposing_geometry_stays_finite(self):
        for points in ([[0., 0.]] * 10, [[1., 0.]] * 5 + [[-1., 0.]] * 5):
            base = FedReTrustV06()
            base.weights = {0: .25}
            result = InnovationContributionRecorder(base).aggregate(clients(points), {"x": torch.zeros(2, dtype=torch.float64)}, 1)
            self.assertTrue(result.diagnostics["bounded_rfa_fallback"])
            self.assertEqual(base.weights, {0: .25})
            self.assertTrue(torch.isfinite(result.state["x"]).all())
            self.assertEqual(result.diagnostics["reference_forward_examples"], 0)
            json.dumps(result.diagnostics, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
