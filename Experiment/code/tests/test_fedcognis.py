"""Independent arithmetic, history and information-boundary checks."""
from pathlib import Path
import math
import sys
import unittest
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiment_core import ClientSubmission, FedAvgAggregator, state_sha256
from fedcognis import FedCognisConfig, FedCognisAggregator
from scenario_metrics import ContributionRecorder


def item(cid, values, n=1):
    return ClientSubmission(cid, n, {'w': torch.tensor(values, dtype=torch.float64)})


class FedCognisTests(unittest.TestCase):
    def test_norm_and_squared_interpretations_match_independent_values(self):
        state = {'w': torch.zeros(2, dtype=torch.float64)}
        submissions = [item(0, [3., 4.]), item(1, [0., 0.])]
        for power, distance in [(1, 5.), (2, 25.)]:
            r = FedCognisAggregator(FedCognisConfig(distance_power=power)).aggregate(submissions, state, 1)
            expected = .5 + .5 * math.exp(-.5 * distance)
            self.assertAlmostEqual(r.diagnostics['client_metrics']['0']['trust'], expected)
            self.assertAlmostEqual(r.coefficients[0], expected / (expected + 1))

    def test_current_updated_trust_and_sample_counts_determine_average(self):
        state = {'w': torch.tensor([10.], dtype=torch.float64)}
        r = FedCognisAggregator(FedCognisConfig(alpha=.25, beta=.7)).aggregate([item(3, [12.], 2), item(8, [9.], 5)], state, 1)
        q0, q1 = .25 + .75 * math.exp(-1.4), .25 + .75 * math.exp(-.7)
        expected = (2*q0*12 + 5*q1*9) / (2*q0 + 5*q1)
        self.assertAlmostEqual(float(r.state['w'][0]), expected)

    def test_zero_penalty_matches_fedavg_across_unequal_sample_counts(self):
        state = {'w': torch.tensor([2., 7.], dtype=torch.float64)}
        submissions = [item(3, [8., 1.], 3), item(9, [-2., 9.], 11), item(0, [3., 4.], 7)]
        actual = FedCognisAggregator(FedCognisConfig(beta=0)).aggregate(submissions, state, 1)
        expected = FedAvgAggregator().aggregate(submissions, state, 1)
        torch.testing.assert_close(actual.state['w'], expected.state['w'], rtol=0, atol=1e-12)

    def test_equal_magnitudes_receive_equal_trust_despite_opposite_directions(self):
        state = {'w': torch.zeros(2, dtype=torch.float64)}
        r = FedCognisAggregator().aggregate([item(0, [3., 4.]), item(1, [-3., -4.])], state, 1)
        self.assertEqual(r.coefficients, {0: .5, 1: .5})
        torch.testing.assert_close(r.state['w'], state['w'])

    def test_memory_decay_and_recovery_follow_same_ema(self):
        agg = FedCognisAggregator(); state = {'w': torch.zeros(1, dtype=torch.float64)}
        for t in (1, 2): agg.aggregate([item(0, [2000.]), item(1, [0.])], state, t)
        r = agg.aggregate([item(0, [0.]), item(1, [0.])], state, 3)
        self.assertAlmostEqual(r.diagnostics['client_metrics']['0']['previous_trust'], .25)
        self.assertAlmostEqual(r.diagnostics['client_metrics']['0']['trust'], .625)

    def test_absence_does_not_rehabilitate_trust_and_reset_clears_memory(self):
        agg = FedCognisAggregator(); state = {'w': torch.zeros(1, dtype=torch.float64)}
        agg.aggregate([item(4, [20.]), item(7, [0.])], state, 1); old = agg.log_trust[4]
        agg.aggregate([item(7, [0.])], state, 10)
        self.assertEqual(agg.log_trust[4], old)
        agg.reset(); self.assertEqual(agg.log_trust, {}); self.assertEqual(agg.last_round, 0)

    def test_no_memory_ignores_earlier_scores(self):
        cfg = FedCognisConfig(alpha=0); a = FedCognisAggregator(cfg); b = FedCognisAggregator(cfg)
        state = {'w': torch.zeros(1, dtype=torch.float64)}
        a.aggregate([item(0, [20.]), item(1, [0.])], state, 1)
        b.aggregate([item(0, [0.]), item(1, [20.])], state, 1)
        submissions = [item(0, [2.]), item(1, [4.])]
        self.assertEqual(a.aggregate(submissions, state, 2).coefficients, b.aggregate(submissions, state, 2).coefficients)

    def test_logspace_handles_all_exponentials_underflowing(self):
        state = {'w': torch.zeros(1, dtype=torch.float64)}
        r = FedCognisAggregator(FedCognisConfig(alpha=0)).aggregate([item(0, [2000.], 1), item(1, [-2000.], 3)], state, 1)
        self.assertAlmostEqual(r.coefficients[0], .25)
        self.assertAlmostEqual(float(r.state['w'][0]), -1000.)
        self.assertTrue(all(math.isfinite(x) for x in r.coefficients.values()))

    def test_permutation_input_immutability_rng_and_recording(self):
        state = {'w': torch.tensor([2., 3.], dtype=torch.float64)}
        submissions = [item(8, [3., 1.]), item(2, [4., 2.])]
        before = [state_sha256(state)] + [state_sha256(s.state) for s in submissions]
        rng = torch.random.get_rng_state().clone()
        a = ContributionRecorder(FedCognisAggregator()).aggregate(submissions, state, 1)
        b = FedCognisAggregator().aggregate(list(reversed(submissions)), state, 1)
        self.assertEqual(a.coefficients, b.coefficients)
        self.assertEqual(before, [state_sha256(state)] + [state_sha256(s.state) for s in submissions])
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        self.assertEqual(a.diagnostics['contribution_measurements']['reconstruction_max_abs_error'], 0)
        self.assertEqual(a.diagnostics['reference_forward_examples'], 0)

    def test_zero_updates_preserve_broadcast(self):
        state = {'w': torch.tensor([5., -7.], dtype=torch.float64)}
        r = FedCognisAggregator().aggregate([item(0, [5., -7.]), item(1, [5., -7.], 3)], state, 1)
        torch.testing.assert_close(r.state['w'], state['w'], rtol=0, atol=0)
        np.testing.assert_allclose(list(r.coefficients.values()), [.25, .75])

    def test_invalid_config_rejected(self):
        for kwargs in ({'alpha':-1}, {'beta':float('nan')}, {'distance_power':3}, {'distance_power':True}, {'initial_trust':0}):
            with self.assertRaises(ValueError): FedCognisConfig(**kwargs)

    def test_invalid_submission_and_repeated_round_rejected(self):
        state = {'w': torch.zeros(1, dtype=torch.float64)}
        agg = FedCognisAggregator()
        for bad in ([], [item(0,[1]),item(0,[2])], [item(0,[1],0)], [item(0,[float('nan')])]):
            with self.assertRaises((ValueError,FloatingPointError)): agg.aggregate(bad,state,1)
        agg.aggregate([item(0,[0])],state,1)
        with self.assertRaises(ValueError): agg.aggregate([item(0,[0])],state,1)


if __name__ == '__main__': unittest.main()
