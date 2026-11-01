"""Boundary checks for exploratory follow-up pairing and descriptive claims."""
import copy
import math
import unittest

import numpy as np

from v05_followup_analysis import (ACCURACY, HONEST, DAMAGE, BLOCKS, T_DF4,
    describe, contrasts, evaluate_gates, fragility, validate_failed_pilot_decision,
    validate_metric_grid)

MAIN = 'FedReTrust-v05'
OLD = 'FedReTrust-Direction-loose'
OTHERS = ('FedAvg', 'RFA', OLD)


def metrics():
    result = []
    for regime in ('IID', 'non-IID'):
        for method in (MAIN, *OTHERS):
            clean = 90. if method in ('FedAvg', 'RFA') else 89.5
            stripe = {'FedAvg': 85., 'RFA': 75., OLD: 84.5, MAIN: 84.5}[method]
            damage = {'FedAvg': 15., 'RFA': .5, OLD: 5., MAIN: 3.5}[method]
            if regime == 'IID' and method in (MAIN, OLD):
                damage = .8
            for block in BLOCKS:
                for scenario, metric, value in [('clean', ACCURACY, clean),
                    ('abrupt_shot_noise', HONEST, clean), ('gradual_stripe', HONEST, stripe),
                    ('label_flip', DAMAGE, damage), ('label_flip', ACCURACY, clean - damage)]:
                    result.append({'method': method, 'regime': regime, 'block': block,
                        'seed': 713000 + block, 'scenario': scenario, 'metric': metric, 'value': value})
    return result


class FollowupAnalysisTests(unittest.TestCase):
    def test_five_block_interval_uses_sample_sd_and_df4(self):
        result = describe([1., 2., 3., 4., 5.], BLOCKS)
        half = T_DF4 * math.sqrt(2.5) / math.sqrt(5)
        self.assertEqual(result['n'], 5)
        self.assertAlmostEqual(result['sd'], math.sqrt(2.5))
        self.assertAlmostEqual(result['ci_low'], 3. - half)
        self.assertAlmostEqual(result['ci_high'], 3. + half)

    def test_five_block_zero_variance_interval(self):
        result = describe([2.] * 5, BLOCKS)
        self.assertEqual(result['ci_low'], 2.)
        self.assertEqual(result['ci_high'], 2.)

    def test_missing_duplicate_reordered_or_nonfinite_blocks_rejected(self):
        for values, blocks in [([1.] * 4, (1, 2, 3, 4)),
            ([1.] * 5, (1, 2, 3, 4, 4)), ([1.] * 5, (2, 1, 3, 4, 5)),
            ([1., 2., np.nan, 4., 5.], BLOCKS), ([1., 2., np.inf, 4., 5.], BLOCKS)]:
            with self.subTest(blocks=blocks, values=values), self.assertRaises(ValueError):
                describe(values, blocks)

    def test_subsets_report_values_and_ranges_without_confidence_intervals(self):
        result = describe([2., 7., 3.], (3, 4, 5), (3, 4, 5), False)
        self.assertEqual(result['n'], 3)
        self.assertEqual(result['range'], 5.)
        self.assertEqual(result['block_5'], 3.)
        self.assertNotIn('ci_low', result)
        with self.assertRaises(ValueError):
            describe([2., 7., 3.], (3, 4, 5), (3, 4, 5), True)

    def test_pairing_rejects_missing_rows_duplicate_rows_and_seed_mismatch(self):
        original = metrics()
        missing = original[1:]
        duplicated = original + [original[0]]
        wrong_seed = copy.deepcopy(original)
        wrong_seed[0]['seed'] += 100
        for rows in (missing, duplicated, wrong_seed):
            with self.subTest(size=len(rows)), self.assertRaises(ValueError):
                contrasts(rows, MAIN, OTHERS)

    def test_block_metric_coverage_rejects_partial_or_missing_method(self):
        rows = metrics()
        validate_metric_grid(rows, (MAIN, *OTHERS))
        for bad in (rows[1:], [r for r in rows if r['method'] != 'RFA']):
            with self.assertRaises(ValueError):
                validate_metric_grid(bad, (MAIN, *OTHERS))

    def test_effects_are_candidate_minus_same_block_comparator(self):
        rows = metrics()
        for row in rows:
            if row['method'] == MAIN and row['scenario'] == 'clean':
                row['value'] += row['block']
        effects = contrasts(rows, MAIN, OTHERS)
        effect = next(e for e in effects if e['comparator'] == OLD and e['regime'] == 'non-IID' and e['scenario'] == 'clean')
        self.assertEqual([effect[f'block_{b}'] for b in BLOCKS], [1., 2., 3., 4., 5.])
        self.assertEqual(effect['mean'], 3.)

    def test_focused_margin_pass_is_separate_from_full_target_pass(self):
        gates, decision = evaluate_gates(contrasts(metrics(), MAIN, OTHERS), OLD)
        self.assertEqual(len(gates), 16)
        self.assertEqual(decision['focused_fix_checks_passed'], 6)
        self.assertTrue(decision['five_block_focused_and_honest_margins_met'])
        self.assertFalse(decision['five_block_all_point_targets_met'])
        self.assertFalse(decision['automatic_continuation_authorized'])
        self.assertNotIn('complete_pilot_point_targets_met', decision)
        self.assertTrue(all('block_5_pass' in gate for gate in gates))

    def test_exact_honest_margin_is_inclusive_but_absolute_improvement_strict(self):
        effects = contrasts(metrics(), MAIN, OTHERS)
        honest = next(e for e in effects if e['comparator'] == 'FedAvg' and e['regime'] == 'non-IID' and e['scenario'] == 'abrupt_shot_noise')
        attack = next(e for e in effects if e['comparator'] == OLD and e['regime'] == 'non-IID' and e['scenario'] == 'label_flip' and e['metric'] == ACCURACY)
        honest['mean'] = honest['ci_low'] = honest['ci_high'] = -1.
        attack['mean'] = attack['ci_low'] = attack['ci_high'] = 0.
        for block in BLOCKS:
            honest[f'block_{block}'] = -1.
            attack[f'block_{block}'] = 0.
        gates, decision = evaluate_gates(effects, OLD)
        self.assertTrue(next(g for g in gates if g['regime'] == 'non-IID' and g['scenario'] == 'abrupt_shot_noise')['point_pass'])
        self.assertFalse(next(g for g in gates if g['criterion'] == 'nonIID_absolute_attack_accuracy_improves_v03')['point_pass'])
        self.assertFalse(decision['five_block_focused_and_honest_margins_met'])

    def test_fragility_reports_onset_and_later_burst_separately(self):
        curve = {r: 90. for r in range(4, 25)}
        curve[5] = 60.
        curve[12] = 40.
        result = fragility(curve)
        self.assertEqual(result['onset_drop_pp'], 30.)
        self.assertEqual(result['maximum_active_one_round_drop_pp'], 50.)
        self.assertEqual(result['maximum_drop_end_round'], 12)
        self.assertEqual(result['minimum_active_accuracy_pp'], 40.)
        self.assertEqual(result['minimum_accuracy_round'], 12)
        with self.assertRaises(ValueError):
            fragility({r: v for r, v in curve.items() if r != 4})

    def test_improving_trajectory_has_zero_maximum_drop(self):
        result = fragility({r: float(r) for r in range(4, 25)})
        self.assertEqual(result['maximum_active_one_round_drop_pp'], 0.)
        self.assertLess(result['onset_drop_pp'], 0.)

    def test_failed_pilot_decision_cannot_be_overwritten_as_passed(self):
        decision = {'advancement_point_criterion_met': False,
            'complete_pilot_point_targets_met': False, 'automatic_continuation_authorized': False}
        validate_failed_pilot_decision(decision)
        for name in decision:
            with self.subTest(name=name), self.assertRaises(ValueError):
                validate_failed_pilot_decision({**decision, name: True})
        with self.assertRaises(ValueError):
            validate_failed_pilot_decision({})


if __name__ == '__main__':
    unittest.main()
