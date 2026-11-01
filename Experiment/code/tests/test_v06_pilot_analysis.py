"""Stress-screen gates, own-clean onset control and affine audit boundaries."""
import copy
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import v06_pilot_analysis as a


def passing_effects():
    effects = []
    for comparator in ('FedAvg', 'RFA', 'FedReTrust-Direction-loose', a.PREVIOUS):
        for regime in ('IID', 'non-IID'):
            for scenario, metric in a.ALL_CORE:
                vals = [0., 0.]
                if metric == a.DAMAGE and comparator == 'FedAvg': vals = [-3., -3.]
                if regime == 'non-IID' and comparator == a.PREVIOUS:
                    if metric == a.DAMAGE: vals = [-2., -2.]
                    if scenario == 'label_flip' and metric == a.ACCURACY: vals = [1., 1.]
                    if metric == a.ONSET_DAMAGE: vals = [0., -5.]
                    if metric == a.ONSET_ACCURACY: vals = [0., 5.]
                effects.append({'candidate': 'FedReTrust-v06', 'comparator': comparator,
                    'regime': regime, 'scenario': scenario, 'metric': metric, 'n': 2,
                    'block_3': vals[0], 'block_5': vals[1], 'mean': float(np.mean(vals)),
                    'min': min(vals), 'max': max(vals), 'range': max(vals) - min(vals)})
    return effects


def affine_example():
    raw = np.arange(20, dtype=float).reshape(10, 2) / 20.
    previous = np.flip(raw, axis=0).copy()
    current = .05 * raw; history = .05 * previous; full = current + history
    metrics = {str(i): {'innovation_factor': .5, 'filtered_effective_coefficient': .1,
        'current_delta_coefficient': .05, 'history_initialized_before_round': True,
        'applied_contribution_scope': 'full_current_plus_client_history',
        'applied_contribution_l2_norm': float(np.linalg.norm(full[i]))} for i in range(10)}
    measures = {'gram_matrix_convention': 'full_applied_current_plus_history_contributions',
        'gram_matrix': (full @ full.T).tolist(), 'current_gram_matrix': (current @ current.T).tolist(),
        'historical_gram_matrix': (history @ history.T).tolist(),
        'current_history_cross_gram_matrix': (current @ history.T).tolist(),
        'aggregate_l2_norm': float(np.linalg.norm(full.sum(0))), 'client_order': list(range(10)),
        'affine_reconstruction_max_abs_error': 0., 'reconstruction_max_abs_error': 0.,
        'historical_term_present': True, 'historical_aggregate_l2_norm': float(np.linalg.norm(history.sum(0))),
        'current_aggregate_l2_norm': float(np.linalg.norm(current.sum(0))),
        'effective_coefficient_mass': .5, 'filtered_effective_coefficient_mass': 1.,
        'historical_coefficient_mass': .5}
    return {'coefficient_convention': 'affine_current_raw_deltas_plus_client_history',
        'contribution_measurements': measures, 'client_metrics': metrics,
        'innovation_radius': 1., 'median_innovation_l2_norm': .5,
        'zero_median_innovation_bypass': False, 'history_clients': 10, 'history_tensor_bytes': 160}


class ScreenAnalysisTests(unittest.TestCase):
    def test_exact27_checks_and_all_boundaries_pass(self):
        gates, decision = a.evaluate_gates(passing_effects())
        self.assertEqual(len(gates), 27)
        self.assertEqual(decision['original_practical_targets_total'], 10)
        self.assertEqual(decision['IID_retention_checks_total'], 10)
        self.assertEqual(decision['nonIID_focused_checks_total'], 7)
        self.assertTrue(decision['complete_selected_screen_point_targets_met'])
        self.assertFalse(decision['automatic_continuation_authorized'])

    def test_IID_block_regression_cannot_be_hidden_by_other_block_gain(self):
        effects = passing_effects()
        row = next(e for e in effects if e['comparator'] == a.PREVIOUS and e['regime'] == 'IID' and e['scenario'] == 'clean')
        row.update(block_3=-.6, block_5=.6, mean=0.)
        gates, decision = a.evaluate_gates(effects)
        self.assertFalse(decision['IID_retention_point_criteria_met'])
        failed = [g for g in gates if not g['point_pass']]
        self.assertEqual(len(failed), 1)
        self.assertEqual(failed[0]['gate_block'], 3)
        self.assertEqual(failed[0]['checked_difference_pp'], -.6)

    def test_attack_accuracy_must_improve_strictly_in_both_nonIID_blocks(self):
        effects = passing_effects()
        row = next(e for e in effects if e['comparator'] == a.PREVIOUS and e['regime'] == 'non-IID' and e['scenario'] == 'label_flip' and e['metric'] == a.ACCURACY)
        row.update(block_3=0., block_5=4., mean=2.)
        gates, decision = a.evaluate_gates(effects)
        self.assertFalse(decision['nonIID_focused_point_criteria_met'])
        self.assertEqual([g['gate_block'] for g in gates if not g['point_pass']], [3])

    def test_lower_clean_baseline_cannot_alone_pass_onset_focus(self):
        effects = passing_effects()
        row = next(e for e in effects if e['comparator'] == a.PREVIOUS and e['regime'] == 'non-IID' and e['metric'] == a.ONSET_ACCURACY)
        row.update(block_3=0., block_5=0., mean=0.)
        _, decision = a.evaluate_gates(effects)
        self.assertFalse(decision['nonIID_focused_point_criteria_met'])

    def test_missing_block_or_confidence_interval_claim_rejected(self):
        for mutation in ('missing_block', 'ci'):
            effects = passing_effects()
            if mutation == 'missing_block': effects[0].pop('block_5')
            else: effects[0]['ci_low'] = -1.
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                a.evaluate_gates(effects)

    def test_selected_block_summary_has_two_values_and_no_CI(self):
        rows = [{'method': 'candidate', 'regime': 'IID', 'block': b, 'scenario': 'clean',
            'metric': a.ACCURACY, 'value': v} for b, v in [(3, 80.), (5, 90.)]]
        result = a.summarize(rows, expected=(3, 5), interval=False)[0]
        self.assertEqual(result['mean'], 85.)
        self.assertEqual(result['range'], 10.)
        self.assertEqual(result['n'], 2)
        self.assertNotIn('ci_low', result)
        with self.assertRaises(ValueError):
            a.summarize(rows[:1], expected=(3, 5), interval=False)

    def test_onset_uses_own_clean_round5_and_reports_raw_drop_separately(self):
        attack = {r: 60. for r in range(4, 25)}; attack[4] = 50.
        clean = {5: 80.}
        result = a.onset_metrics(attack, clean)
        self.assertEqual(result[a.ONSET_DAMAGE], 20.)
        self.assertEqual(result[a.ONSET_ACCURACY], 60.)
        self.assertEqual(result[a.ONSET_DROP], -10.)
        with self.assertRaises(ValueError):
            a.onset_metrics(attack, {})

    def test_affine_audit_preserves_full_current_plus_history_accounting(self):
        values = affine_example(); result = a.audit_affine_round(values)
        self.assertTrue(result['historical_term_present'])
        self.assertEqual(result['clipped_clients'], 10)
        self.assertEqual(result['history_initialized_clients'], 10)

    def test_current_only_gram_cannot_masquerade_as_full_contribution(self):
        values = affine_example(); m = values['contribution_measurements']
        m['gram_matrix'] = copy.deepcopy(m['current_gram_matrix'])
        with self.assertRaises(AssertionError):
            a.audit_affine_round(values)

    def test_wrong_coefficient_contract_or_current_factor_rejected(self):
        values = affine_example(); values['coefficient_convention'] = 'raw_only'
        with self.assertRaises(ValueError):
            a.audit_affine_round(values)
        values = affine_example(); values['client_metrics']['0']['current_delta_coefficient'] = .1
        with self.assertRaises(AssertionError):
            a.audit_affine_round(values)


if __name__ == '__main__':
    unittest.main()
