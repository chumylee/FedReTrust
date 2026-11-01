"""Five-block uncertainty, extended retention and preserved-pilot boundaries."""
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import v06_followup_analysis as a
from v05_followup_analysis import describe


def passing_effects():
    rows = []
    for other in ('FedAvg','RFA','FedReTrust-Direction-loose',a.PREVIOUS):
        for regime in ('IID','non-IID'):
            for scenario, metric in a.ALL_CORE:
                vals = [0.] * 5
                if metric == a.DAMAGE and other == 'FedAvg': vals = [-3.] * 5
                if regime == 'non-IID' and other == a.PREVIOUS:
                    if metric == a.DAMAGE: vals = [-2.] * 5
                    if scenario == 'label_flip' and metric == a.ACCURACY: vals = [1.] * 5
                    if metric == a.ONSET_DAMAGE: vals = [0.,0.,0.,0.,-5.]
                    if metric == a.ONSET_ACCURACY: vals = [0.,0.,0.,0.,5.]
                rows.append({'candidate':'FedReTrust-v06','comparator':other,
                    'regime':regime,'scenario':scenario,'metric':metric,
                    **describe(vals,a.BLOCKS)})
    return rows


class FollowupAnalysisTests(unittest.TestCase):
    def test_exact51_prespecified_checks(self):
        gates,d = a.evaluate_gates(passing_effects())
        self.assertEqual(len(gates),51)
        self.assertEqual(d['original_practical_targets_total'],10)
        self.assertEqual(d['IID_retention_checks_total'],25)
        self.assertEqual(d['nonIID_focused_checks_total'],16)
        self.assertTrue(d['five_block_all_point_targets_met'])
        self.assertFalse(d['automatic_continuation_authorized'])

    def test_one_IID_regression_is_not_hidden_by_mean(self):
        e=passing_effects()
        r=next(x for x in e if x['comparator']==a.PREVIOUS and x['regime']=='IID' and x['scenario']=='clean')
        r.update(describe([1.,1.,1.,1.,-1.],a.BLOCKS))
        gates,d=a.evaluate_gates(e)
        failed=[x for x in gates if not x['point_pass']]
        self.assertEqual(len(failed),1)
        self.assertEqual(failed[0]['gate_block'],5)
        self.assertEqual(failed[0]['checked_difference_pp'],-1.)
        self.assertFalse(d['IID_retention_point_criteria_met'])

    def test_new_block_attack_accuracy_must_improve(self):
        e=passing_effects()
        r=next(x for x in e if x['comparator']==a.PREVIOUS and x['regime']=='non-IID' and x['scenario']=='label_flip' and x['metric']==a.ACCURACY)
        r.update(describe([1.,0.,1.,1.,1.],a.BLOCKS))
        gates,d=a.evaluate_gates(e)
        self.assertFalse(d['nonIID_focused_point_criteria_met'])
        self.assertEqual([x['gate_block'] for x in gates if not x['point_pass']],[2])

    def test_onset_requires_absolute_accuracy_as_well_as_damage(self):
        e=passing_effects()
        r=next(x for x in e if x['comparator']==a.PREVIOUS and x['regime']=='non-IID' and x['metric']==a.ONSET_ACCURACY)
        r.update(describe([0.,0.,0.,0.,0.],a.BLOCKS))
        _,d=a.evaluate_gates(e)
        self.assertFalse(d['nonIID_focused_point_criteria_met'])

    def test_missing_block_or_nominal_interval_rejected(self):
        for missing in ('block_4','ci_low'):
            e=passing_effects();e[0].pop(missing)
            with self.assertRaises(ValueError): a.evaluate_gates(e)

    def test_nominal_interval_arithmetic_and_subset_limits(self):
        x=describe([1.,2.,3.,4.,5.],a.BLOCKS)
        half=2.7764451051977987*np.std([1.,2.,3.,4.,5.],ddof=1)/np.sqrt(5)
        self.assertAlmostEqual(x['ci_low'],3.-half)
        self.assertAlmostEqual(x['ci_high'],3.+half)
        with self.assertRaises(ValueError): describe([1.,2.],(3,5),expected=(3,5),interval=True)
        self.assertNotIn('ci_low',describe([1.,2.],(3,5),expected=(3,5),interval=False))

    def test_paired_blocks_match_seeds_and_missing_rows_rejected(self):
        rows=[{'method':m,'regime':'IID','block':b,'seed':100+b,'scenario':'clean',
            'metric':a.ACCURACY,'value':float(b)+(1 if m=='candidate' else 0)}
            for m in ('candidate','other') for b in a.BLOCKS]
        effect=a.contrasts(rows,'candidate',('other',))[0]
        self.assertEqual(effect['mean'],1.)
        self.assertEqual(effect['ci_low'],1.)
        rows[-1]['seed']+=1
        with self.assertRaises(ValueError): a.contrasts(rows,'candidate',('other',))


if __name__=='__main__': unittest.main()
