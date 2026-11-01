from pathlib import Path
import sys,unittest
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from blockwise_study import pilot_jobs,validation_jobs,expected_pilot_keys,key,MAIN,CONTROL,BASELINES,SEEDS,FRESH_SEEDS
from blockwise_analysis import continuation_checks,paired_t_df4,holm_adjust


class StudyTests(unittest.TestCase):
    def test_registered_grid_and_distinct_continuation(self):
        jobs=pilot_jobs();self.assertEqual(len(jobs),50);self.assertEqual(len({key(j) for j in jobs}),50)
        self.assertEqual(sum(j['method']==MAIN for j in jobs),30)
        self.assertEqual(sum(j['method']==CONTROL for j in jobs),12)
        self.assertEqual(len(expected_pilot_keys()),252)
        self.assertTrue(all(j['rounds']==24 and j['train_images']==512 for j in jobs))
        v=validation_jobs();self.assertEqual(len(v),240);self.assertEqual(len({key(j) for j in v}),240)
        self.assertTrue(set(FRESH_SEEDS).isdisjoint(SEEDS));self.assertTrue({key(j) for j in jobs}.isdisjoint({key(j) for j in v}))

    def test_gate_uses_best_prior_and_both_regimes(self):
        rows=[]
        for g in ('IID','non-IID'):
            for method in (MAIN,)+BASELINES:
                for scenario,metric in [('clean','overall_mean_5_24_pp'),('gradual_stripe','affected_honest_event_pp'),
                                       ('label_flip','attack_active_signed_damage_pp'),('label_flip','attack_active_accuracy_pp')]:
                    rows.append({'method':method,'regime':g,'scenario':scenario,'metric':metric,'panel':'pilot','attack_policy':'standard',
                        'mean':0. if 'damage' in metric else (92. if method==MAIN else 90.),'count':2})
        frame=pd.DataFrame(rows);checks=continuation_checks(frame)
        self.assertEqual(len(checks),10);self.assertTrue(all(r['met'] for r in checks))
        mask=(frame.method=='FLTrust')&(frame.regime=='non-IID')&(frame.metric=='attack_active_signed_damage_pp')
        frame.loc[mask,'mean']=-.1
        checks=continuation_checks(frame);self.assertEqual(sum(not c['met'] for c in checks),1)
        frame.loc[mask,'count']=1
        with self.assertRaises(ValueError):continuation_checks(frame)

    def test_five_seed_statistics_use_seed_differences(self):
        result=paired_t_df4([-2.,-1.,0.,1.,2.]);self.assertAlmostEqual(result['one_sided_p'],.5)
        self.assertLess(result['ci95_low_pp'],0);self.assertGreater(result['ci95_high_pp'],0)
        t=2.7764451051977987
        # For this vector se=sqrt(0.5); shift makes t the known df4 97.5% quantile.
        shift=t*(.5**.5);result=paired_t_df4([shift+i for i in (-2.,-1.,0.,1.,2.)])
        self.assertAlmostEqual(result['one_sided_p'],.025,places=9)
        with self.assertRaises(ValueError):paired_t_df4([1.,2.])
        with self.assertRaises(ValueError):paired_t_df4([1.,2.,3.,4.,float('nan')])

    def test_multiple_comparison_correction(self):
        self.assertEqual(holm_adjust([.01,.03,.04]),[.03,.06,.06])
        self.assertEqual(holm_adjust([.04,.01,.03]),[.06,.03,.06])
        with self.assertRaises(ValueError):holm_adjust([float('nan')])


if __name__=='__main__':unittest.main()
