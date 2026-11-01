"""Strength arithmetic, endpoint equivalence and non-oracle selection checks."""
import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import strength_validation_local as local
import strength_validation_analysis as analysis
import client_validation_local as original_local
from aggregate_validation_local import build_aggregate_counterfactuals, evaluate_aggregate_contributions
from experiment_core import TrainConfig, state_sha256
from test_client_validation_local import LookupModel, IndexedView


def fixture():
    origin={'table':torch.zeros((4,10),dtype=torch.float64)}
    probability=torch.full((4,10),.2/9,dtype=torch.float64)
    for i in range(4):probability[i,i]=.8
    full={'table':probability.log()}
    models={s:{cid:{'table':s*full['table']} for cid in range(10)} for s in local.SCALES}
    report=IndexedView([0,1],[0,1,2,3]);audit=IndexedView([2,3],[0,1,2,3])
    return models,report,audit


def synthetic_cases():
    result={}
    job={'regime':'IID','seed':20261014,'round':12,'scenario':'gradual_stripe'}
    for s in local.SCALES:
        score={0.:0.,.25:.04,.5:.1,1.:.05}[s]
        result[s]=[{'job':{**job,'scale':s},'reports':[[score]*10 for _ in range(10)],
            'flipped':[[-score]*10 for _ in range(10)],
            'audit':[[.1*s if vid<3 else -.05*s for vid in range(10)] for cid in range(10)],
            'poisoned':set(),'changed':{0,1,2},'repaired':set(),
            'groups':{i:'changed' if i<3 else 'unchanged' for i in range(10)}}]
    return result


class StrengthValidationTests(unittest.TestCase):
    def test_fixed_weights_endpoint_hashes_and_fractional_arithmetic(self):
        w={'x':torch.tensor([1e6,.125],dtype=torch.float32)}
        submitted={i:{'x':w['x']+torch.tensor([i*.25,i-4.],dtype=torch.float32)} for i in range(10)}
        before=state_sha256(w);states=local.build_strength_models(w,submitted)
        full,without=build_aggregate_counterfactuals(w,submitted)
        for i in range(10):
            self.assertEqual(state_sha256(states[0.][i]),state_sha256(without[i]))
            self.assertEqual(state_sha256(states[1.][i]),state_sha256(full))
            for s in (.25,.5):
                expected=w['x'].double()+sum(.1*(s if j==i else 1.)*(submitted[j]['x'].double()-w['x'].double()) for j in range(10))
                torch.testing.assert_close(states[s][i]['x'],expected.float(),rtol=0,atol=0)
        self.assertEqual(state_sha256(w),before)
        self.assertLess(float(states[.25][9]['x'][1]),float(states[1.][9]['x'][1]))

    def test_fresh_endpoint_equivalence_zero_class_gains_and_local_cost(self):
        models,report,audit=fixture();cfg=TrainConfig(evaluation_batch_size=128)
        rng=torch.random.get_rng_state().clone();LookupModel.calls=0
        with patch.object(local,'MnistCNN',LookupModel):
            measured=local.evaluate_strength_models(models,report,audit,cfg,4)
        self.assertEqual(LookupModel.calls,62)
        self.assertTrue(torch.equal(rng,torch.random.get_rng_state()))
        self.assertEqual(measured['cost']['total_forward_examples'],124)
        with patch.object(original_local,'MnistCNN',LookupModel):
            old=evaluate_aggregate_contributions(models[1.][0],models[0.],report,audit,cfg,4)
        for role in ('report_rows','audit_rows'):
            expanded=[];previous=[]
            for v in range(10):
                expanded.extend({**r,'validator_id':v} for r in measured[role])
                previous.extend({**r,'validator_id':v} for r in old[role])
            self.assertEqual(local.verify_endpoint_rows(expanded,previous),100)
            corrupt=copy.deepcopy(expanded)
            next(r for r in corrupt if r['scale']==1.)['mean_brier_gain']+=.01
            with self.assertRaises(AssertionError):local.verify_endpoint_rows(corrupt,previous)
        rows=[r for r in measured['report_rows'] if r['candidate_id']==0]
        self.assertEqual([r['scale'] for r in rows],list(local.SCALES))
        self.assertEqual(rows[0]['mean_brier_gain'],0.)
        self.assertTrue(0<rows[1]['mean_brier_gain']<rows[2]['mean_brier_gain']<rows[3]['mean_brier_gain'])
        self.assertLess(rows[1]['flipped_mean_brier_gain'],0.)
        self.assertTrue(all(x is None for x in rows[1]['class_brier_gains'][2:]))
        self.assertFalse({'logits','images','indices','targets'}&set(rows[0]))
        self.assertNotIn('flipped_mean_brier_gain',measured['audit_rows'][0])

    def test_invalid_scales_and_overlapping_roles_are_rejected(self):
        models,report,audit=fixture()
        with self.assertRaises(ValueError):local.evaluate_strength_models({s:v for s,v in models.items() if s!=.25},report,audit,TrainConfig(),0)
        with self.assertRaises(ValueError):local.evaluate_strength_models(models,report,report,TrainConfig(),0)
        models[1.][1]['table']=models[1.][1]['table']+1
        with self.assertRaises(ValueError):local.evaluate_strength_models(models,report,audit,TrainConfig(),0)

    def test_selector_has_known_zero_and_prefers_smaller_exact_ties(self):
        self.assertEqual(analysis.select_scale({.25:0.,.5:-.1,1.:0.}),0.)
        self.assertEqual(analysis.select_scale({.25:.2,.5:.2,1.:.2}),.25)
        self.assertEqual(analysis.select_scale({.25:.2,.5:.3,1.:.1}),.5)
        for scores in ({.25:float('nan'),.5:0.,1.:0.},{0.:1.,.25:.2,.5:.1,1.:.2}):
            with self.assertRaises(ValueError):analysis.select_scale(scores)

    def test_tradeoff_requires_benefit_retention_not_proportional_shrinkage(self):
        self.assertTrue(analysis.tradeoff(.075,-.05,.1,-.1))
        self.assertFalse(analysis.tradeoff(.05,-.05,.1,-.1))
        self.assertFalse(analysis.tradeoff(0.,0.,.1,-.1))
        self.assertTrue(analysis.tradeoff(.01,0.,-.1,-.1))
        self.assertFalse(analysis.tradeoff(-.01,0.,-.1,-.1))
        self.assertTrue(analysis.tradeoff(.1,0.,.1,0.))

    def test_force_full_report_attack_targets_weight_and_respects_self_exclusion(self):
        cases=synthetic_cases()
        for s in local.SCALES:
            c=cases[s][0];c['job']['scenario']='label_flip';c['poisoned']={0,1,2}
        low=analysis.profile_matrix(cases[.25][0],'force_full')
        high=analysis.profile_matrix(cases[1.][0],'force_full')
        self.assertEqual(low[0][:3],[-1.]*3)
        self.assertEqual(high[0][:3],[1.]*3)
        self.assertEqual(high[3][:3],[-1.]*3)
        self.assertEqual(high[0][3],.05)
        score=analysis.base.score_reports(high,list(range(10)),list(range(10)),'peer_top3_mean')[0]
        self.assertAlmostEqual(score,(2+.05)/3)

    def test_audit_never_changes_selected_strength(self):
        cases=synthetic_cases(); before=analysis.adaptive_rows(cases)
        for group in cases.values():
            group[0]['audit']=[[-.2]*10 for _ in range(10)]
        after=analysis.adaptive_rows(cases)
        keys=('selected_scale','score','score_025','score_050','score_100')
        self.assertEqual([{k:r[k] for k in keys} for r in before],[{k:r[k] for k in keys} for r in after])
        truthful=[r for r in before if r['profile']=='truthful']
        self.assertTrue(all(r['selected_scale']==.5 for r in truthful))

    def test_feasibility_is_candidate_existence_not_pooled_scale_success(self):
        curves=analysis.curve_rows(synthetic_cases())
        feasible=analysis.feasibility_rows(curves)
        self.assertEqual(len(feasible),3)
        self.assertTrue(all(not r['any_scale_feasible'] for r in feasible))
        first=next(r for r in curves if r['candidate_id']==0 and r['scale']==.25)
        first['utility_tradeoff_met']=True
        feasible=analysis.feasibility_rows(curves);rates=analysis.utility_rates(feasible)
        self.assertEqual(rates[0]['n_candidates'],3)
        self.assertEqual(rates[0]['successes'],1)
        self.assertFalse(rates[0]['pass'])

    def test_protocol_preserves_intent_gate_and_declares_new_utility_gate(self):
        self.assertEqual(analysis.DIAGNOSTIC_PROTOCOL['fixed_controls']['primary_gate'],analysis.base.DIAGNOSTIC_PROTOCOL['primary_gate'])
        self.assertEqual(analysis.DIAGNOSTIC_PROTOCOL['scales'],[0.,.25,.5,1.])
        self.assertEqual(analysis.DIAGNOSTIC_PROTOCOL['utility_tradeoff']['required_rate'],.8)
        self.assertEqual(analysis.ADAPTIVE_PROFILES,(*analysis.PROFILES,'force_full'))


if __name__=='__main__':unittest.main()
