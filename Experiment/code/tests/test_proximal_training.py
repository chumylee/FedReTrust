"""Independent gradient, budget, dispatch, matching and restoration checks."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import copy,json,sys,unittest
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import experiment_core as core
import proximal_training as prox
from proximal_study import jobs
from proximal_analysis import target_rows


def fixture():
    g=torch.Generator().manual_seed(42)
    model=nn.Linear(2,2)
    with torch.no_grad():model.weight.copy_(torch.tensor([[.2,-.1],[-.3,.4]]));model.bias.copy_(torch.tensor([.1,-.2]))
    data=TensorDataset(torch.randn((32,2),generator=g),torch.randint(0,2,(32,),generator=g))
    cfg=core.TrainConfig(batch_size=8,local_epochs=1,learning_rate=.01,momentum=.9)
    return model,data,cfg,core.clone_state(model)


def target_fixture():
    metrics={'clean':['overall_mean_5_24_pp'],
             'abrupt_shot_noise':['affected_honest_event_pp','unchanged_honest_mean_5_24_pp'],
             'gradual_stripe':['affected_honest_event_pp','unchanged_honest_mean_5_24_pp'],
             'label_flip':['attack_active_signed_damage_pp','attack_active_accuracy_pp']}
    rows=[]
    for method in ('FedAvg','RFA','FedProx','FedProx+RFA'):
        for regime in ('IID','non-IID'):
            for scenario,names in metrics.items():
                policies=(('compliant','bypass') if scenario=='label_flip' else ('compliant',)) if method.startswith('FedProx') else ('standard',)
                for policy in policies:
                    for metric in names:rows.append({'method':method,'regime':regime,'scenario':scenario,'metric':metric,'attack_policy':policy,'mean':80.,'count':2})
    return pd.DataFrame(rows)


class ProximalTrainingTests(unittest.TestCase):
    def test_penalty_gradient_includes_bias_and_is_not_parameter_averaged(self):
        model=nn.Linear(2,2)
        with torch.no_grad():model.weight.fill_(2.);model.bias.fill_(-3.)
        anchor={n:torch.ones_like(p).detach() for n,p in model.named_parameters()}
        penalty=prox.proximal_penalty(model,anchor,.7)
        self.assertAlmostEqual(float(penalty.detach()),.35*(4*1.+2*16.),places=5)
        penalty.backward()
        for name,p in model.named_parameters():torch.testing.assert_close(p.grad,.7*(p.detach()-anchor[name]))
        self.assertTrue(all(v.grad is None for v in anchor.values()))

    def test_current_nonzero_broadcast_has_zero_penalty_and_gradient(self):
        model,_,_,state=fixture()
        penalty=prox.proximal_penalty(model,state,1.);penalty.backward()
        self.assertEqual(float(penalty.detach()),0.)
        self.assertTrue(all(torch.equal(p.grad,torch.zeros_like(p)) for p in model.parameters()))
        self.assertGreater(float(prox.proximal_penalty(model,{n:torch.zeros_like(t) for n,t in state.items()},1.).detach()),0.)

    def test_zero_mu_exactly_reproduces_original_sgd_and_preserves_server(self):
        model,data,cfg,state=fixture();digest=core.state_sha256(state)
        old,old_metrics=prox.ORIGINAL_TRAIN_CLIENT(model,data,cfg,2,8,state)
        new,metrics=prox.train_prox_client(model,data,cfg,2,8,state,0.)
        self.assertEqual(core.state_sha256(old),core.state_sha256(new))
        self.assertEqual(old_metrics,{k:metrics[k] for k in old_metrics})
        self.assertEqual(core.state_sha256(core.clone_state(model)),digest)
        self.assertEqual(core.state_sha256(state),digest)
        self.assertEqual(metrics['online_proximal_penalty'],0.)
        self.assertEqual(metrics['optimizer_steps'],4)

    def test_nonzero_mu_changes_learning_without_extra_data_or_steps(self):
        model,data,cfg,state=fixture()
        zero,z=prox.train_prox_client(model,data,cfg,1,12,state,0.)
        nonzero,n=prox.train_prox_client(model,data,cfg,1,12,state,1.)
        self.assertNotEqual(core.state_sha256(zero),core.state_sha256(nonzero))
        self.assertEqual(z['optimizer_steps'],n['optimizer_steps']);self.assertEqual(z['shuffle_seed'],n['shuffle_seed'])
        self.assertGreater(n['online_proximal_penalty'],0.)
        self.assertEqual(n['first_step_proximal_penalty'],0.)
        self.assertAlmostEqual(n['online_total_objective'],n['online_train_loss']+n['online_proximal_penalty'],places=6)

    def test_bypass_only_when_active_and_penalty_returns_after_attack(self):
        self.assertEqual(prox.effective_mu(1.,'bypass',True),0.)
        self.assertEqual(prox.effective_mu(1.,'bypass',False),1.)
        self.assertEqual(prox.effective_mu(1.,'compliant',True),1.)
        with patch.object(prox,'train_prox_client',return_value=({},{})) as learner:
            with prox.installed_trainer(1.,'bypass'):
                _,m=core.train_client(None,SimpleNamespace(flip_labels=True),None,0,5,{})
                self.assertEqual(learner.call_args.args[-1],0.);self.assertTrue(m['proximal_penalty_bypassed'])
                _,m=core.train_client(None,SimpleNamespace(flip_labels=False),None,0,11,{})
                self.assertEqual(learner.call_args.args[-1],1.);self.assertFalse(m['proximal_penalty_bypassed'])
        self.assertIs(core.train_client,prox.ORIGINAL_TRAIN_CLIENT)

    def test_trainer_restored_after_failure_and_nested_replacement_rejected(self):
        with self.assertRaisesRegex(RuntimeError,'deliberate'):
            with prox.installed_trainer(1.,'compliant'):
                with self.assertRaises(RuntimeError):
                    with prox.installed_trainer(1.,'compliant'):pass
                raise RuntimeError('deliberate')
        self.assertIs(core.train_client,prox.ORIGINAL_TRAIN_CLIENT)
        for mu in (-1.,float('nan'),True):
            with self.assertRaises(ValueError):
                with prox.installed_trainer(mu,'compliant'):pass

    def test_treatment_wrapper_does_not_change_aggregation(self):
        result=object();base=SimpleNamespace(name='RFA',settings={'x':1},aggregate=lambda *args:result)
        treatment=prox.TrainingTreatment(base,'FedProx+RFA',1.)
        self.assertIs(treatment.aggregate([],{},5),result)
        self.assertEqual(treatment.settings['local_training_objective']['mu'],1.)
        self.assertEqual(base.settings,{'x':1})

    def test_complete_declared_grid_and_no_bypass_on_honest_only_scenarios(self):
        all_jobs=jobs();self.assertEqual(len(all_jobs),58)
        self.assertEqual(len({json.dumps(j,sort_keys=True) for j in all_jobs}),58)
        candidates=[j for j in all_jobs if j['role']=='candidate'];self.assertEqual(len(candidates),56)
        self.assertTrue(all(j['scenario'] in ('label_flip','attack_stops') for j in candidates if j['attack_policy']=='bypass'))
        self.assertEqual(sum(j['attack_policy']=='bypass' for j in candidates),16)
        self.assertTrue(all(j['mu']==1. and j['rounds']==24 and j['train_images']==512 for j in candidates))

    def test_targets_require_bypass_protection_and_both_matched_seeds(self):
        means=target_fixture();rows=target_rows(means)
        self.assertEqual(len(rows),36);self.assertTrue(all(r['met'] for r in rows))
        mask=(means.method=='FedProx+RFA')&(means.regime=='non-IID')&(means.scenario=='label_flip')&(means.attack_policy=='bypass')&(means.metric=='attack_active_accuracy_pp')
        means.loc[mask,'mean']=79.
        rows=target_rows(means);self.assertEqual(sum(not r['met'] for r in rows),1)
        means.loc[mask,'count']=1
        with self.assertRaises(ValueError):target_rows(means)


if __name__=='__main__':unittest.main()
