"""Matched trajectory analysis; no evaluation signal feeds back into training."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from experiment_core import file_sha256,write_json
from research_runs import ROOT,source_hashes
from research_analysis import extract_run,cross_method_pair,cost_record,contribution_records
from research_study import task_series
from scenario_metrics import longer_scenarios,model_recovery
from directional_study import checked_result

SCENARIOS=('clean','abrupt_shot_noise','gradual_stripe','label_flip','attack_stops')
SEEDS=(20261014,20261015)
REGIMES=('IID','non-IID')
CANDIDATES=('FedProx','FedProx+RFA')


def key(r):
    j=r['job'];return j['method'],j['regime'],j['scenario'],j['seed'],j.get('attack_policy','standard')


def tags(r):
    j=r['job'];return {'attack_policy':j.get('attack_policy','standard'),'mu':j.get('mu',0.),
                       'evidence_origin':'new_training' if j.get('role')=='candidate' else 'verified_reused_baseline'}


def annotate(rows,r):return [{**row,**tags(r)} for row in rows]


def target_rows(means):
    """Strict descriptive two-seed targets; no fitted tolerance or selection."""
    def value(method,regime,scenario,metric,policy):
        sub=means[(means.method==method)&(means.regime==regime)&(means.scenario==scenario)&
                  (means.metric==metric)&(means.attack_policy==policy)]
        if len(sub)!=1 or int(sub.iloc[0]['count'])!=2:raise ValueError('Target comparison lacks two matched seeds.')
        return float(sub.iloc[0]['mean'])
    rows=[]
    conditions=[('clean','overall_mean_5_24_pp','FedAvg',1),
                ('abrupt_shot_noise','affected_honest_event_pp','FedAvg',1),
                ('gradual_stripe','affected_honest_event_pp','FedAvg',1),
                ('abrupt_shot_noise','unchanged_honest_mean_5_24_pp','FedAvg',1),
                ('gradual_stripe','unchanged_honest_mean_5_24_pp','FedAvg',1)]
    for method in CANDIDATES:
        for regime in REGIMES:
            for scenario,metric,baseline,direction in conditions:
                a=value(method,regime,scenario,metric,'compliant');b=value(baseline,regime,scenario,metric,'standard')
                rows.append({'method':method,'regime':regime,'scenario':scenario,'metric':metric,'attack_policy':'compliant',
                             'baseline':baseline,'value_pp':a,'target_pp':b,'advantage_pp':direction*(a-b),'met':direction*(a-b)>=0})
            for policy in ('compliant','bypass'):
                for metric,direction in (('attack_active_signed_damage_pp',-1),('attack_active_accuracy_pp',1)):
                    a=value(method,regime,'label_flip',metric,policy);b=value('RFA',regime,'label_flip',metric,'standard')
                    rows.append({'method':method,'regime':regime,'scenario':'label_flip','metric':metric,'attack_policy':policy,
                                 'baseline':'RFA','value_pp':a,'target_pp':b,'advantage_pp':direction*(a-b),'met':direction*(a-b)>=0})
    return rows


def analyse_study(study_path):
    study=Path(study_path);manifest=json.loads((study/'result_manifest.json').read_text())
    protocol=json.loads((study/'protocol.json').read_text());records=manifest['records']
    if len(records)!=96 or len(manifest['new_records'])!=58:raise ValueError('Incomplete trajectory study.')
    indexed={key(r):r for r in records}
    expected={(m,g,s,seed,'standard') for m in ('FedAvg','RFA') for g in REGIMES for s in SCENARIOS for seed in SEEDS}
    expected|={(m,g,s,seed,p) for m in CANDIDATES for g in REGIMES for s in SCENARIOS for seed in SEEDS
               for p in (('compliant','bypass') if s in ('label_flip','attack_stops') else ('compliant',))}
    if len(indexed)!=96 or set(indexed)!=expected:raise ValueError('Wrong scientific condition grid.')
    for r in records+manifest['zero_mu_records']:
        checked_result(r)
        expected_sources=protocol['source_sha256'] if r['job'].get('role') else source_hashes()
        if r['source_sha256']!=expected_sources:raise ValueError('Wrong training source fingerprint.')
    def clean(r):
        j=r['job'];policy='compliant' if j['method'] in CANDIDATES else 'standard'
        return indexed[j['method'],j['regime'],'clean',j['seed'],policy]
    metrics,curves,episodes,costs,recovery=[],[],[],[],[]
    contributions,contribution_episodes,reductions=[],[],[]
    pairing,pre_event_audits,local_training,raw_tasks=[],[],[],[]
    for r in records:
        j=r['job'];ref=clean(r);path=ROOT/r['run_path'];control_path=ROOT/ref['run_path']
        # Every new and reused trajectory has the same original inputs and evaluation.
        baseline=indexed['FedAvg',j['regime'],j['scenario'],j['seed'],'standard']
        cross_method_pair(r,baseline)
        pairing.append({'job_id':r['job_id'],'reference_job_id':baseline['job_id'],'matched_inputs':True})
        a,b,c=extract_run(r,ref if j['scenario']!='clean' else None)
        metrics+=annotate(a,r);curves+=annotate(b,r);episodes+=annotate(c,r);costs.append({**cost_record(r),**tags(r)})
        frame=pd.read_csv(path/'client_metrics.csv')
        identity={'method':j['method'],'regime':j['regime'],'seed':j['seed'],'scenario':j['scenario'],'job_id':r['job_id'],**tags(r)}
        for row in frame.to_dict('records'):
            row.setdefault('effective_prox_mu',0.);row.setdefault('online_proximal_penalty',0.)
            row.setdefault('online_total_objective',row['online_train_loss']);row.setdefault('proximal_penalty_bypassed',False)
            local_training.append({**identity,**row})
        task=pd.read_csv(path/'task_metrics.csv');raw_tasks+= [{**identity,**row} for row in task.to_dict('records')]
        if j['scenario']!='clean':
            a,b,c=contribution_records(r,ref)
            contributions+=annotate(a,r);contribution_episodes+=annotate(b,r);reductions+=annotate(c,r)
            history=pd.read_csv(path/'history.csv');control=pd.read_csv(control_path/'history.csv')
            if history[history['round']<=4].global_state_sha256.tolist()!=control[control['round']<=4].global_state_sha256.tolist():
                raise AssertionError('Pre-event trajectory differs from its own clean control.')
            control_task=pd.read_csv(control_path/'task_metrics.csv')
            pd.testing.assert_frame_equal(task[task['round']<=4].reset_index(drop=True),
                control_task[control_task['round']<=4].reset_index(drop=True),check_exact=True)
            pre_event_audits.append({'job_id':r['job_id'],'clean_job_id':ref['job_id'],'first_four_states_and_tasks_exact':True})
        if j['scenario'] in ('label_flip','attack_stops'):
            windows=longer_scenarios()[j['scenario']].events[0].windows
            active=[t for start,end in windows for t in range(start,end+1)]
            observed=task_series(r);reference=task_series(ref)
            for metric,value in (('attack_active_accuracy_pp',100*np.mean([observed[t] for t in active])),
                                 ('clean_reference_attack_window_accuracy_pp',100*np.mean([reference[t] for t in active]))):
                metrics.append({'phase':'development',**identity,'metric':metric,'value':float(value)})
            if j['scenario']=='attack_stops':
                for group,clients in (('all_clients',None),('former_attackers',[0,1,2])):
                    obs={k:v for k,v in task_series(r,client_ids=clients).items() if k>0}
                    reference={k:v for k,v in task_series(ref,client_ids=clients).items() if k>0}
                    recovery.append({**identity,'group':group,**model_recovery(reference,obs,10,horizon=24,tolerance_pp=1.,consecutive=3)})
    # Audit the actual training dispatch against the scenario, not just job labels.
    training_checks=[]
    for r in manifest['new_records']:
        j=r['job'];path=ROOT/r['run_path'];frame=pd.read_csv(path/'client_metrics.csv')
        if len(frame)!=240 or not frame.optimizer_steps.eq(8).all() or not frame.examples.eq(512).all():raise AssertionError('Training budget changed.')
        if not frame.first_step_proximal_penalty.eq(0.).all():raise AssertionError('Wrong proximal anchor.')
        scenario=longer_scenarios()[j['scenario']]
        expected_bypass=[]
        for row in frame.to_dict('records'):
            active=any(e.kind=='label_flip' and e.intent=='malicious' for e in scenario.active(int(row['client']),int(row['round'])))
            bypass=active and j['attack_policy']=='bypass' and j['mu']>0
            want=0. if bypass else j['mu'];expected_bypass.append(bypass)
            if row['effective_prox_mu']!=want or bool(row['proximal_penalty_bypassed'])!=bypass:raise AssertionError('Wrong attacker penalty dispatch.')
        diagnostics=json.loads((path/'aggregation_diagnostics.json').read_text())
        if any(x['values'].get('reference_forward_examples',0)!=0 for x in diagnostics):raise AssertionError('Unexpected server-reference inference.')
        if j['mu']>0 and not (frame.online_proximal_penalty>0).any():raise AssertionError('Proximal term was a no-op.')
        training_checks.append({'job_id':r['job_id'],'client_updates':240,'optimizer_steps':1920,'train_forward_examples':122880,
            'bypassed_client_updates':sum(expected_bypass),'anchor_first_step_zero':True,'reference_forward_examples':0,
            'development_task_forward_examples':75000,'checkpoint_replay_forward_examples':6000})
    zero=[r['zero_mu_audit'] for r in manifest['zero_mu_records']]
    if len(zero)!=2 or any(z['global_states_identical']!=24 or z['submitted_states_identical']!=240 or not z['all_task_metrics_identical'] for z in zero):raise AssertionError('Zero-mu audit incomplete.')
    table=pd.DataFrame(metrics)
    means=table.groupby(['method','attack_policy','mu','regime','scenario','metric']).value.agg(['mean','count']).reset_index()
    if not means['count'].eq(2).all():raise ValueError('Incomplete two-seed metric coverage.')
    effects=[]
    for r in records:
        j=r['job']
        if j['method'] not in CANDIDATES:continue
        baseline='RFA' if j['method']=='FedProx+RFA' else 'FedAvg'
        one=table[table.job_id==r['job_id']]
        old=table[(table.method==baseline)&(table.regime==j['regime'])&(table.seed==j['seed'])&(table.scenario==j['scenario'])]
        paired=one.merge(old,on='metric',suffixes=('_new','_baseline'),validate='one_to_one')
        for row in paired.to_dict('records'):
            effects.append({'method':j['method'],'attack_policy':j['attack_policy'],'mu':j['mu'],'baseline':baseline,
                'regime':j['regime'],'seed':j['seed'],'scenario':j['scenario'],'metric':row['metric'],
                'new_value':row['value_new'],'baseline_value':row['value_baseline'],'difference':row['value_new']-row['value_baseline']})
    targets=target_rows(means)
    out=study/'analysis';out.mkdir(exist_ok=True)
    tables={'seed_metrics':metrics,'accuracy_curves':curves,'model_episodes':episodes,'costs':costs,'recovery':recovery,
            'contributions':contributions,'contribution_episodes':contribution_episodes,'contribution_reductions':reductions,
            'local_training':local_training,'raw_task_metrics':raw_tasks,'paired_effects':effects,'target_checks':targets}
    for name,rows in tables.items():pd.DataFrame(rows).to_csv(out/(name+'.csv'),index=False)
    means.to_csv(out/'two_seed_means.csv',index=False)
    summary={'source_sha256':protocol['source_sha256'],'analysis_source_sha256':file_sha256(__file__),
        'counts':{'new_runs':58,'candidate_runs':56,'zero_mu_runs':2,'reused_baselines':40,'scientific_trajectories':96,
                  'new_global_rounds':1392,'new_client_updates':13920},
        'mu':1.,'target_results':{m:{'met':all(t['met'] for t in targets if t['method']==m),
                    'targets_met':sum(t['met'] for t in targets if t['method']==m),'targets':18} for m in CANDIDATES},
        'zero_mu_audits':zero,'pairing_checks':pairing,'pre_event_checks':pre_event_audits,
        'interpretation':'Two reused development seeds; descriptive comparisons, no significance or optimal-mu claim. CE training loss, proximal penalty, clean-relative damage and absolute attacked accuracy are distinct. Recovery has the original three-round criterion. This is a four-way mechanism comparison, not a new FedReTrust version.'}
    write_json(out/'summary.json',summary)
    write_json(study/'access_audit.json',{'new_runs':58,'trusted_server_images':0,'official_test_used':False,
        'global_development_evaluation':'reporting_only_not_server_feedback','new_client_updates':13920,
        'train_forward_examples':sum(r['train_forward_examples'] for r in training_checks),
        'development_task_forward_examples':sum(r['development_task_forward_examples'] for r in training_checks),
        'checkpoint_replay_forward_examples':sum(r['checkpoint_replay_forward_examples'] for r in training_checks),
        'bypassed_client_updates':sum(r['bypassed_client_updates'] for r in training_checks),
        'training_dispatch_checks':training_checks,'zero_mu_audits':zero})
    return summary
