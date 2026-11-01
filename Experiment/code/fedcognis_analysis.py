"""Offline matched comparisons, contribution checks and access/workload audit."""
from pathlib import Path
from dataclasses import asdict
import json
import numpy as np
import pandas as pd
from experiment_core import file_sha256, write_json
from research_runs import ROOT
from research_analysis import extract_run, cross_method_pair, contribution_records, cost_record
from research_study import task_series
from scenario_metrics import longer_scenarios, matched_reference_name, model_recovery
from directional_study import checked_result


def record_key(r):
    j=r['job']
    return j['method'],j['regime'],j['scenario'],j['seed'],j.get('attack_policy','standard')


def analyse(study):
    from fedcognis_study import MAIN,CONTROL,SQUARED,BASELINES,SETTINGS,jobs,key,verify
    study=Path(study);verify(study)
    manifest=json.loads((study/'result_manifest.json').read_text())
    scientific=manifest['records'];context=manifest['context_records'];records=scientific+context
    new=manifest['new_records']
    if len(new)!=70 or {key(r['job']) for r in new}!={key(j) for j in jobs()}:
        raise ValueError('Incomplete declared new-run grid.')
    if len(scientific)!=352 or len(context)!=72:raise ValueError('Incomplete comparison records.')
    indexed={record_key(r):r for r in records}
    if len(indexed)!=len(records):raise ValueError('Duplicate policy-aware records.')
    context_ids={r['job_id'] for r in context}
    for r in records:
        checked_result(r)
        for name,h in r['source_sha256'].items():
            if file_sha256(ROOT/'code'/name)!=h:raise ValueError('Source changed: '+name)
    def ident(r):
        j=r['job'];m=j['method']
        return {'method':m,'regime':j['regime'],'scenario':j['scenario'],'seed':j['seed'],'job_id':r['job_id'],
                'attack_policy':j.get('attack_policy','standard'),
                'panel':'context' if r['job_id'] in context_ids else 'control' if m==CONTROL else 'interpretation' if m==SQUARED else 'sensitivity' if m in SETTINGS and m!=MAIN else 'primary'}
    def reference(r):
        j=r['job'];policy='compliant' if j.get('attack_policy') in ('compliant','bypass') else 'standard'
        return indexed[j['method'],j['regime'],matched_reference_name(j['scenario']),j['seed'],policy]
    metrics=[];curves=[];episodes=[];costs=[];training=[];tasks_out=[];recoveries=[]
    contributions=[];restorations=[];reductions=[];trust=[];pairs=[];prefix=[]
    for r in records:
        j=r['job'];identity=ident(r);path=ROOT/r['run_path'];ref=reference(r)
        base=indexed['FedAvg',j['regime'],j['scenario'],j['seed'],'standard']
        cross_method_pair(r,base);pairs.append({'job_id':r['job_id'],'fedavg_job_id':base['job_id'],'matched':True})
        scenario=longer_scenarios()[j['scenario']];event=bool(scenario.events)
        a,b,c=extract_run(r,ref if event else None)
        metrics.extend({**row,**identity} for row in a);curves.extend({**row,**identity} for row in b);episodes.extend({**row,**identity} for row in c)
        costs.append({**cost_record(r),**identity})
        local=pd.read_csv(path/'client_metrics.csv');tasks=pd.read_csv(path/'task_metrics.csv')
        training.extend({**row,**identity} for row in local.to_dict('records'))
        tasks_out.extend({**row,**identity} for row in tasks.to_dict('records'))
        if event:
            a,b,c=contribution_records(r,ref)
            contributions.extend({**row,**identity} for row in a);restorations.extend({**row,**identity} for row in b);reductions.extend({**row,**identity} for row in c)
            h=pd.read_csv(path/'history.csv');hc=pd.read_csv(ROOT/ref['run_path']/'history.csv')
            if h[h['round']<=4].global_state_sha256.tolist()!=hc[hc['round']<=4].global_state_sha256.tolist():raise AssertionError('Pre-event trajectory mismatch.')
            tc=pd.read_csv(ROOT/ref['run_path']/'task_metrics.csv')
            pd.testing.assert_frame_equal(tasks[tasks['round']<=4].reset_index(drop=True),tc[tc['round']<=4].reset_index(drop=True),check_exact=True)
            prefix.append({'job_id':r['job_id'],'reference_job_id':ref['job_id'],'exact':True})
        attacks=[e for e in scenario.events if e.intent=='malicious']
        if attacks:
            windows=attacks[0].windows;active=sorted({t for lo,hi in windows for t in range(lo,hi+1)})
            obs=task_series(r);ctl=task_series(ref,scenario_name=j['scenario'])
            for name,value in [('attack_active_accuracy_pp',100*np.mean([obs[t] for t in active])),
                               ('matched_reference_attack_window_accuracy_pp',100*np.mean([ctl[t] for t in active]))]:
                metrics.append({**identity,'phase':'development','metric':name,'value':float(value)})
            for i,(lo,hi) in enumerate(windows):
                if hi>=24:continue
                next_attack=windows[i+1][0] if i+1<len(windows) else None
                for group,clients in [('all_clients',None),('affected_attackers',list(attacks[0].clients))]:
                    observed={t:v for t,v in task_series(r,client_ids=clients).items() if t>0}
                    control={t:v for t,v in task_series(ref,scenario_name=j['scenario'],client_ids=clients).items() if t>0}
                    recoveries.append({**identity,'group':group,'episode':i+1,**model_recovery(control,observed,hi,next_attack=next_attack)})
        if j['method'] in SETTINGS:
            diagnostics=json.loads((path/'aggregation_diagnostics.json').read_text())
            old=json.loads((ROOT/ref['run_path']/'aggregation_diagnostics.json').read_text())
            honest={i for e in scenario.events if e.intent=='honest' for i in e.clients}
            malicious={i for e in scenario.events if e.intent=='malicious' for i in e.clients}
            for row,prior in zip(diagnostics,old):
                for cid,item in row['values']['client_metrics'].items():
                    cid=int(cid);events=scenario.active(cid,row['round'])
                    role='attacking' if any(e.intent=='malicious' for e in events) else 'honest_changed' if any(e.intent=='honest' for e in events) else 'currently_unchanged'
                    other=prior['values']['client_metrics'][str(cid)]
                    trust.append({**identity,'round':row['round'],'client':cid,'role':role,
                        'affected_honest':cid in honest,'affected_attacker':cid in malicious,
                        'update_l2_norm':item['update_l2_norm'],'distance_penalty':item['distance_penalty'],
                        'trust':item['trust'],'previous_trust':item['previous_trust'],
                        'instantaneous_trust':item['instantaneous_trust'],'log_trust':item['log_trust'],
                        'excluded':item['excluded'],'coefficient':item['applied_delta_coefficient'],
                        'reference_coefficient':other['applied_delta_coefficient'],
                        'applied_norm':item['applied_contribution_l2_norm'],
                        'reference_norm':other['applied_contribution_l2_norm']})
    table=pd.DataFrame(metrics)
    means=table.groupby(['panel','method','attack_policy','regime','scenario','metric']).value.agg(['mean','count','std']).reset_index()
    effects=[]
    for r in scientific:
        j=r['job']
        if j['method']!=MAIN:continue
        one=table[table.job_id==r['job_id']]
        for method in BASELINES+(CONTROL,SQUARED):
            other=table[(table.method==method)&(table.regime==j['regime'])&(table.seed==j['seed'])&(table.scenario==j['scenario'])]
            if other.empty:continue
            paired=one.merge(other,on='metric',suffixes=('_main','_baseline'),validate='one_to_one')
            effects.extend({'method':MAIN,'baseline':method,'regime':j['regime'],'scenario':j['scenario'],'seed':j['seed'],
                'metric':x['metric'],'main_value':x['value_main'],'baseline_value':x['value_baseline'],'difference_pp':x['value_main']-x['value_baseline']}
                for x in paired.to_dict('records'))
    audits=[];exact_first_round=0;reconstruction_errors=[]
    for r in new:
        j=r['job'];path=ROOT/r['run_path'];local=pd.read_csv(path/'client_metrics.csv')
        if len(local)!=240 or not local.optimizer_steps.eq(8).all() or not local.examples.eq(512).all():raise AssertionError('Training workload changed.')
        if not np.isfinite(local.online_train_loss).all():raise AssertionError('Nonfinite loss.')
        info=json.loads((path/'run.json').read_text());access=json.loads((path/'fedcognis_access.json').read_text())
        setting=SETTINGS[j['method']]
        assert info['method']['settings']['config']==asdict(setting)
        assert not info['official_test_used'] and not access['official_test_used']
        assert access['server_reference_examples']==access['reference_forward_examples']==access['reference_backward_examples']==access['calibration_runs']==0
        assert access['defence_inputs']==['submitted_parameters','sample_counts','broadcast','round']
        assert info['method']['settings']['QSA_implemented'] is False and info['method']['settings']['SALSTM_implemented'] is False
        base=indexed['FedAvg',j['regime'],j['scenario'],j['seed'],'standard']
        old=pd.read_csv(ROOT/base['run_path']/'client_metrics.csv')
        cols=['client','submitted_state_sha256','broadcast_state_sha256','shuffle_seed','optimizer_steps','online_train_loss']
        pd.testing.assert_frame_equal(local[local['round']==1][cols].reset_index(drop=True),old[old['round']==1][cols].reset_index(drop=True),check_exact=True)
        exact_first_round+=10
        diag=json.loads((path/'aggregation_diagnostics.json').read_text());assert len(diag)==24
        previous={str(i):1. for i in range(10)}
        for row in diag:
            values=row['values'];items=values['client_metrics'];ordered=[items[str(i)] for i in range(10)]
            assert values['reference_forward_examples']==values['reference_backward_examples']==0
            assert values['hard_trust_filter'] is False and all(not v['excluded'] for v in ordered)
            norms=np.array([v['update_l2_norm'] for v in ordered]);signal=-setting.beta*norms**setting.distance_power
            before=np.array([previous[str(i)] for i in range(10)])
            np.testing.assert_allclose([v['previous_trust'] for v in ordered],before,rtol=1e-12,atol=1e-15)
            logq=signal if setting.alpha==0 else np.logaddexp(np.log(before)+np.log(setting.alpha),signal+np.log1p(-setting.alpha))
            np.testing.assert_allclose([v['log_trust'] for v in ordered],logq,rtol=1e-12,atol=1e-12)
            weights=np.exp(logq-logq.max());weights/=weights.sum()
            np.testing.assert_allclose([v['applied_delta_coefficient'] for v in ordered],weights,rtol=1e-12,atol=1e-14)
            assert all(v['applied_delta_coefficient']>=0 for v in ordered)
            for i,v in enumerate(ordered):previous[str(i)]=v['trust']
            reconstruction_errors.append(values['contribution_measurements']['reconstruction_max_abs_error'])
        audits.append({'job_id':r['job_id'],'method':j['method'],'rounds':24,'client_updates':240,
            'optimizer_steps':1920,'train_forward_examples':122880,'reference_forward_examples':0,
            'development_task_forward_examples':75000,'checkpoint_replay_forward_examples':6000})
    assert exact_first_round==700 and len(reconstruction_errors)==1680 and max(reconstruction_errors)<=1e-6
    assert len(pairs)==424 and len(prefix)==356
    out=study/'analysis';out.mkdir(exist_ok=True)
    tables={'seed_metrics':metrics,'accuracy_curves':curves,'model_episodes':episodes,'costs':costs,
            'local_training':training,'raw_task_metrics':tasks_out,'recovery':recoveries,'contributions':contributions,
            'contribution_restoration':restorations,'contribution_reduction':reductions,'trust':trust,'paired_effects':effects}
    for name,rows in tables.items():pd.DataFrame(rows).to_csv(out/(name+'.csv'),index=False)
    means.to_csv(out/'means.csv',index=False)
    summary={'analysis_source_sha256':file_sha256(__file__),'scientific_trajectories':352,'primary_trajectories':312,
        'control_trajectories':16,'interpretation_trajectories':12,'sensitivity_trajectories':12,'context_trajectories':72,
        'new_runs':70,'matching_checks':pairs,'pre_event_checks':prefix,'exact_first_round_client_updates_vs_fedavg':exact_first_round,
        'audited_new_rounds':len(reconstruction_errors),'maximum_reconstruction_error':max(reconstruction_errors),
        'interpretation':'Trust-aggregation adaptation on MNIST; not full FedCognis, QSA or SALSTM. Two reused development seeds; first-seed penalty sensitivities. No statistical superiority or universal failure claim.'}
    write_json(out/'summary.json',summary)
    totals={k:sum(a[k] for a in audits) for k in ['client_updates','optimizer_steps','train_forward_examples','reference_forward_examples','development_task_forward_examples','checkpoint_replay_forward_examples']}
    write_json(study/'access_audit.json',{'new_runs':70,'official_test_used':False,'server_reference_examples':0,
        'calibration_runs':0,'defence_uses_evaluation_data':False,'authentication':'valid-participant assumption, not measured QSA',
        'totals':totals,'runs':audits})
    return summary
