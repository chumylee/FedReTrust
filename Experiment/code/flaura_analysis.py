"""Offline matched comparisons, contribution checks and access/workload audit."""
from pathlib import Path
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
    from flaura_study import MAIN,CONTROL,CALIBRATION,BASELINES,SETTINGS,jobs,key,verify,freeze_scale
    study=Path(study);verify(study);freeze_scale(study)
    manifest=json.loads((study/'result_manifest.json').read_text())
    scientific=manifest['records'];context=manifest['context_records'];records=scientific+context
    new=manifest['new_records']
    if len(new)!=68 or {key(r['job']) for r in new}!={key(j) for j in jobs()}:
        raise ValueError('Incomplete declared new-run grid.')
    if len(scientific)!=306 or len(context)!=72:raise ValueError('Incomplete comparison records.')
    indexed={record_key(r):r for r in records}
    if len(indexed)!=len(records):raise ValueError('Duplicate policy-aware records.')
    context_ids={r['job_id'] for r in context}
    for r in records+manifest['calibration_records']:
        checked_result(r)
        for name,h in r['source_sha256'].items():
            if file_sha256(ROOT/'code'/name)!=h:raise ValueError('Source changed: '+name)
    def ident(r):
        j=r['job'];m=j['method']
        return {'method':m,'regime':j['regime'],'scenario':j['scenario'],'seed':j['seed'],'job_id':r['job_id'],
                'attack_policy':j.get('attack_policy','standard'),
                'panel':'context' if r['job_id'] in context_ids else 'control' if m==CONTROL else 'sensitivity' if m in SETTINGS and m!=MAIN else 'primary'}
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
                        'threshold':row['values']['threshold'],'neighbour_count':len(item['neighbours']),
                        'global_score':item['global_score'],'local_score':item['local_score'],
                        'trust':item['trust'],'previous_trust':item['previous_trust'],'instantaneous_trust':item['instantaneous_trust'],
                        'plr_center_distance':item['plr_center_distance'],'excluded':item['excluded'],
                        'coefficient':item['applied_delta_coefficient'],'reference_coefficient':other['applied_delta_coefficient'],
                        'applied_norm':item['applied_contribution_l2_norm'],'reference_norm':other['applied_contribution_l2_norm'],
                        'kernel_bandwidth_squared':row['values']['kernel']['bandwidth_squared'],
                        'median_termination':row['values']['median_solver']['termination']})
    table=pd.DataFrame(metrics)
    means=table.groupby(['panel','method','attack_policy','regime','scenario','metric']).value.agg(['mean','count','std']).reset_index()
    effects=[]
    for r in scientific:
        j=r['job']
        if j['method']!=MAIN:continue
        one=table[table.job_id==r['job_id']]
        for method in BASELINES+(CONTROL,):
            other=table[(table.method==method)&(table.regime==j['regime'])&(table.seed==j['seed'])&(table.scenario==j['scenario'])]
            if other.empty:continue
            paired=one.merge(other,on='metric',suffixes=('_main','_baseline'),validate='one_to_one')
            effects.extend({'method':MAIN,'baseline':method,'regime':j['regime'],'scenario':j['scenario'],'seed':j['seed'],
                'metric':x['metric'],'main_value':x['value_main'],'baseline_value':x['value_baseline'],'difference_pp':x['value_main']-x['value_baseline']}
                for x in paired.to_dict('records'))
    audits=[];exact_first_round=0
    for r in new:
        j=r['job'];path=ROOT/r['run_path'];rounds=j['rounds'];local=pd.read_csv(path/'client_metrics.csv')
        if len(local)!=10*rounds or not local.optimizer_steps.eq(8).all() or not local.examples.eq(512).all():raise AssertionError('Training workload changed.')
        if not np.isfinite(local.online_train_loss).all():raise AssertionError('Nonfinite loss.')
        info=json.loads((path/'run.json').read_text());access=json.loads((path/'flaura_access.json').read_text())
        if info['official_test_used'] or access['official_test_used']:raise AssertionError('Official-test access.')
        aux=access['reference'];indices=set(aux['original_indices'])
        partition=json.loads((ROOT/'data/partitions'/info['config']['partition_id']/'indices.json').read_text())['indices']
        forbidden=set(partition['development_validation'])|{i for x in partition['clients'].values() for i in x}
        if len(indices)!=100 or indices & forbidden or not indices<=set(partition['server_reference']):raise AssertionError('Auxiliary data leak.')
        if aux['class_counts']!={str(i):10 for i in range(10)}:raise AssertionError('Unbalanced auxiliary data.')
        diag=json.loads((path/'aggregation_diagnostics.json').read_text())
        if sum(x['values']['reference_forward_examples'] for x in diag)!=1000*rounds:raise AssertionError('Missing auxiliary forwards.')
        if j['method']!=CALIBRATION:
            base=indexed['FedAvg',j['regime'],j['scenario'],j['seed'],'standard']
            old=pd.read_csv(ROOT/base['run_path']/'client_metrics.csv')
            columns=['client','submitted_state_sha256','broadcast_state_sha256','shuffle_seed','optimizer_steps','online_train_loss']
            pd.testing.assert_frame_equal(local[local['round']==1][columns].reset_index(drop=True),old[old['round']==1][columns].reset_index(drop=True),check_exact=True)
            exact_first_round+=10
            for row in diag:
                values=row['values'];items=values['client_metrics'];weights=[v['applied_delta_coefficient'] for v in items.values()]
                if not np.isclose(sum(weights),1.) and sum(weights)!=0.:raise AssertionError('Nonconvex aggregate.')
                if not all(v['applied_delta_coefficient']==0 for v in items.values() if v['excluded']):raise AssertionError('Excluded update contributed.')
                if j['method']==CONTROL and any(v['excluded'] for v in items.values()):raise AssertionError('No-filter control excluded client.')
        audits.append({'job_id':r['job_id'],'method':j['method'],'rounds':rounds,'client_updates':10*rounds,
            'optimizer_steps':80*rounds,'train_forward_examples':5120*rounds,'reference_forward_examples':1000*rounds,
            'development_task_forward_examples':3000*(rounds+1),'checkpoint_replay_forward_examples':6000})
    out=study/'analysis';out.mkdir(exist_ok=True)
    tables={'seed_metrics':metrics,'accuracy_curves':curves,'model_episodes':episodes,'costs':costs,
            'local_training':training,'raw_task_metrics':tasks_out,'recovery':recoveries,'contributions':contributions,
            'contribution_restoration':restorations,'contribution_reduction':reductions,'trust':trust,'paired_effects':effects}
    for name,rows in tables.items():pd.DataFrame(rows).to_csv(out/(name+'.csv'),index=False)
    means.to_csv(out/'means.csv',index=False)
    summary={'analysis_source_sha256':file_sha256(__file__),'scientific_trajectories':306,'primary_trajectories':270,
        'control_trajectories':12,'sensitivity_trajectories':24,'context_trajectories':72,'calibration_trajectories':2,
        'new_runs':68,'matching_checks':pairs,'pre_event_checks':prefix,'exact_first_round_client_updates_vs_fedavg':exact_first_round,
        'interpretation':'Documented reconstruction and MNIST transfer, not an author-code reproduction. Two reused development seeds; one-seed sensitivities. No statistical superiority, universal failure or backdoor-security claim.'}
    write_json(out/'summary.json',summary)
    totals={k:sum(a[k] for a in audits) for k in ['client_updates','optimizer_steps','train_forward_examples','reference_forward_examples','development_task_forward_examples','checkpoint_replay_forward_examples']}
    write_json(study/'access_audit.json',{'new_runs':68,'official_test_used':False,'server_auxiliary_examples':100,
        'auxiliary_labels_used_in_defence':False,'development_evaluation':'reporting_only','totals':totals,'runs':audits})
    return summary
