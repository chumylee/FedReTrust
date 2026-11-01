"""Offline v0.4 comparisons and predeclared continuation; never feeds the defence."""
from pathlib import Path
import json,math
import numpy as np
import pandas as pd
from experiment_core import file_sha256,write_json
from research_runs import ROOT
from research_analysis import extract_run,cross_method_pair,cost_record,contribution_records
from research_study import task_series
from scenario_metrics import longer_scenarios,matched_reference_name,model_recovery
from directional_study import checked_result

MAIN='FedReTrust-v04';CONTROL='FedReTrust-v04-block-RFA-control'
BASELINES=('FedAvg','RFA','FLTrust','Viceroy','FedReTrust','FedReTrust-RF-recovery','FedReTrust-Direction-loose')
REGIMES=('IID','non-IID');CORE=('clean','gradual_stripe','label_flip')

def identity(r,panel):
    j=r['job'];return {'method':j['method'],'regime':j['regime'],'scenario':j['scenario'],'seed':j['seed'],
        'job_id':r['job_id'],'attack_policy':j.get('attack_policy','standard'),'panel':panel}

def key(r):
    j=r['job'];return j['method'],j['regime'],j['scenario'],j['seed'],j.get('attack_policy','standard')

def continuation_checks(means):
    def value(method,regime,scenario,metric):
        f=means[(means.method==method)&(means.regime==regime)&(means.scenario==scenario)&
                (means.metric==metric)&(means.panel=='pilot')&(means.attack_policy=='standard')]
        if len(f)!=1 or int(f.iloc[0]['count'])!=2:raise ValueError('Continuation needs two matched seeds per method.')
        return float(f.iloc[0]['mean'])
    targets=[]
    for g in REGIMES:
        for scenario,metric,sign in (('clean','overall_mean_5_24_pp',1),('gradual_stripe','affected_honest_event_pp',1),
                ('label_flip','attack_active_signed_damage_pp',-1),('label_flip','attack_active_accuracy_pp',1)):
            values={m:value(m,g,scenario,metric) for m in BASELINES}
            best=max(values,key=lambda m:sign*values[m]);actual=value(MAIN,g,scenario,metric)
            advantage=sign*(actual-values[best])
            targets.append({'regime':g,'scenario':scenario,'metric':metric,'benchmark':best,'target_pp':values[best],
                'value_pp':actual,'advantage_pp':advantage,'met':advantage>=0,'requirement':'match_best_prior'})
        actual=value(MAIN,g,'gradual_stripe','affected_honest_event_pp');target=value('RFA',g,'gradual_stripe','affected_honest_event_pp')+1.
        targets.append({'regime':g,'scenario':'gradual_stripe','metric':'affected_honest_event_pp','benchmark':'RFA + 1pp',
            'target_pp':target,'value_pp':actual,'advantage_pp':actual-target,'met':actual>=target,'requirement':'practical_stripe_gain'})
    return targets

def paired_t_df4(differences):
    x=np.asarray(differences,dtype=float)
    if x.shape!=(5,) or not np.isfinite(x).all():raise ValueError('Exactly five finite paired seed differences required.')
    mean=float(x.mean());sd=float(x.std(ddof=1));se=sd/math.sqrt(5)
    if se:
        t=mean/se;u=t/math.sqrt(t*t+4.);p=max(0.,min(1.,.5-.75*u+.25*u**3))
    else:t=None;p=0. if mean>0 else 1.
    return {'n':5,'mean_difference_pp':mean,'paired_sd_pp':sd,'ci95_low_pp':mean-2.7764451051977987*se,
        'ci95_high_pp':mean+2.7764451051977987*se,'t_df4':t,'one_sided_p':p}

def holm_adjust(values):
    values=np.asarray(values,dtype=float)
    if not np.isfinite(values).all() or ((values<0)|(values>1)).any():raise ValueError('Invalid p-values.')
    out=np.empty(len(values));previous=0.
    for rank,index in enumerate(np.argsort(values,kind='stable')):
        previous=max(previous,min(1.,(len(values)-rank)*float(values[index])));out[index]=previous
    return out.tolist()

def analyse_study(study_path,stage='pilot'):
    from blockwise_study import expected_pilot_keys,validation_jobs,key as job_key,MAIN as EXPECTED_MAIN
    assert MAIN==EXPECTED_MAIN
    study=Path(study_path);p=json.loads((study/'protocol.json').read_text())
    manifest=json.loads((study/'result_manifest.json').read_text())
    scientific=manifest['pilot_records'] if stage=='pilot' else manifest['validation_records']
    expected=expected_pilot_keys() if stage=='pilot' else {job_key(j) for j in validation_jobs()}
    if len(scientific)!=len(expected) or {job_key(r['job']) for r in scientific}!=expected:raise ValueError('Incomplete comparison grid.')
    context=manifest['context_records'] if stage=='pilot' else []
    records=scientific+context;indexed={key(r):r for r in records}
    if len(indexed)!=len(records):raise ValueError('Duplicate policy-aware trajectory.')
    for r in records:
        checked_result(r)
        for name,digest in r['source_sha256'].items():
            if file_sha256(ROOT/'code'/name)!=digest:raise ValueError('Source changed: '+name)
    context_ids={r['job_id'] for r in context}
    def reference(r):
        j=r['job'];policy='compliant' if j.get('attack_policy') in ('compliant','bypass') else 'standard'
        return indexed[j['method'],j['regime'],matched_reference_name(j['scenario']),j['seed'],policy]
    metrics=[];curves=[];episodes=[];costs=[];training=[];raw_tasks=[];recoveries=[]
    contributions=[];restorations=[];reductions=[];block_rows=[];pairing=[];prefix=[]
    for r in records:
        j=r['job'];panel='context' if r['job_id'] in context_ids else stage;ident=identity(r,panel)
        path=ROOT/r['run_path'];ref=reference(r);base=indexed['FedAvg',j['regime'],j['scenario'],j['seed'],'standard']
        cross_method_pair(r,base);pairing.append({'job_id':r['job_id'],'reference_job_id':base['job_id'],'matched':True})
        is_event=bool(longer_scenarios()[j['scenario']].events)
        a,b,c=extract_run(r,ref if is_event else None)
        metrics.extend({**row,**ident} for row in a);curves.extend({**row,**ident} for row in b);episodes.extend({**row,**ident} for row in c)
        costs.append({**cost_record(r),**ident})
        local=pd.read_csv(path/'client_metrics.csv');tasks=pd.read_csv(path/'task_metrics.csv')
        training.extend({**row,**ident} for row in local.to_dict('records'));raw_tasks.extend({**row,**ident} for row in tasks.to_dict('records'))
        if is_event:
            a,b,c=contribution_records(r,ref)
            contributions.extend({**row,**ident} for row in a);restorations.extend({**row,**ident} for row in b);reductions.extend({**row,**ident} for row in c)
            own=ROOT/ref['run_path'];h=pd.read_csv(path/'history.csv');hc=pd.read_csv(own/'history.csv')
            if h[h['round']<=4].global_state_sha256.tolist()!=hc[hc['round']<=4].global_state_sha256.tolist():raise AssertionError('Pre-event models mismatch.')
            tc=pd.read_csv(own/'task_metrics.csv')
            pd.testing.assert_frame_equal(tasks[tasks['round']<=4].reset_index(drop=True),tc[tc['round']<=4].reset_index(drop=True),check_exact=True)
            prefix.append({'job_id':r['job_id'],'reference_job_id':ref['job_id'],'first_four_rounds_exact':True})
        scenario=longer_scenarios()[j['scenario']];attacks=[e for e in scenario.events if e.intent=='malicious']
        if attacks:
            windows=attacks[0].windows;active=sorted({t for lo,hi in windows for t in range(lo,hi+1)})
            observed=task_series(r);control=task_series(ref,scenario_name=j['scenario'])
            for name,value in [('attack_active_accuracy_pp',100*np.mean([observed[t] for t in active])),
                              ('matched_reference_attack_window_accuracy_pp',100*np.mean([control[t] for t in active]))]:
                metrics.append({**ident,'phase':'development','metric':name,'value':float(value)})
            for i,(lo,hi) in enumerate(windows):
                if hi>=24:continue
                next_attack=windows[i+1][0] if i+1<len(windows) else None
                for group,clients in (('all_clients',None),('affected_attackers',list(attacks[0].clients))):
                    obs={t:v for t,v in task_series(r,client_ids=clients).items() if t>0}
                    ctl={t:v for t,v in task_series(ref,scenario_name=j['scenario'],client_ids=clients).items() if t>0}
                    recoveries.append({**ident,'group':group,'episode':i+1,**model_recovery(ctl,obs,hi,next_attack=next_attack)})
        if j['method'] in (MAIN,CONTROL):
            diagnostics=json.loads((path/'aggregation_diagnostics.json').read_text())
            old=json.loads((ROOT/ref['run_path']/'aggregation_diagnostics.json').read_text())
            for row,prior in zip(diagnostics,old):
                for block,item in row['values']['blocks'].items():
                    for cid,values in item['client_metrics'].items():
                        baseline=prior['values']['blocks'][block]['client_metrics'][cid]['applied_contribution_l2_norm']
                        cid_int=int(cid);events=scenario.active(cid_int,row['round'])
                        role='attacking' if any(e.intent=='malicious' for e in events) else 'honest_changed' if any(e.intent=='honest' for e in events) else 'currently_unchanged'
                        block_rows.append({**ident,'round':row['round'],'client':cid_int,'block':block,'role':role,
                            'is_head':item['protected_head'],'norm_cap':item['norm_cap'],'residual_radius':item['residual_radius'],
                            **values,'matched_norm_ratio':values['applied_contribution_l2_norm']/baseline if baseline>1e-12 else None})
    table=pd.DataFrame(metrics)
    means=table.groupby(['panel','method','attack_policy','regime','scenario','metric']).value.agg(['mean','count','std']).reset_index()
    effects=[]
    for r in scientific:
        j=r['job']
        if j['method']!=MAIN:continue
        one=table[table.job_id==r['job_id']]
        for baseline in BASELINES+(CONTROL,):
            other=table[(table.method==baseline)&(table.regime==j['regime'])&(table.seed==j['seed'])&(table.scenario==j['scenario'])]
            if other.empty:continue
            paired=one.merge(other,on='metric',suffixes=('_new','_baseline'),validate='one_to_one')
            for row in paired.to_dict('records'):
                effects.append({'method':MAIN,'baseline':baseline,'regime':j['regime'],'scenario':j['scenario'],'seed':j['seed'],
                    'metric':row['metric'],'new_value':row['value_new'],'baseline_value':row['value_baseline'],
                    'difference_pp':row['value_new']-row['value_baseline']})
    training_audits=[]
    new=[r for r in manifest['new_records'] if r['job']['v04_stage']==stage]
    for r in new:
        j=r['job'];path=ROOT/r['run_path'];f=pd.read_csv(path/'client_metrics.csv')
        if len(f)!=240 or not f.optimizer_steps.eq(8).all() or not f.examples.eq(512).all():raise AssertionError('Local training budget changed.')
        if not np.isfinite(f.online_train_loss).all():raise AssertionError('Nonfinite local loss.')
        info=json.loads((path/'run.json').read_text())
        if info['official_test_used']:raise AssertionError('Official-test access detected.')
        d=json.loads((path/'aggregation_diagnostics.json').read_text())
        if j['method'] in (MAIN,CONTROL):
            if not f.aggregation_coefficient.isna().all():raise AssertionError('Invented global scalar coefficient.')
            for row in d:
                values=row['values'];blocks=values['blocks']
                if set(blocks)!={'features.0','features.3','classifier.1','classifier.3'}:raise AssertionError('Wrong block layout.')
                if values['reference_forward_examples']!=0:raise AssertionError('Reference access in v0.4.')
                gram=np.array(values['contribution_measurements']['gram_matrix'])
                if not np.allclose(gram,sum(np.array(b['contribution_gram_matrix']) for b in blocks.values()),rtol=1e-10,atol=1e-12):raise AssertionError('Wrong block contribution accounting.')
                for name,b in blocks.items():
                    if name=='classifier.3' or j['method']==CONTROL:
                        if b['residual_mass']!=0:raise AssertionError('Protected block received residual correction.')
                    if b['aggregate_l2_norm']>b['norm_cap']+1e-8:raise AssertionError('Block norm cap violated.')
        training_audits.append({'job_id':r['job_id'],'method':j['method'],'client_updates':240,'optimizer_steps':1920,
            'train_forward_examples':122880,'development_task_forward_examples':75000,'checkpoint_replay_forward_examples':6000,
            'recorded_reference_forward_examples':sum(row['values'].get('reference_forward_examples',0) for row in d)})
    targets=continuation_checks(means) if stage=='pilot' else []
    statistics=[]
    if stage=='fresh_seed_validation':
        frame=pd.DataFrame(effects)
        for old in BASELINES[-3:]:
            for g in REGIMES:
                for scenario,metric in [('clean','overall_mean_5_24_pp'),('gradual_stripe','affected_honest_event_pp'),('label_flip','attack_active_accuracy_pp')]:
                    sub=frame[(frame.baseline==old)&(frame.regime==g)&(frame.scenario==scenario)&(frame.metric==metric)]
                    statistics.append({'baseline':old,'regime':g,'scenario':scenario,'metric':metric,**paired_t_df4(sub.difference_pp.to_numpy())})
        assert len(statistics)==18
        for row,adjusted in zip(statistics,holm_adjust([s['one_sided_p'] for s in statistics])):
            row.update({'holm_adjusted_p':adjusted,'statistically_improved':adjusted<.05,
                'practically_and_statistically_improved':adjusted<.05 and row['mean_difference_pp']>=1.})
    out=study/'analysis'/stage;out.mkdir(parents=True,exist_ok=True)
    tables={'seed_metrics':metrics,'accuracy_curves':curves,'model_episodes':episodes,'costs':costs,
        'local_training':training,'raw_task_metrics':raw_tasks,'recovery':recoveries,'contributions':contributions,
        'contribution_restoration':restorations,'contribution_reduction':reductions,'block_contributions':block_rows,
        'paired_effects':effects,'target_checks':targets,'paired_statistics':statistics}
    for name,rows in tables.items():pd.DataFrame(rows).to_csv(out/(name+'.csv'),index=False)
    means.to_csv(out/'means.csv',index=False)
    summary={'stage':stage,'analysis_source_sha256':file_sha256(__file__),'scientific_trajectories':len(scientific),
        'context_trajectories':len(context),'new_runs':len(new),'pairing_checks':pairing,'pre_event_checks':prefix,
        'continuation':{'pass':all(t['met'] for t in targets),'targets_met':sum(t['met'] for t in targets),'targets':len(targets)} if targets else None,
        'paired_statistics':statistics,'statistical_scope':'Pilot is descriptive only. Fresh-seed tests, when triggered, use5 paired seeds and Holm adjustment across18 prior-version contrasts; same data/partitions remain reused.',
        'scope':'Main fixed v0.4 is never replaced by its control after outcomes. Empty global coefficients mean undefined, not zero. All scenario/seed coverage and individual outcomes retained.'}
    write_json(out/'summary.json',summary)
    write_json(study/('access_audit_'+stage+'.json'),{'stage':stage,'new_runs':len(new),'v04_trusted_server_images':0,
        'official_test_used':False,'development_evaluation':'reporting_only_never_aggregation_feedback',
        'new_client_updates':240*len(new),'train_forward_examples':122880*len(new),
        'development_task_forward_examples':75000*len(new),'checkpoint_replay_forward_examples':6000*len(new),
        'training_audits':training_audits})
    return summary
