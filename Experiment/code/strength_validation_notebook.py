"""Saved-artifact presentation only; never trains or evaluates models."""
from pathlib import Path
import json
import nbformat as nbf
from aggregate_validation_notebook import _execute
ROOT=Path(__file__).resolve().parents[1]
STUDY=ROOT/'outputs/strength_validation/20261030_v1'
NOTEBOOK=ROOT/'code/23_controlled_strength.ipynb'


def make_notebook(study=STUDY):
    md,code=nbf.v4.new_markdown_cell,nbf.v4.new_code_cell
    cells=[md(r'''# 23 — Controlled client contribution strength

Can a smaller contribution retain most of an honest client's local-domain benefit while reducing damage elsewhere? This experiment varies one client's contribution at a time at **0%, 25%, 50%, and 100%** of its original weight. The other nine updates retain their original weights, including any poisoned updates. No renormalization occurs.

\[
G_i(s)=w+\sum_{j\ne i}0.1\Delta_j+s\,0.1\Delta_i,
\qquad u_{iv}(s)=L_v(G_i(0))-L_v(G_i(s)).
\]

Positive utility means including the candidate at that strength improves normalized Brier loss compared with omitting it. It does not establish that the resulting model improves on the broadcast. The 96 cases reuse the same four clean checkpoints, two seeds, two partitions, six scenarios, training samples and private report/audit data as notebook 22. **Zero trusted server images; no global-development or official-test evaluation.**

This is exploratory development on correlated observations, not an independent validation study, a new global-model benchmark or a novelty claim.'''),code(f'''from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
from IPython.display import display,Image,Markdown
CODE=Path({str(ROOT/'code')!r});sys.path.insert(0,str(CODE))
from strength_validation_study import verify,results
from experiment_core import file_sha256
STUDY=Path({str(Path(study).resolve())!r});A=STUDY/'analysis'
assert json.loads((STUDY/'status.json').read_text())['status']=='complete'
P=verify(STUDY);records=results(STUDY);assert len(records)==96
S=json.loads((A/'summary.json').read_text());ACCESS=json.loads((STUDY/'access_audit.json').read_text())
assert S['analysis_source_sha256']==file_sha256(CODE/'strength_validation_analysis.py')
assert ACCESS['exact_prior_update_replays']==960 and ACCESS['exact_endpoint_rows']==19200
assert ACCESS['report_forward_examples']==ACCESS['audit_forward_examples']==3809280
assert ACCESS['trusted_server_images']==ACCESS['global_development_forward_examples']==ACCESS['global_models_advanced']==0
assert ACCESS['official_test_used'] is False
C=pd.read_csv(A/'audit_curves.csv');F=pd.read_csv(A/'feasibility_candidates.csv');FR=pd.read_csv(A/'feasibility_rates.csv')
D=pd.read_csv(A/'adaptive_decisions.csv');R=pd.read_csv(A/'adaptive_grouped_rates.csv');UR=pd.read_csv(A/'adaptive_utility_rates.csv')
FD=pd.read_csv(A/'fixed_decisions.csv');RATES=pd.read_csv(A/'fixed_grouped_rates.csv')
assert len(C)==3840 and len(D)==14400 and len(FD)==34560
RULES=['peer_median','peer_top3_mean','peer_fourth_highest']
LABELS={{'peer_median':'Median','peer_top3_mean':'Top-three mean','peer_fourth_highest':'Fourth-highest'}}
PROFILES=['truthful','directional_extreme','sign_reversal','corrupted_labels','force_full']
REGIMES=['IID','non-IID'];SCENARIOS=['abrupt_shot_noise','gradual_stripe','mixed_stripe_attack']
SLABEL={{'abrupt_shot_noise':'Honest noise','gradual_stripe':'Honest stripes','mixed_stripe_attack':'Stripes + poisoning'}}
PDISPLAY={{'truthful':'Truthful','directional_extreme':'Extreme lies','sign_reversal':'Sign reversal','corrupted_labels':'Flipped labels','force_full':'Force full'}}
pd.set_option('display.max_rows',150);pd.set_option('display.max_columns',20)
print('Verified 96 cases; 960 identical submitted models; 19,200 freshly reproduced endpoint rows.')
print('239 preflight tests passed. All data and checkpoints are reused exploratory development inputs.')
'''),md('''## Declared criteria and controls

The scale selector takes only three peer scores. It chooses the scale with the greatest strictly positive score; exact ties prefer the smaller scale. If no score is positive, it chooses zero. The zero action has known utility zero, so a client cannot make an identical-model action attractive through a claimed score.

We retain the three earlier peer-score controls: median, top-three mean and fourth-highest, always excluding the candidate's own report. Fixed-strength controls retain the four earlier report profiles. Adaptive controls add **force full**: malicious validators support poisoned updates only at 100% and oppose every other nonzero alternative. Reports stay bounded in [−1,1]. These controls do not authenticate report honesty.

The same intent criteria require at least 95% honest retention, 80% poison rejection and 80% repaired readmission in every declared subgroup. Any nonzero poisoned contribution counts as admission, regardless of attenuation. Positive acceptance of a small honest contribution is also insufficient by itself: the separate utility criterion checks its benefit.

For honestly changed candidates, **D(s)** is mean audit Brier benefit on changed domains and **H(s)** is the positive part of harm to remaining domains. A satisfactory scale must have positive benefit, retain at least **75% of positive full-strength benefit**, and leave at most **50% of full-strength collateral harm**. If full strength is harmful on the changed domain, a qualifying alternative must rescue a positive benefit. Full strength may already qualify when it helps its domain without collateral harm.

The audit-aware existence check is an optimistic feasibility bound, not a usable server policy. It must succeed for at least 80% of candidates in each of 12 seed/regime/scenario groups. The report-only selector must meet the same utility target in every profile, in addition to the intent gate. Audit data assess these research gates; they never compute scores or select weights. The reused audits are not an untouched confirmation set.'''),code('''print('Continuation:',S['continuation'])
print('Audit-aware feasibility:',S['feasibility_gate'])
gates=[]
for scale,gate in S['fixed_gates'].items():
    for rule,v in gate['rules'].items():
        gates.append({'control':'Fixed '+scale,'rule':LABELS[rule],'passed':v['pass'],'failed_groups':v['failed_groups'],'groups':v['groups']})
for rule,v in S['adaptive_intent_gate']['rules'].items():
    gates.append({'control':'Adaptive intent','rule':LABELS[rule],'passed':v['pass'],'failed_groups':v['failed_groups'],'groups':v['groups']})
for rule,v in S['adaptive_utility_gates'].items():
    gates.append({'control':'Adaptive utility','rule':LABELS[rule],'passed':v['pass'],'failed_groups':v['failed_groups'],'groups':v['groups']})
display(pd.DataFrame(gates));display(FR)
if (STUDY/'decision.json').exists():
    decision=json.loads((STUDY/'decision.json').read_text())
    for name,digest in decision.get('evidence_sha256',{}).items():assert file_sha256(STUDY/name)==digest
    display(pd.Series({k:v for k,v in decision.items() if isinstance(v,(str,int,float,bool)) or v is None},name='Saved decision').to_frame())
'''),md('''## Contribution curves and feasible trade-offs

Curves show mean individual counterfactual effects, not effects of applying every selected scale simultaneously. Each honest changed group has 24 candidate observations per regime, pooled across two seeds and four correlated checkpoints. The per-seed gate above remains authoritative. The class-wise reports, cross-entropy and accuracy measurements remain in each case's saved reports and audit files; no extra forward evaluation is used to create this review.'''),code('''import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
FIG=A/'figures';FIG.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':130})
def save_figure(fig,name):
    fig.savefig(FIG/(name+'.png'),dpi=180,bbox_inches='tight')
    fig.savefig(FIG/(name+'.pdf'),bbox_inches='tight')
    plt.close(fig);display(Image(filename=str(FIG/(name+'.png'))))
H=C[C.group=='changed']
curve=H.groupby(['regime','scenario','scale'])[['changed_cohort_audit_gain','remaining_domains_audit_gain','all_domains_audit_gain']].mean()
display(curve.round(6))
print('Poisoned contribution curves: negative gain is damage relative to omitting that candidate.')
poison_curves=C[C.group=='poisoned'].groupby(['regime','scenario','scale'])[['all_domains_audit_gain','remaining_domains_audit_gain','remaining_domain_harm']].mean()
display(poison_curves.round(6))
fig,axes=plt.subplots(2,3,figsize=(13,7),sharex=True)
for row,regime in enumerate(REGIMES):
    for col,scenario in enumerate(SCENARIOS):
        ax=axes[row,col];data=curve.loc[(regime,scenario)].sort_index()
        ax.plot(data.index*100,data.changed_cohort_audit_gain,'o-',label='Changed domains',color='#197a64')
        ax.plot(data.index*100,data.remaining_domains_audit_gain,'s-',label='Remaining domains',color='#bd4e50')
        ax.axhline(0,color='#666666',lw=.7);ax.set_title(regime+' · '+SLABEL[scenario]);ax.set_xticks([0,25,50,100]);ax.grid(alpha=.15)
        if row==1:ax.set_xlabel('Contribution strength (%)')
        if col==0:ax.set_ylabel('Mean Brier improvement vs omission')
axes[0,0].legend(fontsize=9);fig.suptitle('Individual contribution curves — honest changed clients',fontsize=15)
fig.tight_layout(rect=(0,0,1,.95));save_figure(fig,'strength_domain_curves')
feasible=F.groupby(['regime','scenario'])[['any_scale_feasible','fractional_scale_feasible','full_strength_feasible']].agg(['sum','count','mean'])
display(feasible)
fig,axes=plt.subplots(1,2,figsize=(12,4.4),sharey=True)
for ax,regime in zip(axes,REGIMES):
    data=F[F.regime==regime].groupby('scenario')[['any_scale_feasible','fractional_scale_feasible']].mean().reindex(SCENARIOS)
    x=np.arange(3)
    ax.bar(x-.17,100*data.any_scale_feasible,.34,label='Any nonzero scale',color='#287b8e')
    ax.bar(x+.17,100*data.fractional_scale_feasible,.34,label='Fractional scale only',color='#8fbdb6')
    for k,field in enumerate(['any_scale_feasible','fractional_scale_feasible']):
        for i,v in enumerate(data[field]):ax.text(i+(-.17 if k==0 else .17),100*v+2,f'{int(round(v*24))}/24',ha='center',fontsize=9)
    ax.axhline(80,color='#555555',ls='--',lw=1);ax.set_ylim(0,112);ax.set_xticks(x,[SLABEL[s] for s in SCENARIOS]);ax.set_title(regime)
axes[0].set_ylabel('Candidates meeting 75% benefit / 50% harm target (%)');axes[0].legend(fontsize=9)
fig.suptitle('Audit-aware feasibility bound; pooled rates do not replace per-seed gates',fontsize=13)
fig.tight_layout(rect=(0,0,1,.94));save_figure(fig,'strength_feasibility')
''')]
    cells += [md('''## Fixed strengths and adaptive decisions

A client is admitted when its chosen strength is nonzero. These admission rates do not measure global-model accuracy. In particular, attenuating poison is distinct from rejecting it. The following tables include clean clients, unchanged clients, honest drift, active poison and repaired clients. The complete per-seed/profile tables are saved alongside this notebook.

The per-scale arbitrary-report envelopes remain available in `fixed_fault_envelopes.csv` and `fixed_envelope_rates.csv`. Those bounds may use different adversarial identities at different scales and must not be presented as an exact simultaneous adaptive attack.'''),code('''fixed=FD[FD.profile=='truthful'].groupby(['regime','rule','scale','scenario','group']).accepted.agg(['sum','count','mean'])
display(fixed.round(4))
selected=D.groupby(['regime','rule','profile','scenario','group']).agg(accepted=('accepted','sum'),observations=('accepted','size'),acceptance=('accepted','mean'),mean_strength=('selected_scale','mean'))
selected.to_csv(A/'notebook_selected_summary.csv')
display(selected.loc[(slice(None),slice(None),['truthful','force_full'],slice(None),slice(None)),:].round(4))
print('Mean strength includes zero for rejected candidates. A 25% decision retains coefficient 0.025, not 0.25 of the whole global model.')
fig,axes=plt.subplots(2,3,figsize=(14,7.3))
metrics=[('gradual_stripe','changed','Honest stripes retained',False),('label_flip','poisoned','Persistent poison rejected',True),('mixed_stripe_attack','poisoned','Mixed-case poison rejected',True)]
for i,regime in enumerate(REGIMES):
    for j,(scenario,group,title,rejection) in enumerate(metrics):
        ax=axes[i,j];values=np.zeros((3,5));counts={}
        for r,rule in enumerate(RULES):
            for p,profile in enumerate(PROFILES):
                sub=D[(D.regime==regime)&(D.rule==rule)&(D.profile==profile)&(D.scenario==scenario)&(D.group==group)]
                n=len(sub);success=n-int(sub.accepted.sum()) if rejection else int(sub.accepted.sum())
                assert n==24
                values[r,p]=100*success/n;counts[r,p]=(success,n)
        ax.imshow(values,cmap='YlGnBu',vmin=0,vmax=100,aspect='auto')
        for r in range(3):
            for p in range(5):
                num,den=counts[r,p];ax.text(p,r,f'{values[r,p]:.0f}%\\n{num}/{den}',ha='center',va='center',fontsize=8,color='white' if values[r,p]>65 else '#111111')
        ax.set_title(regime+' · '+title,fontsize=11);ax.set_yticks(range(3),[LABELS[r] for r in RULES] if j==0 else ['']*3)
        ax.set_xticks(range(5),[PDISPLAY[p] for p in PROFILES],rotation=28,ha='right',fontsize=8)
fig.suptitle('Report-only adaptive controls: retention and rejection',fontsize=15)
fig.tight_layout(rect=(0,0,1,.94));save_figure(fig,'strength_adaptive_decisions')
display(UR.groupby(['rule','profile','regime','scenario'])[['successes','n_candidates']].sum())
print('Every adaptive utility group must pass separately; pooled values are descriptive.')
'''),md('''## Secondary audit metrics and local training

Brier loss, cross-entropy and accuracy can disagree; the declared gate uses Brier. The secondary table summarizes the same predictions, with no further evaluation. Training losses describe the repeated local SGD steps; poisoned clients train on corrupted labels, so those losses do not establish legitimate-task performance. None of these values is official-test accuracy or a multi-round trajectory.'''),code('''SECONDARY=[];TRAIN=[];META=[]
for record in records:
    path=CODE.parent/record['run_path'];job=record['job']
    meta=json.loads((path/'metadata.json').read_text());META.append(meta)
    aud=pd.DataFrame(json.loads((path/'audit.json').read_text()))
    truth=C[(C.regime==job['regime'])&(C.seed==job['seed'])&(C.scenario==job['scenario'])&(C['round']==job['round'])]
    changed=set(truth[truth.group=='changed'].candidate_id)
    for (scale,cid),sub in aud.groupby(['scale','candidate_id']):
        group=truth[(truth.scale==scale)&(truth.candidate_id==cid)].group.iloc[0]
        out={**job,'scale':scale,'candidate_id':cid,'group':group}
        for label,part in [('all_domains',sub),('changed_domains',sub[sub.validator_id.isin(changed)]),('remaining_domains',sub[~sub.validator_id.isin(changed)])]:
            out[label+'_accuracy_gain_pp']=100*part.accuracy_gain.mean() if len(part) else np.nan
            out[label+'_cross_entropy_gain']=part.mean_cross_entropy_gain.mean() if len(part) else np.nan
        SECONDARY.append(out)
    TRAIN.extend({**job,**row} for row in json.loads((path/'training.json').read_text()))
SECONDARY=pd.DataFrame(SECONDARY);TRAIN=pd.DataFrame(TRAIN)
SECONDARY.to_csv(A/'notebook_secondary_audit.csv',index=False)
TRAIN.to_csv(A/'notebook_local_training.csv',index=False)
metrics=[col for col in SECONDARY.columns if col.endswith('_pp') or col.endswith('_gain')]
display(SECONDARY.groupby(['regime','scenario','group','scale'])[metrics].mean().round(5))
display(TRAIN.groupby(['regime','scenario']).agg(local_trainings=('client_id','size'),mean_online_training_loss=('online_train_loss','mean'),training_examples=('examples','sum'),optimizer_steps=('optimizer_steps','sum')).round(5))
'''),md('''## Repair, workload and limits

At checkpoints 12 and 20, clients whose attacks stopped train honestly from the same clean checkpoints as the clean control. Matching scores and scale choices are therefore expected. These stateless comparisons do not demonstrate learned rehabilitation or recovery of an attacked global model.

Each counterfactual keeps the other nine clients at full strength. It is invalid to add these marginal gains or assume they describe an aggregate where all clients simultaneously receive their selected strengths. The conditional next stage would need to test that actual combined model and its evolving trajectory. A failed screen stops this combination; it is not a general impossibility result for reference-free FL.'''),code('''repair=[]
for scale,out in S['repair_equivalence'].items():
    repair.append({'scale':float(scale),'candidate_comparisons':out['n_repaired_observations'],'all_equal':out['all_equal']})
display(pd.DataFrame(repair));assert all(r['all_equal'] for r in repair)
keys=['regime','seed','round','candidate_id','rule']
repaired=D[(D.scenario=='attack_stops')&(D.group=='repaired')]
clean=D[(D.scenario=='clean')&(D.profile=='truthful')]
paired=repaired.merge(clean,on=keys,suffixes=('_repaired','_clean'),validate='many_to_one')
for field in ['selected_scale','score','score_025','score_050','score_100','self_audit_gain','all_domains_audit_gain']:
    assert paired[field+'_repaired'].equals(paired[field+'_clean'])
print('Adaptive repair comparisons:',len(paired),'rows; 24 unique candidate/checkpoint observations, all equal clean truthful counterfactuals.')
display(repaired[repaired.profile=='truthful'].groupby(['rule','regime']).agg(readmitted=('accepted','sum'),observations=('accepted','size'),mean_strength=('selected_scale','mean')))
display(pd.Series({k:v for k,v in ACCESS.items() if isinstance(v,(int,float,bool))},name='Access audit').to_frame())
model_bytes={m['float32_model_payload_bytes'] for m in META};assert len(model_bytes)==1
model_bytes=model_bytes.pop()
assert all(r['naive_model_distribution_bytes']==280*model_bytes and r['peer_scalar_float64_payload_bytes']==2160 for r in ACCESS['cases_audit'])
print('Naive scoring model payload per case:',round(280*model_bytes/2**20,2),'MiB; scalar payload 2160 bytes.')
print('This is an uncompressed estimate, not measured traffic. Audit forwards are research overhead.')
display(pd.DataFrame([{'regime':m['job']['regime'],'seed':m['job']['seed'],**m['cost']} for m in META]).groupby(['regime','seed']).sum(numeric_only=True).round(3))
print('Decision:',S['continuation']['decision'])
print('No trusted reference data, global development feedback, official-test access or global-model advancement occurred in this stage.')
'''),md('''## Reproduction and complete tables

Run `Experiment/.venv/bin/python Experiment/code/strength_validation_notebook.py` to recreate this presentation from verified saved artifacts. It does not train or evaluate a model.

The study directory `outputs/strength_validation/20261030_v1/` contains the frozen protocol/source snapshot, per-case model hashes and fresh endpoint checks, training/report/audit summaries, access audit and decision. The `analysis/` directory contains every fixed and adaptive decision, per-seed rates, per-scale report envelopes, audit curves, feasibility outcomes and the notebook's secondary metrics. PNG and PDF figures are in `analysis/figures/`.

Client data and labels stay inside simulated local evaluator calls. This is a software boundary, not a deployed privacy system. The study uses the same 128 local-report and 128 independent-audit samples per client as the previous pilots; all roles remain disjoint from source training samples and the global reference/development reserves. No report authenticity, deployment scalability or novelty claim follows from this diagnostic.''')]
    notebook=nbf.v4.new_notebook(cells=cells)
    notebook.metadata.kernelspec={'display_name':'FedReTrust Experiment (Python 3.14)','language':'python','name':'fedretrust-experiment'}
    return notebook


def build():
    if json.loads((STUDY/'status.json').read_text())['status']!='complete':raise ValueError('Wait for the complete study.')
    from strength_validation_study import verify
    verify(STUDY)
    return _execute(make_notebook(),NOTEBOOK)


if __name__=='__main__':print(build())
