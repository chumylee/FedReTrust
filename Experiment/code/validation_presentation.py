"""Saved-artifact presentation only; never trains or changes frozen analysis.

This file is intentionally outside the execution freeze. It can run only after
all 800 jobs and the frozen analysis have completed. Presentation exports live
outside analysis/ so the original analysis manifest remains unchanged.
"""
from pathlib import Path
import argparse
import json
import os

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / 'outputs/validation_replication/20261030_v03_v1'
NOTEBOOK = ROOT / 'code/29_fedretrust_validation.ipynb'


def make_notebook(study=STUDY):
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    cells = [md('''# 29 — FedReTrust v0.3: prospective validation on fresh partitions and seeds

The selected reference-free v0.3 is compared with existing baselines on five fresh paired blocks in each of IID and non-IID settings. Each block jointly changes the partition, initialization and sampling. The study also removes individual components to test their contribution. It retains all declared outcomes, including failures to meet the targets.

**800 runs; 24 rounds per run; five independent blocks per regime.** These are new replications on known MNIST, whose earlier results informed development. They are not an unseen-dataset confirmation or an established novelty claim.

[Frozen design](../reports/FedReTrust_v03_Validation_Protocol.md) · [Execution freeze](../outputs/validation_replication/20261030_v03_v1/execution_freeze.json) · [Frozen analysis](validation_analysis.py)
'''), code(f'''from pathlib import Path
import sys,json,ast
import numpy as np
import pandas as pd
from IPython.display import display,Image,Markdown
ROOT=Path({str(ROOT)!r});sys.path.insert(0,str(ROOT/'code'))
STUDY=Path({str(Path(study).resolve())!r})
from validation_replication import verify,records,key,MAIN
from experiment_core import file_sha256,write_json
S0=json.loads((STUDY/'status.json').read_text())
assert S0['status']=='complete' and S0['completed_new_runs']==800
F=verify(STUDY);R=records(STUDY)
assert len(R)==800 and {{key(r['job']) for r in R}}=={{key(j) for j in F['jobs']}}
for relative,digest in json.loads((STUDY/'analysis_manifest.json').read_text()).items():
    assert file_sha256(STUDY/relative)==digest,relative
A=STUDY/'analysis';P=STUDY/'presentation';P.mkdir(exist_ok=True)
FIG=P/'figures';FIG.mkdir(exist_ok=True)
S=json.loads((A/'summary.json').read_text())
assert S['runs']==800 and S['independent_blocks_per_regime']==5
assert S['analysis_source_sha256']==file_sha256(ROOT/'code/validation_analysis.py')
assert S['execution_freeze_sha256']==file_sha256(STUDY/'execution_freeze.json')
M=pd.read_csv(A/'means.csv');B=pd.read_csv(A/'block_metrics.csv');E=pd.read_csv(A/'paired_effects.csv')
C=pd.read_csv(A/'accuracy_curves.csv');Q=pd.read_csv(A/'practical_criteria.csv');Z=pd.read_csv(A/'zero_tolerance_targets.csv')
REC=pd.read_csv(A/'recovery.csv');COST=pd.read_csv(A/'costs.csv')
assert M.n.eq(5).all() and E.n.eq(5).all() and len(Q)==10 and len(Z)==6
LABEL={{MAIN:'FedReTrust v0.3','FLAURA-reconstruction':'FLAURA main','FLAURA-sigma-double':'FLAURA doubled scale',
 'FedCognis-aggregation':'FedCognis main','FedCognis-squared-distance':'FedCognis squared distance',
 'FedReTrust-v03-no-restoration':'v0.3 without restoration','FedReTrust-v03-no-clipping':'v0.3 without clipping',
 'FedReTrust-v03-capped-RFA':'Capped RFA control'}}
PRIMARY=['FedAvg','RFA','FLTrust','Viceroy',MAIN]
CONTROLS=[MAIN,'FedReTrust-v03-no-restoration','FedReTrust-v03-no-clipping','FedReTrust-v03-capped-RFA']
RECENT=[MAIN,'FLAURA-reconstruction','FLAURA-sigma-double','FedCognis-aggregation','FedCognis-squared-distance']
pd.set_option('display.max_rows',120);pd.set_option('display.max_columns',16)
def export(frame,name):
    frame.to_csv(P/(name+'.csv'),index=False)
def show(frame):
    display(frame.round(3))
display(pd.DataFrame([{{'runs':len(R),'paired blocks per regime':5,'audited rounds':S['audited_rounds'],
 'paired inputs':S['matched_input_pairs'],'exact first-round updates':S['exact_first_round_updates'],
 'exact pre-event trajectories':S['exact_pre_event_trajectories']}}]))
'''), md('''## Read the result against the declared targets

Accuracy differences are **v0.3 minus comparator**; higher is better. Damage differences use the same subtraction; lower is better. The practical target requires all five criteria in **each** regime, giving ten checks. These margins are design choices, not established standards. A point pass does not make an uncertain interval conclusive.

Intervals are nominal 95% paired t intervals across five blocks (df=4). They are small-sample descriptions, without simultaneous coverage or a table-wide significance claim. Clients and rounds are not additional independent observations. The older zero-tolerance target is retained separately below.
'''), code('''def margin_status(row):
    if row.direction=='>=':
        if row.ci_low>row.bound_pp:return 'Interval entirely meets margin'
        if row.ci_high<row.bound_pp:return 'Interval entirely misses margin'
    else:
        if row.ci_high<row.bound_pp:return 'Interval entirely meets margin'
        if row.ci_low>row.bound_pp:return 'Interval entirely misses margin'
    return 'Uncertain: interval touches/crosses margin'
Q['interval_assessment']=Q.apply(margin_status,axis=1)
pass_count=int(Q.point_estimate_meets_margin.sum());strict_count=int(Z.point_target_met.sum())
assert bool(S['joint_practical_point_target_met'])==(pass_count==10)
headline=f'Practical point targets: {pass_count}/10. Joint target '+('met.' if pass_count==10 else 'not met.')
display(Markdown('**'+headline+'**'))
show(Q[['regime','scenario','comparator','metric','bound_pp','direction','mean','sd','ci_low','ci_high',
        'point_estimate_meets_margin','interval_assessment']])
export(Q,'practical_criteria_with_interpretation')
print(f'Zero-tolerance comparison with the best baseline on each core measure: {strict_count}/6 point targets met.')
show(Z[['regime','scenario','candidate_mean','best_baseline','best_baseline_mean','point_target_met']])
decision={'conclusion':headline,'practical_point_passes':pass_count,'practical_checks':10,
 'joint_practical_point_target_met':pass_count==10,'zero_tolerance_point_passes':strict_count,'zero_tolerance_checks':6,
 'uncertain_practical_intervals':int(Q.interval_crosses_or_touches_margin.sum()),
 'scope':'Five paired blocks per regime on known MNIST. No simultaneous significance, unseen-dataset or established novelty claim.',
 'evidence_sha256':{n:file_sha256(STUDY/n) for n in ['execution_freeze.json','analysis_manifest.json',
   'analysis/summary.json','analysis/practical_criteria.csv','analysis/zero_tolerance_targets.csv','analysis/paired_effects.csv']}}
decision['point_passes_by_regime']={g:int(x.point_estimate_meets_margin.sum()) for g,x in Q.groupby('regime')}
decision['missed_point_targets']=Q[~Q.point_estimate_meets_margin][['regime','scenario','comparator','metric',
    'mean','ci_low','ci_high','bound_pp','interval_assessment']].to_dict('records')
for regime,passed in decision['point_passes_by_regime'].items():
    display(Markdown(f'**{regime}: {passed}/5 practical point targets met.**'))
for row in Q[~Q.point_estimate_meets_margin].itertuples():
    display(Markdown(f'Missed target: **{row.regime}, {row.scenario} versus {row.comparator}**. '
        f'Paired difference {row.mean:+.2f} pp (nominal 95% interval {row.ci_low:+.2f} to {row.ci_high:+.2f}); '
        f'required {row.direction} {row.bound_pp:+.2f} pp. {row.interval_assessment}.'))
write_json(P/'decision.json',decision)
'''), md('''## Comparable clean, honest-drift and poisoning results

These means cover rounds **5–24**. Honest noise/stripe accuracy describes the affected clients' legitimate tasks. Attack damage is each method's own matched-control accuracy minus attacked accuracy, in percentage points. Absolute attacked accuracy is also shown: a low starting accuracy must not be mistaken for strong protection. A blank entry is an untested combination.

FLTrust retains 300 server reference images; FLAURA retains 100 and its frozen calibration. FedReTrust v0.3, its controls, FedAvg, RFA, Viceroy and the FedCognis aggregation adaptation use no server reference images. FLAURA and FedCognis are documented adaptations, not reproductions of every component of the published systems.
'''), code('''spec=[('clean','overall_mean_5_24_pp','Clean accuracy'),
 ('abrupt_shot_noise','affected_honest_event_pp','Honest noise accuracy'),
 ('gradual_stripe','affected_honest_event_pp','Honest stripe accuracy'),
 ('label_flip','attack_active_accuracy_pp','Attacked accuracy'),
 ('label_flip','attack_active_signed_damage_pp','Attack damage (pp)')]
parts=[]
for scenario,metric,label in spec:
    x=M[(M.scenario==scenario)&(M.metric==metric)].copy();x['measurement']=label;parts.append(x)
head=pd.concat(parts,ignore_index=True)
for regime in ['IID','non-IID']:
    print(regime)
    table=head[(head.regime==regime)&head.panel.isin(['primary','recent_methods'])].pivot(
        index='method',columns='measurement',values='mean')
    show(table.rename(index=LABEL))
export(head,'core_means_intervals_and_controls')
print('All five values behind each main comparison:')
spec_frame=pd.DataFrame([(s,m) for s,m,_ in spec],columns=['scenario','metric'])
blocks=B.merge(spec_frame,on=['scenario','metric'])
show(blocks[blocks.panel=='primary'].pivot(index=['regime','method','scenario','metric'],columns='block',values='value'))
export(blocks,'core_block_values')
print('All means and paired contrasts are preserved in analysis/means.csv and analysis/paired_effects.csv.')
'''), md('''## Training trajectories and all anomaly scenarios

Lines are block means; they are descriptive, not additional samples. Shading marks the original attack windows. Honest-drift panels use affected clients' legitimate task accuracy; other panels use all-client legitimate accuracy. Recovery is classified on individual block trajectories, not on these mean curves.
'''), code('''import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
COLORS={'FedAvg':'#777777','RFA':'#2477aa','FLTrust':'#ba852a','Viceroy':'#9365b8',MAIN:'#147e62',
 'FedReTrust-v03-no-restoration':'#ba4d65','FedReTrust-v03-no-clipping':'#5469af','FedReTrust-v03-capped-RFA':'#b77b32',
 'FLAURA-reconstruction':'#a25280','FLAURA-sigma-double':'#dd8a35',
 'FedCognis-aggregation':'#347bab','FedCognis-squared-distance':'#7c69ae'}
def save(fig,name):
    fig.savefig(FIG/(name+'.png'),dpi=160,bbox_inches='tight')
    fig.savefig(FIG/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
    display(Image(filename=str(FIG/(name+'.png'))))
def draw(ax,frame,value='accuracy_pp',methods=PRIMARY):
    for method in methods:
        part=frame[frame.method==method]
        if part.empty:continue
        x=part.groupby('round')[value].agg(['mean','count'])
        assert x['count'].eq(5).all()
        ax.plot(x.index,x['mean'],color=COLORS[method],label=LABEL.get(method,method),
                lw=2.5 if method==MAIN else 1.6,ls='-' if method==MAIN else '--')
    ax.set_xlim(1,24);ax.set_xticks([1,5,10,15,20,24]);ax.grid(alpha=.18)
def legend(fig,ax,cols=3):
    handles,labels=ax.get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=cols,frameon=False)
panels=[('clean','Clean','all_clients'),('abrupt_shot_noise','Honest noise','affected_honest'),
 ('gradual_stripe','Honest stripes','affected_honest'),('label_flip','Persistent label flipping','all_clients'),
 ('gaussian_update','Gaussian update corruption','all_clients'),('attack_stops','Attack stops after round 10','all_clients'),
 ('recurring_attack','Recurring attack','all_clients'),('mixed_drift_and_attack','Honest drift + attack','all_clients'),
 ('honest_change_stops','Honest change stops','affected_honest'),('mixed_drift_only_control','Matched mixed-drift control','all_clients')]
from scenario_metrics import longer_scenarios
windows={name:sorted({window for event in scenario.events if event.intent=='malicious' for window in event.windows})
         for name,scenario in longer_scenarios().items()}
for regime in ['IID','non-IID']:
    fig,axes=plt.subplots(5,2,figsize=(13,16),sharex=True,sharey=True)
    for ax,(scenario,title,group) in zip(axes.flat,panels):
        draw(ax,C[(C.regime==regime)&(C.scenario==scenario)&(C.group==group)])
        ax.set_title(title);ax.set_ylim(0,100);ax.set_ylabel('Legitimate accuracy (%)')
        for start,end in windows.get(scenario,[]):ax.axvspan(start,end,color='#b94747',alpha=.09)
        if scenario!='clean':ax.axvline(5,color='#999999',ls=':',lw=.7)
    for ax in axes[-1]:ax.set_xlabel('Round')
    fig.suptitle(regime+' · five-block means',y=.997)
    legend(fig,axes[0,0]);fig.tight_layout(rect=(0,.055,1,.98));save(fig,'all_scenarios_'+regime)
fig,axes=plt.subplots(2,3,figsize=(14,8),sharex=True,sharey=True)
for i,regime in enumerate(['IID','non-IID']):
    for k,(scenario,title,group) in enumerate([panels[0],panels[2],panels[3]]):
        ax=axes[i,k];draw(ax,C[(C.regime==regime)&(C.scenario==scenario)&(C.group==group)],methods=RECENT)
        ax.set_title(regime+' · '+title);ax.set_ylim(0,100)
        for start,end in windows.get(scenario,[]):ax.axvspan(start,end,color='#b94747',alpha=.09)
        if scenario!='clean':ax.axvline(5,color='#999999',ls=':',lw=.7)
        if k==0:ax.set_ylabel('Legitimate accuracy (%)')
        if i==1:ax.set_xlabel('Round')
legend(fig,axes[0,0]);fig.tight_layout(rect=(0,.09,1,1));save(fig,'recent_methods')
'''), md('''## What do clipping, directional weighting and restoration contribute?

The controls each train their own full trajectory. Removing restoration sets the current evidence directly as the weight. Removing clipping disables the cap in both the anchor and final update. Capped RFA returns the robust capped aggregate directly. The contrast between the memory-free directional rule and capped RFA isolates final directional weighting; full v0.3 versus capped RFA combines weighting and restoration.

These are mechanism checks, not replacement candidates selected after seeing results. The recurring-attack comparison uses the **same second attack window (rounds 15–18)** across methods. A difference between the first and second episodes within one method would confound memory with training time.
'''), code('''component_specs=[('clean','overall_mean_5_24_pp'),('gradual_stripe','affected_honest_event_pp'),
 ('label_flip','attack_active_signed_damage_pp'),('attack_stops','attack_tail_damage_21_24_pp'),
 ('recurring_attack','second_attack_signed_damage_pp'),('honest_change_stops','affected_honest_return_11_24_pp')]
component=E.merge(pd.DataFrame(component_specs,columns=['scenario','metric']),on=['scenario','metric'])
component=component[component.comparator.isin(CONTROLS[1:])]
show(component[['target','comparator','regime','scenario','metric','mean','sd','ci_low','ci_high']])
export(component,'component_contrasts')
def effect(target,other,regime,scenario,metric):
    rows=E[(E.target==target)&(E.comparator==other)&(E.regime==regime)&(E.scenario==scenario)&(E.metric==metric)]
    assert len(rows)==1
    return rows.iloc[0]
def formatted_effect(row):
    return f'{row["mean"]:+.2f} pp (nominal 95% interval {row.ci_low:+.2f} to {row.ci_high:+.2f})'
findings=[]
for regime in ['IID','non-IID']:
    memory=effect(MAIN,CONTROLS[1],regime,'label_flip','attack_active_signed_damage_pp')
    clipping=effect(MAIN,CONTROLS[2],regime,'label_flip','attack_active_signed_damage_pp')
    direction=effect(CONTROLS[1],CONTROLS[3],regime,'gradual_stripe','affected_honest_event_pp')
    direction_damage=effect(CONTROLS[1],CONTROLS[3],regime,'label_flip','attack_active_signed_damage_pp')
    findings.append(f'{regime}: full v0.3 minus the no-restoration control changes persistent label-flip damage by '
        +formatted_effect(memory)+'. Lower damage favours restoration.')
    findings.append(f'{regime}: full v0.3 minus the no-clipping control changes persistent label-flip damage by '
        +formatted_effect(clipping)+'. Lower damage favours clipping.')
    findings.append(f'{regime}: the memory-free directional rule minus capped RFA changes honest-stripe accuracy by '
        +formatted_effect(direction)+' and persistent label-flip damage by '+formatted_effect(direction_damage)
        +'. Higher accuracy and lower damage are desirable; the two outcomes expose a trade-off.')
for finding in findings:display(Markdown(finding))
decision['component_findings']=findings
write_json(P/'decision.json',decision)
fig,axes=plt.subplots(2,3,figsize=(14,8),sharex=True,sharey=True)
for i,regime in enumerate(['IID','non-IID']):
    for k,(scenario,title,group) in enumerate([panels[2],panels[3],panels[6]]):
        ax=axes[i,k];draw(ax,C[(C.regime==regime)&(C.scenario==scenario)&(C.group==group)],methods=CONTROLS)
        ax.set_title(regime+' · '+title);ax.set_ylim(0,100)
        for start,end in windows.get(scenario,[]):ax.axvspan(start,end,color='#b94747',alpha=.09)
        if k==0:ax.set_ylabel('Legitimate accuracy (%)')
        if i==1:ax.set_xlabel('Round')
legend(fig,axes[0,0],2);fig.tight_layout(rect=(0,.10,1,1));save(fig,'component_trajectories')
'''), md('''## Recovery after attacks stop

Recovery means accuracy at least as high as the method's matched control minus **one percentage point**, for **three consecutive post-attack rounds**. Streak start and confirmation are different. A method can already be within tolerance when the attack stops; this is recorded rather than described as repair from a large failure. Later relapses remain visible.

Unconfirmed recovery is right-censored at the next attack or round 24. It is never assigned a fabricated recovery time, and the report does not average recovery times only among successes.
'''), code('''recovery_view=REC.copy()
recovery_view['confirmed']=recovery_view.confirmation_round.notna()
recovery_view['has_later_relapse']=recovery_view.relapse_rounds_after_confirmation.map(lambda v:len(ast.literal_eval(v))>0)
summary=recovery_view.groupby(['method','regime','scenario','group','episode'],as_index=False).agg(
    blocks=('block','count'),confirmed=('confirmed','sum'),censored=('right_censored','sum'),
    already_within_at_attack_end=('criterion_met_at_attack_end','sum'),later_relapse=('has_later_relapse','sum'))
assert summary.blocks.eq(5).all()
show(summary[summary.group=='all_clients'])
export(summary,'recovery_counts')
export(recovery_view,'recovery_all_blocks_and_episodes')
print('Every block, streak start, confirmation round, censoring reason and relapse is retained in recovery_all_blocks_and_episodes.csv.')
show(recovery_view[(recovery_view.method==MAIN)&(recovery_view.group=='all_clients')][[
    'regime','block','scenario','episode','streak_start_round','confirmation_round','right_censored',
    'criterion_met_at_attack_end','relapse_rounds_after_confirmation']])
'''), md('''## Training loss, evaluation loss, final accuracy and class detail

Training loss is the online local cross-entropy seen during SGD; attacked clients can be optimizing corrupted labels. It is not a clean held-out accuracy measurement. Evaluation loss below is on the preserved balanced development views and does not feed back into any defence. Per-class scores are secondary observations from those same views.

The final round and last four rounds complement the predeclared rounds 5–24 mean; they do not replace it after observing an outcome.
'''), code('''TRAIN=pd.read_csv(A/'local_training.csv');TASK=pd.read_csv(A/'raw_task_metrics.csv')
assert len(TRAIN)==192000 and len(TASK)==60000
train_round=TRAIN.groupby(['panel','method','regime','block','scenario','round'],as_index=False).online_train_loss.mean()
train_summary=train_round[train_round['round']>=5].groupby(['panel','method','regime','block','scenario'],as_index=False).online_train_loss.mean()
eval_summary=TASK[TASK['round']>=5].groupby(['panel','method','regime','block','scenario','task'],as_index=False).agg(
    mean_loss=('loss','mean'),balanced_view_accuracy=('accuracy','mean'))
export(train_summary,'training_loss_by_block');export(eval_summary,'evaluation_loss_by_block')
show(train_summary[train_summary.scenario=='clean'].groupby(['regime','method']).online_train_loss.agg(['mean','std']))
show(eval_summary[(eval_summary.scenario=='clean')&(eval_summary.task=='clean')].groupby(['regime','method']).mean_loss.agg(['mean','std']))
fig,axes=plt.subplots(2,2,figsize=(12,8),sharex=True)
for i,regime in enumerate(['IID','non-IID']):
    draw(axes[i,0],train_round[(train_round.regime==regime)&(train_round.scenario=='clean')],value='online_train_loss')
    draw(axes[i,1],TASK[(TASK.regime==regime)&(TASK.scenario=='clean')&(TASK.task=='clean')&(TASK['round']>0)],value='loss')
    axes[i,0].set_title(regime+' · clean local training loss');axes[i,1].set_title(regime+' · clean evaluation loss')
    for ax in axes[i]:ax.set_ylabel('Cross-entropy');ax.set_ylim(bottom=0)
for ax in axes[-1]:ax.set_xlabel('Round')
legend(fig,axes[0,0]);fig.tight_layout(rect=(0,.10,1,1));save(fig,'clean_losses')
tail=M[M.metric.isin(['overall_mean_5_24_pp','overall_tail_21_24_pp','overall_round24_pp'])]
export(tail,'mean_tail_and_final_all_scenarios')
show(tail[tail.scenario=='clean'].pivot(index=['regime','method'],columns='metric',values='mean').rename(index=LABEL))
class_parts=[]
for digit in range(10):
    x=TASK[['panel','method','regime','block','scenario','round','task']].copy()
    count=TASK[f'class_examples_{digit}'];x['digit']=digit
    x['accuracy_pp']=100*TASK[f'class_correct_{digit}']/count.replace(0,np.nan)
    x['cross_entropy']=TASK[f'class_loss_sum_{digit}']/count.replace(0,np.nan)
    class_parts.append(x)
classes=pd.concat(class_parts,ignore_index=True)
class_summary=classes[classes['round']>=5].groupby(['panel','method','regime','block','scenario','task','digit'],as_index=False)[['accuracy_pp','cross_entropy']].mean()
export(class_summary,'per_class_block_means')
print('Per-class and all-scenario mean/tail/final measurements are exported without choosing a favourable endpoint.')
'''), md('''## Applied influence and clipping diagnostics

Applied contribution norms and coefficients measure what reached the aggregate. Their matched-control ratios are descriptive; they do not prove whether an honest update would improve the model. Clipping and restoration activation are reported alongside task outcomes, without treating every weight reduction as a mistaken rejection.
'''), code('''CLIENT=pd.read_csv(A/'client_diagnostics.csv');ROUND=pd.read_csv(A/'round_diagnostics.csv')
CON=pd.read_csv(A/'contributions.csv')
client=CLIENT[CLIENT.method.isin(CONTROLS)].copy()
client['clipping_active']=client.clip_factor.lt(1-1e-12)
client['restoration_limited']=client.restored_weight.lt(client.evidence_weight-1e-12)
client_summary=client.groupby(['method','regime','block','scenario'],as_index=False).agg(
    clipped_submission_fraction=('clipping_active','mean'),restoration_limited_fraction=('restoration_limited','mean'),
    mean_clip_factor=('clip_factor','mean'))
export(client_summary,'component_activation_by_block')
show(client_summary.groupby(['method','regime','scenario'])[['clipped_submission_fraction','restoration_limited_fraction']].mean())
influence=CON.groupby(['method','regime','block','scenario','group'],as_index=False).agg(
    mean_applied_norm=('applied_norm','mean'),mean_matched_norm_ratio=('matched_norm_ratio','mean'),
    mean_coefficient=('coefficient','mean'))
export(influence,'influence_by_block_and_client_group')
fallback=ROUND[ROUND.method.isin(CONTROLS)].copy()
fallback['fallback_used']=fallback.bounded_rfa_fallback.map(lambda v:v is True or str(v).lower()=='true')
fallback=fallback.groupby(['method','regime','block','scenario'],as_index=False).fallback_used.mean()
export(fallback,'fallback_fraction_by_block')
print('Raw client/round diagnostics, contributions and contribution-restoration episodes remain in analysis/.')
'''), md('''## Resource cost and evidence boundaries

Wall times are from two concurrent fresh processes on this machine. Peak RSS includes Python, data and framework memory; it is not isolated algorithm allocation. Aggregation timing is recorded separately from measurement overhead. Server-reference training/forward work is counted explicitly. This is not a network, cryptography or production FLaas benchmark.

All declared jobs are retained. There is no post-result replacement of the candidate, seed resampling, horizon extension or new calibration. The fixed Gaussian update corruption and nonadaptive attacks support only the tested threat settings. No universal attack protection or algorithmic originality follows from these results.
'''), code('''cols=['worker_wall_seconds','process_lifetime_peak_rss_mib','median_defence_seconds',
      'median_recorder_seconds','reference_forward_examples','reference_backward_examples','reference_optimizer_steps','artifact_bytes']
cost_summary=COST.groupby(['method','regime'],as_index=False)[cols].mean()
show(cost_summary);export(cost_summary,'resource_cost_means')
display(pd.DataFrame([S['workload']]))
print('Maximum aggregation reconstruction error:',S['maximum_reconstruction_error'])
print('Fresh runs:',len(R),'| All first-round updates and pre-event trajectories passed the frozen pairing audit.')
print('Full exports:',P)
display(Markdown('**'+headline+'** '+decision['scope']))
''')]
    return nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{
        'display_name':'Python 3','language':'python','name':'python3'}})


def build(study=STUDY, output=NOTEBOOK):
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/fedretrust_validation_matplotlib')
    from aggregate_validation_notebook import _execute
    from experiment_core import file_sha256, write_json
    study, output = Path(study).resolve(), Path(output).resolve()
    status = json.loads((study / 'status.json').read_text())
    if status.get('status') != 'complete' or status.get('completed_new_runs') != 800:
        raise ValueError('Wait for all 800 jobs and frozen analysis before reporting outcomes.')
    notebook = make_notebook(study)
    _execute(notebook, output)
    out = study / 'presentation'
    artifacts = {str(p.relative_to(ROOT)): file_sha256(p)
                 for p in out.rglob('*') if p.is_file() and p.name != 'presentation_manifest.json'}
    artifacts[str(output.relative_to(ROOT))] = file_sha256(output)
    write_json(out / 'presentation_manifest.json', {
        'status': 'rendered_pending_visual_review', 'builder_sha256': file_sha256(__file__),
        'analysis_manifest_sha256': file_sha256(study / 'analysis_manifest.json'),
        'artifacts': artifacts, 'scope': 'Saved-artifact presentation only; no training or model evaluation.'})
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--study', type=Path, default=STUDY)
    parser.add_argument('--output', type=Path, default=NOTEBOOK)
    args = parser.parse_args()
    print(build(args.study, args.output))
