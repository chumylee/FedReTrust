"""Artifact-only v0.4 report. No local training, model inference or selection."""
from pathlib import Path
import json
import nbformat as nbf
from aggregate_validation_notebook import _execute
ROOT=Path(__file__).resolve().parents[1]
STUDY=ROOT/'outputs/reference_free/20261030_v04'
NOTEBOOK=ROOT/'code/25_blockwise_fedretrust.ipynb'


def make_notebook(study=STUDY):
    md,code=nbf.v4.new_markdown_cell,nbf.v4.new_code_cell
    cells=[md(r'''# 25 — FedReTrust v0.4: protected classifier and bounded feature adaptation

This experiment tests whether different parts of a client update should receive different treatment. **The classifier uses robust aggregation. Feature layers may retain more adaptation, with bounded residual corrections and gradual restoration.** Client training stays unchanged. The method uses zero trusted server images, zero client validation reports and no access to evaluation results or simulator intent.

This is a new experimental FedReTrust implementation. It combines existing robust-aggregation and clipping ideas; the version number is not a claim of novelty, a robustness theorem or improved performance. The previous layer-score diagnostic did not establish reliable attack identification. Here the change is the aggregation action, not a claim that the earlier diagnostic succeeded.'''),code(f'''from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
from IPython.display import display,Image,Markdown
CODE=Path({str(ROOT/'code')!r});sys.path.insert(0,str(CODE))
from blockwise_study import verify,records,MAIN,CONTROL,BASELINES
from experiment_core import file_sha256
STUDY=Path({str(Path(study).resolve())!r});A=STUDY/'analysis/pilot'
STATUS=json.loads((STUDY/'status.json').read_text());assert STATUS['status']=='complete'
P=verify(STUDY);new=records(STUDY)
S=json.loads((A/'summary.json').read_text());ACCESS=json.loads((STUDY/'access_audit_pilot.json').read_text())
assert S['analysis_source_sha256']==file_sha256(CODE/'blockwise_analysis.py')
assert S['scientific_trajectories']==252 and S['context_trajectories']==72 and S['new_runs']==50
assert ACCESS['v04_trusted_server_images']==0 and ACCESS['official_test_used'] is False
M=pd.read_csv(A/'means.csv');SEED=pd.read_csv(A/'seed_metrics.csv');C=pd.read_csv(A/'accuracy_curves.csv')
T=pd.read_csv(A/'target_checks.csv');E=pd.read_csv(A/'paired_effects.csv');B=pd.read_csv(A/'block_contributions.csv')
REC=pd.read_csv(A/'recovery.csv');TRAIN=pd.read_csv(A/'local_training.csv');RAW=pd.read_csv(A/'raw_task_metrics.csv')
COST=pd.read_csv(A/'costs.csv')
LABEL={{'FedReTrust':'v0.1','FedReTrust-RF-recovery':'v0.2','FedReTrust-Direction-loose':'v0.3',MAIN:'v0.4',CONTROL:'v0.4 block-RFA control'}}
REGIMES=['IID','non-IID'];FOCUS=['FedAvg','RFA','FedReTrust-RF-recovery','FedReTrust-Direction-loose',MAIN,CONTROL]
COLORS={{'FedAvg':'#777777','RFA':'#2778a4','FedReTrust-RF-recovery':'#b58131','FedReTrust-Direction-loose':'#8c5ca4',MAIN:'#198768',CONTROL:'#b94a50'}}
pd.set_option('display.max_rows',150);pd.set_option('display.max_columns',18)
display(Markdown('**Pilot continuation gate: '+('passed' if S['continuation']['pass'] else 'not met')+' ('+str(S['continuation']['targets_met'])+'/10 checks).**'))
print('50 new pilot runs;202 verified earlier runs;72 additional context trajectories.261 preflight tests passed.')
print('Only a passed pilot gate triggers the predeclared five-new-seed validation. Pilot differences alone are not statistical significance.')
if (STUDY/'decision.json').exists():
    D=json.loads((STUDY/'decision.json').read_text())
    for name,digest in D.get('evidence_sha256',{{}}).items():assert file_sha256(STUDY/name)==digest
    display(pd.Series({{k:v for k,v in D.items() if isinstance(v,(str,int,float,bool)) or v is None}},name='Saved decision').to_frame())
'''),md(r'''## Fixed rule and component control

The model has four blocks: the two convolutional modules, the 128-unit representation module, and the ten-class classifier. Each module's weights and bias are treated jointly. For block updates \(d_i\), sample shares \(a_i\), and current broadcast \(w\):

1. Cap each block update's norm at **6 × the median block norm**, giving \(b_i=c_i d_i\).
2. Compute the original three-iteration RFA estimate \(r=\sum_i u_i b_i\) separately in that block.
3. In feature blocks, set residual radius \(\rho=6\,\mathrm{median}_i\|b_i-r\|\), and current allowance \(e_i=\min(1,\rho/\|b_i-r\|)\). Zero residual has allowance one.
4. Set \(q_i=\min(e_i,q_i^{previous}+0.5)\), starting new client/block pairs at one. Restriction is immediate; absence does not restore allowance.
5. Apply \(g=r+\sum_i a_iq_i(b_i-r)\). The classifier uses \(q_i=0\), so it remains the bounded RFA estimate.

The **block-RFA control** sets every feature correction to zero too. It shares the same block partition, classifier, norm caps and RFA solver. It isolates the effect of allowing feature residuals and cannot replace the main candidate after seeing outcomes.

The effective raw-update coefficient in a block is
\[
\beta_i=c_i\left[a_iq_i+\left(1-\sum_j a_jq_j\right)u_i\right].
\]
The recorder verifies that these coefficients reconstruct the actual model update, including floating-point casting. Since coefficients differ by block, **there is no single global client coefficient**. Blank scalar entries in the original training CSV are intentionally undefined. Global contribution norms and Gram matrices are computed from the actual concatenated block contributions.

The radius is one fixed permissive setting, selected before these outcomes to test feature retention while keeping a protected classifier. The rule operates from round 1; it assumes no known-clean warm-up, no known attacker identities and no fixed attacker fraction.'''),md('''## What was compared

The pilot uses 24 rounds, ten clients, 512 images per client, one epoch, batch size 64, learning rate 0.01 and momentum 0.9. Standard development evaluation uses 1,000 examples per task and never feeds back into the defence. The 300 server-reference examples remain unused by v0.4. FLTrust and v0.1 retain their original reference-data privilege as comparators.

The main v0.4 runs all ten original scenarios with seed 20261014, and repeats clean, honest noise, honest stripes, persistent label flips and stopped attacks with seed 20261015. The control repeats clean/stripe/label-flip in both regimes and seeds. Eight missing second-seed noise/recovery runs complete the comparisons for v0.2/v0.3. This gives 50 new runs and 252 primary comparison trajectories, plus 72 saved NNM/FedProx context trajectories.

**Primary summaries average rounds 5–24.** Honest-drift accuracy measures affected clients, with unchanged clients reported separately. Attack damage is the signed difference from the same method's matched no-attack trajectory, using legitimate labels. Absolute attacked accuracy is retained to prevent a reduced clean baseline from looking like improved protection. Tail means use rounds 21–24; final results use round 24. Warm-up rounds 1–4 remain in the raw data, not the primary window.

The strict continuation gate requires v0.4 to match the best of the seven earlier comparators on clean accuracy, affected stripe accuracy, persistent-attack damage and absolute attacked accuracy, in **both** regimes. It additionally requires at least a 1-point stripe gain over RFA in each regime: ten checks total. Two-seed means can support continuation but not a statistical-superiority claim.'''),code('''primary=[('clean','overall_mean_5_24_pp','Clean accuracy'),('abrupt_shot_noise','affected_honest_event_pp','Honest noise accuracy'),
 ('gradual_stripe','affected_honest_event_pp','Honest stripe accuracy'),('label_flip','attack_active_accuracy_pp','Absolute attacked accuracy'),
 ('label_flip','attack_active_signed_damage_pp','Attack damage (lower better)')]
rows=[]
for scenario,metric,label in primary:
    f=M[(M.scenario==scenario)&(M.metric==metric)].copy();f['measurement']=label;rows.append(f)
primary_table=pd.concat(rows)
core=primary_table[primary_table.panel=='pilot'].pivot(index=['regime','method'],columns='measurement',values='mean')
display(core.round(3));core.to_csv(A/'notebook_primary_comparison.csv')
print('Primary rows have two seeds. The block-RFA control has no noise runs; its blank noise cell is missing coverage, not zero.')
display(T.round(4))
print('Each seed difference versus the previous selected FedReTrust versions:')
effects=E[E.baseline.isin(['FedReTrust','FedReTrust-RF-recovery','FedReTrust-Direction-loose'])].merge(
    pd.DataFrame([(s,m) for s,m,_ in primary],columns=['scenario','metric']),on=['scenario','metric'])
display(effects.pivot(index=['baseline','regime','scenario','metric'],columns='seed',values='difference_pp').round(4))
print('Context comparisons, including both attacker policies from the proximal pilot:')
display(primary_table[primary_table.panel=='context'].pivot(index=['regime','method','attack_policy'],columns='measurement',values='mean').round(3))
print('NNM has no stopped-attack trajectory in this context panel. Context methods are not used to select a new candidate.')
'''),md('''## Learning trajectories and protection

Curves below use only the two-seed scenarios and show descriptive means. Honest noise/stripe panels represent affected clients. The protection panels show both absolute performance and damage relative to each method's own clean run. A mean curve is not a per-seed recovery test.'''),code('''import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
FIG=A/'figures';FIG.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':130})
def save_figure(fig,name):
    fig.savefig(FIG/(name+'.png'),dpi=180,bbox_inches='tight');fig.savefig(FIG/(name+'.pdf'),bbox_inches='tight')
    plt.close(fig);display(Image(filename=str(FIG/(name+'.png'))))
def draw(ax,data):
    for method in FOCUS:
        sub=data[data.method==method]
        if sub.empty:continue
        grouped=sub.groupby('round').accuracy_pp.agg(['mean','count']);assert grouped['count'].eq(2).all()
        ax.plot(grouped.index,grouped['mean'],label=LABEL.get(method,method),color=COLORS[method],
                ls='--' if method==CONTROL else '-',lw=2.2 if method==MAIN else 1.5)
    ax.set_xlim(1,24);ax.set_xticks([1,5,10,15,20,24]);ax.grid(alpha=.15)
pilot=C[C.panel=='pilot']
fig,axes=plt.subplots(2,3,figsize=(13.5,7.4),sharex=True)
for i,g in enumerate(REGIMES):
    for k,(scenario,label) in enumerate([('clean','Clean'),('abrupt_shot_noise','Honest noise'),('gradual_stripe','Honest stripes')]):
        ax=axes[i,k];group='all_clients' if scenario=='clean' else 'affected_honest'
        draw(ax,pilot[(pilot.regime==g)&(pilot.scenario==scenario)&(pilot.group==group)])
        ax.set_title(g+' · '+label);ax.set_ylim(0,100)
        if scenario!='clean':ax.axvline(5,color='#555555',ls=':',lw=1)
        if k==0:ax.set_ylabel('Legitimate accuracy (%)')
        if i==1:ax.set_xlabel('Training round')
handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=3,frameon=False)
fig.suptitle('FedReTrust v0.4 — clean learning and honest adaptation',fontsize=15)
fig.tight_layout(rect=(0,.075,1,.95));save_figure(fig,'v04_honest_trajectories')
attack=pilot[(pilot.group=='all_clients')&pilot.scenario.isin(['label_flip','attack_stops'])].copy()
clean=pilot[(pilot.group=='all_clients')&(pilot.scenario=='clean')][['method','regime','seed','round','accuracy_pp']].rename(columns={'accuracy_pp':'clean_accuracy_pp'})
paired=attack.merge(clean,on=['method','regime','seed','round'],validate='many_to_one');paired['damage_pp']=paired.clean_accuracy_pp-paired.accuracy_pp
paired.to_csv(A/'notebook_paired_damage_curves.csv',index=False)
for scenario,name,title in [('label_flip','v04_persistent_attack','Persistent label flips'),('attack_stops','v04_stopped_attack','Attacks stop after round 10')]:
    fig,axes=plt.subplots(2,2,figsize=(12,7.6),sharex=True)
    for i,g in enumerate(REGIMES):
        part=paired[(paired.regime==g)&(paired.scenario==scenario)]
        draw(axes[i,0],part);draw(axes[i,1],part.drop(columns='accuracy_pp').rename(columns={'damage_pp':'accuracy_pp'}))
        axes[i,0].set_ylabel('Legitimate accuracy (%)');axes[i,0].set_ylim(0,100)
        axes[i,1].set_ylabel('Own-clean minus attacked (pp)');axes[i,1].axhline(0,color='#555555',lw=.7)
        if scenario=='attack_stops':axes[i,1].axhline(1,color='#555555',ls=':',lw=1)
        for k,label in enumerate(['Absolute accuracy','Clean-relative damage']):
            axes[i,k].set_title(g+' · '+label);axes[i,k].axvspan(5,24 if scenario=='label_flip' else 10,color='#b94a50',alpha=.06)
            if i==1:axes[i,k].set_xlabel('Training round')
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=3,frameon=False)
    fig.suptitle(title+' — two-seed means',fontsize=15)
    fig.tight_layout(rect=(0,.075,1,.95));save_figure(fig,name)
'''),md('''## All scenarios, final/tail results and recovery

All ten scenarios are compared below on **seed 20261014 only**, so extra scenarios are not presented as two-seed evidence. These include Gaussian updates, recurring attacks, simultaneous honest drift plus poisoning, honest change that stops, and the mixed scenario's no-attack control.

Recovery requires accuracy within 1 percentage point of the matched no-attack trajectory for three consecutive post-event rounds. A streak start and its confirmation differ by two rounds. Cases that do not confirm within the available window remain right-censored. Whether the criterion was already met at the end of the attack is retained. Model recovery does not by itself prove restoration of trust or correct classification of client intent.'''),code('''wide_metrics=['overall_mean_5_24_pp','overall_tail_21_24_pp','overall_round24_pp','affected_honest_event_pp',
              'unchanged_honest_mean_5_24_pp','attack_active_signed_damage_pp','attack_active_accuracy_pp']
panel=SEED[(SEED.panel=='pilot')&(SEED.seed==20261014)&SEED.metric.isin(wide_metrics)]
all_scenarios=panel.pivot(index=['regime','scenario','method'],columns='metric',values='value')
all_scenarios.to_csv(A/'notebook_all_scenarios_seed14.csv')
display(all_scenarios.loc[(slice(None),slice(None),[MAIN,'FedReTrust','FedReTrust-RF-recovery','FedReTrust-Direction-loose']),:].round(3))
print('All seven earlier methods plus the declared control are retained in notebook_all_scenarios_seed14.csv.')
two_seed=M[(M.panel=='pilot')&M.metric.isin(wide_metrics)&(M['count']==2)]
display(two_seed[two_seed.method.isin([MAIN,CONTROL])].pivot(index=['regime','scenario','method'],columns='metric',values='mean').round(3))
cols=['method','regime','scenario','seed','group','episode','criterion_met_at_attack_end','streak_start_round','confirmation_round','right_censored','censor_reason']
display(REC[(REC.panel=='pilot')&REC.method.isin([MAIN,'FedReTrust','FedReTrust-RF-recovery','FedReTrust-Direction-loose'])][cols].sort_values(['scenario','regime','method','seed','group']))
print('Additional model episodes, contribution-reduction and restoration records remain in the complete CSV outputs.')
'''),md('''## Did the feature residuals change client influence?

These plots show the sum of actual raw-update coefficients for the three affected clients in each block. It is a measure of applied update weighting, not a probability of honesty. The unclipped equal-count FedAvg reference would assign these three clients coefficient mass 0.3 in every block. Norm caps can reduce total coefficient mass below one.

Global norms combine the actual per-block contributions. Averaging block coefficients into a fictitious whole-model coefficient would misdescribe this algorithm. Restoration is observable in the saved residual allowance history; all inference of client roles occurs only in offline analysis.'''),code('''block_order=['features.0','features.3','classifier.1','classifier.3']
block_labels=['Convolution 1','Convolution 2','Representation','Classifier']
fig,axes=plt.subplots(2,4,figsize=(15,7.2),sharex=True,sharey=True)
for i,g in enumerate(REGIMES):
    for k,block in enumerate(block_order):
        ax=axes[i,k]
        for scenario,color,description in [('gradual_stripe','#198768','Honest stripes'),('label_flip','#b94a50','Label poisoning')]:
            for method,style in [(MAIN,'-'),(CONTROL,'--')]:
                part=B[(B.regime==g)&(B.block==block)&(B.scenario==scenario)&(B.method==method)&B.client.isin([0,1,2])]
                per_seed=part.groupby(['seed','round']).effective_coefficient.sum();line=per_seed.groupby('round').mean()
                ax.plot(line.index,100*line.values,color=color,ls=style,lw=1.8,label=description+(' · control' if method==CONTROL else ' · v0.4'))
        ax.axhline(30,color='#777777',ls=':',lw=1);ax.axvline(5,color='#777777',ls=':',lw=.7)
        ax.set_title(g+' · '+block_labels[k],fontsize=10);ax.set_xticks([1,5,10,15,20,24]);ax.grid(alpha=.15)
        if k==0:ax.set_ylabel('Raw coefficient mass (%)')
        if i==1:ax.set_xlabel('Training round')
handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=4,frameon=False,fontsize=9)
fig.suptitle('Block contribution audit — three affected clients, two-seed means',fontsize=14)
fig.tight_layout(rect=(0,.065,1,.95));save_figure(fig,'v04_block_contributions')
summary=B[B['round']>=5].groupby(['method','regime','scenario','block','role']).agg(
    mean_coefficient=('effective_coefficient','mean'),mean_allowance=('residual_allowance','mean'),
    mean_clip_factor=('norm_clip_factor','mean'),mean_applied_norm=('applied_contribution_l2_norm','mean')).reset_index()
summary.to_csv(A/'notebook_block_summary.csv',index=False)
display(summary[(summary.method==MAIN)&summary.scenario.isin(['gradual_stripe','label_flip'])].round(5))
'''),md('''## Training losses, raw tasks and resource use

Local online cross-entropy is measured on the minibatches used for SGD; poisoned clients therefore use corrupted training labels. Held-out task accuracy uses legitimate labels. These quantities have different meanings. Every raw evaluation task remains available even when it is not the active honest domain.

Runtime includes concurrent workers and reused comparisons collected at different times. Treat it as resource reporting rather than an isolated speed benchmark. The new method adds aggregation work but no additional client optimization or inference. The stage-specific access audit records all new work.'''),code('''TRAIN['window']=np.where(TRAIN['round']<=4,'Warm-up 1–4','Rounds 5–24')
train_summary=TRAIN.groupby(['panel','method','regime','scenario','attack_policy','window']).agg(
    mean_online_training_CE=('online_train_loss','mean'),client_updates=('client','size'),optimizer_steps=('optimizer_steps','sum')).reset_index()
train_summary.to_csv(A/'notebook_training_summary.csv',index=False)
display(train_summary[(train_summary.panel=='pilot')&(train_summary.scenario=='clean')].round(5))
raw=[]
for label,mask in [('Mean 5–24',RAW['round']>=5),('Tail 21–24',RAW['round']>=21),('Final 24',RAW['round']==24)]:
    f=RAW[mask].groupby(['panel','method','regime','scenario','attack_policy','task'])[['accuracy','loss']].mean().reset_index()
    f['window']=label;f['accuracy_percent']=100*f.accuracy;raw.append(f.drop(columns='accuracy'))
raw=pd.concat(raw);raw.to_csv(A/'notebook_raw_task_summary.csv',index=False)
display(raw[(raw.panel=='pilot')&(raw.method==MAIN)&(raw.window=='Mean 5–24')].round(4))
print('All methods, tasks, windows and scenarios are in notebook_raw_task_summary.csv; original class-wise records remain in raw_task_metrics.csv.')
display(COST.groupby(['panel','method'])[['worker_wall_seconds','median_training_seconds','median_defence_seconds','median_recorder_seconds','median_evaluation_seconds','process_lifetime_peak_rss_mib']].mean().round(3))
display(pd.Series({k:v for k,v in ACCESS.items() if isinstance(v,(str,int,float,bool))},name='Pilot access audit').to_frame())
assert ACCESS['new_client_updates']==12000 and ACCESS['train_forward_examples']==6144000
assert ACCESS['development_task_forward_examples']==3750000 and ACCESS['checkpoint_replay_forward_examples']==300000
assert len([r for r in new if r['job']['v04_stage']=='pilot'])==50
if STATUS['validation_new_runs']:
    V=STUDY/'analysis/fresh_seed_validation';VS=json.loads((V/'summary.json').read_text())
    stats=pd.read_csv(V/'paired_statistics.csv');assert len(stats)==18
    display(Markdown('**Five-new-seed validation completed.** Each row below uses paired seed differences, not client-round observations.'))
    display(stats.round(6))
    display(pd.read_csv(V/'means.csv').pivot(index=['regime','method','scenario'],columns='metric',values='mean').round(4))
else:
    display(Markdown('**The pilot gate did not pass, so fresh-seed validation was not launched. No statistical-superiority claim is supported by this pilot.**'))
'''),md('''## Reproduction, sources and limits

Rebuild this report from saved artifacts with `Experiment/.venv/bin/python Experiment/code/blockwise_notebook.py`. It does not train or evaluate any model. The frozen protocol, source snapshots, per-run artifacts, access audit and decision are in `outputs/reference_free/20261030_v04/`; complete primary and context tables are in `analysis/pilot/`. Figures have PNG and PDF exports.

Relevant prior work includes [RFA](https://arxiv.org/abs/1912.13445), [centered clipping and worker momentum](https://proceedings.mlr.press/v139/karimireddy21a.html), [FedCC's layer-dependent aggregation](https://arxiv.org/abs/2212.01976), and output-layer attack observations from [FL-Defender](https://arxiv.org/abs/2207.00872) and [LFighter](https://arxiv.org/abs/2207.01982). This implementation does not reproduce their complete algorithms or inherit their guarantees. A protected classifier does not make poisoned feature updates harmless; the attack runs test that concern empirically.

The same partitions and development dataset have informed earlier designs. Two repeated seeds are exploratory evidence. The conditional five-new-seed stage, if triggered, measures seed replication on reused data, not unseen-dataset confirmation. Its primary comparisons use paired t intervals and Holm-adjusted one-sided tests across 18 previous-version contrasts; practical improvement additionally requires at least one percentage point. Those tests assume independent seed outcomes and the paired-t model. Comparisons to other baselines remain descriptive.

Failure of one fixed rule does not prove reference-free robust learning impossible. Success would still need broader datasets, partitions, adaptive attackers and a separate novelty argument. Current simulator client-data boundaries are not a deployed privacy or secure-aggregation system.''')]
    cells[-1:-1]=[md('''## Other earlier design variants — historical appendix

The main comparison uses the previously selected v0.2/v0.3 settings. For completeness, this appendix also shows all three unselected design variants: v0.2 static, v0.3 strict and v0.3 normalized-direction. Each has only one saved seed for the three core scenarios. These are artifact-only historical comparisons, outside the continuation gate; no extra fitting or training is performed.'''),code('''from research_analysis import extract_run
from directional_study import checked_result as check_historical
historical=[]
old_methods=['FedReTrust-RF-static','FedReTrust-Direction-strict','FedReTrust-Unit-direction']
for location in ['reference_free/20261030_v02','reference_free/20261030_v03']:
    for path in (CODE.parent/'outputs'/location/'jobs').glob('*/result.json'):
        r=json.loads(path.read_text());j=r['job']
        if j['method'] in old_methods and j['seed']==20261014 and j['scenario'] in ['clean','gradual_stripe','label_flip']:
            check_historical(r);historical.append(r)
assert len(historical)==18
rows=[]
for r in historical:
    j=r['job'];ref=next(q for q in historical if q['job']['method']==j['method'] and q['job']['regime']==j['regime'] and q['job']['scenario']=='clean')
    values,_,_=extract_run(r,None if j['scenario']=='clean' else ref);rows.extend(values)
extra=pd.DataFrame(rows)
main_seed=SEED[(SEED.method==MAIN)&(SEED.seed==20261014)&SEED.scenario.isin(['clean','gradual_stripe','label_flip'])]
extra=pd.concat([extra,main_seed],ignore_index=True)
selected=[]
for scenario,metric,label in [('clean','overall_mean_5_24_pp','Clean accuracy'),('gradual_stripe','affected_honest_event_pp','Honest stripe accuracy'),
                             ('label_flip','overall_mean_5_24_pp','Absolute attacked accuracy'),('label_flip','attack_active_signed_damage_pp','Attack damage')]:
    part=extra[(extra.scenario==scenario)&(extra.metric==metric)].copy();part['measurement']=label;selected.append(part)
extra=pd.concat(selected);extra.to_csv(A/'notebook_other_design_variants.csv',index=False)
display(extra.pivot(index=['regime','method'],columns='measurement',values='value').round(3))
print('Seed20261014 only. Label-flip attack is active throughout rounds5–24, so its overall window equals the attack-active accuracy.')
''')]
    n=nbf.v4.new_notebook(cells=cells)
    n.metadata.kernelspec={'display_name':'FedReTrust Experiment (Python 3.14)','language':'python','name':'fedretrust-experiment'}
    return n


def build():
    if json.loads((STUDY/'status.json').read_text())['status']!='complete':raise ValueError('Wait for the registered study to finish.')
    from blockwise_study import verify
    verify(STUDY);return _execute(make_notebook(),NOTEBOOK)


if __name__=='__main__':print(build())
