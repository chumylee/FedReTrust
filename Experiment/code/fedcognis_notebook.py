"""Artifact-only report for the completed FedCognis aggregation study."""
from pathlib import Path
import json
import nbformat as nbf
from aggregate_validation_notebook import _execute
from fedcognis_study import ROOT, STUDY


def make_notebook():
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    cells = [md('''# 28 — FedCognis aggregation adaptation: protection, honest drift and recovery

This comparison evaluates a **documented adaptation of the trust-aggregation component**, using the existing MNIST CNN and local training. It does not reproduce the complete FedCognis system, its QSA cryptography, SALSTM time-series architecture or published IIoT scores. No public author implementation was located in the article or targeted searches.

The study permits favourable, unfavourable and mixed outcomes. It retains the fixed main method, a memory control, both distance interpretations and all declared penalty alternatives. Low accuracy alone is not proof that a defence rejected an honest update; applied influence and paired controls accompany it.

[Primary paper](https://www.techscience.com/cmc/v85n1/63554/html) · [Preserved paper/algorithm/tables](../resources/fedcognis_reference/source_manifest.json) · [Frozen protocol](../outputs/fedcognis/20261030_v1/protocol.json) · [Implementation](fedcognis.py)'''), code(f'''from pathlib import Path
import sys,json
import pandas as pd
import numpy as np
from IPython.display import display,Image,Markdown
ROOT=Path({str(ROOT)!r});sys.path.insert(0,str(ROOT/'code'))
from fedcognis_study import STUDY,MAIN,CONTROL,SQUARED,BASELINES,SETTINGS,verify,records
from experiment_core import file_sha256
assert json.loads((STUDY/'status.json').read_text())['status']=='complete'
P=verify(STUDY);assert len(records(STUDY))==70
A=STUDY/'analysis';S=json.loads((A/'summary.json').read_text());ACCESS=json.loads((STUDY/'access_audit.json').read_text())
assert S['analysis_source_sha256']==file_sha256(ROOT/'code/fedcognis_analysis.py')
assert S['scientific_trajectories']==352 and S['context_trajectories']==72
assert ACCESS['server_reference_examples']==0 and ACCESS['official_test_used'] is False
D=json.loads((STUDY/'decision.json').read_text())
for name,digest in D['evidence_sha256'].items():assert file_sha256(STUDY/name)==digest
M=pd.read_csv(A/'means.csv');V=pd.read_csv(A/'seed_metrics.csv');E=pd.read_csv(A/'paired_effects.csv')
C=pd.read_csv(A/'accuracy_curves.csv');T=pd.read_csv(A/'trust.csv');REC=pd.read_csv(A/'recovery.csv')
pd.set_option('display.max_rows',100);pd.set_option('display.max_columns',16)
LABEL={{MAIN:'FedCognis aggregation',CONTROL:'FedCognis without memory',SQUARED:'FedCognis squared distance',
 'FedReTrust':'v0.1','FedReTrust-RF-recovery':'v0.2 selected','FedReTrust-Direction-loose':'v0.3 selected','FedReTrust-v04':'v0.4',
 'FLAURA-reconstruction':'FLAURA main','FLAURA-sigma-double':'FLAURA doubled scale'}}
print('70 new runs: 30 main, 16 no-memory controls, 12 squared-distance interpretations, 12 penalty sensitivities.')
print('282 reused primary comparisons; 72 NNM/FedProx context trajectories. All 291 preflight tests passed.')
for key in ['conclusion','memory_result','distance_interpretation_result','penalty_sensitivity_result','recovery_result','influence_result']:
    display(Markdown('**'+key.replace('_',' ').capitalize()+':** '+D[key]))
'''), md(r'''## Executable rule, data access and reconstruction choices

For submitted local model $w_i$ and the broadcast $w$, this adaptation uses

$$d_i=\|w_i-w\|_2,\quad e_i=\exp(-\beta d_i^p),\quad q_i\leftarrow\alpha q_i+(1-\alpha)e_i,$$
$$a_i=\frac{n_iq_i}{\sum_j n_jq_j},\qquad w_{new}=w+\sum_i a_i(w_i-w).$$

Main settings are **alpha=.5, beta=.5, p=1, initial trust=1**. The memory-free control sets alpha=0. The squared interpretation sets p=2. Beta=.1 and .7 are fixed first-seed alternatives. No setting is selected as a replacement after results. Stable log-space computation implements the same EMA and normalization without exponentials underflowing collectively. Absent clients receive no automatic trust restoration.

These choices are explicit: displayed Eq.20 uses an L2 norm, while nearby prose calls it squared distance; both interpretations are tested. Table 6 calls its penalty delta rather than beta, so mapping its .5 value to beta is an assumption. Alpha and initialization lack complete defaults. We use the current broadcast and update trust before current aggregation, following Algorithm 1's sequence. Eq.15 supplies normalized sample-count/trust weights; Algorithm 1's malformed final expression is not applied literally. The stored EMA scores are not additionally normalized to sum to one before the next round.

**Zero server reference images and zero calibration runs.** The aggregator receives only parameters, sample counts, broadcast and round. Neither labels, evaluation scores, simulation intent nor data enter it. The reserved reference partition remains excluded from clients and is unused by this defence. Existing FLTrust/v0.1 comparators retain 300 reference images; FLAURA retains 100 plus its earlier clean geometry calibration.

All participating identities, including poisoned clients, are assumed authenticated. We do not implement or measure QSA or infer signature failure from malicious intent. A valid identity can submit a poisoned update. There is no added hard trust cutoff: the operational aggregation formula specifies soft weights; the prose does not give a reproducible trust-threshold rule. Numerically negligible coefficients are not re-labelled hard exclusions. The reported paper recovery rate is not invented as an extra update rule.

Local CNN training stays fixed: MNIST, SGD lr .01/momentum .9, one epoch, 512 images/client, batch64, ten clients and 24 rounds. This isolates the aggregation component; it leaves time-series representation, cryptography and communication claims outside the experiment.'''), md('''## Main comparisons

Two-seed means cover rounds **5–24**. Honest noise and stripe scores measure affected clients' legitimate tasks. Damage is percentage points below the same method's matched no-attack trajectory; absolute attacked accuracy is shown alongside it. FLAURA doubled-scale and the squared-distance interpretation cover only clean/stripe/persistent poisoning. Blank entries mean untested coverage.

Earlier selected FedReTrust and FLAURA variants retain their development-selection history. No repeated-seed mean is a formal significance result.'''), code('''spec=[('clean','overall_mean_5_24_pp','Clean accuracy'),('abrupt_shot_noise','affected_honest_event_pp','Honest noise accuracy'),
 ('gradual_stripe','affected_honest_event_pp','Honest stripe accuracy'),('label_flip','attack_active_accuracy_pp','Absolute attacked accuracy'),
 ('label_flip','attack_active_signed_damage_pp','Attack damage (pp)')]
parts=[]
for scenario,metric,label in spec:
    x=M[(M.scenario==scenario)&(M.metric==metric)].copy();x['measurement']=label;parts.append(x)
head=pd.concat(parts)
main=head[head.panel.isin(['primary','control','interpretation'])]
assert main['count'].eq(2).all()
comparison=main.pivot(index=['regime','method'],columns='measurement',values='mean')
display(comparison.round(3));comparison.to_csv(A/'notebook_primary_comparison.csv')
paired=E.merge(pd.DataFrame([(s,m) for s,m,_ in spec],columns=['scenario','metric']),on=['scenario','metric'])
for regime in ['IID','non-IID']:
    print(regime+' paired differences')
    display(paired[paired.regime==regime].pivot(index=['baseline','scenario','metric'],columns='seed',values='difference_pp').round(3))
paired.to_csv(A/'notebook_paired_effects.csv',index=False)
print('Paired differences are main FedCognis minus comparator. Positive accuracy favours main; positive damage favours the comparator.')
context=head[head.panel=='context'].pivot(index=['regime','method','attack_policy'],columns='measurement',values='mean')
display(context.round(3));context.to_csv(A/'notebook_context_comparison.csv')
'''), md('''## Training trajectories

The curves show two-seed means and a readable subset of the full comparison table. Attack shading marks active poisoning. Recovery classification below uses each seed separately, not the mean curve.'''), code('''import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
FIG=A/'figures';FIG.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
COLORS={'FedAvg':'#888888','RFA':'#2b7aaa','FLTrust':'#c28b35','FedReTrust-Direction-loose':'#8c63ad',
 'FLAURA-sigma-double':'#bd5464',MAIN:'#168466',CONTROL:'#2e58a2',SQUARED:'#c57632'}
FOCUS=['FedAvg','RFA','FLTrust','FedReTrust-Direction-loose','FLAURA-sigma-double',MAIN]
def save(fig,name):
    fig.savefig(FIG/(name+'.png'),dpi=180,bbox_inches='tight');fig.savefig(FIG/(name+'.pdf'),bbox_inches='tight')
    plt.close(fig);display(Image(filename=str(FIG/(name+'.png'))))
def draw(ax,frame,value='accuracy_pp',methods=FOCUS):
    for method in methods:
        part=frame[frame.method==method]
        if part.empty:continue
        x=part.groupby('round')[value].agg(['mean','count']);assert x['count'].eq(2).all()
        ax.plot(x.index,x['mean'],color=COLORS[method],label=LABEL.get(method,method),lw=2.4 if method==MAIN else 1.5,ls='-' if method==MAIN else '--')
    ax.set_xlim(1,24);ax.set_xticks([1,5,10,15,20,24]);ax.grid(alpha=.15)
def legend(fig,ax):
    handles,labels=ax.get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=3,frameon=False)
primary=C[C.panel.isin(['primary','control','interpretation'])]
fig,axes=plt.subplots(2,3,figsize=(14,7.8),sharex=True)
for i,regime in enumerate(['IID','non-IID']):
    for k,(scenario,title,group) in enumerate([('clean','Clean','all_clients'),('abrupt_shot_noise','Honest noise','affected_honest'),('gradual_stripe','Honest stripes','affected_honest')]):
        ax=axes[i,k];draw(ax,primary[(primary.regime==regime)&(primary.scenario==scenario)&(primary.group==group)])
        ax.set_title(regime+' · '+title);ax.set_ylim(0,100)
        if k:ax.axvline(5,color='#777777',ls=':',lw=1)
        if k==0:ax.set_ylabel('Legitimate accuracy (%)')
        if i==1:ax.set_xlabel('Training round')
fig.suptitle('FedCognis aggregation adaptation — clean learning and honest drift')
legend(fig,axes[0,0]);fig.tight_layout(rect=(0,.08,1,.95));save(fig,'fedcognis_honest_trajectories')
damages=[]
for scenario in ['label_flip','attack_stops']:
    attacked=primary[(primary.scenario==scenario)&(primary.group=='all_clients')]
    clean=primary[(primary.scenario=='clean')&(primary.group=='all_clients')]
    paired=attacked.merge(clean,on=['method','regime','seed','round'],suffixes=('_attack','_clean'),validate='one_to_one')
    paired['damage_pp']=paired.accuracy_pp_clean-paired.accuracy_pp_attack;paired['scenario']=scenario;damages.append(paired)
    fig,axes=plt.subplots(2,2,figsize=(12.5,7.8),sharex=True)
    for i,regime in enumerate(['IID','non-IID']):
        draw(axes[i,0],attacked[attacked.regime==regime]);draw(axes[i,1],paired[paired.regime==regime],value='damage_pp')
        axes[i,0].set_ylim(0,100);axes[i,0].set_ylabel('Legitimate accuracy (%)');axes[i,1].set_ylabel('Own-clean minus attacked (pp)')
        for k,title in enumerate(['Absolute accuracy','Clean-relative damage']):
            axes[i,k].set_title(regime+' · '+title);axes[i,k].axvspan(5,24 if scenario=='label_flip' else 10,color='#bd3f4e',alpha=.06)
            if i==1:axes[i,k].set_xlabel('Training round')
        axes[i,1].axhline(0,color='#555555',lw=.7)
        if scenario=='attack_stops':axes[i,1].axhline(1,color='#777777',ls=':',lw=1)
    fig.suptitle('FedCognis aggregation adaptation — '+('persistent label flips' if scenario=='label_flip' else 'attack stops after round 10'))
    legend(fig,axes[0,0]);fig.tight_layout(rect=(0,.08,1,.95));save(fig,'fedcognis_'+scenario)
pd.concat(damages).to_csv(A/'notebook_damage_curves.csv',index=False)
'''), md('''## Applied influence and trust memory

The same three clients (0–2) are affected in the stripe and label-flip scenarios; their equal-count FedAvg coefficient mass is 30%. Clean rows track those IDs for comparison. These are actual coefficients on raw submitted updates, not honesty probabilities. No hard trust filter is implemented.

The memory control changes alpha from .5 to zero while retaining the distance calculation and beta. Its trajectories can diverge after the first aggregation. The comparison tests the contribution of memory within this adaptation; it does not itself establish whether every attenuated update was useful.'''), code('''focus=T[T.method.isin([MAIN,CONTROL,SQUARED])&T.scenario.isin(['clean','gradual_stripe','label_flip'])&(T.client<3)]
group=focus.groupby(['method','regime','scenario','seed','round']).agg(coefficient_mass=('coefficient','sum'),mean_trust=('trust','mean'),mean_signal=('instantaneous_trust','mean'),mean_norm=('update_l2_norm','mean'),mean_penalty=('distance_penalty','mean')).reset_index()
group['coefficient_mass_pp']=100*group.coefficient_mass
summary=group[group['round']>=5].groupby(['method','regime','scenario','seed']).mean(numeric_only=True).drop(columns='round')
display(summary.round(4));summary.to_csv(A/'notebook_influence.csv')
fig,axes=plt.subplots(2,3,figsize=(14,7.4),sharex=True)
for i,regime in enumerate(['IID','non-IID']):
    for k,scenario in enumerate(['clean','gradual_stripe','label_flip']):
        ax=axes[i,k];draw(ax,group[(group.regime==regime)&(group.scenario==scenario)],value='coefficient_mass_pp',methods=[MAIN,CONTROL,SQUARED])
        ax.axhline(30,color='#777777',ls=':',lw=1);ax.set_ylim(0,65)
        ax.set_title(regime+' · '+{'clean':'Clean clients 0–2','gradual_stripe':'Honest stripes','label_flip':'Label poisoning'}[scenario])
        if scenario!='clean':ax.axvline(5,color='#777777',ls=':',lw=1)
        if k==0:ax.set_ylabel('Applied coefficient mass (%)')
        if i==1:ax.set_xlabel('Training round')
fig.suptitle('Influence of the three affected clients — two-seed means')
legend(fig,axes[0,0]);fig.tight_layout(rect=(0,.08,1,.95));save(fig,'fedcognis_influence')
'''), md('''## Distance interpretation and penalty sensitivity

The squared-distance alternative repeats the three core scenarios on both seeds. The beta=.1/.7 settings use only the first seed. They are sensitivity checks, not independent repetitions or substitutes selected after observing results. The table uses the first seed for every setting so the comparison is matched.'''), code('''parts=[]
for scenario,metric,label in spec:
    if scenario=='abrupt_shot_noise':continue
    x=V[(V.seed==20261014)&V.method.isin(SETTINGS)&(V.scenario==scenario)&(V.metric==metric)].copy();x['measurement']=label;parts.append(x)
grid=pd.concat(parts).pivot(index=['regime','method'],columns='measurement',values='value')
display(grid.round(3));grid.to_csv(A/'notebook_sensitivity_first_seed.csv')
second=V[(V.seed==20261015)&V.method.isin([MAIN,CONTROL,SQUARED])]
parts=[]
for scenario,metric,label in spec:
    if scenario=='abrupt_shot_noise':continue
    x=second[(second.scenario==scenario)&(second.metric==metric)].copy();x['measurement']=label;parts.append(x)
second_grid=pd.concat(parts).pivot(index=['regime','method'],columns='measurement',values='value')
display(second_grid.round(3));second_grid.to_csv(A/'notebook_interpretation_second_seed.csv')
'''), md('''## All scenarios and recovery

The main method covers all ten scenarios on the first seed; five repeat on the second. Mixed attacks use the corresponding honest-drift-only control. Recovery requires legitimate accuracy within one percentage point of its matched control for three consecutive post-attack rounds. Streak start and confirmation differ; already-within-criterion and censored episodes remain explicit. Model recovery is distinct from restoration of a client's applied contribution.'''), code('''metrics=['overall_mean_5_24_pp','overall_tail_21_24_pp','overall_round24_pp','affected_honest_event_pp','attack_active_accuracy_pp','attack_active_signed_damage_pp']
all_scenarios=V[(V.method==MAIN)&V.metric.isin(metrics)].pivot(index=['regime','scenario','seed'],columns='metric',values='value')
display(all_scenarios.round(3));all_scenarios.to_csv(A/'notebook_all_scenarios.csv')
recovery=REC[(REC.group=='all_clients')&REC.method.isin(['FedAvg','RFA','FLTrust','FedReTrust-Direction-loose','FLAURA-reconstruction',MAIN,CONTROL])]
display(recovery.drop(columns=['job_id','attack_policy','panel']).round(4))
recovery.to_csv(A/'notebook_recovery.csv',index=False)
print('Full attacked-client recovery and contribution restoration/reduction are preserved separately; censored episodes are not reported as recovered.')
'''), md('''## Training, task losses and resource use

Training loss uses actual local labels, including poisoned labels. It is not accuracy against legitimate evaluation labels. Raw task summaries include initialization and pre-event rounds and therefore differ from the primary rounds-5–24 scenario score. Timing comes from concurrent fresh workers and is not isolated algorithm benchmarking.

Two development seeds on fixed partitions and fixed affected IDs support descriptive evidence. They do not establish statistical superiority, generalization to unseen tasks, cryptographic security or replication of the full FedCognis paper.'''), code('''train=pd.read_csv(A/'local_training.csv',low_memory=False);raw=pd.read_csv(A/'raw_task_metrics.csv');cost=pd.read_csv(A/'costs.csv')
training=train[train.method==MAIN].groupby(['regime','scenario','seed']).agg(online_train_loss=('online_train_loss','mean'),client_updates=('client','size'),optimizer_steps=('optimizer_steps','sum'))
display(training.round(4));training.to_csv(A/'notebook_training.csv')
tasks=raw[(raw.method==MAIN)&raw.scenario.isin(['clean','gradual_stripe','label_flip'])].groupby(['regime','scenario','seed','task']).agg(mean_accuracy=('accuracy','mean'),mean_loss=('loss','mean'),recorded_rounds=('round','nunique'))
display(tasks.round(4));tasks.to_csv(A/'notebook_raw_tasks.csv')
display(cost[cost.method.isin(list(BASELINES)+[MAIN,CONTROL,SQUARED])].groupby(['method','regime'])[['median_defence_seconds','median_training_seconds','reference_forward_examples']].mean().round(4))
display(pd.Series(ACCESS['totals'],name='70 new runs').to_frame())
print('Verified',len(S['matching_checks']),'input pairs;',len(S['pre_event_checks']),'exact pre-event trajectories;',S['exact_first_round_client_updates_vs_fedavg'],'exact first-round updates vs FedAvg.')
print('All',S['audited_new_rounds'],'new rounds reconstruct from applied coefficients; maximum error',S['maximum_reconstruction_error'])
'''), md('''## Rebuild

Run `Experiment/.venv/bin/python Experiment/code/fedcognis_notebook.py` after completion. This report reads saved artifacts only; it performs no training, inference or parameter selection. Protocol, source snapshots, all settings and all outcomes remain under `outputs/fedcognis/20261030_v1/`. The earlier FLAURA and baseline studies are preserved.''')]
    notebook = nbf.v4.new_notebook(cells=cells)
    notebook.metadata.kernelspec = {'display_name':'FedReTrust Experiment (Python 3.14)','language':'python','name':'fedretrust-experiment'}
    return notebook


if __name__ == '__main__':
    assert json.loads((STUDY/'status.json').read_text())['status']=='complete'
    print(_execute(make_notebook(),ROOT/'code/28_fedcognis_comparison.ipynb'))
