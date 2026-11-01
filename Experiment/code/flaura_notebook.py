"""Artifact-only FLAURA presentation. No training, inference or parameter selection."""
from pathlib import Path
import json
import nbformat as nbf
from aggregate_validation_notebook import _execute

ROOT=Path(__file__).resolve().parents[1]
STUDY=ROOT/'outputs/flaura/20261030_v1'
NOTEBOOK=ROOT/'code/26_flaura_comparison.ipynb'


def make_notebook():
    md,code=nbf.v4.new_markdown_cell,nbf.v4.new_code_cell
    cells=[md('''# 26 — FLAURA reconstruction: protection, honest drift and recovery

This study asks whether FLAURA's representation-based trust can retain useful honest changes while limiting poisoning in the existing MNIST workload. Both outcomes were allowed: successful adaptation or a protection–adaptation trade-off. **Lower drift accuracy alone is not sufficient evidence of harmful honest exclusion.** We also measure applied client weights, exclusions, model recovery and contribution restoration, and compare a control that disables hard filtering.

The implementation is a **documented reconstruction of Algorithm 1**, not author code and not a reproduction of the paper's numerical results. The paper does not provide several operational defaults, and the pseudocode differs from displayed equations. These choices were recorded before calibration and training. All settings and negative or positive results are retained.

[Primary paper](https://www.nature.com/articles/s41598-026-50985-2) · [Frozen protocol](../outputs/flaura/20261030_v1/protocol.json) · [Local implementation](flaura.py)'''),code(f'''from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
from IPython.display import display,Image,Markdown
CODE=Path({str(ROOT/'code')!r});sys.path.insert(0,str(CODE))
from flaura_study import verify,records,MAIN,CONTROL,BASELINES,SETTINGS,freeze_scale
from experiment_core import file_sha256
STUDY=Path({str(STUDY)!r});A=STUDY/'analysis'
assert json.loads((STUDY/'status.json').read_text())['status']=='complete'
P=verify(STUDY);new=records(STUDY);S=json.loads((A/'summary.json').read_text())
assert len(new)==68 and S['scientific_trajectories']==306 and S['context_trajectories']==72
assert S['analysis_source_sha256']==file_sha256(CODE/'flaura_analysis.py')
SCALE=freeze_scale(STUDY);ACCESS=json.loads((STUDY/'access_audit.json').read_text())
assert ACCESS['official_test_used'] is False and ACCESS['server_auxiliary_examples']==100
M=pd.read_csv(A/'means.csv');SEED=pd.read_csv(A/'seed_metrics.csv');C=pd.read_csv(A/'accuracy_curves.csv')
E=pd.read_csv(A/'paired_effects.csv');T=pd.read_csv(A/'trust.csv');REC=pd.read_csv(A/'recovery.csv')
TRAIN=pd.read_csv(A/'local_training.csv',low_memory=False);RAW=pd.read_csv(A/'raw_task_metrics.csv');COST=pd.read_csv(A/'costs.csv')
LABEL={{'FedReTrust':'v0.1','FedReTrust-RF-recovery':'v0.2','FedReTrust-Direction-loose':'v0.3','FedReTrust-v04':'v0.4',MAIN:'FLAURA reconstruction',CONTROL:'FLAURA without hard filtering'}}
REGIMES=['IID','non-IID']
pd.set_option('display.max_rows',100);pd.set_option('display.max_columns',16)
print('68 new runs: 2 short geometry calibrations, 30 main, 12 no-filter controls, 24 sensitivity runs.')
print('240 verified earlier comparisons and 72 NNM/FedProx context trajectories. 279 preflight tests passed.')
print('Fixed sigma_g:',SCALE['sigma_g'],'; 100 auxiliary images; no official-test access.')
if (STUDY/'decision.json').exists():
    D=json.loads((STUDY/'decision.json').read_text())
    for name,digest in D.get('evidence_sha256',{{}}).items():assert file_sha256(STUDY/name)==digest
    display(pd.Series({{k:v for k,v in D.items() if isinstance(v,(str,int,float,bool))}},name='Saved interpretation').to_frame())
'''),md(r'''## Reconstruction and explicit assumptions

The server extracts the post-ReLU 128-dimensional penultimate activation of every submitted model on the same 100 reserved clean images. Ten images per digit are selected deterministically from the existing server-reference partition. These are disjoint from **all** client and development-evaluation data. Labels are used only for balanced selection; the defence receives copied images without labels and performs no backpropagation on them. This is a server-data privilege. FLTrust and FedReTrust v0.1 retain their existing 300-image privilege; reference-free methods retain zero.

| Item | Implemented interpretation |
|---|---|
| Representation centre | Uniform-client geometric median of flattened activation matrices, approximated by smoothed Weiszfeld; convergence diagnostics retained. |
| Global score | Algorithm 1: $G_i=\exp[-\|R_i-R_c\|_F^2/(2\sigma_g^2)]$. Displayed Eq. 7 lacks the factor 2. |
| MMD | Empirical RKHS mean-embedding norm using an RBF kernel; includes diagonal terms and takes the square root. |
| Kernel bandwidth | Median positive pairwise squared distance among all current pooled activation rows. The paper does not specify a kernel or bandwidth rule. |
| Neighbours | Largest adjacent gap in sorted other-client MMD distances, based on Eq. 11. Flat/equal-gap profiles keep all peers; cutoff ties are included. The paper's precise knee implementation is unspecified. |
| Local score | $L_i=\exp[-\mathrm{mean}_{j\in N_i}D_{ij}]$. |
| Memory | $q_i\leftarrow0.5q_i+0.5[0.5G_i+0.5L_i]$, initialized at one. Absent clients do not recover automatically. This EMA is symmetric, not an asymmetric rehabilitation rule. |
| Filtering | Algorithm 1: retain $q_i\ge\gamma\,\mathrm{mean}(q)$, with main $\gamma=0.8$. |
| Aggregation | Normalize retained trust weights and average raw client deltas. Sample counts are not an additional multiplier. |
| Unspecified rebalancing | The prose mentions it without a formula/threshold; Algorithm 1 omits it. It is not invented here. |

The prose places representation extraction at the server; Algorithm 1 shows clients uploading representations. Here the server recomputes them from submitted weights, preventing fabricated representations. **No public author repository or complete default parameter set was located.** Alpha, beta and gamma are reconstruction choices, not attributed paper defaults.

Before the evaluated runs, two six-round clean FedAvg runs on a separate seed supplied 120 client-to-centre distances. Their pooled median sets one fixed global scale for both regimes and all later rounds. This uses clean development calibration and is disclosed as an additional offline assumption. No accuracy or attack outcomes selected the scale. Main parameters were not replaced after seeing results.

Our common workload remains MNIST, the original CNN, SGD learning rate 0.01/momentum 0.9, one local epoch, 512 images/client, ten clients and 24 rounds. The paper uses other datasets and different local optimization. This study tests transfer of the aggregation rule to our workload; it does not contradict or confirm the paper's original backdoor results.'''),md('''## Main comparison: clean learning, honest drift and poisoning

Each principal entry averages **two development seeds over rounds 5–24**. Honest noise and stripes use affected honest clients' legitimate tasks. Attack damage is percentage points below the same method's matched no-attack trajectory. Absolute attacked accuracy is also shown. All four warm-up rounds remain in the raw records; final and tail results appear separately below.

The no-filter control changes only gamma from 0.8 to zero; representation scoring, memory, scale and soft trust weights stay unchanged. It is a component control, not a separately selected replacement for FLAURA.'''),code('''primary=[('clean','overall_mean_5_24_pp','Clean accuracy'),('abrupt_shot_noise','affected_honest_event_pp','Honest noise accuracy'),
 ('gradual_stripe','affected_honest_event_pp','Honest stripe accuracy'),('label_flip','attack_active_accuracy_pp','Absolute attacked accuracy'),
 ('label_flip','attack_active_signed_damage_pp','Attack damage (pp)')]
parts=[]
for scenario,metric,label in primary:
    x=M[(M.scenario==scenario)&(M.metric==metric)].copy();x['measurement']=label;parts.append(x)
comparison=pd.concat(parts)
main=comparison[comparison.panel.isin(['primary','control'])]
assert main['count'].eq(2).all()
table=main.pivot(index=['regime','method'],columns='measurement',values='mean')
display(table.round(3));table.to_csv(A/'notebook_primary_comparison.csv')
print('Blank noise entries for the control indicate untested coverage, not zero accuracy.')
selected=E[E.baseline.isin(['FedAvg','RFA','FLTrust','FedReTrust-Direction-loose','FedReTrust-v04',CONTROL])].merge(
 pd.DataFrame([(s,m) for s,m,_ in primary if s!='abrupt_shot_noise'],columns=['scenario','metric']),on=['scenario','metric'])
paired=selected.pivot(index=['baseline','regime','scenario','metric'],columns='seed',values='difference_pp')
display(paired.round(3));paired.to_csv(A/'notebook_paired_comparison.csv')
print('Differences are FLAURA reconstruction minus comparator. Positive accuracy differences favour FLAURA; positive damage differences favour the comparator.')
context=comparison[comparison.panel=='context'].pivot(index=['regime','method','attack_policy'],columns='measurement',values='mean')
display(context.round(3));context.to_csv(A/'notebook_context_comparison.csv')
'''),md('''## Training trajectories

The curves show two-seed means. A mean curve is descriptive and cannot replace per-seed recovery classification. All principal baselines are present in the tables; the plots show a readable subset. Dashed curves identify the no-filter control.'''),code('''import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
FIG=A/'figures';FIG.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':130})
FOCUS=['FedAvg','RFA','FLTrust','FedReTrust-Direction-loose',MAIN,CONTROL]
COLORS={'FedAvg':'#777777','RFA':'#2778a4','FLTrust':'#bf8631','FedReTrust-Direction-loose':'#8c5ca4',MAIN:'#198768',CONTROL:'#b94a50'}
def save_figure(fig,name):
    fig.savefig(FIG/(name+'.png'),dpi=180,bbox_inches='tight');fig.savefig(FIG/(name+'.pdf'),bbox_inches='tight')
    plt.close(fig);display(Image(filename=str(FIG/(name+'.png'))))
def draw(ax,frame,value='accuracy_pp'):
    for method in FOCUS:
        part=frame[frame.method==method]
        if part.empty:continue
        grouped=part.groupby('round')[value].agg(['mean','count']);assert grouped['count'].eq(2).all()
        ax.plot(grouped.index,grouped['mean'],label=LABEL.get(method,method),color=COLORS[method],
            ls='--' if method==CONTROL else '-',lw=2.2 if method==MAIN else 1.5)
    ax.set_xlim(1,24);ax.set_xticks([1,5,10,15,20,24]);ax.grid(alpha=.15)
def legend(fig,ax):
    handles,labels=ax.get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=3,frameon=False)
primary_curves=C[C.panel.isin(['primary','control'])]
fig,axes=plt.subplots(2,3,figsize=(14,7.8),sharex=True)
for i,regime in enumerate(REGIMES):
    for k,(scenario,title,group) in enumerate([('clean','Clean','all_clients'),('abrupt_shot_noise','Honest noise','affected_honest'),('gradual_stripe','Honest stripes','affected_honest')]):
        ax=axes[i,k];draw(ax,primary_curves[(primary_curves.regime==regime)&(primary_curves.scenario==scenario)&(primary_curves.group==group)])
        ax.set_title(regime+' · '+title);ax.set_ylim(0,100)
        if k:ax.axvline(5,color='#777777',ls=':',lw=1)
        if k==0:ax.set_ylabel('Legitimate accuracy (%)')
        if i==1:ax.set_xlabel('Training round')
fig.suptitle('FLAURA reconstruction — clean learning and honest adaptation')
legend(fig,axes[0,0]);fig.tight_layout(rect=(0,.08,1,.95));save_figure(fig,'flaura_honest_trajectories')
damage=[]
for scenario in ['label_flip','attack_stops']:
    attacked=primary_curves[(primary_curves.scenario==scenario)&(primary_curves.group=='all_clients')]
    clean=primary_curves[(primary_curves.scenario=='clean')&(primary_curves.group=='all_clients')]
    paired=attacked.merge(clean,on=['method','regime','seed','round'],suffixes=('_attack','_clean'),validate='one_to_one')
    paired['damage_pp']=paired.accuracy_pp_clean-paired.accuracy_pp_attack;paired['scenario']=scenario;damage.append(paired)
    fig,axes=plt.subplots(2,2,figsize=(12,7.8),sharex=True)
    for i,regime in enumerate(REGIMES):
        draw(axes[i,0],attacked[attacked.regime==regime]);draw(axes[i,1],paired[paired.regime==regime],value='damage_pp')
        axes[i,0].set_ylim(0,100);axes[i,0].set_ylabel('Legitimate accuracy (%)');axes[i,1].set_ylabel('Own-clean minus attacked (pp)')
        for k,title in enumerate(['Absolute accuracy','Clean-relative damage']):
            axes[i,k].set_title(regime+' · '+title);axes[i,k].axvspan(5,24 if scenario=='label_flip' else 10,color='#bd3f4e',alpha=.06)
            if i==1:axes[i,k].set_xlabel('Training round')
        axes[i,1].axhline(0,color='#555555',lw=.7)
        if scenario=='attack_stops':axes[i,1].axhline(1,color='#777777',ls=':',lw=1)
    fig.suptitle('FLAURA reconstruction — '+('persistent label flips' if scenario=='label_flip' else 'attack stops after round 10'))
    legend(fig,axes[0,0]);fig.tight_layout(rect=(0,.08,1,.95));save_figure(fig,'flaura_'+scenario)
pd.concat(damage).to_csv(A/'notebook_damage_curves.csv',index=False)
'''),md('''## Honest influence, exclusions and the hard-filter control

The three affected clients are 0, 1 and 2 in the stripe and label-flip scenarios. Their combined equal-count FedAvg coefficient would be 30%. We show **actual coefficients on submitted updates**, not a probability of honesty. Clean rows follow those same client IDs for context. No-filter controls retain soft trust weighting, so their influence need not be 30%.

Compare honest-stripe accuracy with the affected clients' weights and exclusion rate. A lower weight alone does not prove an update was useful; a lower accuracy alone does not prove mistaken exclusion. The paired policy control tests the added effect of hard filtering, while global trajectories can diverge after filtering begins. Simulator intent is used only in this offline analysis.'''),code('''focus=T[T.method.isin([MAIN,CONTROL])&T.scenario.isin(['clean','gradual_stripe','label_flip'])&(T.client<3)].copy()
group_keys=['method','regime','scenario','seed','round']
group=focus.groupby(group_keys).agg(coefficient_mass=('coefficient','sum'),reference_mass=('reference_coefficient','sum'),exclusion_fraction=('excluded','mean'),mean_trust=('trust','mean'),mean_global_score=('global_score','mean'),mean_local_score=('local_score','mean')).reset_index()
group['coefficient_mass_pp']=100*group.coefficient_mass
seed_summary=group[group['round']>=5].groupby(['method','regime','scenario','seed']).mean(numeric_only=True).drop(columns='round').reset_index()
seed_summary.to_csv(A/'notebook_influence_seed_summary.csv',index=False)
display(seed_summary.round(4))
fig,axes=plt.subplots(2,3,figsize=(14,7.4),sharex=True)
for i,regime in enumerate(REGIMES):
    for k,scenario in enumerate(['clean','gradual_stripe','label_flip']):
        ax=axes[i,k];draw(ax,group[(group.regime==regime)&(group.scenario==scenario)],value='coefficient_mass_pp')
        ax.axhline(30,color='#777777',ls=':',lw=1);ax.set_ylim(0,65)
        ax.set_title(regime+' · '+{'clean':'Clean clients 0–2','gradual_stripe':'Honest stripes','label_flip':'Label poisoning'}[scenario])
        if scenario!='clean':ax.axvline(5,color='#777777',ls=':',lw=1)
        if k==0:ax.set_ylabel('Applied coefficient mass (%)')
        if i==1:ax.set_xlabel('Training round')
fig.suptitle('Influence of the three affected clients — two-seed means')
legend(fig,axes[0,0]);fig.tight_layout(rect=(0,.07,1,.95));save_figure(fig,'flaura_influence')
print('Geometric-median termination counts, counted once per round:')
display(T.drop_duplicates(['job_id','round']).groupby(['method','median_termination']).size().unstack(fill_value=0))
'''),md('''## Sensitivity: all settings retained

Four fixed, one-at-a-time alternatives change the global scale by ×0.5/×2 or gamma to 0.5/1.0. They cover clean, stripe and label flipping in both regimes on **seed 20261014 only**. The main and no-filter rows below use that same seed. These settings are not independent repetitions and the best row is not substituted for the main reconstruction. Other unspecified choices, including kernel and knee implementation, remain limitations.'''),code('''sensitivity=SEED[(SEED.seed==20261014)&SEED.method.isin(SETTINGS)&SEED.scenario.isin(['clean','gradual_stripe','label_flip'])]
parts=[]
for scenario,metric,label in primary:
    if scenario=='abrupt_shot_noise':continue
    x=sensitivity[(sensitivity.scenario==scenario)&(sensitivity.metric==metric)].copy();x['measurement']=label;parts.append(x)
grid=pd.concat(parts).pivot(index=['regime','method'],columns='measurement',values='value')
display(grid.round(3));grid.to_csv(A/'notebook_sensitivity.csv')
print('No post-result parameter selection; first-seed results only.')
'''),md('''## All scenarios, final rounds and recovery

The main method runs all ten scenarios on the first seed. Five scenarios repeat on the second seed. The mixed drift-and-attack scenario uses its matching drift-only control, not the clean-input scenario. Recovery requires accuracy within one percentage point of the matched control for three consecutive post-attack rounds. The confirmation round differs from the streak start. Censored episodes have not demonstrated recovery within the observation window.'''),code('''all_metrics=['overall_mean_5_24_pp','overall_tail_21_24_pp','overall_round24_pp','affected_honest_event_pp','attack_active_accuracy_pp','attack_active_signed_damage_pp']
all_scenarios=SEED[(SEED.method==MAIN)&SEED.metric.isin(all_metrics)].pivot(index=['regime','scenario','seed'],columns='metric',values='value')
display(all_scenarios.round(3));all_scenarios.to_csv(A/'notebook_all_scenarios.csv')
recovery=REC[REC.method.isin(['FedAvg','RFA','FLTrust','FedReTrust-Direction-loose',MAIN])&(REC.group=='all_clients')]
display(recovery.drop(columns=['job_id','attack_policy','panel']).round(4))
print('Individual attacked-client recovery and contribution-restoration records are preserved in recovery.csv and contribution_restoration.csv; no censored case is relabelled recovered.')
'''),md('''## Training, raw tasks, workload and interpretation limits

Online training loss is measured on each client's actual local training targets, which can be poisoned. It is distinct from loss on legitimate evaluation labels. The clean/noise/stripe task files include round zero and all four pre-event training rounds. A method can improve its own clean-relative damage by lowering clean accuracy; absolute attacked accuracy prevents that ambiguity.

Two reused development seeds give descriptive paired evidence. There is one fixed partition per regime, with the same affected client IDs across repetitions. These results do not establish statistical superiority, unseen-dataset performance or a universal impossibility result. The current suite does not reproduce FLAURA's backdoor experiments. A failure of this reconstruction must be reported with its kernel, scale, threshold and knee assumptions; success likewise needs broader independent evaluation.'''),code('''selected=TRAIN[TRAIN.method.isin(list(BASELINES)+[MAIN,CONTROL])]
training=selected.groupby(['method','regime','scenario','seed']).agg(online_train_loss=('online_train_loss','mean'),client_updates=('client','size'),optimizer_steps=('optimizer_steps','sum')).reset_index()
training.to_csv(A/'notebook_training_summary.csv',index=False)
display(training[(training.method==MAIN)&training.scenario.isin(['clean','gradual_stripe','label_flip'])].round(4))
raw=RAW[(RAW.method==MAIN)&RAW.scenario.isin(['clean','gradual_stripe','label_flip'])]
raw_summary=raw.groupby(['regime','scenario','seed','task']).agg(mean_accuracy=('accuracy','mean'),mean_loss=('loss','mean'),recorded_rounds=('round','nunique')).reset_index()
display(raw_summary.round(4));raw_summary.to_csv(A/'notebook_raw_task_summary.csv',index=False)
print('Raw-task means above include round zero and pre-event rounds; they are not the primary scenario-weighted scores.')
display(COST[COST.method.isin(list(BASELINES)+[MAIN,CONTROL])].groupby(['method','regime'])[['median_defence_seconds','median_training_seconds','reference_forward_examples']].mean().round(4))
display(pd.Series(ACCESS['totals'],name='All new runs including calibration').to_frame())
print('All',len(S['matching_checks']),'cross-method input pairs verified;',len(S['pre_event_checks']),'pre-event trajectories exactly matched.')
print(S['exact_first_round_client_updates_vs_fedavg'],'new first-round client updates exactly match FedAvg. Evaluation data never enter the defence.')
'''),md('''## Reproduction

Rebuild this report from completed artifacts with `Experiment/.venv/bin/python Experiment/code/flaura_notebook.py`. It performs no training or model inference. The protocol, source snapshots, calibration freeze, per-run artifacts and access audit live under `outputs/flaura/20261030_v1/`. Figures are exported as PNG and PDF.

The preserved [paper and Algorithm 1](../resources/flaura_reference/source_manifest.json) document the reconstruction's provenance. This work changes no earlier baseline implementation or frozen result.''')]
    notebook=nbf.v4.new_notebook(cells=cells)
    notebook.metadata.kernelspec={'display_name':'FedReTrust Experiment (Python 3.14)','language':'python','name':'fedretrust-experiment'}
    return notebook


def build():
    if json.loads((STUDY/'status.json').read_text())['status']!='complete':raise ValueError('Study must complete first.')
    return _execute(make_notebook(),NOTEBOOK)


if __name__=='__main__':print(build())
