"""Build the completed second-seed follow-up report from artifacts only."""
from pathlib import Path
import json
import nbformat as nbf
from aggregate_validation_notebook import _execute
from flaura_second_seed import ROOT, STUDY


def make_notebook():
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    cells = [md('''# 27 — FLAURA doubled-scale follow-up: second seed

Six new trajectories repeat **the unchanged doubled-scale setting** on development seed 20261015: clean, honest stripes and persistent label flipping in IID and non-IID data. The first-seed improvement motivated this follow-up, so this is selected development evidence. Both seeds already feature in the broader project; neither is an untouched final-test set. One partition per regime, fixed affected clients and the same auxiliary images remain limitations.

The original main FLAURA setting remains recorded in notebook 26. No new parameter search, calibration, recovery scenarios, FedCognis implementation or official-test evaluation is included here. The auxiliary-data requirement and reconstruction limitations remain unchanged.

[Frozen follow-up protocol](../outputs/flaura/20261030_sigma_double_seed2/protocol.json) · [Original comparison](26_flaura_comparison.ipynb)'''), code(f'''from pathlib import Path
import sys,json
import pandas as pd
import numpy as np
from IPython.display import display,Image,Markdown
ROOT=Path({str(ROOT)!r});sys.path.insert(0,str(ROOT/'code'))
from flaura_second_seed import STUDY,PARENT,TARGET,MAIN,CONTROL,METHODS,SEEDS,verify
from experiment_core import file_sha256
assert json.loads((STUDY/'status.json').read_text())['status']=='complete'
P=verify(STUDY);A=STUDY/'analysis';S=json.loads((A/'summary.json').read_text())
assert S['source_sha256']==file_sha256(ROOT/'code/flaura_second_seed.py')
ACCESS=json.loads((STUDY/'access_audit.json').read_text())
assert ACCESS['official_test_used'] is False
D=json.loads((STUDY/'decision.json').read_text())
for name,digest in D['evidence_sha256'].items():assert file_sha256(STUDY/name)==digest
M=pd.read_csv(A/'means.csv');V=pd.read_csv(A/'seed_metrics.csv');E=pd.read_csv(A/'paired_effects.csv')
C=pd.read_csv(A/'accuracy_curves.csv');T=pd.read_csv(A/'trust.csv')
pd.set_option('display.max_rows',100);pd.set_option('display.max_columns',16)
LABEL={{MAIN:'FLAURA main',TARGET:'FLAURA doubled scale',CONTROL:'FLAURA no filter','FedReTrust':'v0.1','FedReTrust-RF-recovery':'v0.2 selected','FedReTrust-Direction-loose':'v0.3 selected','FedReTrust-v04':'v0.4'}}
print('Six new runs; 126 verified earlier trajectories; 279 preflight tests passed.')
print('Fixed sigma_g:',P['config']['sigma_g'],'; 100 clean auxiliary images; no recalibration or official-test access.')
display(pd.Series({{k:v for k,v in D.items() if isinstance(v,(str,bool,int,float))}},name='Saved interpretation').to_frame())
'''), md('''## Each seed and the two-seed comparison

Accuracy is percent; damage is percentage points below the same method's own matched clean run. Primary means cover rounds 5–24. Stripe accuracy measures affected honest clients. Both clean and absolute attacked accuracy accompany damage, because a lower clean baseline can make damage look smaller.

Positive paired accuracy differences favour the doubled scale; negative damage differences favour it. A repeated direction across two seeds is descriptive replication, not a statistical-superiority finding.'''), code('''spec=[('clean','overall_mean_5_24_pp','Clean accuracy'),('gradual_stripe','affected_honest_event_pp','Honest stripe accuracy'),('label_flip','attack_active_accuracy_pp','Absolute attacked accuracy'),('label_flip','attack_active_signed_damage_pp','Attack damage (pp)')]
parts=[];means=[]
for scenario,metric,label in spec:
    x=V[(V.scenario==scenario)&(V.metric==metric)].copy();x['measurement']=label;parts.append(x)
    x=M[(M.scenario==scenario)&(M.metric==metric)].copy();x['measurement']=label;means.append(x)
HEAD=pd.concat(parts);MEAN=pd.concat(means)
assert MEAN['count'].eq(2).all()
each=HEAD[HEAD.method.isin([MAIN,TARGET])].pivot(index=['regime','seed','method'],columns='measurement',values='value')
display(each.round(3));each.to_csv(A/'notebook_flaura_each_seed.csv')
table=MEAN.pivot(index=['regime','method'],columns='measurement',values='mean')
display(table.round(3));table.to_csv(A/'notebook_all_baselines.csv')
chosen=E[(E.comparator==MAIN)].merge(pd.DataFrame([(s,m) for s,m,_ in spec],columns=['scenario','metric']),on=['scenario','metric'])
paired=chosen.pivot(index=['regime','scenario','metric'],columns='seed',values='difference_pp')
display(paired.round(3));paired.to_csv(A/'notebook_scale_effects.csv')
'''), md('''## Seed comparisons and trajectories

The first figure separates the two seeds; lines connect the same method's results and do not imply a temporal trend. The second figure shows descriptive two-seed mean training curves. All comparators remain in the tables; the plots use a readable subset.'''), code('''import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
FIG=A/'figures';FIG.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
colors={MAIN:'#198768',TARGET:'#275cac','FedAvg':'#888888','RFA':'#ce8339','FLTrust':'#a95164','FedReTrust-Direction-loose':'#8b63ad'}
focus=list(colors)
def save(fig,name):
    fig.savefig(FIG/(name+'.png'),dpi=180,bbox_inches='tight');fig.savefig(FIG/(name+'.pdf'),bbox_inches='tight')
    plt.close(fig);display(Image(filename=str(FIG/(name+'.png'))))
fig,axes=plt.subplots(2,3,figsize=(13,7.5))
for row,regime in enumerate(['IID','non-IID']):
    for col,measurement in enumerate(['Clean accuracy','Honest stripe accuracy','Attack damage (pp)']):
        ax=axes[row,col]
        for method in focus:
            x=HEAD[(HEAD.regime==regime)&(HEAD.method==method)&(HEAD.measurement==measurement)].sort_values('seed')
            assert len(x)==2
            ax.plot([0,1],x.value,marker='o',color=colors[method],label=LABEL.get(method,method),lw=2.5 if method==TARGET else 1.4,ls='-' if method in (MAIN,TARGET) else '--')
        ax.set_xticks([0,1],['20261014','20261015']);ax.set_xlim(-.15,1.15);ax.grid(alpha=.15)
        ax.set_title(regime+' · '+measurement);ax.set_ylabel('Percentage points' if col==2 else 'Accuracy (%)')
        if col==2:ax.axhline(0,color='#777777',lw=.6)
        if row==1:ax.set_xlabel('Development seed')
fig.suptitle('FLAURA doubled scale — does the first-seed improvement repeat?')
handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=3,frameon=False)
fig.tight_layout(rect=(0,.09,1,.95));save(fig,'flaura_seed2_comparison')
poison=C[(C.scenario=='label_flip')&(C.group=='all_clients')]
clean=C[(C.scenario=='clean')&(C.group=='all_clients')]
damage=poison.merge(clean,on=['method','regime','seed','round'],suffixes=('_poison','_clean'),validate='one_to_one')
damage['damage_pp']=damage.accuracy_pp_clean-damage.accuracy_pp_poison
damage.to_csv(A/'notebook_damage_curves.csv',index=False)
fig,axes=plt.subplots(2,2,figsize=(12,7.5))
for row,regime in enumerate(['IID','non-IID']):
    for col in range(2):
        frame=C[(C.regime==regime)&(C.scenario=='gradual_stripe')&(C.group=='affected_honest')] if col==0 else damage[damage.regime==regime]
        value='accuracy_pp' if col==0 else 'damage_pp';ax=axes[row,col]
        for method in focus:
            x=frame[frame.method==method].groupby('round')[value].agg(['mean','count']);assert x['count'].eq(2).all()
            ax.plot(x.index,x['mean'],color=colors[method],label=LABEL.get(method,method),lw=2.5 if method==TARGET else 1.5,ls='-' if method in (MAIN,TARGET) else '--')
        ax.set_xlim(1,24);ax.set_xticks([1,5,10,15,20,24]);ax.axvline(5,color='#777777',ls=':',lw=1);ax.grid(alpha=.15)
        ax.set_title(regime+' · '+('Honest stripes' if col==0 else 'Persistent label-flip damage'))
        ax.set_ylabel('Affected-client accuracy (%)' if col==0 else 'Own-clean minus attacked (pp)')
        if col==0:ax.set_ylim(0,100)
        else:ax.axhline(0,color='#777777',lw=.6)
        if row==1:ax.set_xlabel('Training round')
fig.suptitle('Honest adaptation and attack protection — two-seed mean curves')
handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=3,frameon=False)
fig.tight_layout(rect=(0,.09,1,.95));save(fig,'flaura_seed2_trajectories')
'''), md('''## Influence, training, final rounds and access audit

Clients 0–2 are the affected group. Their combined coefficients represent actual applied update weights, not probabilities of honesty. We retain per-seed exclusions and trust scores. Local training loss uses actual local targets, including poisoned labels; legitimate evaluation accuracy is separate.

Only clean, persistent stripes and persistent label flipping were repeated. **No stopped-attack recovery claim can be made for the doubled-scale setting from this follow-up.**'''), code('''group=T[(T.client<3)&(T['round']>=5)].groupby(['method','regime','scenario','seed','round']).agg(coefficient_mass=('coefficient','sum'),exclusion_fraction=('excluded','mean'),mean_trust=('trust','mean')).reset_index()
influence=group.groupby(['method','regime','scenario','seed'])[['coefficient_mass','exclusion_fraction','mean_trust']].mean()
display(influence.round(4));influence.to_csv(A/'notebook_influence.csv')
tail=V[V.method.isin([MAIN,TARGET])&V.metric.isin(['overall_mean_5_24_pp','overall_tail_21_24_pp','overall_round24_pp','affected_honest_event_pp','attack_active_signed_damage_pp'])].pivot(index=['regime','method','scenario','seed'],columns='metric',values='value')
display(tail.round(3));tail.to_csv(A/'notebook_final_tail.csv')
training=pd.read_csv(A/'local_training.csv',low_memory=False)
train=training[training.method==TARGET].groupby(['regime','scenario','seed']).agg(online_train_loss=('online_train_loss','mean'),client_updates=('client','size'),optimizer_steps=('optimizer_steps','sum'))
display(train.round(4));train.to_csv(A/'notebook_training.csv')
raw=pd.read_csv(A/'raw_task_metrics.csv');raw=raw[raw.method==TARGET]
raw_summary=raw.groupby(['regime','scenario','seed','task']).agg(mean_accuracy=('accuracy','mean'),mean_loss=('loss','mean'),rounds=('round','nunique'))
display(raw_summary.round(4));raw_summary.to_csv(A/'notebook_raw_tasks.csv')
print('Raw task means include round zero and the four pre-event rounds; they are not the primary scenario-weighted result.')
cost=pd.read_csv(A/'costs.csv');display(cost[cost.method.isin([MAIN,TARGET])].groupby(['method','regime'])[['median_defence_seconds','median_training_seconds','reference_forward_examples']].mean().round(4))
display(pd.Series(ACCESS['totals'],name='Six new runs').to_frame())
print('Verified:',len(S['matching_checks']),'input pairs;',len(S['pre_event_checks']),'exact pre-event trajectories;',S['exact_first_round_updates'],'exact first-round client updates vs FedAvg.')
print('All',S['audited_new_rounds'],'new rounds reconstruct from applied weights; max error:',S['maximum_reconstruction_error'])
'''), md('''## Rebuild

Run `Experiment/.venv/bin/python Experiment/code/flaura_second_seed_notebook.py` after the study and interpretation are complete. This report reads saved artifacts only and performs no training or inference. All original FLAURA study artifacts and notebook 26 are preserved by hash checks. Fresh seeds and broader datasets would be needed before stronger generalization claims.''')]
    notebook=nbf.v4.new_notebook(cells=cells)
    notebook.metadata.kernelspec={'display_name':'FedReTrust Experiment (Python 3.14)','language':'python','name':'fedretrust-experiment'}
    return notebook


if __name__ == '__main__':
    assert json.loads((STUDY/'status.json').read_text())['status']=='complete'
    print(_execute(make_notebook(), ROOT/'code/27_flaura_second_seed.ipynb'))
