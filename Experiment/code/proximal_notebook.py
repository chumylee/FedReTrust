"""Presentation of saved four-way training results; no model evaluation."""
from pathlib import Path
import json
import nbformat as nbf
from aggregate_validation_notebook import _execute

ROOT=Path(__file__).resolve().parents[1]
STUDY=ROOT/'outputs/proximal_training/20261030_v1'
NOTEBOOK=ROOT/'code/24_proximal_training.ipynb'


def make_notebook(study=STUDY):
    md,code=nbf.v4.new_markdown_cell,nbf.v4.new_code_cell
    cells=[md(r'''# 24 — FedProx local training with FedAvg and RFA

**Observed result:** neither proximal treatment meets the joint target at μ=1. FedProx+RFA improves non-IID label-flip protection under both attacker policies, but reduces honest-stripe accuracy in both regimes. With penalty-bypassing attackers, its non-IID damage is 3.41 pp versus RFA's 4.17 pp. Its stripe accuracy is 65.88% IID / 77.41% non-IID, versus RFA's 69.10% / 79.25% and FedAvg's 80.67% / 80.60%. These are two-seed means, not significance claims.

Does keeping local training closer to the current shared model improve the balance between honest adaptation and poison resistance? This four-way experiment compares **FedAvg, FedProx, RFA, and FedProx+RFA** over actual 24-round global-model trajectories.

The two proximal treatments minimize minibatch cross-entropy plus
\(\frac{\mu}{2}\sum_p(\theta_p-w_{t,p})^2\), with **μ = 1**, including biases and without parameter-count normalization. The anchor is the current round's broadcast, reset for every client. The optimizer, samples, initialization, aggregation and training budget otherwise remain unchanged. Original SGD momentum 0.9 is retained to isolate the objective change. This is a FedProx-objective adaptation, not an exact reproduction of the author's optimizer or published benchmark.

**Zero trusted server images.** Global development measurements are used only for research reporting, never as optimizer or aggregation feedback. The 300 reserved reference images remain unused. No official-test data are loaded.

This fixed-setting pilot uses two reused development seeds and fixed IID/non-IID partitions. It is exploratory, not a tuned FedProx comparison, an independent confirmation, a new FedReTrust version or a novelty claim.'''),code(f'''from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
from IPython.display import display,Image,Markdown
CODE=Path({str(ROOT/'code')!r});sys.path.insert(0,str(CODE))
from proximal_study import verify,new_records
from experiment_core import file_sha256
STUDY=Path({str(Path(study).resolve())!r});A=STUDY/'analysis'
assert json.loads((STUDY/'status.json').read_text())['status']=='complete'
P=verify(STUDY);records=new_records(STUDY);assert len(records)==58
S=json.loads((A/'summary.json').read_text());ACCESS=json.loads((STUDY/'access_audit.json').read_text())
assert S['analysis_source_sha256']==file_sha256(CODE/'proximal_analysis.py')
assert len(S['pairing_checks'])==96 and len(S['pre_event_checks'])==80
assert ACCESS['trusted_server_images']==0 and ACCESS['official_test_used'] is False
assert ACCESS['bypassed_client_updates']==624
M=pd.read_csv(A/'two_seed_means.csv');SEED=pd.read_csv(A/'seed_metrics.csv')
C=pd.read_csv(A/'accuracy_curves.csv');T=pd.read_csv(A/'target_checks.csv')
E=pd.read_csv(A/'paired_effects.csv');REC=pd.read_csv(A/'recovery.csv')
TRAIN=pd.read_csv(A/'local_training.csv');RAW=pd.read_csv(A/'raw_task_metrics.csv')
CON=pd.read_csv(A/'contributions.csv');COST=pd.read_csv(A/'costs.csv')
METHODS=['FedAvg','FedProx','RFA','FedProx+RFA'];REGIMES=['IID','non-IID']
SCENARIOS=['clean','abrupt_shot_noise','gradual_stripe','label_flip','attack_stops']
SLABEL={{'clean':'Clean','abrupt_shot_noise':'Honest noise','gradual_stripe':'Honest stripes','label_flip':'Persistent label flip','attack_stops':'Attack stops'}}
POLICY={{'standard':'Original baseline','compliant':'Penalty followed','bypass':'Attacker bypasses penalty'}}
COLORS={{'FedAvg':'#7b7b7b','FedProx':'#bc7628','RFA':'#2778a4','FedProx+RFA':'#25856b'}}
pd.set_option('display.max_rows',150);pd.set_option('display.max_columns',18)
print('Verified 58 new runs: 56 treatments + 2 zero-penalty replay audits; 40 verified baseline runs reused.')
print('96 scientific trajectories; 248 preflight tests passed; two matched seeds per condition.')
display(pd.DataFrame(S['zero_mu_audits']))
'''),md('''## Design and declared comparison

Every trajectory has ten clients, 512 training images per client, one local epoch, batch size 64, learning rate 0.01 and 24 rounds. Rounds 1–4 are clean warm-up. Clients 0–2 receive honest shot noise from round 5, honest stripes that gradually increase from round 5, or label poisoning from round 5. Persistent poisoning lasts through round 24; stopped poisoning lasts through round 10. Poisoned labels are mapped to `9 − label`, but evaluation always uses legitimate labels.

For both poisoning scenarios, **penalty followed** applies μ=1 even to poisoned clients. **Attacker bypasses penalty** lets active poisoned clients use μ=0; they resume μ=1 once the attack stops. The server receives no attack flags. This tests an adversary's freedom to ignore a prescribed local objective. Bypass runs are unnecessary in clean and honest-drift conditions because no client is poisoning labels there.

The strict descriptive target requires each proximal treatment to match FedAvg on clean accuracy, affected-client honest noise/stripe accuracy, and unchanged-client accuracy during both drifts. It must also match RFA on both absolute persistent-attack accuracy and damage relative to its own clean run, under both attacker policies. Both data regimes must meet every requirement: **18 checks per treatment**. Recovery is reported separately. Tiny differences are numerical comparisons, not evidence of statistical superiority.

This pilot does not include simultaneous drift plus attack, Gaussian updates or recurring attacks. It compares four methods, so passing its gate would not establish superiority over FLTrust, Viceroy or every earlier baseline.'''),code('''primary=[('clean','overall_mean_5_24_pp','Clean accuracy'),
 ('abrupt_shot_noise','affected_honest_event_pp','Honest noise: affected accuracy'),
 ('gradual_stripe','affected_honest_event_pp','Honest stripes: affected accuracy'),
 ('label_flip','attack_active_accuracy_pp','Persistent attack: absolute accuracy'),
 ('label_flip','attack_active_signed_damage_pp','Persistent attack: damage (lower better)')]
rows=[]
for scenario,metric,label in primary:
    part=M[(M.scenario==scenario)&(M.metric==metric)].copy();part['measurement']=label
    rows.append(part)
PRIMARY=pd.concat(rows)
display(PRIMARY.pivot(index=['regime','method','attack_policy'],columns='measurement',values='mean').round(3))
print('Accuracy values are percentages; damage is percentage points. All are means across two seeds.')
display(pd.DataFrame(S['target_results']).T)
display(T[['method','regime','scenario','attack_policy','metric','value_pp','target_pp','advantage_pp','met']].round(4))
print('Positive advantage means the target was exceeded. Damage reverses the sign because lower is better.')
print('Per-seed treatment effects relative to the same aggregation method:')
selected=E.merge(pd.DataFrame([(s,m) for s,m,_ in primary],columns=['scenario','metric']),on=['scenario','metric'])
display(selected.pivot(index=['method','attack_policy','regime','scenario','metric'],columns='seed',values='difference').round(4))
'''),md('''## Learning and honest adaptation

Legitimate-task accuracy weights each client's original class proportions and its current honest input domain. Clients have equal population weight. The honest-stripe task therefore changes during its ramp. Drift panels below show only honestly affected clients; the tables retain both affected and unchanged clients. A lower loss on flipped training labels is not evidence of good legitimate-task predictions.

The main accuracy summaries use **rounds 5–24**. Tail accuracy averages rounds 21–24; final accuracy uses round 24 only. Warm-up is not included in these summaries. Round-zero task metrics and every training/evaluation round remain in the saved raw data.'''),code('''import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
FIG=A/'figures';FIG.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':130})
def save_figure(fig,name):
    fig.savefig(FIG/(name+'.png'),dpi=180,bbox_inches='tight')
    fig.savefig(FIG/(name+'.pdf'),bbox_inches='tight')
    plt.close(fig);display(Image(filename=str(FIG/(name+'.png'))))
def draw(ax,frame):
    for method in METHODS:
        for policy in ('standard','compliant','bypass'):
            sub=frame[(frame.method==method)&(frame.attack_policy==policy)]
            if sub.empty:continue
            line=sub.groupby('round').accuracy_pp.mean()
            label=method+(' (bypass)' if policy=='bypass' else '')
            ax.plot(line.index,line.values,color=COLORS[method],ls='--' if policy=='bypass' else '-',label=label,lw=1.8)
    ax.set_xlim(1,24);ax.set_xticks([1,5,10,15,20,24]);ax.grid(alpha=.15)
fig,axes=plt.subplots(2,3,figsize=(13.5,7.2),sharex=True)
for i,regime in enumerate(REGIMES):
    for k,scenario in enumerate(SCENARIOS[:3]):
        ax=axes[i,k];group='all_clients' if scenario=='clean' else 'affected_honest'
        draw(ax,C[(C.regime==regime)&(C.scenario==scenario)&(C.group==group)])
        ax.set_title(regime+' · '+SLABEL[scenario]);ax.set_ylim(0,100)
        if scenario!='clean':ax.axvline(5,color='#444444',ls=':',lw=1)
        if k==0:ax.set_ylabel('Legitimate accuracy (%)')
        if i==1:ax.set_xlabel('Training round')
handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=4,frameon=False)
fig.suptitle('Actual training trajectories — two-seed means',fontsize=15)
fig.tight_layout(rect=(0,.05,1,.95));save_figure(fig,'proximal_honest_trajectories')
metrics=['overall_mean_5_24_pp','overall_tail_21_24_pp','overall_round24_pp',
         'affected_honest_event_pp','unchanged_honest_mean_5_24_pp']
display(M[M.metric.isin(metrics)].pivot(index=['regime','scenario','method','attack_policy'],columns='metric',values='mean').round(3))
'''),md('''## Protection and recovery

Damage is the paired clean accuracy minus attacked accuracy over active attack rounds. Negative signed damage means the attacked trajectory did better over that window; it is not clipped. Absolute attack accuracy is also shown so a method cannot appear protective merely by reducing its clean baseline.

For stopped attacks, recovery requires accuracy to be no more than **1 percentage point below the same method's clean trajectory for three consecutive rounds after round 10**. The first round of that streak and its third-round confirmation are distinct. Failure to confirm by round 24 is right-censored, not an invented recovery time. A method already within the criterion at round 10 is identified explicitly. Both all-client and former-attacker populations are reported.'''),code('''# Pair each curve with its own method/seed clean trajectory; bypass shares its compliant clean run.
attack=C[(C.group=='all_clients')&C.scenario.isin(['label_flip','attack_stops'])].copy()
clean=C[(C.group=='all_clients')&(C.scenario=='clean')][['method','regime','seed','round','accuracy_pp']].rename(columns={'accuracy_pp':'clean_accuracy_pp'})
paired=attack.merge(clean,on=['method','regime','seed','round'],validate='many_to_one')
paired['damage_pp']=paired.clean_accuracy_pp-paired.accuracy_pp
paired.to_csv(A/'notebook_paired_damage_curves.csv',index=False)
for scenario,filename in [('label_flip','proximal_persistent_attack'),('attack_stops','proximal_stopped_attack')]:
    fig,axes=plt.subplots(2,2,figsize=(12,7.6),sharex=True)
    for i,regime in enumerate(REGIMES):
        sub=paired[(paired.regime==regime)&(paired.scenario==scenario)]
        draw(axes[i,0],sub)
        damage=sub.drop(columns='accuracy_pp').rename(columns={'damage_pp':'accuracy_pp'})
        draw(axes[i,1],damage)
        axes[i,0].set_ylabel('Legitimate accuracy (%)');axes[i,0].set_ylim(0,100)
        axes[i,1].set_ylabel('Own-clean minus attacked (pp)');axes[i,1].axhline(0,color='#555555',lw=.8)
        for k,description in enumerate(['Absolute accuracy','Damage relative to own clean']):
            ax=axes[i,k];ax.set_title(regime+' · '+description)
            ax.axvspan(5,24 if scenario=='label_flip' else 10,color='#b95555',alpha=.07)
            if i==1:ax.set_xlabel('Training round')
        if scenario=='attack_stops':axes[i,1].axhline(1,color='#555555',ls=':',lw=1,label='Recovery threshold')
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=3,frameon=False)
    fig.suptitle(SLABEL[scenario]+' — solid: prescribed objective; dashed: attacker bypass',fontsize=13)
    fig.tight_layout(rect=(0,.075,1,.95));save_figure(fig,filename)
cols=['regime','method','attack_policy','seed','group','criterion_met_at_attack_end',
      'streak_start_round','confirmation_round','rounds_to_streak_start','rounds_to_confirmation','right_censored']
display(REC[cols].sort_values(['regime','method','attack_policy','seed','group']))
print('Recovery rows are individual seed outcomes; the plots show means and cannot establish a three-round per-seed streak.')
display(M[M.metric.str.startswith('attack_')].pivot(index=['regime','scenario','method','attack_policy'],columns='metric',values='mean').round(3))
'''),md('''## Clean training, losses and raw task performance

Online training cross-entropy is averaged across the local minibatches visited during optimization. The proximal penalty and total objective are reported separately. They are not held-out accuracy measurements. In poisoned clients, cross-entropy is computed against the poisoned targets.

The raw task table evaluates each fixed development task separately, without the client-population weighting of the primary accuracy measure. The clean task remains available even during drift. Results from different definitions should not be substituted for one another.'''),code('''TRAIN['window']=np.select([TRAIN['round']<=4,TRAIN['round']<=10],['Warm-up 1–4','Rounds 5–10'],default='Rounds 11–24')
train_summary=TRAIN.groupby(['regime','scenario','method','attack_policy','window']).agg(
    client_updates=('client','size'),online_CE=('online_train_loss','mean'),
    proximal_penalty=('online_proximal_penalty','mean'),total_objective=('online_total_objective','mean'),
    bypassed_updates=('proximal_penalty_bypassed','sum'),optimizer_steps=('optimizer_steps','sum')).reset_index()
train_summary.to_csv(A/'notebook_training_summary.csv',index=False)
print('Clean training, with warm-up retained separately:')
display(train_summary[train_summary.scenario=='clean'].round(6))
print('All scenario training, averaged over all 24 rounds (full windows saved in notebook_training_summary.csv):')
display(TRAIN.groupby(['regime','scenario','method','attack_policy'])[['online_train_loss','online_proximal_penalty','online_total_objective']].mean().round(6))
raw=[]
for label,mask in [('Mean rounds 5–24',RAW['round']>=5),('Tail rounds 21–24',RAW['round']>=21),('Final round 24',RAW['round']==24)]:
    part=RAW[mask].groupby(['regime','scenario','method','attack_policy','task'])[['accuracy','loss']].mean().reset_index()
    part['window']=label;part['accuracy_percent']=100*part.accuracy;raw.append(part.drop(columns='accuracy'))
raw=pd.concat(raw);raw.to_csv(A/'notebook_raw_task_summary.csv',index=False)
display(raw[(raw.scenario=='clean')&(raw.window=='Mean rounds 5–24')].round(4))
print('Every scenario/task/window is saved in notebook_raw_task_summary.csv and every original observation in raw_task_metrics.csv.')
'''),md('''## Contribution diagnostics, workload and interpretation

These contribution coefficients and norms describe the updates actually applied. Smaller contributions are not automatically proof of correctly detecting malicious intent. Matched norm ratios compare with the same client's own-method clean run and can be undefined when the denominator is zero. The complete contribution-reduction and restoration episodes remain in the analysis directory.

Worker times include concurrent CPU work and different run dates for reused baselines. They describe measured resource use, not an isolated algorithm speed benchmark. The two zero-penalty audit runs add verification work but are excluded from the scientific comparison. No extra private validation reports or server reference passes are introduced by the proximal objective.'''),code('''active=CON[CON['round']>=5].copy()
active=active[(active.scenario!='attack_stops')|(active['round']<=10)]
con_summary=active.groupby(['regime','scenario','method','attack_policy','group']).agg(
    mean_coefficient=('coefficient','mean'),mean_applied_norm=('applied_norm','mean'),
    mean_matched_norm_ratio=('matched_norm_ratio','mean'),defined_ratios=('matched_norm_ratio','count')).reset_index()
con_summary.to_csv(A/'notebook_contribution_summary.csv',index=False)
display(con_summary[con_summary.group!='unchanged_honest'].round(5))
display(COST.groupby(['method','attack_policy','evidence_origin'])[['worker_wall_seconds','median_round_seconds',
    'median_training_seconds','median_defence_seconds','median_evaluation_seconds','process_lifetime_peak_rss_mib']].mean().round(3))
display(pd.Series({k:v for k,v in ACCESS.items() if isinstance(v,(int,float,bool,str))},name='Access and workload audit').to_frame())
assert ACCESS['new_client_updates']==13920
assert ACCESS['train_forward_examples']==7127040
assert ACCESS['development_task_forward_examples']==4350000
assert ACCESS['checkpoint_replay_forward_examples']==348000
if (STUDY/'decision.json').exists():
    decision=json.loads((STUDY/'decision.json').read_text())
    for name,digest in decision.get('evidence_sha256',{}).items():assert file_sha256(STUDY/name)==digest
    display(pd.Series({k:v for k,v in decision.items() if isinstance(v,(str,int,float,bool)) or v is None},name='Saved interpretation').to_frame())
'''),md('''## Reproduction and limits

`Experiment/.venv/bin/python Experiment/code/proximal_notebook.py` rebuilds this executed review from saved artifacts without training or model evaluation. The frozen protocol, source snapshot, all requests, run manifests, zero-penalty exact-replay audits and access audit are in `outputs/proximal_training/20261030_v1/`. Complete per-seed metrics, curves, losses, recovery episodes, target checks, paired effects and resource measurements are in its `analysis/` directory. Figures are saved as PNG and PDF.

The [FedProx paper](https://arxiv.org/abs/1812.06127) motivates the proximal objective. The [author implementation](https://github.com/litian96/FedProx) recommends tuning μ across several orders of magnitude; μ=1 here is one predeclared pilot value, not a universal default. The [reference optimizer](https://github.com/litian96/FedProx/blob/master/flearn/optimizer/pgd.py) anchors the penalty to the broadcast. This experiment keeps the project's original momentum to isolate the objective's effect.

A failed joint gate means this tested setting did not meet all declared requirements. It cannot rule out other coefficients, longer local training, other proximal optimizers or unrelated mechanisms. A passed gate would warrant fresh-seed validation and broader scenarios, not an immediate claim of superiority over all baselines. All observations here concern a single shared global model; personalized-model accuracy is a different research target.''')]
    notebook=nbf.v4.new_notebook(cells=cells)
    notebook.metadata.kernelspec={'display_name':'FedReTrust Experiment (Python 3.14)','language':'python','name':'fedretrust-experiment'}
    return notebook


def build():
    if json.loads((STUDY/'status.json').read_text())['status']!='complete':raise ValueError('Wait for complete training and analysis.')
    from proximal_study import verify
    verify(STUDY)
    return _execute(make_notebook(),NOTEBOOK)


if __name__=='__main__':print(build())
