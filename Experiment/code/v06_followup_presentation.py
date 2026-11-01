"""Saved-artifact presentation for the unchanged v0.6 five-block follow-up."""
from pathlib import Path
import json
import os
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / 'outputs/reference_free/20261031_v06_followup'
NOTEBOOK = ROOT / 'code/33_fedretrust_v06_followup.ipynb'


def build():
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    cells = [md('''# 33 — FedReTrust v0.6: completed five-block development comparison

**24 new candidate runs and 176 matching saved comparisons: 200 trajectories in total.** The unchanged v0.6 candidate now covers all five existing paired blocks. The additional blocks are 1, 2 and 4; the selected pilot blocks 3 and 5 are reused exactly. Each method is compared on both IID and non-IID data with clean training, honest noise, honest stripes and persistent label flipping.

All original block outcomes had already informed development. This is an exploratory follow-up, not untouched confirmation. The failed 25/27-check pilot is preserved. The same aggregation rule operates in both regimes; no parameters were changed for these additional runs.

v0.6 tracks a vector for each client, initially zero. It clips the change from that tracked vector to the current update using a radius of twice the current cohort's median innovation norm. It then applies the v0.5 rule to the filtered updates. History commits only after a valid aggregate. An effectively zero median bypasses this extra clipping; absent clients retain their history. No trusted warmup, attack schedule, server reference data or evaluation result enters the rule.

The mechanism is inspired by [Clip21](https://arxiv.org/abs/2305.18929v1) and [Byz-Clip21](https://arxiv.org/html/2603.23472v2). This server-side adaptation to local-SGD updates and v0.5 is not their algorithm or theorem. The adaptive radius and multiplier 2 are declared engineering choices. History can become poisoned, genuine drift can be delayed, and the original directional gate can still suppress minority clients.
'''), code(f'''from pathlib import Path
import json,sys
import numpy as np
import pandas as pd
from IPython.display import display,Image,Markdown
ROOT=Path({str(ROOT)!r});STUDY=Path({str(STUDY)!r})
sys.path.insert(0,str(ROOT/'code'))
from v06_followup import verify,records,MAIN
from experiment_core import file_sha256
assert json.loads((STUDY/'status.json').read_text())['status']=='complete'
F=verify(STUDY);assert len(records(STUDY))==24
AM=json.loads((STUDY/'analysis_manifest.json').read_text())
assert AM['execution_freeze_sha256']==file_sha256(STUDY/'execution_freeze.json')
for name,digest in AM['artifact_sha256'].items():assert file_sha256(STUDY/name)==digest,name
A=STUDY/'analysis';P=STUDY/'presentation';P.mkdir(exist_ok=True)
FIG=P/'figures';FIG.mkdir(exist_ok=True)
B=pd.read_csv(A/'block_metrics.csv');M=pd.read_csv(A/'means.csv');E=pd.read_csv(A/'paired_effects.csv')
G=pd.read_csv(A/'gates.csv');C=pd.read_csv(A/'accuracy_curves.csv');D=json.loads((A/'decision.json').read_text())
V03='FedReTrust-Direction-loose';V05='FedReTrust-v05'
ORDER=['FedAvg','RFA',V03,V05,MAIN]
LABEL={{'FedAvg':'FedAvg','RFA':'RFA',V03:'FedReTrust v0.3',V05:'FedReTrust v0.5',MAIN:'FedReTrust v0.6'}}
assert M.n.eq(5).all() and E.n.eq(5).all()
assert 'ci_low' in M.columns and 'ci_low' in E.columns
pd.set_option('display.max_rows',120);pd.set_option('display.max_columns',15)
SPEC=[('clean','overall_mean_5_24_pp','Clean accuracy'),
 ('abrupt_shot_noise','affected_honest_event_pp','Honest noise accuracy'),
 ('gradual_stripe','affected_honest_event_pp','Honest stripe accuracy'),
 ('label_flip','attack_active_signed_damage_pp','Label-flip damage'),
 ('label_flip','overall_mean_5_24_pp','Attacked accuracy')]
'''), md('''## Accuracy, protection and individual blocks

Primary outcomes average rounds 5–24. Honest drift uses affected clients' legitimate tasks. Damage is each method's own matched clean-control accuracy minus its attacked accuracy, in percentage points. Absolute attacked accuracy is shown so a lower clean reference cannot masquerade as protection.

All five paired block values and observed ranges are reported. Nominal two-sided 95% t intervals (df4) assume independent, approximately normal block differences. They can be unstable at n=5 and have no simultaneous coverage; these intervals are exploratory because development used known blocks. Client rounds are not independent repetitions. Comparisons below use blocks 1–5 for every method.
'''), code('''parts=[]
for scenario,metric,label in SPEC:
    part=M[(M.scenario==scenario)&(M.metric==metric)].copy();part['measurement']=label;parts.append(part)
CORE=pd.concat(parts,ignore_index=True)
for regime in ['IID','non-IID']:
    display(Markdown('**'+regime+' · all five matched blocks**'))
    table=CORE[CORE.regime==regime].pivot(index='method',columns='measurement',values='mean')
    display(table.reindex(ORDER).rename(index=LABEL).round(3))
display(CORE[['regime','method','measurement','block_1','block_2','block_3','block_4','block_5','mean','ci_low','ci_high','min','max']].round(3))
CORE.to_csv(P/'core_results.csv',index=False)
selected=pd.DataFrame(SPEC,columns=['scenario','metric','measurement'])
PAIR=E[E.comparator==V05].merge(selected,on=['scenario','metric'])
print('v0.6 minus v0.5: positive accuracy is better; negative damage is better.')
display(PAIR[['regime','measurement','block_1','block_2','block_3','block_4','block_5','mean','ci_low','ci_high']].round(3))
PAIR.to_csv(P/'v06_minus_v05.csv',index=False)
'''), md('''## Fixed success and retention checks

Ten original practical checks assess mean clean/noise/stripe accuracy against FedAvg (at least FedAvg minus 1 point), attack damage against RFA (at most RFA plus 1 point), and damage reduction versus FedAvg (at least 2 points), separately by regime.

Twenty-five IID checks require each of the five blocks' clean/noise/stripe/attacked accuracy to remain within 0.5 points below v0.5 and damage within 0.5 points above it. Sixteen non-IID checks require a five-block mean damage reduction of at least 1 point, better absolute attacked accuracy in every block, at least 5 points better absolute onset accuracy and own-clean-relative onset damage in block 5, and no worsening of those two onset measures in blocks 1–4.

These margins, including the new 5-point onset hurdle, are development choices. Complete five-block point success requires all 51 checks. This expands coverage while preserving the original margins; it cannot reverse the separate pilot outcome. A partial onset improvement does not imply that balance, IID retention, other attacks or production safety have been established.
'''), code('''for group,part in G.groupby('gate_group'):
    print(group,':',int(part.point_pass.sum()),'/',len(part))
print('Five-block point targets met:',D['five_block_all_point_targets_met'])
print('Original pilot met all point targets:',D['pilot_complete_point_targets_met'])
print('IID retention criteria met:',D['IID_retention_point_criteria_met'])
print('Non-IID focused criteria met:',D['nonIID_focused_point_criteria_met'])
display(G[['gate_group','regime','scenario','criterion','comparator','gate_block','assessment_unit',
           'checked_difference_pp','bound_pp','direction','point_pass']].round(3))
'''), md('''## Attack onset and subsequent bursts

Onset damage compares attacked round-5 accuracy with the same method's clean round-5 accuracy. The raw round-4-to-5 fall is also retained, but alone can look better merely because pre-attack learning became worse. These are distinct measurements.
'''), code('''ON=pd.read_csv(A/'onset_fragility.csv')
display(ON[ON.regime=='non-IID'].round(3))
OE=pd.read_csv(A/'onset_paired_effects.csv')
display(OE[OE.comparator==V05].round(3))
'''), code('''import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
COLORS={'FedAvg':'#777777','RFA':'#216a9e',V03:'#b67a27',V05:'#16846c',MAIN:'#804db2'}
def save(fig,name):
    fig.savefig(FIG/(name+'.png'),dpi=160,bbox_inches='tight')
    fig.savefig(FIG/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
    display(Image(filename=str(FIG/(name+'.png'))))
fig,axes=plt.subplots(2,4,figsize=(16,7.5))
for i,regime in enumerate(['IID','non-IID']):
    for j,(scenario,metric,title) in enumerate(SPEC[:4]):
        ax=axes[i,j]
        for k,method in enumerate(ORDER):
            vals=B[(B.regime==regime)&(B.method==method)&(B.scenario==scenario)&(B.metric==metric)].sort_values('block').value.to_numpy()
            assert len(vals)==5
            ax.plot([k,k],[vals.min(),vals.max()],color=COLORS[method],lw=2)
            ax.scatter(k+np.linspace(-.10,.10,5),vals,color=COLORS[method],s=25,zorder=3)
            ax.plot([k-.14,k+.14],[vals.mean(),vals.mean()],color=COLORS[method],lw=2)
        ax.set_xticks(range(5),['FedAvg','RFA','v0.3','v0.5','v0.6'])
        ax.set_title(title+(' ↓' if j==3 else ' ↑'));ax.grid(axis='y',alpha=.18)
        ax.set_ylabel(regime+' · '+('pp' if j==3 else '%'))
        if j==3:ax.axhline(0,color='#999999',lw=.7)
fig.suptitle('Five development blocks · dots = individual blocks; tick = mean; line = observed range',y=.995)
fig.tight_layout(rect=(0,0,1,.96));save(fig,'core_results')
fig,axes=plt.subplots(2,5,figsize=(20,8),sharex=True,sharey=True)
for i,regime in enumerate(['IID','non-IID']):
    for j,block in enumerate([1,2,3,4,5]):
        ax=axes[i,j]
        for method in ORDER:
            part=C[(C.regime==regime)&(C.block==block)&(C.method==method)&(C.scenario=='label_flip')&(C.group=='all_clients')].sort_values('round')
            assert len(part)==24
            ax.plot(part['round'],part.accuracy_pp,color=COLORS[method],label=LABEL[method],lw=2.3 if method==MAIN else 1.4)
        ax.axvspan(5,24,color='#b94747',alpha=.08);ax.axvline(5,color='#999999',ls=':',lw=.7)
        ax.set_title(regime+' · block '+str(block));ax.set_xlim(1,24);ax.set_ylim(0,100)
        ax.set_xticks([1,5,10,15,20,24]);ax.set_xlabel('Round');ax.set_ylabel('Legitimate accuracy (%)');ax.grid(alpha=.15)
handles,labels=axes[0,0].get_legend_handles_labels()
fig.legend(handles,labels,loc='lower center',ncol=3,frameon=False)
fig.suptitle('Persistent label flipping · all five development blocks',y=.995)
fig.tight_layout(rect=(0,.09,1,.97));save(fig,'attack_trajectories')
fig,axes=plt.subplots(2,5,figsize=(20,7.5))
REFS=[V05,'RFA','FedAvg']
for i,regime in enumerate(['IID','non-IID']):
    for j,(scenario,metric,title) in enumerate(SPEC):
        ax=axes[i,j]
        for k,other in enumerate(REFS):
            row=E[(E.regime==regime)&(E.scenario==scenario)&(E.metric==metric)&(E.comparator==other)].iloc[0]
            vals=np.array([row['block_'+str(b)] for b in range(1,6)])
            ax.scatter(vals,k+np.linspace(-.1,.1,5),color=COLORS[other],alpha=.6,s=22)
            ax.plot([row.ci_low,row.ci_high],[k,k],color=COLORS[other],lw=2)
            ax.scatter(row['mean'],k,color=COLORS[other],marker='D',s=38)
        ax.axvline(0,color='#777777',ls=':',lw=1)
        ax.set_yticks(range(3),['v0.5','RFA','FedAvg']);ax.set_ylim(-.5,2.5)
        ax.set_title(regime+' · '+title+(' ↓' if j==3 else ' ↑'))
        ax.set_xlabel('v0.6 minus comparator (pp)');ax.grid(axis='x',alpha=.15)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
fig.suptitle('Paired differences · dots = five blocks; diamond = mean; line = nominal 95% t interval (df4)',y=.995)
fig.tight_layout(rect=(0,0,1,.96));save(fig,'paired_effects')

'''), md('''## History accounting, cost and limits

v0.6's applied update contains both current raw updates and historical vectors. Its current raw coefficients alone therefore do not describe the complete update. The new recorder separately measures current, historical and full filtered contributions and reconstructs the actual model update including history. Direct comparisons of raw-current coefficient mass with older methods must respect this difference.

The defense receives submitted updates, counts, identifiers, broadcast and round only. Offline evaluators can read simulator truth; the defense cannot. All 200 matched runs use the same local-training and evaluation workloads. The new history storage and aggregation costs are reported separately; concurrent wall times are descriptive rather than isolated benchmarks.

Known limits include poisoned or stale history, slower legitimate adaptation, the zero-innovation fallback and the original directional gate's risk of suppressing honest minority updates. This screen covers fixed persistent attacks and full participation on MNIST. It does not establish attack-stop recovery, adaptive-attack protection, secure aggregation, differential privacy or novelty.
'''), code('''print('New candidate jobs:',len(records(STUDY)))
print('Compared trajectories:',B[['method','regime','block','scenario']].drop_duplicates().shape[0])
print('Scientific sources verified:',len(F['source_sha256']))
print('Source freeze:',file_sha256(STUDY/'execution_freeze.json'))
display(pd.read_csv(A/'costs.csv').groupby('method')[['median_defence_seconds','worker_wall_seconds']].mean().rename(index=LABEL).round(4))
H=pd.read_csv(A/'history_activation.csv')
storage=H.groupby('regime').agg(history_tensor_bytes=('history_tensor_bytes','max'),
    fraction_rounds_with_history_term=('historical_term_present','mean'),
    mean_clipped_clients_per_round=('clipped_clients','mean'),
    maximum_affine_error=('affine_reconstruction_max_abs_error','max'),
    maximum_model_error=('reconstruction_max_abs_error','max'))
storage['history_tensor_MiB']=storage.history_tensor_bytes/(1024**2)
print('Persistent history tensor payload; Python object and temporary workspace overhead excluded:')
display(storage)
print('Full per-round history and innovation diagnostics remain in analysis/round_diagnostics.csv and raw aggregation artifacts.')
''')]
    notebook=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec': {
        'display_name':'Python 3','language':'python','name':'python3'}})
    os.environ.setdefault('MPLCONFIGDIR','/tmp/fedretrust_v06_matplotlib')
    from aggregate_validation_notebook import _execute
    _execute(notebook,NOTEBOOK)
    from experiment_core import file_sha256,write_json
    files=[p for p in (STUDY/'presentation').rglob('*') if p.is_file()]
    write_json(STUDY/'presentation_manifest.json',{
        'builder_sha256':file_sha256(__file__),'notebook_sha256':file_sha256(NOTEBOOK),
        'analysis_manifest_sha256':file_sha256(STUDY/'analysis_manifest.json'),
        'artifact_sha256':{str(p.relative_to(STUDY)):file_sha256(p) for p in files},
        'code_cells_executed':sum(c.cell_type=='code' for c in notebook.cells),
        'visual_review':'pending_separate_completion_verification'})
    print(json.dumps({'notebook':str(NOTEBOOK),'figures':3,'executed':True}))


if __name__=='__main__':
    build()
