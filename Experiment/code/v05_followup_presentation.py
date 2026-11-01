"""Presentation of saved v0.5 follow-up artifacts; no training or scientific edits."""
from pathlib import Path
import json
import os
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / 'outputs/reference_free/20261031_v05_followup'
NOTEBOOK = ROOT / 'code/31_fedretrust_v05_followup.ipynb'


def build():
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    cells = [md('''# 31 — FedReTrust v0.5: five-block exploratory follow-up

**24 new runs, 16 reused v0.5 pilot runs, 120 reused baseline runs: 160 matched runs in total.** This separately declared follow-up adds blocks 3–5 to the unchanged v0.5 candidate. It compares v0.5, v0.3, FedAvg and RFA on clean training, honest noise, honest stripes and persistent label flipping, keeping IID and non-IID separate.

The original pilot **failed its overall advancement criterion**. Its six focused improvement/retention checks passed, but the non-IID honest-noise margin failed; the practical RFA protection target also failed. The user then explicitly authorized this follow-up. That later decision does not turn the pilot into a pass.

All five original block outcomes informed development, and the follow-up was chosen after seeing the pilot. These are exploratory results on known MNIST, not independent confirmation. No parameter, seed, partition or previous result has been replaced. No official test images or server reference dataset are used by the candidate.

The candidate remains `m × v0.3_update + (1−m) × capped_RFA`, with sample-weighted trust mass `m`. Both estimates are computed from v0.5's current submitted updates in each round. v0.3's evidence, clipping and restoration rules are unchanged. Fully trusted harmful updates remain a known limitation, and rejected updates can retain indirect RFA influence.
'''), code(f'''from pathlib import Path
import json,sys
import numpy as np
import pandas as pd
from IPython.display import display,Image,Markdown
ROOT=Path({str(ROOT)!r});STUDY=Path({str(STUDY)!r})
sys.path.insert(0,str(ROOT/'code'))
from v05_followup import verify,records,MAIN
from experiment_core import file_sha256
assert json.loads((STUDY/'status.json').read_text())['status']=='complete'
F=verify(STUDY);assert len(records(STUDY))==24
AM=json.loads((STUDY/'analysis_manifest.json').read_text())
assert AM['execution_freeze_sha256']==file_sha256(STUDY/'execution_freeze.json')
for name,digest in AM['artifact_sha256'].items():assert file_sha256(STUDY/name)==digest,name
A=STUDY/'analysis';P=STUDY/'presentation';P.mkdir(exist_ok=True)
FIG=P/'figures';FIG.mkdir(exist_ok=True)
B=pd.read_csv(A/'block_metrics.csv');M=pd.read_csv(A/'means.csv')
E=pd.read_csv(A/'paired_effects.csv');G=pd.read_csv(A/'gates.csv');C=pd.read_csv(A/'accuracy_curves.csv')
D=json.loads((A/'decision.json').read_text())
PILOT=json.loads((ROOT/'outputs/reference_free/20261031_v05/analysis/decision.json').read_text())
assert PILOT['advancement_point_criterion_met'] is False
assert M.n.eq(5).all() and E.n.eq(5).all()
OLD='FedReTrust-Direction-loose';ORDER=['FedAvg','RFA',OLD,MAIN]
LABEL={{'FedAvg':'FedAvg','RFA':'RFA',OLD:'FedReTrust v0.3',MAIN:'FedReTrust v0.5'}}
pd.set_option('display.max_rows',100);pd.set_option('display.max_columns',16)
SPEC=[('clean','overall_mean_5_24_pp','Clean accuracy'),
 ('abrupt_shot_noise','affected_honest_event_pp','Honest noise accuracy'),
 ('gradual_stripe','affected_honest_event_pp','Honest stripe accuracy'),
 ('label_flip','attack_active_signed_damage_pp','Label-flip damage'),
 ('label_flip','overall_mean_5_24_pp','Attacked accuracy')]
'''), md('''## Five-block outcomes and uncertainty

All primary outcomes average rounds 5–24. Honest-drift accuracy measures affected clients' legitimate tasks. Damage equals each method's own matched clean accuracy minus attacked accuracy, in percentage points; absolute attacked accuracy is shown alongside it.

Intervals are nominal 95% Student-t intervals across five blocks (df=4). Paired effects subtract the comparator within each block. These small-sample, descriptive intervals do not adjust for all comparisons or for choosing this follow-up after seeing the pilot. Rounds and clients are not independent repetitions. No universal superiority or established novelty claim follows.
'''), code('''parts=[]
for scenario,metric,label in SPEC:
    part=M[(M.scenario==scenario)&(M.metric==metric)].copy();part['measurement']=label;parts.append(part)
CORE=pd.concat(parts,ignore_index=True)
for regime in ['IID','non-IID']:
    display(Markdown('**'+regime+' · five-block means**'))
    table=CORE[CORE.regime==regime].pivot(index='method',columns='measurement',values='mean')
    display(table.reindex(ORDER).rename(index=LABEL).round(3))
display(CORE[['regime','method','measurement','mean','sd','ci_low','ci_high']].round(3))
CORE.to_csv(P/'core_results.csv',index=False)
selected=pd.DataFrame(SPEC,columns=['scenario','metric','measurement'])
PAIR=E[E.comparator==OLD].merge(selected,on=['scenario','metric'])
print('v0.5 minus v0.3: positive accuracy is better; negative damage is better.')
display(PAIR[['regime','measurement','block_1','block_2','block_3','block_4','block_5',
              'mean','sd','ci_low','ci_high']].round(3))
PAIR.to_csv(P/'v05_minus_v03.csv',index=False)
'''), md('''## Original pilot and added blocks

The following split makes the decision sequence visible. The two pilot blocks and three added blocks are descriptive subsets, not independent confirmation datasets. The candidate was unchanged throughout. These subgroup means and observed ranges have no confidence intervals.
'''), code('''subsets=[]
coreblocks=B.merge(selected,on=['scenario','metric'])
for name,blocks in [('Pilot blocks 1–2',[1,2]),('Follow-up blocks 3–5',[3,4,5])]:
    subset=coreblocks[coreblocks.block.isin(blocks)]
    stats=subset.groupby(['regime','method','scenario','metric','measurement'],as_index=False).value.agg(['mean','min','max','count']).reset_index()
    stats['subset']=name;subsets.append(stats)
SUB=pd.concat(subsets,ignore_index=True)
display(SUB[SUB.method.isin([OLD,MAIN])][['subset','regime','method','measurement','mean','min','max','count']].round(3))
SUB.to_csv(P/'pilot_and_added_blocks.csv',index=False)
'''), md('''## Declared targets, assessed again without rewriting the pilot

The ten practical targets require clean/noise/stripe accuracy within 1 point of FedAvg in each regime, label-flip damage within 1 point of RFA, and damage at least 2 points below FedAvg. The six focused checks require non-IID damage to improve by at least 1 point over v0.3, absolute attacked accuracy to improve, and IID clean/noise/stripe/damage to remain within 0.5 points of v0.3.

The combined advancement-style criterion requires the six FedAvg honest-performance checks plus the six focused checks. Complete target success requires all sixteen checks. These are point-estimate assessments using previously declared design margins. The interval columns show how uncertain those assessments remain.
'''), code('''print('Preserved pilot advancement criterion:',PILOT['advancement_point_criterion_met'])
print('Preserved pilot practical targets:',PILOT['original_practical_targets_passed'],'/',PILOT['original_practical_targets_total'])
for group,part in G.groupby('gate_group'):
    print('Five-block',group,':',int(part.point_pass.sum()),'/',len(part))
honest=G[G.criterion.eq('honest_accuracy_within_1pp_FedAvg')]
focused=G[G.gate_group.eq('focused_fix')]
print('Five-block advancement-style criterion:',bool(honest.point_pass.all() and focused.point_pass.all()))
print('Five-block complete point-target success:',bool(G.point_pass.all()))
display(G[['gate_group','regime','scenario','comparator','criterion','mean','ci_low','ci_high',
           'bound_pp','direction','point_pass']].round(3))
display(Markdown('The original failed pilot gate remains unchanged. These five-block results describe the separately authorized exploratory follow-up.'))
'''), code('''import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
COLORS={'FedAvg':'#777777','RFA':'#216a9e',OLD:'#b67a27',MAIN:'#16846c'}
def save(fig,name):
    fig.savefig(FIG/(name+'.png'),dpi=160,bbox_inches='tight')
    fig.savefig(FIG/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
    display(Image(filename=str(FIG/(name+'.png'))))
fig,axes=plt.subplots(1,2,figsize=(13,5.5))
for ax,regime in zip(axes,['IID','non-IID']):
    for i,(scenario,metric,label) in enumerate(SPEC):
        row=PAIR[(PAIR.regime==regime)&(PAIR.scenario==scenario)&(PAIR.metric==metric)].iloc[0]
        vals=[row[f'block_{b}'] for b in range(1,6)]
        ax.scatter(vals,np.linspace(i-.13,i+.13,5),color='#999999',s=22,zorder=2)
        ax.errorbar(row['mean'],i,xerr=[[row['mean']-row.ci_low],[row.ci_high-row['mean']]],
                    fmt='o',color='#216a9e',capsize=4,lw=2,zorder=3)
    ax.axvline(0,color='#555555',ls=':',lw=1)
    ax.set_yticks(range(5),[x[2]+(' ↓' if x[1]=='attack_active_signed_damage_pp' else ' ↑') for x in SPEC])
    ax.invert_yaxis();ax.set_title(regime);ax.set_xlabel('v0.5 minus v0.3 (percentage points)');ax.grid(axis='x',alpha=.15)
fig.suptitle('Paired effects · grey dots = five blocks; blue = mean and nominal 95% t interval',y=1.01)
fig.tight_layout();save(fig,'paired_effects')
'''), md('''## Mean training trajectories

Curves are five-block means. Honest noise and stripe panels use affected clients' legitimate task accuracy. The label-flip panel uses all-client legitimate accuracy, with attack rounds 5–24 shaded. Curves are descriptive and cannot establish attack-stop recovery, which is outside this four-scenario follow-up.
'''), code('''fig,axes=plt.subplots(2,4,figsize=(15,7.5),sharex=True,sharey=True)
PANELS=[('clean','all_clients','Clean'),('abrupt_shot_noise','affected_honest','Honest noise'),
        ('gradual_stripe','affected_honest','Honest stripes'),('label_flip','all_clients','Label flipping')]
for i,regime in enumerate(['IID','non-IID']):
    for j,(scenario,group,title) in enumerate(PANELS):
        ax=axes[i,j]
        for method in ORDER:
            part=C[(C.regime==regime)&(C.method==method)&(C.scenario==scenario)&(C.group==group)]
            stats=part.groupby('round').accuracy_pp.agg(['mean','count']);assert stats['count'].eq(5).all()
            ax.plot(stats.index,stats['mean'],color=COLORS[method],label=LABEL[method],lw=2.3 if method==MAIN else 1.5)
        if scenario=='label_flip':ax.axvspan(5,24,color='#b94747',alpha=.08)
        if scenario!='clean':ax.axvline(5,color='#999999',ls=':',lw=.7)
        ax.set_title(regime+' · '+title);ax.set_xlim(1,24);ax.set_ylim(0,100)
        ax.set_xticks([1,5,10,15,20,24]);ax.grid(alpha=.15)
        if j==0:ax.set_ylabel('Legitimate accuracy (%)')
        if i==1:ax.set_xlabel('Round')
handles,labels=axes[0,0].get_legend_handles_labels()
fig.legend(handles,labels,loc='lower center',ncol=4,frameon=False)
fig.tight_layout(rect=(0,.06,1,1));save(fig,'all_scenarios')
'''), md('''## Attack fragility beyond the mean

This secondary diagnostic retains the original concern about fully trusted poisoning bursts. It records the accuracy at rounds 4 and 5, the largest one-round decline ending within attack rounds 5–24, and minimum attack-period accuracy. These quantities are descriptive and do not change the declared primary success criteria.
'''), code('''FRAG=pd.read_csv(A/'label_flip_fragility.csv')
display(FRAG[FRAG.regime=='non-IID'].round(3))
fig,ax=plt.subplots(figsize=(11,4.8))
for method in ORDER:
    part=C[(C.regime=='non-IID')&(C.block==5)&(C.method==method)&(C.scenario=='label_flip')&(C.group=='all_clients')].sort_values('round')
    assert len(part)==24
    ax.plot(part['round'],part.accuracy_pp,color=COLORS[method],label=LABEL[method],lw=2.3 if method==MAIN else 1.5)
ax.axvspan(5,24,color='#b94747',alpha=.08);ax.axvline(5,color='#999999',ls=':',lw=.7)
ax.set_title('Non-IID block 5 · persistent label flipping · individual block')
ax.set_xlim(1,24);ax.set_ylim(0,100);ax.set_xticks([1,5,10,15,20,24])
ax.set_xlabel('Round');ax.set_ylabel('Legitimate accuracy (%)');ax.grid(alpha=.15)
ax.legend(loc='lower right',ncol=2,frameon=False);fig.tight_layout();save(fig,'block5_attack')
'''), md('''## Verification and interpretation boundaries

This report reads completed artifacts only. Candidate source and configuration match the frozen pilot; previous scientific files, completed jobs and the failed pilot gate remain intact. Paired inputs, first-round local updates and own-method pre-event trajectories are checked across the full comparison.

All four methods here use zero trusted server reference images. The candidate requires individual client updates and stable identifiers; it does not implement secure aggregation or differential privacy. Four MNIST scenarios with full participation and fixed attacks do not establish production FLaaS performance, universal protection, recovery behavior or novelty.
'''), code('''print('New follow-up jobs:',len(records(STUDY)))
print('Combined comparison runs:',B[['method','regime','block','scenario']].drop_duplicates().shape[0])
print('Verified scientific sources:',len(F['source_sha256']))
print('Source freeze:',file_sha256(STUDY/'execution_freeze.json'))
display(pd.read_csv(A/'costs.csv').groupby('method')[['median_defence_seconds','worker_wall_seconds']].mean().rename(index=LABEL).round(4))
print('Concurrent-process costs are descriptive, not isolated performance benchmarks.')
''')]
    notebook = nbf.v4.new_notebook(cells=cells, metadata={'kernelspec': {
        'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}})
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/fedretrust_v05_followup_matplotlib')
    from aggregate_validation_notebook import _execute
    _execute(notebook, NOTEBOOK)
    from experiment_core import file_sha256, write_json
    files = [p for p in (STUDY/'presentation').rglob('*') if p.is_file()]
    write_json(STUDY/'presentation_manifest.json', {
        'builder_sha256': file_sha256(__file__), 'notebook_sha256': file_sha256(NOTEBOOK),
        'analysis_manifest_sha256': file_sha256(STUDY/'analysis_manifest.json'),
        'artifact_sha256': {str(p.relative_to(STUDY)): file_sha256(p) for p in files},
        'code_cells_executed': sum(c.cell_type=='code' for c in notebook.cells),
        'visual_review': 'pending_separate_completion_verification'})
    print(json.dumps({'notebook': str(NOTEBOOK), 'figures': 3, 'executed': True}))


if __name__ == '__main__':
    build()
