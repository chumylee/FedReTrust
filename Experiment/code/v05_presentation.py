"""Executed notebook from the completed v0.5 pilot; no training or analysis edits."""
from pathlib import Path
import json
import os

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / 'outputs/reference_free/20261031_v05'
NOTEBOOK = ROOT / 'code/30_fedretrust_v05_pilot.ipynb'


def build():
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    cells = [md('''# 30 — FedReTrust v0.5: a focused development pilot

**16 new runs, 48 reused matched comparisons, two paired blocks per regime.** This screen compares v0.5 with v0.3, FedAvg and RFA on clean training, honest noise, honest stripes and persistent label flipping. The training seeds, partitions and workloads match the original study exactly.

v0.5 keeps v0.3's norm cap, evidence and restoration rules. It changes the final update to `m × v0.3_update + (1−m) × capped_RFA`, where `m` is the sample-weighted trust mass. There is no new tuning parameter or trusted server dataset. Rejected updates can still contribute indirectly through RFA.

This is **development evidence**: the previous five-block results informed the design. It is not independent confirmation, a significance test or an established novelty claim. When every trust weight equals one, the rule is unchanged; the severe original block-5 onset failure lies outside this two-block screen.
'''), code(f'''from pathlib import Path
import json,sys
import pandas as pd
import numpy as np
from IPython.display import display,Image,Markdown
ROOT=Path({str(ROOT)!r});STUDY=Path({str(STUDY)!r})
sys.path.insert(0,str(ROOT/'code'))
from v05_pilot import verify,records,MAIN
from experiment_core import file_sha256
assert json.loads((STUDY/'status.json').read_text())['status']=='complete'
F=verify(STUDY);assert len(records(STUDY))==16
AM=json.loads((STUDY/'analysis_manifest.json').read_text())
assert AM['execution_freeze_sha256']==file_sha256(STUDY/'execution_freeze.json')
for name,digest in AM['artifact_sha256'].items():
    assert file_sha256(STUDY/name)==digest,name
A=STUDY/'analysis';P=STUDY/'presentation';P.mkdir(exist_ok=True)
FIG=P/'figures';FIG.mkdir(exist_ok=True)
B=pd.read_csv(A/'block_metrics.csv');M=pd.read_csv(A/'means.csv')
E=pd.read_csv(A/'paired_effects.csv');G=pd.read_csv(A/'gates.csv')
C=pd.read_csv(A/'accuracy_curves.csv');D=json.loads((A/'decision.json').read_text())
OLD='FedReTrust-Direction-loose'
ORDER=['FedAvg','RFA',OLD,MAIN]
LABEL={{'FedAvg':'FedAvg','RFA':'RFA',OLD:'FedReTrust v0.3',MAIN:'FedReTrust v0.5'}}
assert M.n.eq(2).all() and E.n.eq(2).all()
pd.set_option('display.max_rows',100);pd.set_option('display.max_columns',12)
'''), md('''## Declared decision criteria

The original ten practical targets require, in each regime, clean/noise/stripe accuracy within 1 percentage point of FedAvg, label-flip damage within 1 point of RFA, and damage at least 2 points below FedAvg. Six focused checks require non-IID damage to improve by at least 1 point over v0.3, absolute attacked accuracy to improve, and IID performance to stay within 0.5 points of v0.3.

**Advancement** requires the six FedAvg honest-performance checks plus all six focused checks. It can pass while an original attack target fails. **Complete pilot target success** requires all ten original targets plus all six focused checks. These are declared design margins, not industry standards.
'''), code('''headline=D['interpretation']
if D['focused_fix_checks_passed']==D['focused_fix_checks_total'] and not D['advancement_point_criterion_met']:
    headline='The focused improvement and retention checks passed, but overall advancement failed because an honest-performance margin against FedAvg was missed.'
display(Markdown('**'+headline+'**'))
print(f"Original practical targets: {D['original_practical_targets_passed']}/{D['original_practical_targets_total']}")
print(f"Focused fix checks: {D['focused_fix_checks_passed']}/{D['focused_fix_checks_total']}")
print('Advancement criterion met:',D['advancement_point_criterion_met'])
print('Complete pilot point targets met:',D['complete_pilot_point_targets_met'])
display(G[['gate_group','regime','scenario','comparator','criterion','mean','bound_pp','direction',
           'point_pass','block_1_pass','block_2_pass']].round(3))
'''), md('''## Accuracy and attack damage

All results average rounds 5–24. Honest-drift accuracy measures the affected clients' legitimate task. Attack damage equals each method's own clean-control accuracy minus attacked accuracy; the absolute attacked accuracy is included to distinguish genuine protection from a lower clean starting point.

The two observed block values and ranges describe variation only. They are not confidence intervals. Rounds and clients do not create additional independent repetitions.
'''), code('''SPEC=[('clean','overall_mean_5_24_pp','Clean accuracy (%)'),
 ('abrupt_shot_noise','affected_honest_event_pp','Honest noise accuracy (%)'),
 ('gradual_stripe','affected_honest_event_pp','Honest stripe accuracy (%)'),
 ('label_flip','attack_active_signed_damage_pp','Label-flip damage (pp)'),
 ('label_flip','overall_mean_5_24_pp','Attacked accuracy (%)')]
parts=[]
for scenario,metric,label in SPEC:
    part=M[(M.scenario==scenario)&(M.metric==metric)].copy()
    part['measurement']=label;parts.append(part)
CORE=pd.concat(parts,ignore_index=True)
for regime in ['IID','non-IID']:
    display(Markdown('**'+regime+' — two-block means**'))
    table=CORE[CORE.regime==regime].pivot(index='method',columns='measurement',values='mean')
    display(table.reindex(ORDER).rename(index=LABEL).round(3))
print('Both underlying block values:')
display(CORE[['regime','method','measurement','block_1','block_2','mean','min','max']].round(3))
CORE.to_csv(P/'core_results.csv',index=False)
print('Paired v0.5 minus v0.3 effects (positive accuracy is better; negative damage is better):')
selected=pd.DataFrame([(s,m) for s,m,_ in SPEC],columns=['scenario','metric'])
display(E[E.comparator==OLD].merge(selected,on=['scenario','metric'])[
    ['regime','scenario','metric','block_1','block_2','mean','min','max']].round(3))
'''), code('''import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
COLORS={'FedAvg':'#777777','RFA':'#216a9e',OLD:'#b67a27',MAIN:'#16846c'}
def save(fig,name):
    fig.savefig(FIG/(name+'.png'),dpi=160,bbox_inches='tight')
    fig.savefig(FIG/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
    display(Image(filename=str(FIG/(name+'.png'))))
fig,axes=plt.subplots(2,4,figsize=(15,7.5))
for row,regime in enumerate(['IID','non-IID']):
    for col,(scenario,metric,title) in enumerate(SPEC[:4]):
        ax=axes[row,col]
        for i,method in enumerate(ORDER):
            vals=B[(B.regime==regime)&(B.method==method)&(B.scenario==scenario)&(B.metric==metric)].sort_values('block').value.to_numpy()
            assert len(vals)==2
            ax.plot([i,i],[vals.min(),vals.max()],color=COLORS[method],lw=2)
            ax.scatter([i-.045,i+.045],vals,color=COLORS[method],s=28,zorder=3)
            ax.plot([i-.14,i+.14],[vals.mean(),vals.mean()],color=COLORS[method],lw=2)
        ax.set_xticks(range(4),['FedAvg','RFA','v0.3','v0.5'])
        ax.set_title(title+' '+('↓' if col==3 else '↑'))
        ax.set_ylabel(regime+' · '+('pp' if col==3 else '%'));ax.grid(axis='y',alpha=.2)
        if col==3:ax.axhline(0,color='#aaaaaa',lw=.6)
fig.suptitle('Two paired development blocks · dots = block values; bars = observed range; ticks = mean',y=.995)
fig.tight_layout(rect=(0,0,1,.96));save(fig,'core_results')
'''), md('''## Training trajectories

Curves show the mean of the same two blocks. The shaded interval marks persistent label flipping (rounds 5–24); honest stripes also begin at round 5. These descriptive curves cannot establish recovery after an attack stops, which this pilot does not test.
'''), code('''fig,axes=plt.subplots(2,2,figsize=(12,8),sharex=True,sharey=True)
for row,regime in enumerate(['IID','non-IID']):
    for col,(scenario,group,title) in enumerate([
        ('gradual_stripe','affected_honest','Honest stripes'),('label_flip','all_clients','Persistent label flipping')]):
        ax=axes[row,col]
        for method in ORDER:
            part=C[(C.regime==regime)&(C.method==method)&(C.scenario==scenario)&(C.group==group)]
            stats=part.groupby('round').accuracy_pp.agg(['mean','count']);assert stats['count'].eq(2).all()
            ax.plot(stats.index,stats['mean'],color=COLORS[method],label=LABEL[method],lw=2.4 if method==MAIN else 1.6)
        if scenario=='label_flip':ax.axvspan(5,24,color='#b94747',alpha=.08)
        ax.axvline(5,color='#999999',ls=':',lw=.7)
        ax.set_title(regime+' · '+title);ax.set_ylim(0,100);ax.set_xlim(1,24)
        ax.set_ylabel('Legitimate accuracy (%)');ax.set_xlabel('Round');ax.grid(alpha=.15)
handles,labels=axes[0,0].get_legend_handles_labels()
fig.legend(handles,labels,loc='lower center',ncol=4,frameon=False)
fig.tight_layout(rect=(0,.055,1,1));save(fig,'trajectories')
'''), md('''## How much the new rule activated

The robust-complement mass is the share assigned to the capped RFA estimate before clipping. It is computed from submitted updates and trust weights; evaluation results do not enter the rule. These diagnostics describe the candidate's own training trajectories, not proof that individual rejected clients were malicious.
'''), code('''ROUND=pd.read_csv(A/'round_diagnostics.csv')
activation=[]
for row in ROUND[(ROUND.method==MAIN)&(ROUND['round']>=5)].itertuples():
    values=json.loads(row.values_json)
    activation.append({'regime':row.regime,'block':row.block,'scenario':row.scenario,
      'robust_complement_mass':values['robust_complement_mass'],
      'negligible_complement':values['robust_complement_mass']<=1e-12,
      'fallback':values['bounded_rfa_fallback']})
ACT=pd.DataFrame(activation).groupby(['regime','block','scenario'],as_index=False).agg(
    mean_robust_complement=('robust_complement_mass','mean'),
    fraction_negligible_complement=('negligible_complement','mean'),
    fallback_fraction=('fallback','mean'))
display(ACT.round(4));ACT.to_csv(P/'complement_activation.csv',index=False)
'''), md('''## Verification and limits

All candidate and reused runs have their inputs, initial updates and own-method pre-event trajectories checked. Original scientific sources and results remain unchanged. All methods in this screen use zero trusted server images. v0.5 still exposes individual updates to the server and inherits v0.3's honest-majority/identity assumptions; it is not a privacy protocol or a guarantee against malicious clients.

Blocks 3–5, other attacks, recovery, partial participation and other datasets remain outside this pilot. A successful screen would justify a further declared evaluation; a failed screen remains a useful negative result. No further training is launched by this notebook.
'''), code('''display(pd.DataFrame([{k:D[k] for k in ['new_runs','reused_runs','matched_input_pairs',
 'exact_first_round_updates','exact_own_method_pre_event_trajectories']}]))
COST=pd.read_csv(A/'costs.csv')
display(COST.groupby('method')[['median_defence_seconds','worker_wall_seconds']].mean().rename(index=LABEL).round(4))
print('Costs are from concurrent runs, not isolated timing benchmarks.')
print('Frozen source files:',len(F['source_sha256']))
print('Freeze SHA256:',file_sha256(STUDY/'execution_freeze.json'))
''')]
    notebook = nbf.v4.new_notebook(cells=cells, metadata={
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python'}})
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/fedretrust_v05_matplotlib')
    from aggregate_validation_notebook import _execute
    _execute(notebook, NOTEBOOK)
    from experiment_core import file_sha256, write_json
    files = [p for p in (STUDY / 'presentation').rglob('*') if p.is_file()]
    write_json(STUDY / 'presentation_manifest.json', {
        'builder_sha256': file_sha256(__file__), 'notebook_sha256': file_sha256(NOTEBOOK),
        'analysis_manifest_sha256': file_sha256(STUDY / 'analysis_manifest.json'),
        'artifact_sha256': {str(p.relative_to(STUDY)): file_sha256(p) for p in files},
        'code_cells_executed': sum(c.cell_type == 'code' for c in notebook.cells),
        'visual_review': 'pending_separate_completion_verification'})
    print(json.dumps({'notebook': str(NOTEBOOK), 'figures': 2, 'executed': True}))


if __name__ == '__main__':
    build()
