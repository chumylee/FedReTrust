"""Execute a read-only review of the separately declared v0.3 pilot."""
from pathlib import Path
import json
import os

import nbformat as nbf

from research_runs import ROOT


def build():
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    target = json.loads((ROOT / "outputs/reference_free/20261030_v03/analysis/target_check.json").read_text())
    count = sum(row["target_met_descriptively"] for row in target["comparisons"])
    outcome = f"**Outcome: {count}/6 core mean targets met descriptively. The combined target was {'met' if count == 6 else 'not met'}.** Two development seeds do not establish statistical superiority."
    cells = [
        md(f"""# 19 — Direction-based, reference-free FedReTrust

{outcome}

This is the second reference-free development attempt following supervisor discussion. The aim is to retain useful honest drift while improving attack protection, especially under non-IID data. **Viceroy is included.** All comparisons use matched development runs; original v0.1, v0.2 and frozen final results are preserved.

The protocol declared three candidates before their training. One configuration is selected jointly across IID and non-IID using the first seed. The selected configuration is then evaluated on all ten scenarios on that seed and the three core scenarios on a second seed. Both seeds and the evaluation data have been used previously: these are exploratory results, not independent confirmatory evidence. No official-test evaluation is performed."""),
        code("""from pathlib import Path
import sys, json
import numpy as np
import pandas as pd
from IPython.display import display, Image
CODE = Path.cwd() if (Path.cwd()/'directional_study.py').exists() else Path.cwd()/'Experiment/code'
sys.path.insert(0, str(CODE.resolve()))
from directional_study import STUDY, BASELINES, CANDIDATES, PREVIOUS_METHOD, SCREEN_SEED, CHECK_SEED, verify
from research_runs import ROOT
from research_analysis import cross_method_pair
P = verify(STUDY)
S = json.loads((STUDY/'selection.json').read_text())
STATUS = json.loads((STUDY/'status.json').read_text())
assert STATUS['status'] == 'complete'
SELECTED = S['selected']
KEEP = [*BASELINES, PREVIOUS_METHOD, SELECTED]
M = pd.read_csv(STUDY/'analysis/seed_metrics.csv')
records = json.loads((STUDY/'result_manifest.json').read_text())['records']
lookup = {(r['job']['method'],r['job']['regime'],r['job']['scenario'],r['job']['seed']):r for r in records}
for r in records:
    j = r['job']
    cross_method_pair(r, lookup[('FedAvg',j['regime'],j['scenario'],j['seed'])])
audit = json.loads((STUDY/'candidate_access_audit.json').read_text())
assert audit['candidate_runs'] == 38 and audit['candidate_rounds'] == 912
assert len(records) == 194 and audit['trusted_examples'] == 0
pd.set_option('display.max_rows', 180)
pd.set_option('display.max_columns', 12)
print('Selected candidate:', SELECTED)
print('42 new runs: 38 candidate runs and 4 Viceroy controls; 152 verified reused runs.')
print('All 194 cross-method provenance checks passed. Zero reference use in 912 candidate aggregation rounds.')
def comparison(metric, seed, scenarios=None):
    x = M[(M.metric == metric) & (M.seed == seed) & M.method.isin(KEEP)]
    if scenarios is not None: x = x[x.scenario.isin(scenarios)]
    return x.pivot(index=['regime','scenario'], columns='method', values='value').reindex(columns=KEEP).round(2)
"""),
        md(r"""## Change from v0.2

The earlier prototype required both unusual distance and poor alignment to restrict an update. Non-IID label-flip updates often opposed the anchor but stayed too close to trigger the distance gate. v0.3 removes that gate. This also risks restricting useful honest non-IID updates, so clean and honest-task performance remain explicit targets.

For raw updates $\Delta_i$ and sample shares $p_i$:

1. Bound update norms at $C=6\operatorname{median}_i\|\Delta_i\|$, giving $\bar\Delta_i=c_i\Delta_i$.
2. Form an anchor $a$ with three RFA iterations. The loose and strict candidates use capped updates; the unit-direction candidate uses $\Delta_i/\max(\|\Delta_i\|,\epsilon)$, preventing magnitude from affecting its directional anchor.
3. Compute $u_i=\cos(\Delta_i,a)$ and $m=\operatorname{median}(u_i)$. Evidence is $g_i=[u_i/(\tau m)]_0^1$, with $\tau=.25$ (loose/unit) or $.5$ (strict). Negative alignment gives zero evidence regardless of distance. Full weight only requires modest positive alignment.
4. Use $q_i=\min(g_i,q_i^{old}+.5)$, then aggregate $\sum_i(p_iq_i/\sum_jp_jq_j)\bar\Delta_i$. Restriction is immediate; two fully supportive participations can restore a zero weight. Missing clients retain memory and each new submission is reassessed.

An undefined anchor or nonpositive median cosine falls back to bounded RFA on parameter updates, preserving memory. A unit vector is never applied as a model delta. Clipping follows normalization so normalization cannot undo the norm cap.

The aggregator receives client parameters, authentic sample counts, broadcast parameters and a round number. It receives no data, labels, evaluator feedback, scenario identity or attacker identity. The legacy job field `root_steps=8` is ignored by its scalar-only factory. The 300 reserved images remain excluded from clients to preserve matched selections; they are unused by this method. FLTrust and original FedReTrust retain their original 300-image access.

This combines the existing [RFA solver](https://arxiv.org/abs/1912.13445), [FLTrust-inspired directional weighting](https://arxiv.org/abs/2012.13995), and the previous prototype's clipping/restoration rules. It does not reproduce FLTrust or inherit its guarantees. Novelty is unestablished."""),
        code("""print('All declared settings; one selected configuration is used in both regimes.')
display(pd.DataFrame(P['candidate_configs']).T)
screen_rows = []
for candidate in S['candidates']:
    for detail in candidate['details']:
        screen_rows.append({'candidate':candidate['candidate'], **detail,
                           'largest_shortfall_pp':candidate['largest_violation_pp'],
                           'all_screen_targets_met':candidate['all_screen_guards_pass']})
display(pd.DataFrame(screen_rows).drop(columns='violations_pp').round(3))
print('Selection minimizes the largest shortfall against the BEST of five baselines, with no 1pp allowance.')
print('Selection is a development decision; it is not a success finding.')
"""),
        md("""## Two-seed core results and target checks

Accuracy is the mean over rounds 5–24. Honest stripe accuracy measures affected clients on their legitimate changed task. Attack damage is the loss against the same method's paired clean trajectory, in percentage points. Lower damage is better. Damage must be interpreted alongside clean and absolute attacked accuracy; lowering clean performance is not useful protection.

The six targets use the best included baseline for each metric and regime. Matching or exceeding all six means is the descriptive core target; beating every method across all conditions is a stronger requirement. These two-seed means do not establish statistical superiority."""),
        code("""means = pd.read_csv(STUDY/'analysis/two_seed_core_means.csv')
display(means.set_index(['regime','scenario','metric']).reindex(columns=KEEP).round(2))
target = json.loads((STUDY/'analysis/target_check.json').read_text())
T = pd.DataFrame(target['comparisons'])
display(T.drop(columns='metric').round(3))
print('ALL SIX CORE TARGETS MET:', target['core_mean_targets_all_met'])
print('Positive advantage means better than the best baseline; zero means equal.')
for seed in (SCREEN_SEED, CHECK_SEED):
    print('Individual development seed', seed)
    x = M[(M.seed == seed) & M.method.isin(KEEP) & (
        ((M.scenario == 'clean') & (M.metric == 'overall_mean_5_24_pp')) |
        ((M.scenario == 'gradual_stripe') & (M.metric == 'affected_honest_event_pp')) |
        ((M.scenario == 'label_flip') & (M.metric == 'attack_active_signed_damage_pp')))]
    display(x.pivot(index=['regime','scenario','metric'],columns='method',values='value').reindex(columns=KEEP).round(2))
"""),
        code("""import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
figure_dir = STUDY/'analysis/figures'; figure_dir.mkdir(exist_ok=True)
labels = {'FedAvg':'FedAvg','RFA':'RFA','FLTrust':'FLTrust','Viceroy':'Viceroy','FedReTrust':'Original FedReTrust',PREVIOUS_METHOD:'Previous prototype',SELECTED:'New prototype'}
colors = dict(zip(KEEP, ['#777777','#2166ac','#d6604d','#8c6bb1','#b38b00','#1b9e77','#101010']))
fig, axes = plt.subplots(1,2,figsize=(12,5.8),sharey=True)
for ax, regime in zip(axes,['IID','non-IID']):
    for method in KEEP:
        xs, ys = [], []
        for seed, marker in [(SCREEN_SEED,'o'),(CHECK_SEED,'^')]:
            sub = M[(M.method == method) & (M.regime == regime) & (M.seed == seed)]
            x = sub[(sub.scenario == 'label_flip') & (sub.metric == 'attack_active_signed_damage_pp')].value.item()
            y = sub[(sub.scenario == 'gradual_stripe') & (sub.metric == 'affected_honest_event_pp')].value.item()
            xs.append(x); ys.append(y)
            ax.scatter(x,y,marker=marker,color=colors[method],s=65 if method==SELECTED else 45,zorder=5 if method==SELECTED else 3)
        ax.plot(xs,ys,color=colors[method],alpha=.35,lw=1)
    ax.set_title(regime)
    ax.set_xlabel('Label-flip damage (pp; lower is better)')
    ax.grid(alpha=.2)
axes[0].set_ylabel('Affected-client stripe accuracy (%; higher is better)')
fig.legend(handles=[Line2D([0],[0],marker='o',color=colors[m],linestyle='',label=labels[m]) for m in KEEP],loc='lower center',ncol=4,frameon=False)
axes[1].legend(handles=[Line2D([0],[0],marker=k,color='black',linestyle='',label=str(s)) for s,k in [(SCREEN_SEED,'o'),(CHECK_SEED,'^')]],title='Development seed',frameon=False,loc='lower right')
fig.suptitle('Second reference-free attempt: protection and honest drift')
fig.tight_layout(rect=(0,.14,1,.94))
fig.savefig(figure_dir/'protection_drift.png',dpi=180)
fig.savefig(figure_dir/'protection_drift.pdf')
plt.close(fig)
display(Image(filename=str(figure_dir/'protection_drift.png')))
"""),
        md("""## Wider scenario panel — first development seed only

All ten scenarios are included below, including clean training and the matched mixed-drift control. Mixed attack damage is measured against each method's mixed-drift-only control. Gaussian perturbations in this workload were weak; near-zero damage alone is limited evidence of protection. Round 24 is reported separately from the main period average."""),
        code("""for metric, title in [
    ('overall_mean_5_24_pp','Overall legitimate-task accuracy, mean rounds 5–24 (%)'),
    ('overall_round24_pp','Overall legitimate-task accuracy, final round (%)'),
    ('affected_honest_event_pp','Affected honest clients during change (%)'),
    ('affected_honest_return_11_24_pp','Formerly changed honest clients after change stops (%)'),
    ('attack_active_signed_damage_pp','Attack-active damage (pp)'),
    ('second_attack_signed_damage_pp','Second recurring attack damage (pp)'),
    ('attack_tail_damage_21_24_pp','Remaining damage in rounds 21–24 (pp)')]:
    print(title)
    display(comparison(metric, SCREEN_SEED))
panel = []
for (regime, scenario, metric), group in M[(M.seed == SCREEN_SEED) & M.metric.isin(['affected_honest_event_pp','attack_active_signed_damage_pp'])].groupby(['regime','scenario','metric']):
    base = group[group.method.isin(BASELINES)]
    direction = -1 if metric == 'attack_active_signed_damage_pp' else 1
    best = base.loc[(base.value * direction).idxmax()]
    value = group[group.method == SELECTED].value.item()
    panel.append({'regime':regime,'scenario':scenario,'metric':metric,'candidate_pp':value,
                  'best_baseline':best.method,'target_pp':best.value,'advantage_pp':direction*(value-best.value)})
F = pd.DataFrame(panel)
display(F.round(3))
F.to_csv(STUDY/'analysis/full_panel_target_check.csv',index=False)
print('All wider honest/attack targets met on first seed:', bool((F.advantage_pp >= 0).all()))
"""),
        md("""## Recovery, influence and compute

Model recovery requires accuracy within one percentage point of the paired reference for three consecutive post-event rounds. The reported delay is to confirmation. Already-good and censored episodes are different outcomes; absence of observed recovery within this short window is not proof of permanent failure. Weight restoration is a separate mechanism, not proof of recovered accuracy.

Simulator truth is used only in the offline group diagnostics below. Policy weights precede normalization and clipping and are not final contribution shares. Costs are descriptive because baseline timings were collected earlier and workers run concurrently. The raw cost counter is not a valid way to infer FLTrust reference use; the explicit reference-image column records its actual requirement."""),
        code("""E = pd.read_csv(STUDY/'analysis/model_episodes.csv')
display(E[(E.seed == SCREEN_SEED) & E.method.isin(KEEP)][['regime','scenario','method','intent','episode','status','observed_restoration_delay_rounds']].sort_values(['regime','scenario','method','episode']))
rows = []
for result in records:
    j = result['job']
    if j['method'] != SELECTED or j['seed'] != SCREEN_SEED: continue
    ds = json.loads((ROOT/result['run_path']/'aggregation_diagnostics.json').read_text())
    for group, ids in [('clients_0_1_2',[0,1,2]),('clients_3_to_9',list(range(3,10)))]:
        frame = pd.DataFrame([d['values']['client_metrics'][str(i)] for d in ds if d['round'] >= 5 for i in ids])
        last = ds[-1]['values']['client_metrics']
        rows.append({'regime':j['regime'],'scenario':j['scenario'],'group':group,
                     'mean_cosine':frame.anchor_cosine.mean(),'mean_evidence':frame.evidence_weight.mean(),
                     'mean_policy_weight':frame.restored_weight.mean(),'mean_clip_factor':frame.clip_factor.mean(),
                     'zero_weight_fraction':float((frame.restored_weight == 0).mean()),
                     'below_half_weight_fraction':float((frame.restored_weight < .5).mean()),
                     'zero_policy_weights_round24':sum(last[str(i)]['restored_weight']==0 for i in ids),
                     'fallback_rounds':sum(d['values']['bounded_rfa_fallback'] for d in ds)})
D = pd.DataFrame(rows).sort_values(['regime','scenario','group'])
D.to_csv(STUDY/'analysis/mechanism_diagnostics.csv',index=False)
display(D[D.scenario.isin(['clean','gradual_stripe','label_flip'])].round(3))
C = pd.read_csv(STUDY/'analysis/costs.csv')
cost = C[C.method.isin(KEEP)].groupby('method')[['median_defence_seconds','median_training_seconds','median_evaluation_seconds']].median()
cost['trusted_reference_images'] = [300 if m in ['FLTrust','FedReTrust'] else 0 for m in cost.index]
cost.to_csv(STUDY/'analysis/costs_and_reference_access.csv')
display(cost.round(4))
"""),
        md("""## Interpretation limits and reproduction

All candidate settings were declared before training and remain fixed after selection. This is a small adaptive research programme: v0.2 results informed v0.3, selection uses the first development seed, and the second seed is also previously used development data. The table establishes measured outcomes only for these runs. One fixed partition per regime, MNIST, ten fully participating clients, fixed affected cohorts and 24 rounds limit generalization.

Geometry does not reveal intent. Honest minority clients may oppose consensus and coordinated attackers may align with it. No adaptive/backdoor study, secure aggregation, false-count defence, Sybil defence, formal robustness theorem or novelty claim is provided. A positive two-seed mean margin does not establish statistical superiority. Every baseline must be included in any claim about beating every baseline; this pilot includes Viceroy with its original rate adapter. FLAURA and FedCognis remain future comparisons.

Reproduce from the project root:

```sh
Experiment/.venv/bin/python -m unittest discover -s Experiment/code/tests -v
Experiment/.venv/bin/python Experiment/code/directional_study.py prepare
Experiment/.venv/bin/python Experiment/code/directional_study.py run
Experiment/.venv/bin/python Experiment/code/directional_notebook.py
```

The runner verifies sources and saved artifacts before reusing results. The review notebook reads saved outputs and does not train models. Preserve this separate exploratory study alongside both earlier versions."""),
    ]
    notebook = nbf.v4.new_notebook(cells=cells)
    notebook.metadata.kernelspec = {"display_name": "FedReTrust Experiment (Python 3.14)",
                                    "language": "python", "name": "fedretrust-experiment"}
    path = ROOT / "code/19_directional_fedretrust.ipynb"
    os.environ.setdefault("IPYTHONDIR", "/tmp/fedretrust_ipython")
    from IPython.core.interactiveshell import InteractiveShell
    from IPython.utils.capture import capture_output
    shell = InteractiveShell.instance()
    previous = Path.cwd(); os.chdir(path.parent)
    try:
        count = 0
        for cell in notebook.cells:
            if cell.cell_type != "code": continue
            count += 1
            with capture_output() as captured:
                result = shell.run_cell(cell.source, store_history=False)
            cell.execution_count = count
            cell.outputs = []
            if captured.stdout: cell.outputs.append(nbf.v4.new_output("stream", name="stdout", text=captured.stdout))
            if captured.stderr: cell.outputs.append(nbf.v4.new_output("stream", name="stderr", text=captured.stderr))
            for item in captured.outputs:
                cell.outputs.append(nbf.v4.new_output("display_data", data=item.data, metadata=item.metadata))
            if result.error_before_exec or result.error_in_exec:
                raise RuntimeError(f"Cell {count}: {result.error_before_exec or result.error_in_exec}")
        notebook.metadata["execution_note"] = "All cells executed with IPython InteractiveShell; reads saved development artifacts only."
        nbf.validate(notebook)
        nbf.write(notebook, path)
    finally:
        os.chdir(previous)
    return path


if __name__ == "__main__":
    print(build())
