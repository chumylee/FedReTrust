"""Build an executed, read-only review of the reference-free development pilot."""
from pathlib import Path
import os

import nbformat as nbf

from research_runs import ROOT


def build():
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    cells = [
        md("""# 18 — FedReTrust without a clean server reference

**Exploratory revision following the supervisor discussion.** This pilot asks whether a combination of robust peer aggregation, directional evidence and bounded client influence can preserve honest drift while protecting against the existing attacks. It uses **zero trusted server examples**. Original FedReTrust v0.1 and the frozen five-seed study remain unchanged.

**Outcome:** neither candidate passed the screening guards. The selected recovery variant improved honest stripe accuracy over RFA, but its non-IID attack protection remained substantially weaker. This combination did not achieve the desired balance. It remains a separate prototype and does not replace v0.1 or alter its frozen claims.

These are development results, collected after earlier final results had been examined. The second seed is a replication on previously used development data, not an untouched final test. The pilot does not establish novelty or general superiority."""),
        code("""from pathlib import Path
import sys, json
import pandas as pd
from IPython.display import display
CODE = Path.cwd() if (Path.cwd()/'reference_free_study.py').exists() else Path.cwd()/'Experiment/code'
sys.path.insert(0, str(CODE.resolve()))
from reference_free_study import STUDY, BASELINES, CANDIDATES, SCREEN_SEED, CHECK_SEED, verify
from research_runs import ROOT
P = verify(STUDY)
S = json.loads((STUDY/'selection.json').read_text())
STATUS = json.loads((STUDY/'status.json').read_text())
assert STATUS['status'] == 'complete'
SELECTED = S['selected']
M = pd.read_csv(STUDY/'analysis/seed_metrics.csv')
KEEP = [*BASELINES, SELECTED]
print('Selected for follow-up:', SELECTED)
print('Trusted examples used by either prototype:', P['trusted_examples_for_candidates'])
print('Source and partition hashes verified.')
pd.set_option('display.max_rows', 150)
pd.set_option('display.max_columns', 12)
def comparison(metric, seed, scenarios=None):
    x = M[(M.metric == metric) & (M.seed == seed) & M.method.isin(KEEP)]
    if scenarios is not None: x = x[x.scenario.isin(scenarios)]
    return x.pivot(index=['regime','scenario'], columns='method', values='value').reindex(columns=KEEP).round(2)
"""),
        md(r"""## How the prototype works

For submitted updates $\Delta_i$ and sample shares $p_i$:

1. Cap update norms at $C=6\,\mathrm{median}_i\|\Delta_i\|$. Set $\bar\Delta_i=c_i\Delta_i$, where $c_i=\min(1,C/\|\Delta_i\|)$.
2. Compute a three-step **RFA** anchor $a$ from the capped updates. This replaces the clean-root direction; it is not a known-honest direction.
3. Assess each raw update using distance $d_i=\|\Delta_i-a\|$ and cosine $u_i=\cos(\Delta_i,a)$. A large update alone is not enough to restrict it.
4. Let $o_i=[d_i/(2\operatorname{median}(d))-1]_0^1$ and $s_i=[(u_i-.25\operatorname{median}(u))/(.5\operatorname{median}(u))]_0^1$. The fresh evidence weight is $g_i=1-o_i(1-s_i)$. Thus only the combination of distance and weak alignment reduces weight.
5. The **static** version uses $q_i=g_i$. The **recovery** version uses $q_i=\min(g_i,q_i^{old}+.5)$: restriction is immediate; two fully supportive participations can restore a zero weight. Every submitted update is reassessed regardless of its old weight.
6. Apply $\sum_i \frac{p_iq_i}{\sum_jp_jq_j}\bar\Delta_i$. Trust is normalized before magnitude clipping. An undefined anchor direction or nonpositive median alignment falls back to bounded RFA and preserves previous memory; a zero trust denominator also falls back to bounded RFA.

The median is the lower middle value for an even cohort. Epsilon handling and tensor validation are specified in `fedretrust_reference_free.py`. Updates and counts are visible to the server; secure aggregation and protection against false sample counts are not implemented.

This borrows the geometric-median anchor from [RFA](https://arxiv.org/abs/1912.13445), directional screening from the idea behind [FLTrust](https://arxiv.org/abs/2012.13995), and bounded-influence motivation from [robust clipping](https://proceedings.mlr.press/v139/karimireddy21a.html). It is **not** the published centered-clipping algorithm and does not inherit those methods' theoretical guarantees. The specific combination is a hypothesis; originality has not been established."""),
        md("""## Declared experiment and selection

Two candidates were fixed before new training: identical geometry and clipping, with restoration disabled/enabled. Both were screened on clean, stripe drift and persistent label flipping, in IID and non-IID, using seed 20261014. One candidate was selected jointly across regimes by the smallest worst violation of three 1-point guards: clean and stripe accuracy relative to FedAvg, and damage relative to the better of RFA/FLTrust.

The selected candidate then ran all ten original scenarios on that seed, plus the three screening scenarios on seed 20261015. No thresholds or candidates were added after seeing outcomes. A selected candidate can still fail the guards.

All runs use 24 rounds, ten clients, 512 images/client, one local epoch and 1,000 development images per evaluation task. The original 300 reserved root images remain excluded from clients to preserve pairing; the prototypes never use them. FLTrust and v0.1 retain their original reference access.

The experiment evaluator still needs labelled held-out data to measure accuracy. That data is never passed to the prototype's aggregator. The common job records retain a legacy `root_steps=8` field, but only the baseline factory uses it; candidate construction accepts scalar settings only, and candidate reference-forward counts must be zero."""),
        code("""display(pd.DataFrame([{k:v for k,v in c.items() if k != 'details'} for c in S['candidates']]))
display(pd.DataFrame([{'candidate':c['candidate'], **d} for c in S['candidates'] for d in c['details']]))
print('Selection rule:', P['screen_selection'])
manifest = json.loads((STUDY/'result_manifest.json').read_text())
assert manifest['original_training_sources_unchanged'] and not manifest['official_test_used']
records = manifest['records']
print('Records:', len(records), '; candidate runs:', sum(r['job']['method'] in CANDIDATES for r in records))
print('Original sources unchanged:', manifest['original_training_sources_unchanged'])
print((STUDY/'preflight_tests.txt').read_text().strip())
"""),
        md("""## Core outcomes: show both seeds separately

Accuracy is a percentage; attack damage is a percentage-point loss against the same method's matched clean trajectory. Clean accuracy averages rounds 5–24. Stripe accuracy evaluates affected honest clients' legitimate tasks over rounds 5–24. Persistent attack damage uses the same rounds. A small negative damage means the attack run happened to score above its control; it does not mean the attack is beneficial in general."""),
        code("""for seed in (SCREEN_SEED, CHECK_SEED):
    print()
    print('DEVELOPMENT SEED', seed, '(selection)' if seed == SCREEN_SEED else '(replication)')
    display(comparison('overall_mean_5_24_pp', seed, ['clean']))
    display(comparison('affected_honest_event_pp', seed, ['gradual_stripe']))
    display(comparison('attack_active_signed_damage_pp', seed, ['label_flip']))
core = M[M.method.isin(KEEP) & (
    ((M.scenario == 'clean') & (M.metric == 'overall_mean_5_24_pp')) |
    ((M.scenario == 'gradual_stripe') & (M.metric == 'affected_honest_event_pp')) |
    ((M.scenario == 'label_flip') & (M.metric == 'attack_active_signed_damage_pp')))]
assert (core.groupby(['regime','scenario','metric','method']).size() == 2).all()
core_means = core.pivot_table(index=['regime','scenario','metric'], columns='method', values='value', aggfunc='mean').reindex(columns=KEEP)
print('Means of the two development seeds above; one seed was used for selection:')
display(core_means.round(2))
core_means.to_csv(STUDY/'analysis/two_seed_core_means.csv')
"""),
        md("""## Protection–adaptation trade-off

Each marker is one development seed. Higher stripe accuracy and lower attack damage are preferable. Lines connect the two seeds for the same method; they are not confidence intervals. The plot compares two separate scenarios, not a simultaneous mixed attack-and-drift test."""),
        code("""import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from IPython.display import Image
figure_dir = STUDY/'analysis/figures'
figure_dir.mkdir(exist_ok=True)
colors = dict(zip(KEEP, ['#777777','#0072B2','#E69F00','#CC79A7','#009E73']))
labels = {m: m for m in KEEP}
labels['FedReTrust'] = 'FedReTrust v0.1'
labels[SELECTED] = 'Reference-free v0.2'
fig, axes = plt.subplots(1,2, figsize=(10.5,4.8), sharey=True)
for ax, regime in zip(axes, ['IID','non-IID']):
    for method in KEEP:
        x, y = [], []
        for seed, marker in [(SCREEN_SEED,'o'),(CHECK_SEED,'^')]:
            subset = M[(M.regime == regime) & (M.method == method) & (M.seed == seed)]
            damage = subset[(subset.scenario == 'label_flip') & (subset.metric == 'attack_active_signed_damage_pp')].value.item()
            drift = subset[(subset.scenario == 'gradual_stripe') & (subset.metric == 'affected_honest_event_pp')].value.item()
            x.append(damage); y.append(drift)
            ax.scatter(damage, drift, marker=marker, color=colors[method], s=52, zorder=3)
        ax.plot(x, y, color=colors[method], alpha=.45, linewidth=1)
    ax.set_title(regime)
    ax.set_xlabel('Persistent label-flip damage (pp; lower is better)')
    ax.grid(alpha=.2)
axes[0].set_ylabel('Affected-client stripe accuracy (%; higher is better)')
handles = [Line2D([0],[0], color=colors[m], marker='o', linestyle='', label=labels[m]) for m in KEEP]
fig.legend(handles=handles, loc='lower center', ncol=3, frameon=False)
axes[1].legend(handles=[Line2D([0],[0], color='black', marker=mark, linestyle='', label=str(seed)) for seed,mark in [(SCREEN_SEED,'o'),(CHECK_SEED,'^')]], title='Development seed', loc='lower right', frameon=False)
fig.suptitle('Reference-free pilot: two development seeds, fixed partitions')
fig.tight_layout(rect=(0,.13,1,.94))
fig.savefig(figure_dir/'protection_drift.png', dpi=180)
fig.savefig(figure_dir/'protection_drift.pdf')
plt.close(fig)
display(Image(filename=str(figure_dir/'protection_drift.png')))
"""),
        md("""## All ten scenarios: first seed only

The following tables are complete for the first development seed. They are not five-seed estimates. The overall accuracy table includes the endpoint separately; the period average remains the main summary. Honest drift and attack damage have their own tables so that a strong aggregate score cannot hide harm to changed clients."""),
        code("""print('Overall legitimate-task accuracy, mean rounds 5–24 (%)')
display(comparison('overall_mean_5_24_pp', SCREEN_SEED))
print('Overall legitimate-task accuracy, round 24 (%)')
display(comparison('overall_round24_pp', SCREEN_SEED))
print('Affected honest clients during their event (%)')
display(comparison('affected_honest_event_pp', SCREEN_SEED))
print('Formerly changed honest clients after their change ends, rounds 11–24 (%)')
display(comparison('affected_honest_return_11_24_pp', SCREEN_SEED))
print('Attack-active damage versus matched no-attack trajectory (pp; lower is better)')
display(comparison('attack_active_signed_damage_pp', SCREEN_SEED))
print('Second recurring attack only, rounds 15–18 (pp; lower is better)')
display(comparison('second_attack_signed_damage_pp', SCREEN_SEED))
"""),
        md("""## Model recovery and costs

Model recovery keeps the original definition: accuracy within 1 percentage point of the matched reference for three consecutive post-event rounds. Confirmation delay is reported only for observed restoration. Already-good, delayed-impairment and censored cases remain distinct. This is separate from restoring a client's aggregation weight.

Mixed attack damage uses each method's mixed-drift-only control, not its clean run. Runtime is descriptive under two concurrent workers; reused baseline timings were collected in the earlier development study, so they are not a controlled speed benchmark. The legacy `reference_forward_examples` counter is not recorded by FLTrust; its default zero in the raw cost export must not be interpreted as absence of reference access. The table instead reports each method's actual reference-dataset requirement."""),
        code("""E = pd.read_csv(STUDY/'analysis/model_episodes.csv')
E = E[(E.seed == SCREEN_SEED) & E.method.isin(KEEP)]
cols = ['regime','scenario','method','intent','episode','status','observed_restoration_delay_rounds']
display(E[cols].sort_values(['regime','scenario','method','episode']))
C = pd.read_csv(STUDY/'analysis/costs.csv')
cost_table = C.groupby('method')[['median_defence_seconds','median_training_seconds','median_evaluation_seconds']].median()
cost_table['trusted_reference_images'] = [300 if m in ['FLTrust','FedReTrust'] else 0 for m in cost_table.index]
display(cost_table.round(4))
cost_table.to_csv(STUDY/'analysis/costs_and_reference_access.csv')
assert (C[C.method.isin(CANDIDATES)].reference_forward_examples == 0).all()
"""),
        md("""## Check what the mechanism actually did

These are descriptive diagnostics from the first seed; means cover rounds 5–24. Simulator group labels are used only here, after training. `restored_weight` is the policy weight before sample-share normalization and magnitude clipping; it is not the final contribution fraction. A full restored weight therefore does not imply an unclipped contribution.

This table checks whether honest clients still end at zero weight and whether malicious updates escaped the distance gate. A low outlier score means little restriction, even when directional alignment is poor."""),
        code("""diagnostic_rows = []
for result in records:
    j = result['job']
    if j['method'] != SELECTED or j['seed'] != SCREEN_SEED or j['scenario'] not in ['clean','gradual_stripe','label_flip','honest_change_stops']: continue
    ids = list(range(10)) if j['scenario'] == 'clean' else [0,1,2]
    diagnostics = json.loads((ROOT/result['run_path']/'aggregation_diagnostics.json').read_text())
    observations = [row['values']['client_metrics'][str(i)] for row in diagnostics if row['round'] >= 5 for i in ids]
    frame = pd.DataFrame(observations)
    last = diagnostics[-1]['values']['client_metrics']
    diagnostic_rows.append({'regime':j['regime'], 'scenario':j['scenario'],
        'mean_evidence_weight':frame.evidence_weight.mean(),
        'mean_restored_weight':frame.restored_weight.mean(),
        'mean_outlier_score':frame.outlier_score.mean(),
        'mean_alignment_score':frame.alignment_score.mean(),
        'mean_clip_factor':frame.clip_factor.mean(),
        'zero_policy_weights_at_round24':sum(last[str(i)]['restored_weight'] == 0 for i in ids),
        'undefined_policy_weights_at_round24':sum(last[str(i)]['restored_weight'] is None for i in ids),
        'fallback_rounds':sum(row['values']['bounded_rfa_fallback'] for row in diagnostics),
        'clients_in_group':len(ids)})
D = pd.DataFrame(diagnostic_rows).sort_values(['regime','scenario'])
display(D.round(3))
D.to_csv(STUDY/'analysis/mechanism_diagnostics.csv', index=False)
"""),
        md("""## What this pilot can and cannot establish

The prototype removes the trusted-dataset dependency and is executable in the same simulator. Performance must be assessed from every displayed condition; a good stripe result alone is insufficient. Two core seeds and one seed for the full scenario panel cannot establish reliable superiority. One partition per regime, MNIST, fixed affected clients, full participation and the short horizon remain limitations. The known Gaussian perturbation was weak against FedAvg in the original study. Changes from v0.1 include geometry, normalization and restoration as well as reference removal; the gains cannot be attributed to reference removal alone.

Client geometry is not an oracle for intent: adaptive attackers may align with the peer anchor, and useful honest minority updates may disagree with it. Removing reference data changes the assumptions; it does not make peer evidence known-honest. Formal novelty, adaptive/backdoor protection and performance on different partitions/datasets remain untested. FLAURA and FedCognis are not implemented in this pilot.

Reproduction (from the project root):

```sh
Experiment/.venv/bin/python Experiment/code/reference_free_study.py prepare
Experiment/.venv/bin/python Experiment/code/reference_free_study.py run
Experiment/.venv/bin/python Experiment/code/reference_free_notebook.py
```

The runner validates and reuses completed artifacts; it does not repeat the old official-test experiment. Keep this pilot separate from the original frozen claims."""),
    ]
    notebook = nbf.v4.new_notebook(cells=cells)
    notebook.metadata.kernelspec = {"display_name": "FedReTrust Experiment (Python 3.14)",
                                    "language": "python", "name": "fedretrust-experiment"}
    path = ROOT / "code/18_reference_free_fedretrust.ipynb"
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
        notebook.metadata["execution_note"] = "All cells executed with IPython InteractiveShell in the project environment; reads saved development artifacts only."
        nbf.validate(notebook)
        nbf.write(notebook, path)
    finally:
        os.chdir(previous)
    return path


if __name__ == "__main__":
    print(build())
