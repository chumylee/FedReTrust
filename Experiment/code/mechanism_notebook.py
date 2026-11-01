"""Build an executed, artifact-only review after the mechanism pilot completes.

This presentation module is outside the frozen training source set. It never
trains, evaluates model checkpoints, refits scores, or changes the declared gate.
"""
from pathlib import Path
import argparse
import json
import os

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STUDY = ROOT / "outputs/mechanism_pilot/20261030_v1"
DEFAULT_NOTEBOOK = ROOT / "code/20_mechanism_diagnostics.ipynb"


def make_notebook(study=DEFAULT_STUDY):
    """Construct cells without reading results or executing any cell."""
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    setup = f"""from pathlib import Path
import sys, json
import numpy as np
import pandas as pd
from IPython.display import display, Image, Markdown
CODE = Path({json.dumps(str(ROOT / 'code'))})
sys.path.insert(0, str(CODE))
from mechanism_pilot import verify
from directional_study import checked_result
from research_analysis import cross_method_pair
from experiment_core import file_sha256
ROOT = CODE.parent
STUDY = Path({json.dumps(str(Path(study).resolve()))})
STATUS = json.loads((STUDY/'status.json').read_text())
assert STATUS['status'] == 'complete' and STATUS['new_runs'] == 32
P = verify(STUDY)
A = STUDY/'analysis'
DG = A/'diagnostics'
MANIFEST = json.loads((STUDY/'result_manifest.json').read_text())
records = MANIFEST['records']
observed = MANIFEST['observed_records']
assert len(observed) == 32 and MANIFEST['official_test_used'] is False
lookup = {{(r['job']['method'],r['job']['regime'],r['job']['scenario'],r['job']['seed']):r for r in records}}
for r in records:
    checked_result(r)
    j = r['job']
    cross_method_pair(r, lookup[('FedAvg',j['regime'],j['scenario'],j['seed'])])
AUDIT = json.loads((STUDY/'observation_access_audit.json').read_text())
assert AUDIT['new_runs'] == 32 and AUDIT['new_rounds'] == 768 and AUDIT['trusted_examples'] == 0
assert all(r.get('all24_global_states_and_all_task_metrics_identical') is True
           for r in AUDIT['runs'] if r['method'] == 'FedAvg')
M = pd.DataFrame(json.loads((A/'seed_metrics.json').read_text()))
C = pd.DataFrame(json.loads((A/'accuracy_curves.json').read_text()))
TARGET = json.loads((A/'target_check.json').read_text())
DIAG = json.loads((DG/'diagnostic_summary.json').read_text())
GROUPS = pd.read_csv(DG/'diagnostic_groups.csv')
PROBE = json.loads((DG/'probe_score_summary.json').read_text())
PROBE_MANIFEST = json.loads((STUDY/'probes/summary.json').read_text())
assert PROBE_MANIFEST['status'] == 'complete' and PROBE_MANIFEST['local_trainings'] == 640
assert PROBE_MANIFEST['official_test_used'] is False and PROBE_MANIFEST['evaluation_forward_examples'] == 0
assert PROBE['probe_manifest_sha256'] == file_sha256(STUDY/'probes/summary.json')
assert PROBE['trajectory_gate_modified'] is False
assert PROBE['trajectory_gate_pass'] == DIAG['gate']['pass']
assert PROBE['selected_score'] == DIAG['selected_score']
KEEP = ['FedAvg','RFA','FLTrust','Viceroy','FedReTrust',
        'FedReTrust-RF-recovery','FedReTrust-Direction-loose','NNM-RFA']
LABELS = {{'FedAvg':'FedAvg','RFA':'RFA','FLTrust':'FLTrust','Viceroy':'Viceroy',
          'FedReTrust':'FedReTrust v0.1','FedReTrust-RF-recovery':'FedReTrust v0.2',
          'FedReTrust-Direction-loose':'FedReTrust v0.3','NNM-RFA':'NNM–RFA'}}
REGIMES = ['IID','non-IID']
SEEDS = [20261014,20261015]
SCENARIOS = ['clean','abrupt_shot_noise','gradual_stripe','label_flip']
SCENARIO_LABELS = {{'clean':'Clean','abrupt_shot_noise':'Honest noise',
                   'gradual_stripe':'Honest stripe','label_flip':'Label flip'}}
assert set(M.method) == set(KEEP)
pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 14)
pd.set_option('display.width', 180)
def comparison(metric, seed):
    rows = M[(M.metric == metric) & (M.seed == seed)]
    return rows.pivot(index=['regime','scenario'], columns='method', values='value').reindex(columns=KEEP).rename(columns=LABELS).round(2)
SUMMARY = M.groupby(['regime','scenario','metric','method']).value.agg(mean='mean',n='count').reset_index()
TWO = SUMMARY[SUMMARY.n == 2]
def two_seed_comparison(metric):
    rows = TWO[TWO.metric == metric]
    return rows.pivot(index=['regime','scenario'], columns='method', values='mean').reindex(columns=KEEP).rename(columns=LABELS).round(2)
print(f'Verified {{len(records)}} comparison runs, 32 new trajectories, 768 observed rounds and 640 probe client trainings.')
print('All 16 observed FedAvg trajectories reproduce every original global-state hash and task metric.')
"""
    cells = [
        md("""# 20 — Mechanism diagnostics and NNM–RFA

This experiment asks whether richer observations distinguish honest change from label poisoning, and whether a published heterogeneity-aware wrapper improves robust aggregation. **These are separate questions.** The observations do not affect any aggregation decision. NNM–RFA is an independent aggregation control, not a new FedReTrust prototype.

All five original baselines, including Viceroy, and both reference-free prototypes remain in the comparison. This is an exploratory study of previously used development seeds, partitions and images. There is no new official-test evaluation, confirmatory superiority finding, general security guarantee or novelty claim."""),
        code(setup),
        code("""target_count = sum(row['target_met_descriptively'] for row in TARGET['comparisons'])
display(Markdown(f"**NNM–RFA: {target_count}/6 baseline mean targets met descriptively. "
                 f"Previous v0.3 IID stripe result preserved: {TARGET['preserves_v03_IID_stripe']}.**"))
display(Markdown(f"**Diagnostic gate: {'PASS' if DIAG['gate']['pass'] else 'NO GO'}. "
                 f"Selected observation: `{DIAG['selected_score']}`.**"))
print('Common-broadcast selected-score status:', PROBE['selected_score_status'])
print('Common-broadcast descriptive contradiction:', PROBE['descriptive_probe_contradiction'])
print(DIAG['gate']['interpretation'])
decision_path = STUDY/'decision.json'
if decision_path.exists():
    DECISION = json.loads(decision_path.read_text())
    for name, digest in DECISION['evidence_sha256'].items():
        assert file_sha256(STUDY/name) == digest
    if not DECISION['new_combined_FedReTrust_prototype_built'] and not DIAG['gate']['pass']:
        display(Markdown('**No combined FedReTrust prototype was built because the planned diagnostic gate failed.**'))
    else:
        print('New combined FedReTrust prototype built:', DECISION['new_combined_FedReTrust_prototype_built'])
    print(DECISION['reason'])
"""),
        md("""## Workload, access and coverage

There are 32 new trajectories: FedAvg and NNM–RFA, two development seeds, two fixed partitions, four scenarios, and 24 rounds. All ten clients participate, each uses the same 512 training images and one local epoch. The three affected clients begin honest noise, gradual stripe, or persistent label flipping at round 5. The label attack maps every digit with `y → 9 − y`.

The observer reads model updates, client IDs, sample counts and broadcasts. It uses no images, labels, simulator intent, or evaluation feedback. Simulator labels are joined only in offline analysis. FLTrust and original FedReTrust retain their historical 300-image references; those data are unused by NNM–RFA and the observer and remain excluded from clients to preserve matching.

The two older reference-free prototypes have no second-seed honest-noise run. Missing entries remain missing: one-seed prototype noise results are never labelled two-seed means. The coverage table and per-seed tables below make this explicit."""),
        code("""coverage = pd.DataFrame([{k:r['job'][k] for k in ['method','regime','scenario','seed']} for r in records])
coverage_table = coverage.groupby(['regime','scenario','method']).seed.nunique().unstack('method').reindex(columns=KEEP).rename(columns=LABELS)
display(coverage_table)
missing = []
for regime in REGIMES:
    for scenario in SCENARIOS:
        for method in KEEP:
            present = set(coverage[(coverage.regime == regime) & (coverage.scenario == scenario) & (coverage.method == method)].seed)
            for seed in sorted(set(SEEDS)-present):
                missing.append({'regime':regime,'scenario':scenario,'method':LABELS[method],'missing_seed':seed})
display(pd.DataFrame(missing))
print('Counts are distinct development seeds, not independent partitions or client-round replications.')
"""),
        md("""## What changes and what is only observed

**NNM–RFA:** with a declared maximum `f=3`, each client update is replaced by the arithmetic mean of its seven nearest updates, including itself. Original sample shares then weight the mixtures in the unchanged RFA solver, using up to three iterations and its original stopping tolerance. The budget stays three on clean runs too. The final aggregate remains robust aggregation; RFA is not merely an anchor for a subsequent unconstrained mean. This is an empirical adaptation to local SGD, not a transfer of gradient-method guarantees. [NNM primary paper](https://proceedings.mlr.press/v206/allouah23a/allouah23a.pdf).

**Passive observations:** whole-update cosine, penultimate-layer cosine, output-layer cosine, the mean and minimum cosine across all ten output rows including their biases, head-relative norm, same-client temporal cosine and relative change, and disagreement across four layers. Every directional score uses slices of the same current, sample-weighted whole-update RFA reference. The head-relative norm is descriptive only and is excluded from score selection.

A zero vector has no defined direction. Undefined cosines and dependent summaries remain null. Temporal history is causal and client-specific, with no assumed clean warmup. The observer never supplies weights, drops clients or changes the returned aggregate."""),
        md("""## Two-seed core results and targets

Clean accuracy is the legitimate-task mean over rounds 5–24. Honest stripe accuracy concerns the affected clients' legitimate changed task. Label-flip damage is the signed loss against the same method's paired clean run, in percentage points. A small damage value can be misleading when clean accuracy is already low, so absolute attacked accuracy appears alongside it.

Targets compare NNM–RFA against the best of the five original baselines in each regime. The additional guard preserves v0.3's IID stripe result. Matching a two-seed mean target is descriptive, not proof of superiority."""),
        code("""core = TWO[((TWO.scenario == 'clean') & (TWO.metric == 'overall_mean_5_24_pp')) |
           ((TWO.scenario == 'gradual_stripe') & (TWO.metric == 'affected_honest_event_pp')) |
           ((TWO.scenario == 'label_flip') & TWO.metric.isin(['attack_active_signed_damage_pp','overall_mean_5_24_pp']))]
display(core.pivot(index=['regime','scenario','metric'],columns='method',values='mean').reindex(columns=KEEP).rename(columns=LABELS).round(2))
display(pd.DataFrame(TARGET['comparisons']).drop(columns='metric').round(3))
print('All six baseline targets met:', TARGET['all_six_baseline_targets_met'])
print('Preserves v0.3 IID stripe:', TARGET['preserves_v03_IID_stripe'])
print('A positive advantage is favourable; zero meets a descriptive target by equality.')
"""),
        code("""import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
FIGURES = A/'figures'; FIGURES.mkdir(exist_ok=True)
COLORS = dict(zip(KEEP,['#767676','#0072B2','#D55E00','#8E6CBE','#9C7100','#009E73','#CC79A7','#111111']))
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
def export_figure(fig, stem):
    fig.savefig(FIGURES/(stem+'.png'),dpi=180,bbox_inches='tight')
    fig.savefig(FIGURES/(stem+'.pdf'),bbox_inches='tight')
    plt.close(fig)
    display(Image(filename=str(FIGURES/(stem+'.png'))))
fig, axes = plt.subplots(1,2,figsize=(13,5.5),sharey=True)
for ax, regime in zip(axes,REGIMES):
    for method in KEEP:
        xs, ys = [], []
        for seed, marker in zip(SEEDS,['o','^']):
            rows = M[(M.regime == regime) & (M.method == method) & (M.seed == seed)]
            x = rows[(rows.scenario == 'label_flip') & (rows.metric == 'attack_active_signed_damage_pp')].value.item()
            y = rows[(rows.scenario == 'gradual_stripe') & (rows.metric == 'affected_honest_event_pp')].value.item()
            xs.append(x); ys.append(y)
            ax.scatter(x,y,marker=marker,color=COLORS[method],s=62 if method=='NNM-RFA' else 42,zorder=5)
        ax.plot(xs,ys,color=COLORS[method],alpha=.35,lw=1)
    ax.set(title=regime,xlabel='Label-flip damage (pp; lower is better)')
    ax.grid(alpha=.2)
axes[0].set_ylabel('Affected-client stripe accuracy (%; higher is better)')
fig.legend(handles=[Line2D([0],[0],marker='o',color=COLORS[m],linestyle='',label=LABELS[m]) for m in KEEP],loc='lower center',ncol=4,frameon=False)
axes[1].legend(handles=[Line2D([0],[0],marker=k,color='black',linestyle='',label=str(s)) for s,k in zip(SEEDS,['o','^'])],title='Development seed',frameon=False)
fig.suptitle('Protection and honest drift — individual development seeds')
fig.tight_layout(rect=(0,.15,1,.94))
export_figure(fig,'mechanism_protection_drift')
"""),
        md("""## Every tested scenario: means, tails, final rounds and individual seeds

The new pilot covers clean training, honest noise, honest stripe and persistent label flips. It does not add repaired-client, stopping-attack, recurring-attack, mixed or Gaussian runs. Earlier notebooks retain those results; this pilot cannot supply a new recovery claim.

The period mean, final round and rounds 21–24 answer different questions. Round 24 includes the influence of earlier clean rounds on training, but is one measurement after round 24; it is not an average that adds the clean rounds. Honest-change overall accuracy mixes legitimate tasks, so affected-client and unchanged-client figures are also included."""),
        code("""METRICS = [
    ('overall_mean_5_24_pp','Overall legitimate-task accuracy, rounds 5–24 (%)'),
    ('overall_tail_21_24_pp','Overall legitimate-task accuracy, rounds 21–24 (%)'),
    ('overall_round24_pp','Overall legitimate-task accuracy, round 24 (%)'),
    ('affected_honest_event_pp','Affected honest-client accuracy during change (%)'),
    ('unchanged_honest_mean_5_24_pp','Unchanged honest-client accuracy, rounds 5–24 (%)'),
    ('attack_active_signed_damage_pp','Label-flip damage, rounds 5–24 (pp)'),
    ('attack_tail_damage_21_24_pp','Label-flip damage, rounds 21–24 (pp)')]
for metric, title in METRICS:
    display(Markdown('**'+title+' — two-seed means**'))
    display(two_seed_comparison(metric))
for seed in SEEDS:
    display(Markdown(f'**Individual seed {seed} — all recorded metrics**'))
    x = M[(M.seed == seed) & M.metric.isin([item[0] for item in METRICS])]
    display(x.pivot(index=['regime','scenario','metric'],columns='method',values='value').reindex(columns=KEEP).rename(columns=LABELS).round(2))
print('NaN means no matching result; it is not zero and has not been imputed.')
"""),
        code("""fig, axes = plt.subplots(2,4,figsize=(18,8),sharex=True)
panels = [('clean','all_clients','Clean task'),('abrupt_shot_noise','affected_honest','Affected honest noise'),
          ('gradual_stripe','affected_honest','Affected honest stripe'),('label_flip','all_clients','Absolute accuracy under label flips')]
for row_index, regime in enumerate(REGIMES):
    for column, (scenario, group, title) in enumerate(panels):
        ax = axes[row_index,column]
        for method in KEEP:
            x = C[(C.regime==regime) & (C.scenario==scenario) & (C.group==group) & (C.method==method)]
            if set(x.seed) != set(SEEDS): continue
            mean = x.groupby('round').accuracy_pp.agg(['mean','count'])
            if not (mean['count']==2).all(): raise ValueError('Incomplete curve seed coverage')
            ax.plot(mean.index,mean['mean'],color=COLORS[method],lw=2.2 if method=='NNM-RFA' else 1.4,
                    linestyle='--' if method in ['FedReTrust-RF-recovery','FedReTrust-Direction-loose'] else '-')
        if scenario != 'clean': ax.axvline(5,color='#777777',linestyle=':',lw=1)
        ax.set_title(regime+' · '+title)
        ax.set_xlim(1,24); ax.set_xticks([1,5,10,15,20,24]); ax.grid(alpha=.2)
        if column==0: ax.set_ylabel('Accuracy (%)')
        if row_index==1: ax.set_xlabel('Communication round')
fig.legend(handles=[Line2D([0],[0],color=COLORS[m],label=LABELS[m]) for m in KEEP],loc='lower center',ncol=4,frameon=False)
fig.suptitle('Two-seed mean learning curves; prototype noise curves omitted for incomplete coverage')
fig.tight_layout(rect=(0,.12,1,.95))
export_figure(fig,'mechanism_learning_curves')
"""),
        md("""## Fixed diagnostic selection and held-out slices

Eight predeclared candidate scores use fixed directions; cosine scores become `1 − cosine`. Honest examples from the first FedAvg development seed set one threshold per score, conservatively limiting empirical false flags to at most 5% in every honest subgroup and regime. Selection chooses the alternative with the strongest worst-regime attack recall on that calibration slice; ties follow the declared order.

The selected score and threshold then stay fixed. Checks use FedAvg's second seed and both NNM–RFA seeds. Each check must retain at most 5% honest false flags per subgroup, at least 80% attack recall in each regime, and at least a ten-percentage-point improvement in worst-regime recall over the separately calibrated whole-cosine baseline. Any undefined required observation fails that check. These thresholds are research gates, not estimated deployment guarantees.

Client-round observations and checkpoints are correlated. No binomial confidence intervals or significance claims are attached. The NNM–RFA first-seed check tests transfer across a different trajectory on the calibration seed, not independent validation."""),
        code("""calibration = []
for name, row in DIAG['calibration'].items():
    calibration.append({'score':name,'eligible':row['eligible'],'undefined_rows':row['undefined_calibration_rows'],
                        'threshold':row.get('threshold'),'minimum_attack_recall':row.get('minimum_attack_recall'),
                        **{'attack_recall_'+regime:row.get('attack_recall_by_regime',{}).get(regime) for regime in REGIMES}})
display(pd.DataFrame(calibration).set_index('score'))
print('Selected once on calibration:',DIAG['selected_score'])
display(pd.DataFrame([{'method':r['method'],'seed':r['seed'],'pass':r['pass'],'reasons':'; '.join(r['reasons']),
                       'worst_regime_recall_improvement':r['metrics'].get('worst_regime_recall_improvement')}
                      for r in DIAG['gate']['checks']]))
chosen = [DIAG['baseline_score']] + ([DIAG['selected_score']] if DIAG['selected_score'] is not None else [])
selected_groups = GROUPS[GROUPS.score.isin(chosen)].copy()
display(selected_groups[['method','seed','phase','regime','scenario','group','score','n_rows','n_undefined','threshold','flagged','rate','rate_name']])
print('Full candidate/group table:', DG/'diagnostic_groups.csv')
print('No alternate candidate is substituted if the fixed selection fails.')
"""),
        code("""fig, axes = plt.subplots(1,2,figsize=(15,5.5))
check_pairs = [(r['method'],r['seed']) for r in DIAG['gate']['checks']]
positions = [(method,seed,regime) for method,seed in check_pairs for regime in REGIMES]
ticks = [LABELS[method]+'\\n'+str(seed)+' · '+regime for method,seed,regime in positions]
for column, score in enumerate(chosen):
    offset = (column-(len(chosen)-1)/2)*.28
    color = '#0072B2' if score==DIAG['baseline_score'] else '#D55E00'
    recalls, fprs = [], []
    for method,seed,regime in positions:
        rows = selected_groups[(selected_groups.method==method)&(selected_groups.seed==seed)&(selected_groups.regime==regime)&(selected_groups.score==score)]
        valid = not rows.empty and rows.n_undefined.eq(0).all() and rows.rate.notna().all()
        attacks = rows[rows.attack]
        honest = rows[~rows.attack]
        recalls.append(float(attacks.rate.iloc[0]) if valid and len(attacks)==1 else np.nan)
        fprs.append(float(honest.rate.max()) if valid and len(honest)==6 else np.nan)
    for ax, values in zip(axes,[recalls,fprs]):
        coordinates = np.arange(len(positions))+offset
        ax.bar(coordinates,values,width=.26,color=color,label=score)
        for position, value in zip(coordinates,values):
            label = (f'{100*value:.1f}'.rstrip('0').rstrip('.')+'%') if np.isfinite(value) else 'NA'
            height = min(value+.012,1.01) if np.isfinite(value) else .012
            ax.text(position,height,label,ha='center',va='bottom',fontsize=8,color=color)
for ax,title,line in zip(axes,['Attack recall','Worst honest-subgroup false-positive rate'],[.8,.05]):
    ax.axhline(line,color='#555555',linestyle='--',lw=1,label='Declared bound')
    ax.set_title(title); ax.set_xticks(range(len(positions)),ticks,rotation=35,ha='right')
    ax.set_ylim(0,1.03); ax.grid(axis='y',alpha=.2); ax.legend(frameon=False,fontsize=8)
fig.suptitle('Fixed-score trajectory checks; labels show observed rates')
if not np.isfinite(selected_groups.rate).all() or selected_groups.n_undefined.gt(0).any():
    print('NA denotes undefined or incomplete evidence; 0% is an observed zero rate.')
fig.tight_layout()
export_figure(fig,'mechanism_diagnostic_checks')
"""),
        md("""## Common-broadcast cross-check

The clean FedAvg broadcasts immediately before rounds 5, 8, 12 and 20 are reused for counterfactual clean, honest-noise, honest-stripe and label-flip local training. For each checkpoint, the model, client images and minibatch seeds match. Clean updates must exactly replay the original clean submissions. The resulting aggregates are discarded: this is not another learning trajectory or accuracy evaluation.

Only the previously selected score and whole-cosine baseline are examined at their frozen thresholds. Temporal scores remain undefined because these independent probes have no history. A contradiction requires investigation; a favourable probe cannot rescue a failed trajectory gate."""),
        code("""print('Probe observations:',PROBE['n_observations'])
print('Selected score:',PROBE['selected_score'],'; status:',PROBE['selected_score_status'])
print('Frozen thresholds:',PROBE['thresholds'])
print('Trajectory gate unchanged:',not PROBE['trajectory_gate_modified'])
print('Descriptive contradiction:',PROBE['descriptive_probe_contradiction'])
display(pd.DataFrame(PROBE['selected_score_slices']))
probe_groups = pd.DataFrame(PROBE['groups'])
display(probe_groups[['seed','regime','scenario','group','score','role','n_rows','n_undefined','threshold','flagged','rate','rate_name']])
print(PROBE['interpretation'])
"""),
        md("""## Optional offline interpretation: where NNM's coefficients come from

The table below sums effective coefficients on the raw updates of actively attacking or honestly changed clients, then averages rounds 5–24 within each seed. Entries report the available-seed mean and seed count: prototype noise results with `n=1` are not two-seed evidence. Clipping can reduce total coefficient mass below one, so raw masses are not necessarily normalized shares.

The NNM decomposition shows the fraction of a group's coefficient mass arriving through mixtures anchored by clients in another group. This explains how mixing redistributes coefficients; it is accounting, not accuracy, intent detection, or proof that an attack succeeded. This post-pilot interpretation cannot replace or rescue the planned diagnostic gate."""),
        code("""mixing_path = A/'nnm_mixing_audit.json'
if mixing_path.exists():
    mixing = json.loads(mixing_path.read_text())
    assert mixing['manifest_sha256'] == file_sha256(STUDY/'result_manifest.json')
    assert mixing['source_sha256'] == file_sha256(CODE/'nnm_mixing_audit.py')
    audit_means = pd.DataFrame(mixing['available_seed_group_means'])
    audit_methods = ['FedAvg','RFA','FLTrust','FedReTrust-Direction-loose','NNM-RFA']
    selected_mass = audit_means[audit_means.method.isin(audit_methods) & audit_means.group.isin(['attack','honest_changed'])].copy()
    selected_mass['mass_and_seed_count'] = selected_mass.apply(
        lambda row: f"{row['group_effective_coefficient_mass']:.4f} (n={int(row['seed_count'])})", axis=1)
    display(Markdown('**Effective group coefficient mass — available-seed means with explicit coverage**'))
    display(selected_mass.pivot(index=['regime','scenario','group'],columns='method',values='mass_and_seed_count')
            .reindex(columns=audit_methods).rename(columns=LABELS))
    through_others = selected_mass[selected_mass.method=='NNM-RFA'].copy()
    through_others['mass_through_other_group_anchors_pct'] = 100*through_others.group_coefficient_fraction_from_other_group_anchors
    display(Markdown('**NNM fraction of each group’s coefficient mass routed through other-group anchors**'))
    display(through_others[['regime','scenario','group','seed_count','mass_through_other_group_anchors_pct']]
            .set_index(['regime','scenario','group']).round(2))
    print('Maximum independently checked A-transpose-w coefficient expansion error:', mixing['maximum_nnm_coefficient_expansion_error'])
else:
    print('Optional NNM mixing audit is unavailable; no explanation is inferred from missing coefficients.')
"""),
        md("""## Compute, influence and evidence limits

The observer adds measurement work, including an extra RFA reference calculation, to both new methods. Algorithm timings from the inner contribution recorder exclude this outer observer; round and worker timings include it. They must not be confused. Historical costs were collected earlier, and workers share CPU resources, so these are descriptive timings rather than isolated performance benchmarks.

NNM's original-client contribution coefficients are reconstructed through the mixing matrix. A mixture indexed by client i is not exclusively that client's contribution. This bounded persistent-scenario pilot does not measure new recovery behaviour, and it does not justify a recovery claim from trust or coefficients alone."""),
        code("""costs = pd.DataFrame(json.loads((A/'costs.json').read_text()))
cost = costs.groupby('method')[['median_defence_seconds','median_recorder_seconds','median_training_seconds','median_evaluation_seconds','median_round_seconds']].median().reindex(KEEP)
overhead = []
for result in observed:
    diagnostics = json.loads((ROOT/result['run_path']/'aggregation_diagnostics.json').read_text())
    overhead.append({'method':result['job']['method'],
                     'observer_seconds':float(np.median([r['values']['update_observation']['diagnostics_seconds'] for r in diagnostics]))})
cost['median_observer_seconds'] = pd.DataFrame(overhead).groupby('method').observer_seconds.median()
cost['trusted_reference_images'] = [300 if method in ['FLTrust','FedReTrust'] else 0 for method in cost.index]
cost.index = [LABELS[method] for method in cost.index]
display(cost.round(5))
cost.to_csv(A/'notebook_costs_and_access.csv')
print('Observer overhead is available only for the two newly observed methods; blanks are not zero.')
print('Figures exported as PNG and PDF:',FIGURES)
"""),
        md("""## Interpretation and reproduction

The aggregation comparison and diagnostic gate must be interpreted separately. NNM–RFA can improve or fail without validating a new layer-aware FedReTrust mechanism. A passing score screen would justify a separately declared candidate study; it would not establish useful aggregation, reliable rehabilitation, general separation of honest and malicious behaviour, or novelty. A failed screen rules out this fixed selection under the declared gate, not every possible mechanism.

This study uses MNIST, ten fully participating clients, one fixed IID and one fixed non-IID partition, the same affected cohort, 24 rounds, and two previously used development seeds. It adds no adaptive, backdoor, Sybil, fabricated-count, unseen-domain or privacy guarantee. Honest minority clients can differ from consensus and coordinated attackers can resemble it. Published layer weighting, clustering, temporal evidence and mixing mechanisms are existing ideas; merely combining them is not a novelty argument.

Reproduce the presentation only after the declared pilot has completed:

```sh
Experiment/.venv/bin/python Experiment/code/mechanism_notebook.py
```

The builder verifies completed study status, frozen source hashes and saved comparison artifacts before execution. Notebook cells read saved results and export figures; they do not launch training, evaluate model checkpoints or change the diagnostic decision. Earlier studies remain preserved."""),
    ]
    notebook = nbf.v4.new_notebook(cells=cells)
    notebook.metadata.kernelspec = {"display_name": "FedReTrust Experiment (Python 3.14)",
                                    "language": "python", "name": "fedretrust-experiment"}
    return notebook


def build(study=DEFAULT_STUDY, output=DEFAULT_NOTEBOOK, execute=True):
    """Require completed results before writing or executing this review."""
    study, output = Path(study).resolve(), Path(output).resolve()
    status = json.loads((study / "status.json").read_text())
    if status.get("status") != "complete" or status.get("new_runs") != 32:
        raise ValueError("The 32-trajectory pilot and common-broadcast analysis must complete first.")
    for name in ("protocol.json", "result_manifest.json", "analysis/target_check.json",
                 "analysis/diagnostics/diagnostic_summary.json",
                 "analysis/diagnostics/probe_score_summary.json", "probes/summary.json"):
        if not (study / name).is_file():
            raise ValueError(f"Missing completed review artifact: {name}")
    from mechanism_pilot import verify
    verify(study)
    notebook = make_notebook(study)
    if execute:
        os.environ.setdefault("IPYTHONDIR", "/tmp/fedretrust_ipython")
        from IPython.core.interactiveshell import InteractiveShell
        from IPython.utils.capture import capture_output
        shell = InteractiveShell.instance()
        previous = Path.cwd()
        os.chdir(ROOT / "code")
        try:
            execution_count = 0
            for cell in notebook.cells:
                if cell.cell_type != "code":
                    continue
                execution_count += 1
                with capture_output() as captured:
                    result = shell.run_cell(cell.source, store_history=False)
                cell.execution_count = execution_count
                cell.outputs = []
                if captured.stdout:
                    cell.outputs.append(nbf.v4.new_output("stream", name="stdout", text=captured.stdout))
                if captured.stderr:
                    cell.outputs.append(nbf.v4.new_output("stream", name="stderr", text=captured.stderr))
                for item in captured.outputs:
                    cell.outputs.append(nbf.v4.new_output("display_data", data=item.data, metadata=item.metadata))
                if result.error_before_exec or result.error_in_exec:
                    raise RuntimeError(f"Notebook cell {execution_count}: {result.error_before_exec or result.error_in_exec}")
        finally:
            os.chdir(previous)
        notebook.metadata["execution_note"] = (
            "All cells executed with IPython InteractiveShell in the project environment after completed pilot verification. "
            "Saved development and probe artifacts only; no model training or evaluation.")
    nbf.validate(notebook)
    output.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, output)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", type=Path, default=DEFAULT_STUDY)
    parser.add_argument("--output", type=Path, default=DEFAULT_NOTEBOOK)
    parser.add_argument("--no-execute", action="store_true")
    arguments = parser.parse_args()
    print(build(arguments.study, arguments.output, execute=not arguments.no_execute))
