"""Reporting-only consolidation of the final frozen research evidence.

Reads saved measurements. Never trains, evaluates checkpoints, changes scientific
sources, alters gates, or writes into an existing scientific study directory.
Run from any working directory with Experiment/.venv/bin/python.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import html
import importlib.util
import json
import math
import os
import shutil

os.environ.setdefault('MPLCONFIGDIR', '/tmp/fedretrust_final_results_matplotlib')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nbformat as nbf

PROJECT = Path(__file__).resolve().parents[2]
EXP = PROJECT / 'Experiment'
OUT = EXP / 'outputs/final_results/20261031_v06'
FREEZE = EXP / 'outputs/research_freeze/20261031_v06_final'
ORIGINAL = EXP / 'outputs/validation_replication/20261030_v03_v1'
V05 = EXP / 'outputs/reference_free/20261031_v05_followup'
V06 = EXP / 'outputs/reference_free/20261031_v06_followup'
NOTEBOOK = EXP / 'code/34_fedretrust_final_results.ipynb'
T4 = 2.7764451051977987
KEY = ['method', 'regime', 'block', 'seed', 'scenario', 'job_id']
V3, V5, V6 = 'FedReTrust-Direction-loose', 'FedReTrust-v05', 'FedReTrust-v06'
MAIN = ['FedAvg', 'RFA', 'FLTrust', 'Viceroy', V3, V5, V6,
        'FLAURA-reconstruction', 'FLAURA-sigma-double',
        'FedCognis-aggregation', 'FedCognis-squared-distance']
BASELINES = [m for m in MAIN if m not in (V3, V5, V6)]
LABEL = {m: m for m in MAIN}
LABEL.update({V3: 'FedReTrust v0.3', V5: 'FedReTrust v0.5', V6: 'FedReTrust v0.6',
              'FLAURA-reconstruction': 'FLAURA reconstruction',
              'FLAURA-sigma-double': 'FLAURA double sigma',
              'FedCognis-aggregation': 'FedCognis aggregation',
              'FedCognis-squared-distance': 'FedCognis squared distance'})
CORE = [('clean', 'overall_mean_5_24_pp', 'Clean accuracy (%)', 'higher'),
        ('abrupt_shot_noise', 'affected_honest_event_pp', 'Honest noise accuracy (%)', 'higher'),
        ('gradual_stripe', 'affected_honest_event_pp', 'Honest stripe accuracy (%)', 'higher'),
        ('label_flip', 'overall_mean_5_24_pp', 'Attacked accuracy (%)', 'higher'),
        ('label_flip', 'attack_active_signed_damage_pp', 'Label-flip damage (pp)', 'lower')]
PALETTE = {'FedAvg': '#707070', 'RFA': '#246a9b', 'FLTrust': '#b66d1d',
           V3: '#697f22', V5: '#16836e', V6: '#8043ad', 'Viceroy': '#9c4d48'}
FIGURES = []
INPUTS = {}
TABLE_INFO = {}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda: f.read(1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def rel(path):
    return str(Path(path).relative_to(PROJECT))


def read_json(path):
    INPUTS[rel(path)] = sha(path)
    return json.loads(Path(path).read_text())


def read_csv(path):
    INPUTS[rel(path)] = sha(path)
    return pd.read_csv(path)


def save_json(name, value):
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def csv(name, frame, description):
    path = OUT / 'tables' / (name + '.csv')
    frame.to_csv(path, index=False, float_format='%.15g', na_rep='NA')
    TABLE_INFO[name] = {'path': str(path.relative_to(OUT)), 'rows': len(frame),
                        'description': description}
    return frame


def stats(group, col='value'):
    g = group.sort_values('block')
    assert g.block.tolist() == [1, 2, 3, 4, 5], 'Incomplete/duplicated block group'
    vals = g[col].to_numpy(dtype=float)
    assert np.isfinite(vals).all()
    mean, sd = float(vals.mean()), float(vals.std(ddof=1))
    half = T4 * sd / math.sqrt(5)
    return {'n': 5, **{f'block_{i}': float(v) for i, v in enumerate(vals, 1)},
            'mean': mean, 'sd': sd, 'min': float(vals.min()), 'max': float(vals.max()),
            'ci_low': mean - half, 'ci_high': mean + half}


def summarise(frame, groups, col='value'):
    return pd.DataFrame([{**dict(zip(groups, identity)), **stats(g, col)}
                         for identity, g in frame.groupby(groups, dropna=False)])


def combine(name):
    pieces = []
    for study, method, origin in [(ORIGINAL, None, 'original_800'),
                                  (V05, V5, 'v05_40'), (V06, V6, 'v06_40')]:
        d = read_csv(study / 'analysis' / (name + '.csv'))
        if method:
            d = d[d.method.eq(method)].copy()
        d['source_table'] = rel(study / 'analysis' / (name + '.csv'))
        d['evidence_origin'] = origin
        pieces.append(d)
    return pd.concat(pieces, ignore_index=True)


def integrity():
    spec = importlib.util.spec_from_file_location('final_freeze_check', FREEZE / 'verify_freeze.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.verify()


def paired(metrics):
    rows = []
    # Match only scenarios measured for both methods. Never silently discard a block.
    groups = ['regime', 'scenario', 'metric']
    join = groups + ['block', 'seed']
    for target, others in [(V6, [m for m in MAIN if m != V6]),
                           (V5, ['FedAvg', 'RFA', V3])]:
        a = metrics[metrics.method.eq(target)]
        for comparator in others:
            b = metrics[metrics.method.eq(comparator)]
            shared = a[groups].drop_duplicates().merge(b[groups].drop_duplicates(), on=groups,
                                                     validate='one_to_one')
            aa, bb = a.merge(shared, on=groups), b.merge(shared, on=groups)
            d = aa.merge(bb, on=join, how='outer', suffixes=('_target', '_other'),
                         validate='one_to_one', indicator=True)
            assert len(d) and d['_merge'].eq('both').all()
            d['difference'] = d.value_target - d.value_other
            for identity, g in d.groupby(groups):
                rows.append({'candidate': target, 'comparator': comparator,
                             **dict(zip(groups, identity)), **stats(g, 'difference')})
    return pd.DataFrame(rows)


def save_figure(fig, name, title, caption, sources):
    fig.savefig(OUT / 'figures' / (name + '.png'), dpi=180, bbox_inches='tight', facecolor='white')
    fig.savefig(OUT / 'figures' / (name + '.pdf'), bbox_inches='tight', facecolor='white',
                metadata={'Title': title, 'Subject': caption, 'Creator': 'Frozen-results presentation builder'})
    plt.close(fig)
    FIGURES.append({'name': name, 'title': title, 'caption': caption, 'sources': sources,
                    'png': f'figures/{name}.png', 'pdf': f'figures/{name}.pdf'})


def figures(means, curves, effects):
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'ps.fonttype': 42})
    for regime in ['IID', 'non-IID']:
        fig, axes = plt.subplots(1, 5, figsize=(19, 7), sharey=True)
        for ax, (scenario, metric, title, direction) in zip(axes, CORE):
            for i, method in enumerate(MAIN):
                d = means[(means.method == method) & (means.regime == regime) &
                          (means.scenario == scenario) & (means.metric == metric)]
                if d.empty:
                    ax.text(.5, i, 'Not tested', transform=ax.get_yaxis_transform(),
                            ha='center', va='center', fontsize=9, color='#777777')
                    continue
                r = d.iloc[0]
                color = PALETTE.get(method, '#455e73')
                ax.plot([r.ci_low, r.ci_high], [i, i], color=color, lw=1.5)
                ax.scatter(r['mean'], i, s=44 if method == V6 else 24, marker='D' if method == V6 else 'o', color=color)
            ax.set_title(title.replace(' accuracy', '\naccuracy').replace(' damage', '\ndamage'), fontsize=11)
            ax.set_xlabel(direction + ' is better')
            ax.grid(axis='x', alpha=.18)
            ax.set_yticks(range(len(MAIN)), [LABEL[m] for m in MAIN])
        axes[0].invert_yaxis()
        fig.suptitle(f'{regime}: five paired development blocks', fontsize=16)
        fig.text(.5, .02, 'Points = means; lines = nominal 95% t intervals (df4). Rounds 5–24. Damage uses each method’s own clean control.', ha='center')
        fig.tight_layout(rect=(0, .06, 1, .94))
        save_figure(fig, 'core_' + regime.replace('-', '_'), f'{regime} core results',
                    'Accuracy is percent; damage is percentage points. Honest noise/stripe accuracy uses affected clients’ legitimate tasks. '
                    'All plotted cells have five blocks. FLAURA/FedCognis noise cells were not tested. Intervals are exploratory, not simultaneous.',
                    ['tables/core_results.csv', 'tables/means_all_metrics.csv'])

    fig, axes = plt.subplots(2, 4, figsize=(17, 8), sharex=True)
    for i, regime in enumerate(['IID', 'non-IID']):
        for j, (scenario, group, title) in enumerate([
                ('clean', 'all_clients', 'Clean'), ('abrupt_shot_noise', 'affected_honest', 'Honest noise'),
                ('gradual_stripe', 'affected_honest', 'Honest stripes'), ('label_flip', 'all_clients', 'Persistent label flipping')]):
            ax = axes[i, j]
            for method in ['FedAvg', 'RFA', V5, V6]:
                d = curves[(curves.method == method) & (curves.regime == regime) &
                           (curves.scenario == scenario) & (curves['group'] == group)]
                t = d.groupby('round').accuracy_pp.agg(['mean', 'std', 'count'])
                assert t['count'].eq(5).all() and len(t) == 24
                half = T4 * t['std'] / np.sqrt(5)
                ax.plot(t.index, t['mean'], label=LABEL[method], color=PALETTE[method], lw=1.7)
                ax.fill_between(t.index, t['mean'] - half, t['mean'] + half, color=PALETTE[method], alpha=.09)
            ax.axvline(5, ls=':', color='#444444', lw=1)
            ax.set_title(f'{regime}: {title}')
            ax.set_xlabel('Round'); ax.set_ylabel('Legitimate-task accuracy (%)')
            ax.set_xticks([1, 5, 10, 15, 20, 24]); ax.grid(alpha=.15)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, frameon=False)
    fig.suptitle('Accuracy trajectories: mean and nominal 95% intervals across five blocks', fontsize=15)
    fig.tight_layout(rect=(0, .06, 1, .95))
    save_figure(fig, 'core_trajectories', 'Core learning trajectories',
                'Four clean warm-up rounds precede interventions at round 5. Curves show all clients for clean/attack and affected honest clients for noise/stripes. '
                'Shading is pointwise across five blocks, not a simultaneous confidence band. Five blocks are repeated runs, not additional rounds.', ['tables/accuracy_curves.csv'])

    fig, axes = plt.subplots(2, 4, figsize=(17, 7))
    refs = [V5, 'RFA', 'FLTrust', 'FedAvg']
    for i, regime in enumerate(['IID', 'non-IID']):
        for j, (scenario, metric, title, direction) in enumerate(CORE[1:]):
            ax = axes[i, j]
            for k, comparator in enumerate(refs):
                r = effects[(effects.candidate == V6) & (effects.comparator == comparator) &
                            (effects.regime == regime) & (effects.scenario == scenario) & (effects.metric == metric)].iloc[0]
                color = PALETTE[comparator]
                ax.scatter([r[f'block_{b}'] for b in range(1, 6)], k + np.linspace(-.12, .12, 5), s=18, color=color, alpha=.5)
                ax.plot([r.ci_low, r.ci_high], [k, k], color=color, lw=2)
                ax.scatter(r['mean'], k, marker='D', color=color, s=32)
            ax.axvline(0, color='#555555', ls=':', lw=1)
            ax.set_yticks(range(4), ['v0.5', 'RFA', 'FLTrust', 'FedAvg']); ax.set_ylim(-.5, 3.5)
            ax.set_title(f'{regime}: ' + title.split(' (')[0] + '\n' + ('positive favours v0.6' if direction == 'higher' else 'negative favours v0.6'), fontsize=10)
            ax.set_xlabel('v0.6 − comparator (pp)'); ax.grid(axis='x', alpha=.15)
    fig.suptitle('Paired differences: five block dots, mean diamond and nominal 95% t interval', fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, .94))
    save_figure(fig, 'paired_v06', 'v0.6 paired comparisons',
                'Differences use the same five block/seed pairs. Accuracy benefits are positive; damage benefits are negative. '
                'Intervals are exploratory, assume approximately normal independent block differences and have no table-wide coverage.', ['tables/paired_effects.csv'])

    fig, axes = plt.subplots(2, 5, figsize=(18, 7), sharex=True, sharey=True)
    for i, regime in enumerate(['IID', 'non-IID']):
        for j, block in enumerate(range(1, 6)):
            ax = axes[i, j]
            for method in ['FedAvg', 'RFA', V5, V6]:
                d = curves[(curves.method == method) & (curves.regime == regime) & (curves.block == block) &
                           (curves.scenario == 'label_flip') & (curves['group'] == 'all_clients')].sort_values('round')
                ax.plot(d['round'], d.accuracy_pp, color=PALETTE[method], label=LABEL[method], lw=1.5)
            ax.axvline(5, color='#555555', ls=':', lw=1)
            ax.set_title(f'{regime}, block {block}'); ax.set_xticks([1, 5, 12, 18, 24])
            ax.set_xlabel('Round'); ax.grid(alpha=.15)
            if j == 0: ax.set_ylabel('Attacked accuracy (%)')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, frameon=False)
    fig.suptitle('Persistent label flipping: every block, including onset and later drops', fontsize=15)
    fig.tight_layout(rect=(0, .06, 1, .95))
    save_figure(fig, 'label_flip_blocks', 'Every label-flip trajectory',
                'All five known development blocks are shown separately. Attack starts at round 5 and continues through round 24. '
                'No block is omitted and there is no inference from a single final round.', ['tables/accuracy_curves.csv', 'tables/onset_by_block.csv'])


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    for name in ['tables', 'figures', 'evidence']:
        (OUT / name).mkdir(exist_ok=True)
    before = integrity()
    evidence = read_json(FREEZE / 'evidence_manifest.json')
    selection = read_json(FREEZE / 'selection.json')
    protocol = read_json(EXP / 'outputs/validation_plan/20261030_v03_v1/protocol.json')
    frozen = read_json(FREEZE / 'research_freeze.json')
    original_freeze = read_json(ORIGINAL / 'execution_freeze.json')
    records = evidence['records']
    index = pd.DataFrame([{**r['job'], 'job_id': r['job_id'], 'run_path': r['run_path'],
                           'evidence_origin': r['evidence_origin']} for r in records])
    assert len(index) == 880 and not index.duplicated(KEY[:3] + ['scenario']).any()
    csv('run_index', index, 'One row per distinct frozen trajectory. Raw paths are relative to the project root.')

    metrics, curves, costs, local = [combine(n) for n in ['block_metrics', 'accuracy_curves', 'costs', 'local_training']]
    assert set(metrics.job_id) == set(index.job_id) == set(curves.job_id) == set(costs.job_id) == set(local.job_id)
    assert len(costs) == 880 and len(local) == 211200
    assert not metrics.duplicated(KEY + ['metric']).any()
    assert not curves.duplicated(KEY + ['group', 'round']).any()
    assert not local.duplicated(['job_id', 'client', 'round']).any()
    assert local.groupby('job_id').size().eq(240).all()
    assert local.examples.eq(512).all() and local.optimizer_steps.eq(8).all()

    # New versions' identical zero-reference workload has no root backward pass.
    mask = costs.method.isin([V5, V6])
    for col in ['reference_backward_examples', 'reference_optimizer_steps']:
        costs.loc[mask, col] = 0
    metrics['derivation'] = 'unchanged_frozen_analysis'
    extra, onset = [], []
    def append_metric(ident, name, value, source):
        hit = metrics[(metrics.job_id == ident['job_id']) & (metrics.metric == name)]
        if len(hit):
            assert len(hit) == 1 and np.isclose(hit.iloc[0].value, value, atol=1e-10, rtol=0), name
        else:
            extra.append({**ident, 'metric': name, 'value': value, 'source_table': source,
                          'derivation': 'saved_curve_arithmetic_equivalent_metric'})
    for identity, g in curves[(curves.scenario == 'label_flip') & (curves['group'] == 'all_clients')].groupby(KEY):
        ident = dict(zip(KEY, identity)); g = g.set_index('round').sort_index()
        clean = curves[(curves.method == ident['method']) & (curves.regime == ident['regime']) &
                       (curves.block == ident['block']) & (curves.scenario == 'clean') &
                       (curves['group'] == 'all_clients')].set_index('round').sort_index()
        assert len(clean) == len(g) == 24
        a, c = g.accuracy_pp, clean.accuracy_pp
        vals = {'overall_round5_pp': a.loc[5], 'attack_onset_signed_damage_pp': c.loc[5] - a.loc[5],
                'onset_raw_drop_pp': a.loc[4] - a.loc[5], 'attack_active_accuracy_pp': a.loc[5:24].mean(),
                'matched_reference_attack_window_accuracy_pp': c.loc[5:24].mean()}
        for name, val in vals.items(): append_metric(ident, name, float(val), 'tables/accuracy_curves.csv')
        losses = (a.shift(1) - a).loc[5:24]
        onset.append({**ident, **vals, 'round4_accuracy_pp': a.loc[4],
                      'maximum_active_one_round_drop_pp': max(0., losses.max()),
                      'minimum_active_accuracy_pp': a.loc[5:24].min(),
                      'scope': 'secondary_descriptive; fixed onset gates preserved separately'})
    metrics = pd.concat([metrics, pd.DataFrame(extra)], ignore_index=True)
    means = summarise(metrics, ['method', 'regime', 'scenario', 'metric'])
    effects = paired(metrics)
    csv('block_metrics', metrics, 'All scalar frozen metrics plus explicitly labelled equivalent label-flip onset/active-window arithmetic. One measurement per block.')
    csv('means_all_metrics', means, 'All measured method/regime/scenario/metric cells. Five block values, sample SD, range and nominal 95% t interval.')
    csv('paired_effects', effects, 'Candidate minus comparator in shared measured scenarios. Five matched block/seed differences. No pooling across regimes.')
    csv('accuracy_curves', curves, 'Saved legitimate-task accuracy (%) by round, method, scenario, cohort and block.')
    csv('onset_by_block', pd.DataFrame(onset), 'Label-flip round-5 accuracy and own-clean damage, raw onset change and later fragility. All 14 methods.')

    # Verify imported means against every original numeric summary and the final selection.
    comparisons = 0
    for path, selected in [(ORIGINAL / 'analysis/means.csv', None), (V05 / 'analysis/means.csv', V5),
                           (V06 / 'analysis/means.csv', V6)]:
        old = read_csv(path)
        if selected: old = old[old.method.eq(selected)]
        match = old.merge(means, on=['method', 'regime', 'scenario', 'metric'], suffixes=('_old', '_new'), validate='one_to_one')
        assert len(match) == len(old)
        for c in ['mean', 'sd', 'ci_low', 'ci_high']:
            assert np.allclose(match[c + '_old'], match[c + '_new'], atol=1e-10, rtol=0), (path, c)
        comparisons += len(match)
    # Independently check whole-trajectory means, tail and final values from saved curves.
    for (job_id, metric), g in metrics[metrics.metric.isin(['overall_mean_5_24_pp', 'overall_tail_21_24_pp', 'overall_round24_pp'])].groupby(['job_id', 'metric']):
        c = curves[(curves.job_id == job_id) & (curves['group'] == 'all_clients')]
        start = {'overall_mean_5_24_pp': 5, 'overall_tail_21_24_pp': 21, 'overall_round24_pp': 24}[metric]
        assert np.isclose(g.value.iloc[0], c[c['round'].ge(start)].accuracy_pp.mean(), atol=1e-10, rtol=0)

    core_rows = []
    for regime in ['IID', 'non-IID']:
        for method in MAIN:
            for scenario, metric, title, direction in CORE:
                d = means[(means.method == method) & (means.regime == regime) & (means.scenario == scenario) & (means.metric == metric)]
                ident = {'regime': regime, 'method': method, 'display_name': LABEL[method],
                         'measurement': title, 'direction': direction, 'scenario': scenario, 'metric': metric}
                core_rows.append({**ident, 'status': 'measured' if len(d) else 'not_tested',
                                  **(d.iloc[0].to_dict() if len(d) else {})})
    core = csv('core_results', pd.DataFrame(core_rows), 'Main 11-method table. Explicit not_tested cells with NA numbers. Means and five-block uncertainty.')
    for regime in ['IID', 'non-IID']:
        q = core[core.regime.eq(regime)].pivot(index='method', columns='measurement', values='mean').reindex(MAIN)
        csv('core_' + regime.replace('-', '_'), q.rename(index=LABEL).reset_index(), 'Compact mean table; precision retained. Use core_results for SD, CI and each block.')
    coverage = []
    for method in sorted(index.method.unique()):
        for regime in ['IID', 'non-IID']:
            for scenario in sorted(index.scenario.unique()):
                d = index[(index.method == method) & (index.regime == regime) & (index.scenario == scenario)]
                assert len(d) in (0, 5)
                coverage.append({'method': method, 'regime': regime, 'scenario': scenario,
                                 'runs': len(d), 'status': 'measured_five_blocks' if len(d) else 'not_tested',
                                 'role': 'v03_component_control' if method not in MAIN else ('proposed_version' if method in [V3, V5, V6] else 'baseline')})
    csv('coverage', pd.DataFrame(coverage), 'Full method × regime × scenario coverage. Zero here means no runs, never zero performance.')

    strict = []
    for regime in ['IID', 'non-IID']:
        for scenario, metric, title, direction in CORE:
            eligible = means[(means.method.isin(BASELINES)) & (means.regime == regime) & (means.scenario == scenario) & (means.metric == metric)]
            best = eligible.sort_values('mean', ascending=direction == 'lower').iloc[0]
            candidate = means[(means.method == V6) & (means.regime == regime) & (means.scenario == scenario) & (means.metric == metric)].iloc[0]
            strict.append({'regime': regime, 'scenario': scenario, 'metric': metric, 'measurement': title,
                           'candidate_mean': candidate['mean'], 'best_measured_baseline': best.method,
                           'best_baseline_mean': best['mean'], 'candidate_minus_best_pp': candidate['mean'] - best['mean'],
                           'candidate_point_at_least_as_good': bool(candidate['mean'] <= best['mean'] if direction == 'lower' else candidate['mean'] >= best['mean']),
                           'eligible_baselines': ';'.join(sorted(eligible.method)),
                           'scope': 'posthoc_descriptive_ranking_only; not_a_new_or_revised_gate'})
    csv('descriptive_best_baseline', pd.DataFrame(strict), 'Best mean among baselines actually measured for each cell. Post hoc description, not a significance claim or altered success criterion.')

    # Summarise local online loss by the original affected/unchanged client roles.
    local['client_group'] = np.where(local.client.isin([0, 1, 2]), 'clients_0_1_2', 'clients_3_9')
    train_rows = []
    for window, start, end in [('all_1_24', 1, 24), ('primary_5_24', 5, 24), ('tail_21_24', 21, 24), ('final_24', 24, 24)]:
        d = local[local['round'].between(start, end)]
        for group in ['all_clients', 'clients_0_1_2', 'clients_3_9']:
            part = d if group == 'all_clients' else d[d.client_group.eq(group)]
            t = part.groupby(KEY, as_index=False).agg(value=('online_train_loss', 'mean'), client_rounds=('client', 'size'))
            t['window'] = window; t['client_group'] = group
            train_rows.append(t)
    training = pd.concat(train_rows, ignore_index=True)
    csv('training_loss_by_block', training, 'Online local cross-entropy on the batches used during SGD. May use poisoned labels. No training-accuracy metric was recorded. Group names denote client IDs, not inferred intent.')
    csv('training_loss_summary', summarise(training, ['method', 'regime', 'scenario', 'window', 'client_group']), 'Five-block summaries of online training loss, separate from held-out legitimate-task accuracy.')

    task_rows, access_rows, config_rows, class_rows = [], [], [], []
    common_configs, scenarios = {}, {}
    for r in records:
        ident = {k: r['job'][k] if k != 'job_id' else r['job_id'] for k in KEY}
        runpath = PROJECT / r['run_path']
        run = read_json(runpath / 'run.json')
        access = read_json(runpath / 'validation_access.json')
        assert run['official_test_used'] is False and access['official_test_used'] is False
        assert access['defence_uses_evaluation_data'] is False
        assert run['parameters'] == 421642 and run['device'] == 'cpu'
        config = {k: v for k, v in run['config'].items() if k not in ['development_seed', 'partition_id']}
        common_configs[json.dumps(config, sort_keys=True)] = config
        s = run['scenario']; scenarios.setdefault(s['name'], s)
        assert scenarios[s['name']] == s
        config_rows.append({**ident, 'method_settings_json': json.dumps(run['method'], sort_keys=True),
                            'partition_id': run['config']['partition_id'], 'partition_sha256': run['partition_sha256'],
                            'initial_state_sha256': run['initial_state_sha256'], 'selection_sha256': run['selection_sha256']})
        access_rows.append({**ident, **access})
        tasks = read_csv(runpath / 'task_metrics.csv')
        assert len(tasks) == 75 and tasks.examples.eq(1000).all()
        for window, start, end in [('initial_0', 0, 0), ('warmup_1_4', 1, 4), ('primary_5_24', 5, 24), ('tail_21_24', 21, 24), ('final_24', 24, 24)]:
            d = tasks[tasks['round'].between(start, end)]
            for task, g in d.groupby('task'):
                task_rows.append({**ident, 'window': window, 'task': task,
                                  'accuracy_percent': 100 * g.accuracy.mean(), 'cross_entropy': g.loss.mean(),
                                  'evaluation_examples_per_round': 1000})
                if window == 'primary_5_24':
                    for digit in range(10):
                        assert g[f'class_examples_{digit}'].eq(100).all()
                        class_rows.append({**ident, 'task': task, 'digit': digit,
                                           'accuracy_percent': 100 * g[f'class_correct_{digit}'].sum() / g[f'class_examples_{digit}'].sum(),
                                           'cross_entropy': g[f'class_loss_sum_{digit}'].sum() / g[f'class_examples_{digit}'].sum()})
    assert len(common_configs) == 1
    task_table = csv('evaluation_tasks_by_block', pd.DataFrame(task_rows), 'Fixed balanced development tasks, 1000 images/task. These unweighted task scores differ from population-weighted legitimate-task scores in the main comparison.')
    for value in ['accuracy_percent', 'cross_entropy']:
        csv('evaluation_tasks_' + value + '_summary', summarise(task_table, ['method', 'regime', 'scenario', 'window', 'task'], value), 'Five-block fixed-task summaries. Do not substitute for affected-cohort or population-weighted main outcomes.')
    csv('per_class_evaluation_by_block', pd.DataFrame(class_rows), 'Per-digit development task accuracy and cross-entropy over rounds 5–24. All 880 trajectories retained.')
    csv('run_settings', pd.DataFrame(config_rows), 'Exact saved per-run algorithm settings and pairing identifiers. JSON fields preserve algorithm-specific parameters.')
    access = pd.DataFrame(access_rows)
    csv('data_access_by_run', access, 'Verified saved constructor/reference access for every run. Evaluator data and official test are not aggregation inputs.')
    methods = []
    for method, g in access.groupby('method'):
        refs = sorted(g.server_reference_examples.unique().tolist())
        assert len(refs) == 1
        methods.append({'method': method, 'display_name': LABEL.get(method, method), 'runs': len(g),
                        'server_reference_images': refs[0],
                        'reference_role': '300 clean root-training images' if method == 'FLTrust' else ('100 auxiliary images plus earlier frozen calibration' if method.startswith('FLAURA') else 'no server reference images'),
                        'implementation_scope': 'documented reconstruction/sensitivity variant; not full-paper reproduction' if method.startswith(('FLAURA', 'FedCognis')) else ('v0.3-only component control' if method not in MAIN else 'frozen project implementation'),
                        'official_test_used': False, 'defence_uses_evaluation_data': False})
    csv('methods_and_access', pd.DataFrame(methods), 'Method names, reference-data requirements and reproduction scope.')
    csv('costs_by_run', costs, 'All timing/resource measurements. Concurrent wall time and process RSS are descriptive; RSS includes interpreter, data and framework.')
    cost_cols = ['worker_wall_seconds', 'process_lifetime_peak_rss_mib', 'median_round_seconds', 'median_training_seconds',
                 'median_evaluation_seconds', 'median_defence_seconds', 'median_recorder_seconds', 'artifact_bytes',
                 'reference_forward_examples', 'reference_backward_examples', 'reference_optimizer_steps']
    long_cost = costs.melt(id_vars=KEY, value_vars=cost_cols, var_name='resource', value_name='value')
    csv('costs_by_scenario', summarise(long_cost, ['method', 'regime', 'scenario', 'resource']), 'Five-block costs kept per scenario to avoid unequal-coverage pooling.')
    common_cost = costs[costs.scenario.isin(['clean', 'gradual_stripe', 'label_flip']) & costs.method.isin(MAIN)]
    assert len(common_cost) == 330
    common_cost = common_cost.groupby(['method', 'regime', 'block', 'seed'], as_index=False)[cost_cols].mean()
    csv('costs_common_three_scenarios', summarise(common_cost.melt(id_vars=['method', 'regime', 'block', 'seed'], var_name='resource', value_name='value'), ['method', 'regime', 'resource']), 'Each block averages the same three scenarios: clean, stripes and label flip. No causal speed ranking across different study dates.')

    # Preserve complete original gates, recovery/censoring and component contrasts.
    appendices = {
        'v03_practical_targets': ORIGINAL / 'presentation/practical_criteria_with_interpretation.csv',
        'v03_strict_targets': ORIGINAL / 'analysis/zero_tolerance_targets.csv',
        'v03_component_contrasts': ORIGINAL / 'presentation/component_contrasts.csv',
        'recovery_all_blocks': ORIGINAL / 'analysis/recovery.csv',
        'recovery_counts': ORIGINAL / 'presentation/recovery_counts.csv',
        'model_episodes': ORIGINAL / 'analysis/model_episodes.csv',
        'original_contribution_summary': ORIGINAL / 'presentation/influence_by_block_and_client_group.csv',
        'original_component_activation': ORIGINAL / 'presentation/component_activation_by_block.csv',
        'original_fallbacks': ORIGINAL / 'presentation/fallback_fraction_by_block.csv',
        'v05_followup_gates': V05 / 'analysis/gates.csv',
        'v06_followup_gates': V06 / 'analysis/gates.csv',
        'v06_history_activation': V06 / 'analysis/history_activation.csv',
    }
    for name, path in appendices.items():
        # Byte-for-byte copies preserve the original missingness and precision.
        d = read_csv(path)
        destination = OUT / 'tables' / (name + '.csv')
        shutil.copyfile(path, destination)
        assert sha(path) == sha(destination)
        TABLE_INFO[name] = {'path': str(destination.relative_to(OUT)), 'rows': len(d), 'original': rel(path),
                            'description': 'Unchanged frozen source table. Original/v0.3 tables do not establish v0.6 recovery or ablation outcomes.'}
    gates = read_csv(V06 / 'analysis/gates.csv')
    assert len(gates) == 51 and int(gates.point_pass.sum()) == 45
    csv('v06_failed_targets', gates[~gates.point_pass], 'All six failed unchanged follow-up checks. Engineering margins, not statistical hypothesis tests.')
    recovery = read_csv(ORIGINAL / 'analysis/recovery.csv')
    assert not recovery.method.isin([V5, V6]).any()
    assert recovery[recovery.right_censored].confirmation_round.isna().all()
    counts = read_csv(ORIGINAL / 'presentation/recovery_counts.csv')
    assert counts.blocks.eq(5).all() and (counts.confirmed + counts.censored).eq(5).all()
    for identity, g in recovery.groupby(['method', 'regime', 'scenario', 'group', 'episode']):
        row = counts
        for k, v in zip(['method', 'regime', 'scenario', 'group', 'episode'], identity): row = row[row[k].eq(v)]
        assert len(row) == 1 and row.iloc[0].confirmed == int((~g.right_censored).sum())

    docs(selection, protocol, frozen, common_configs, scenarios, original_freeze)
    figures(means, curves, effects)
    save_json('figure_catalog.json', FIGURES)
    save_json('table_catalog.json', TABLE_INFO)
    save_json('source_index.json', {'research_freeze': rel(FREEZE / 'research_freeze.json'),
        'evidence_manifest': rel(FREEZE / 'evidence_manifest.json'),
        'raw_run_index': 'tables/run_index.csv',
        'all_frozen_study_and_report_artifacts': evidence['study_and_report_artifacts_sha256'],
        'note': 'Large original client/round diagnostics and checkpoints are indexed in place, not duplicated or omitted from the frozen evidence.'})
    notebook(core, effects)
    render_html(core, effects, selection, pd.DataFrame(strict))
    # These reads and presentation writes must not change the scientific record.
    after = integrity()
    assert before == after
    save_json('numerical_verification.json', {'status': 'passed', 'before': before, 'after': after,
        'matched_original_summary_rows': comparisons, 'derived_overall_curve_checks': 2640,
        'local_client_rounds': len(local), 'raw_task_rows_read': 880 * 75,
        'core_measured_cells': int(core.status.eq('measured').sum()),
        'core_explicit_not_tested_cells': int(core.status.eq('not_tested').sum()),
        'recovery_rows': len(recovery), 'recovery_counts_match_blocks': True,
        'v06_followup_targets_passed': 45, 'v06_followup_targets_total': 51,
        'source_settings_and_access_checked_for_all_880_runs': True,
        'no_training_or_checkpoint_evaluation_performed': True})
    manifest = {'schema_version': 1, 'created_at': datetime.now(timezone.utc).isoformat(),
        'status': 'built_numerically_verified_visual_review_pending',
        'research_freeze_sha256': sha(FREEZE / 'research_freeze.json'),
        'builder': rel(Path(__file__)), 'builder_sha256': sha(__file__),
        'input_sha256': INPUTS,
        'artifact_sha256': {rel(p): sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name not in ['package_manifest.json', 'completion_verification.json']},
        'notebook': rel(NOTEBOOK), 'notebook_sha256': sha(NOTEBOOK),
        'figures': len(FIGURES), 'tables': len(TABLE_INFO)}
    save_json('package_manifest.json', manifest)
    print(json.dumps({'output': str(OUT), 'notebook': str(NOTEBOOK), 'figures': len(FIGURES), 'tables': len(TABLE_INFO)}))


def docs(selection, protocol, frozen, configs, scenarios, original_freeze):
    definitions = {
        'overall_mean_5_24_pp': 'Mean legitimate-task accuracy (%) across rounds 5–24, all clients weighted by their population class counts and current legitimate task.',
        'overall_tail_21_24_pp': 'Mean of the same legitimate-task accuracy over rounds 21–24.',
        'overall_round24_pp': 'Legitimate-task accuracy after the final aggregation at round 24. No extra clean rounds are appended.',
        'unchanged_honest_mean_5_24_pp': 'Rounds 5–24 accuracy on the legitimate tasks of clients whose inputs and labels did not change.',
        'affected_honest_event_pp': 'Affected honest clients’ legitimate-task accuracy during their input-change window: 5–24, except 5–10 for honest_change_stops.',
        'affected_honest_return_11_24_pp': 'Affected honest clients’ legitimate-task accuracy after their input change stops, rounds 11–24.',
        'attack_active_signed_damage_pp': 'Mean matched-control legitimate-task accuracy minus attacked accuracy over attack-active rounds. Negative values are retained. Mixed attacks use mixed_drift_only_control; others use own-method clean.',
        'attack_active_positive_area_pp_rounds': 'Sum of positive matched-control-minus-attack accuracy deficits over attack-active rounds (pp-rounds). This is not a mean accuracy or signed damage.',
        'attack_tail_damage_21_24_pp': 'Mean matched-control-minus-attack legitimate-task accuracy over rounds 21–24 (pp).',
        'attack_active_accuracy_pp': 'Absolute legitimate-task accuracy (%) over attack-active rounds. For persistent label_flip this equals overall_mean_5_24_pp.',
        'matched_reference_attack_window_accuracy_pp': 'Matched-control accuracy (%) over the same attack-active rounds and observed scenario’s legitimate task weights.',
        'second_attack_signed_damage_pp': 'Mean signed matched-control damage in the recurring attack’s second window, rounds 15–18 (pp).',
        'overall_round5_pp': 'Absolute all-client legitimate-task accuracy (%) after round 5 under persistent label flipping.',
        'attack_onset_signed_damage_pp': 'Own-method clean round-5 accuracy minus attacked round-5 accuracy (pp).',
        'onset_raw_drop_pp': 'Attacked-run round-4 accuracy minus round-5 accuracy (pp). Negative means accuracy rose. This is not own-clean-relative damage.'}
    save_json('metric_dictionary.json', {'scalar_metrics': definitions,
        'accuracy_formula': 'Per-class held-out correct/count, then the frozen client-population/class/task weighting. Fixed balanced task accuracy in evaluation_tasks tables is correct/1000 and is a different denominator.',
        'training_loss': 'Mean online local SGD batch cross-entropy. Each client has 512 examples in eight equal batches. Local labels may be poisoned. Training accuracy was not recorded.',
        'evaluation_loss': 'Held-out cross-entropy on each fixed development task. Evaluation data is not a defence input.',
        'statistics': 'Five block values; sample SD ddof=1; mean ± 2.7764451051977987 × SD / sqrt(5). Paired intervals apply this formula to block differences. No rounds/clients treated as independent replicates. No multiplicity correction; known-development evidence.',
        'recovery': 'After attack cessation, observed accuracy must be at least matched control minus 1 percentage point for three consecutive eligible post-attack rounds. Confirmation is the third round. Stop eligibility before the next attack or at round 24. Censored rows have no observed confirmation, not a zero recovery time.',
        'recovery_caveat': 'A trajectory may already be within the tolerance at attack end; report this separately. First restoration does not imply permanent recovery. v0.6 and v0.5 were not tested in recovery scenarios.',
        'contribution_diagnostics': 'Applied update norms/coefficients are descriptive, not proof of malicious intent or causal utility. v0.6 is affine in current deltas plus historical vectors; current coefficients alone cannot reconstruct it.',
        'missingness': 'not_tested/NA means no measurement. A measured zero remains numeric zero.',
        'units': 'Accuracy columns ending _pp retain historical identifiers but are percentages. Accuracy differences/damage are percentage points. Area is percentage-point-rounds. Times are seconds, memory MiB.'})
    save_json('protocol_and_scenarios.json', {'training_config_verified_all_runs': list(configs.values())[0],
        'model': 'MNIST_CNN_conv32_conv64_dense128', 'parameters': 421642,
        'training': protocol['training'], 'blocks': protocol['blocks'],
        'partition_design': protocol['partition_design'], 'partitions': original_freeze['partitions'],
        'scenarios': scenarios, 'statistics': protocol['statistics'],
        'matched_control': 'mixed_drift_and_attack uses mixed_drift_only_control; other scenarios use clean.',
        'v06_scope': selection['v06_coverage'], 'original_limits': protocol['limits'],
        'component_definitions_v03_only': protocol['component_definitions']})
    save_json('final_method_and_claims.json', selection)
    save_json('v06_exact_settings.json', {'candidate': frozen['candidate_settings'], 'recorder': frozen['recorder_settings']})
    decisions = {}
    for name, folder in [('original_v03', ORIGINAL), ('v05_pilot', EXP / 'outputs/reference_free/20261031_v05'),
                         ('v05_followup', V05), ('v06_pilot', EXP / 'outputs/reference_free/20261031_v06'), ('v06_followup', V06)]:
        path = folder / ('presentation/decision.json' if name == 'original_v03' else 'analysis/decision.json')
        d = read_json(path); decisions[name] = {'original': rel(path), 'sha256': sha(path)}
        save_json('evidence/' + name + '_decision.json', d)
    save_json('decision_history.json', decisions)
    text = '''FEDRETRUST FINAL RESULTS PACKAGE — 31 October 2026

Open START_HERE.html for the readable guide. Notebook 34 provides an executed,
saved-artifact review. tables/ contains machine-readable results; figures/ contains
PNG and vector PDF versions; catalogs define their scope and provenance.

Selected project candidate: FedReTrust v0.6. Keep v0.5 as the earlier-version
comparison. This is one unchanged rule for both IID and non-IID.

The package indexes 880 distinct trajectories: original 800 plus 40 v0.5 and 40
v0.6 runs. There are 14 method identifiers, including three v0.3 component controls.
The main comparison has 11 non-control methods. Each measured cell has five paired
blocks. No previous two-seed result is mixed into this corpus.

Coverage differs. v0.5/v0.6 cover clean, honest noise, honest stripes and persistent
label flipping. FLAURA/FedCognis have five-block clean/stripe/label-flip evidence,
but no five-block noise result. The original five methods cover ten scenarios.
v0.3 component tests and original recovery studies do not become v0.6 evidence.

All runs use 24 rounds, with four clean warm-up rounds and interventions from round
5. Five blocks are repetitions with paired seed/partition conditions, not extra
clean rounds. Official test data was not used. All five original block conditions
informed development; the final version choice is not untouched confirmation.

Main evidence: v0.6 improves non-IID label-flip damage over v0.5 by 1.7684 pp
(nominal paired 95% interval 0.6225 to 2.9142). It loses 1.1176 pp IID stripe
accuracy (loss interval 0.0148 to 2.2203). RFA retains better average attack protection;
v0.6 retains better stripe accuracy. Follow-up passes 45/51 unchanged checks, so
complete success is not claimed. No universal best, malicious-intent proof,
production-readiness or established novelty claim is supported.

The large original per-client/per-round diagnostics and model checkpoints remain
in place, with hashes and paths in source_index.json and the research freeze.
This is a results package, not the cleaned code submission or final report.

Integrity: run the frozen verify_freeze.py, then verify this package's manifest.
Rebuild presentation only: Experiment/.venv/bin/python Experiment/code/final_results_package.py
Run that command from the project root. It reads saved artifacts only and does
not train or evaluate checkpoints. A rebuild needs renewed visual verification.

Next: step 4 in a new chat, GPT-6 Astra — High. Read NEXT_CHAT_1.txt.
'''
    (OUT / 'README.txt').write_text(text)
    handoff = f'''Continue my COMP6002 project, steps 4–6, in this new chat.

At every NEW STEP, remind me which model/reasoning setting to use.
Step 4 (understand experiments through questions): GPT-6 Astra — High.
Steps 5–6 (code cleanup/documentation and reproducibility): GPT-6.1 Sol — High.
Keep explanations short and work one step at a time. Start step 4 by helping me
understand the experimental setup and ask which question I want to explore first.
Do not skip my questions and begin code cleanup immediately.

Workspace: {PROJECT}
Read first: {OUT / 'START_HERE.html'}
Research selection: {FREEZE / 'selection.json'}
Research freeze: {FREEZE / 'research_freeze.json'}
Freeze verifier: {FREEZE / 'verify_freeze.py'}
Evidence index: {FREEZE / 'evidence_manifest.json'}
Executed final notebook: {NOTEBOOK}
Workflow/deadlines: {EXP / 'outputs/submission_planning/20261031/requirements_and_sequence.json'}

Steps 1–3 are complete. Final project candidate is v0.6, keep v0.5 as the important
earlier comparison. The implementation is 0.6-exploratory. Research is frozen:
preserve 73 scientific sources, 880 distinct runs, all original failures and gates.
No new variants, tuning, seeds, partitions, scenarios or performance experiments.

v0.6 has 40 runs: 5 blocks × 2 regimes × 4 scenarios. FLAURA/FedCognis lack a
five-block noise comparison. v0.3 recovery/component findings do not establish
v0.6 recovery or ablations. Keep IID/non-IID separate. Use paired block intervals,
and report absolute attacked accuracy alongside own-clean-relative damage.
The 45/51 follow-up result is partial success. All blocks were known development
conditions; no universal superiority, independent confirmation or established
novelty claim. All official test data remains unused in this evidence.

For step 5, create a separate submission copy. Never clean in place or overwrite
the frozen source/output trees. Preserve numerical behaviour and document all
changes. Include setup/dependencies/data instructions, commands, a practical demo
and an accessible GitHub/Bitbucket code link as required. Decide inclusion of large
artifacts/data/checkpoints explicitly. Do not publish private data or huge outputs
by default. Do not fabricate project-management snapshots or contributions.

For step 6, verify integrity, use saved-artifact reproduction and appropriate
small implementation checks. Any separately labelled demo/reproduction run must
not overwrite study evidence or be added to final means. Do not rerun the full
880-run study just to rebuild results. Leave explicit reproducibility limitations.

Report due 1 November 2026 at 11:59 pm Perth, confirmed by user. Report checks and
submission must happen immediately after step 7, before presentation completion.
Step 7 is a different chat (GPT-6 Astra — Extra High) and must begin with the actual
submitted proposal, whose exact identity still needs confirmation. Presentation
is already booked for the 10–11 November sessions. Main demo and backup recording
are required. Referencing-style and presentation/demo minute split placeholders
remain unresolved in the guideline. Report originality declaration must be signed
by the student. Do not fabricate signatures, supervisor confirmations or submission.

Stop at the agreed chat boundary after step 6 with a concrete handoff for report
writing. Do not start writing the report or presentation in this chat.
'''
    (OUT / 'NEXT_CHAT_1.txt').write_text(handoff)


def notebook(core, effects):
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    cells = [md('''# 34 — Final FedReTrust research results

Final project candidate: **v0.6**, with v0.5 retained as an earlier-version comparison.
This notebook reads completed, frozen evidence only. No training or checkpoint evaluation.
The corpus has 880 distinct runs, 14 method identifiers and five paired blocks per measured cell.
The main comparison excludes three v0.3 component controls. IID and non-IID remain separate.

All intervals are nominal 95% t intervals across five known development blocks (df4).
They are exploratory, may be unstable at n=5, and have no simultaneous coverage.
The version selection used known block results; this is not untouched confirmation.
'''), code(f'''from pathlib import Path
import json, hashlib
import pandas as pd
from IPython.display import display, Image, Markdown
HERE = Path.cwd()
EXP = next((p for p in [HERE, *HERE.parents] if (p/'outputs/final_results/20261031_v06').is_dir()), None)
if EXP is None:
    candidate = HERE/'Experiment'
    EXP = candidate if (candidate/'outputs/final_results/20261031_v06').is_dir() else Path({str(EXP)!r})
PACKAGE = EXP/'outputs/final_results/20261031_v06'
assert PACKAGE.is_dir()
CORE = pd.read_csv(PACKAGE/'tables/core_results.csv')
PAIRS = pd.read_csv(PACKAGE/'tables/paired_effects.csv')
ORDER = {MAIN!r}
LABELS = {LABEL!r}
print('Selected method: FedReTrust v0.6. Evidence: 880 distinct trajectories; 73 frozen scientific sources.')
print('Main outcomes use rounds 5–24 of 24. Five blocks are repetitions, not extra rounds.')
assert CORE[CORE.status.eq('measured')].n.eq(5).all()
assert CORE[CORE.status.eq('not_tested')]['mean'].isna().all()
'''), md('''## Main results

Clean and attacked accuracy use all clients' legitimate tasks. Honest noise and stripes
use the affected honest clients' tasks. Damage is each method's matched clean accuracy
minus attacked accuracy, in percentage points. Absolute attacked accuracy is shown alongside it.
FLAURA/FedCognis noise cells are untested. They are never filled using earlier two-seed results.
'''), code('''for regime in ['IID', 'non-IID']:
    display(Markdown('### '+regime))
    table = CORE[CORE.regime.eq(regime)].pivot(index='method', columns='measurement', values='mean')
    display(table.reindex(ORDER).rename(index=LABELS).round(3).fillna('Not tested'))
    display(Image(filename=str(PACKAGE/'figures'/('core_'+regime.replace('-','_')+'.png'))))
'''), md('''## The selected compromise

v0.6 reduces non-IID label-flip damage relative to v0.5 in all five blocks, with a
mean reduction of 1.7684 pp (nominal paired interval 0.6225–2.9142). IID stripe
accuracy falls by 1.1176 pp (loss interval 0.0148–2.2203). RFA retains better mean
non-IID attack protection; v0.6 retains better stripe accuracy. No scalar balance
score or universal ranking was predeclared.
'''), code('''keep = PAIRS.candidate.eq('FedReTrust-v06') & PAIRS.comparator.isin(['FedReTrust-v05','RFA','FLTrust','FedAvg'])
spec = CORE[['scenario','metric']].drop_duplicates()
view = PAIRS[keep].merge(spec,on=['scenario','metric'])
display(view[['regime','comparator','scenario','metric','block_1','block_2','block_3','block_4','block_5','mean','ci_low','ci_high']].round(3))
display(Image(filename=str(PACKAGE/'figures/paired_v06.png')))
display(Image(filename=str(PACKAGE/'figures/core_trajectories.png')))
display(Image(filename=str(PACKAGE/'figures/label_flip_blocks.png')))
'''), md('''## Targets and coverage limits

45/51 unchanged follow-up checks pass: original practical 8/10, IID retention 21/25,
non-IID focus 16/16. The pilot remains failed. These are engineering margins,
not statistical proof of superiority. v0.6 has no five-block attack-stop, recurring,
mixed-event or Gaussian-update result. Historical component controls concern v0.3 only.
'''), code('''G = pd.read_csv(PACKAGE/'tables/v06_followup_gates.csv')
assert len(G)==51 and int(G.point_pass.sum())==45
display(G.groupby('gate_group').point_pass.agg(['sum','count']))
display(pd.read_csv(PACKAGE/'tables/v06_failed_targets.csv')[['gate_group','regime','scenario','gate_block','criterion','checked_difference_pp','bound_pp','direction']])
display(pd.read_csv(PACKAGE/'tables/methods_and_access.csv'))
coverage = pd.read_csv(PACKAGE/'tables/coverage.csv')
display(coverage[coverage.regime.eq('IID')].pivot(index='method',columns='scenario',values='status'))
print('Recovery: within 1 pp of matched control for three consecutive eligible post-attack rounds; censored outcomes remain censored.')
display(pd.read_csv(PACKAGE/'tables/recovery_counts.csv'))
display(pd.read_csv(PACKAGE/'tables/v03_component_contrasts.csv').round(3))
'''), md('''## Training, evaluation and costs

Training loss is online local cross-entropy; local labels can be poisoned. Training
accuracy was not recorded. Fixed balanced task evaluation and population-weighted
legitimate-task evaluation have different denominators. Concurrent timing and
process RSS are descriptive; v0.6 history alone adds 32.17 MiB of tensor payload.
No method reads evaluation scores during aggregation. FLTrust uses 300 clean root
images. FLAURA uses 100 auxiliary images and earlier frozen calibration. v0.6 uses none.
'''), code('''for filename in ['training_loss_summary','evaluation_tasks_cross_entropy_summary','costs_common_three_scenarios']:
    data = pd.read_csv(PACKAGE/'tables'/(filename+'.csv'))
    data = data[data.method.isin(['FedAvg','RFA','FLTrust','FedReTrust-v05','FedReTrust-v06'])]
    if 'scenario' in data: data = data[data.scenario.eq('clean')]
    if 'window' in data: data = data[data.window.eq('primary_5_24')]
    if 'client_group' in data: data = data[data.client_group.eq('all_clients')]
    if 'task' in data: data = data[data.task.eq('clean')]
    if 'resource' in data: data = data[data.resource.isin(['median_defence_seconds','process_lifetime_peak_rss_mib','worker_wall_seconds'])]
    display(Markdown('**'+filename.replace('_',' ')+'**'))
    display(data.round(4))
print('Full tables, definitions, source hashes, original decisions and next-chat handoff are in the results package.')
'''), md('''## Interpretation boundaries

This controlled MNIST/CNN study supports a measured protection/adaptation compromise.
It does not prove malicious intent, broad concept-drift handling, secure aggregation,
differential privacy, adaptive-attack resistance, production readiness or established novelty.
The official test set was not used. Historical raw data and failures remain preserved.
''')]
    nb = nbf.v4.new_notebook(cells=cells, metadata={'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}})
    from aggregate_validation_notebook import _execute
    _execute(nb, NOTEBOOK)
    for cell in nb.cells:
        if cell.cell_type == 'code':
            assert cell.execution_count is not None
            assert all(o.output_type != 'error' and not (o.output_type == 'stream' and o.name == 'stderr') for o in cell.outputs)


def render_html(core, effects, selection, strict):
    css = '''body{font:16px/1.55 system-ui,sans-serif;color:#243247;background:#f5f7fa;margin:0}main{max-width:1250px;margin:auto;padding:42px 28px}h1{font-size:34px;line-height:1.2}h2{margin-top:40px}h3{margin-top:28px}p{max-width:95ch}a{color:#215b92}table{border-collapse:collapse;background:white;font-size:14px;width:100%}th,td{padding:9px 12px;border-bottom:1px solid #dde3eb;text-align:right}th:first-child,td:first-child{text-align:left}th{background:#e9eef5}figure{margin:28px 0;background:white;padding:15px}img{max-width:100%;height:auto}figcaption{font-size:14px;color:#526077}nav{display:flex;gap:20px;flex-wrap:wrap;padding:18px 0}summary{cursor:pointer;font-weight:600}details{margin:20px 0}.scroll{overflow-x:auto}.note{border-left:4px solid #8043ad;padding:5px 20px;background:#eee9f5}code{font-size:.9em;overflow-wrap:anywhere}li{margin:7px 0}@media print{body{background:white}main{padding:0}details{display:block}figure{break-inside:avoid}}'''
    parts = ['<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
        '<title>FedReTrust final results</title><style>' + css + '</style><main>',
        '<h1>FedReTrust final results</h1><p>Final project candidate: <strong>v0.6</strong>. Frozen 31 October 2026. Five paired development blocks per measured cell.</p>',
        '<nav><a href="#results">Main comparison</a><a href="#evidence">Figures</a><a href="#targets">Targets</a><a href="#definitions">Definitions</a><a href="#files">All tables</a><a href="NEXT_CHAT_1.txt">Next chat handoff</a></nav>',
        '<div class="note"><p>v0.6 improves non-IID poisoning protection over v0.5, at a cost to IID stripe adaptation. RFA retains better average attack protection; v0.6 retains better honest-stripe accuracy. This is a measured compromise, not a universal winner.</p></div>',
        '<p>880 distinct runs: the original 800, plus 40 v0.5 and 40 v0.6 trajectories. There are 14 method identifiers including three historical v0.3 component controls. The main table compares 11 non-control methods.</p>',
        '<h2 id="results">Main comparison</h2><p>Mean [nominal 95% interval], using five blocks and rounds 5–24. Accuracy is percent; damage is percentage points. Honest noise and stripe scores concern affected honest clients. Damage uses each method’s own clean control. Intervals are exploratory, not simultaneous; all block conditions were known during development.</p>']
    for regime in ['IID', 'non-IID']:
        display = []
        for method in MAIN:
            row = {'Method': LABEL[method]}
            for _, _, label, _ in CORE:
                r = core[(core.regime == regime) & (core.method == method) & (core.measurement == label)].iloc[0]
                row[label] = 'Not tested' if r.status == 'not_tested' else f'{r["mean"]:.2f} [{r.ci_low:.2f}, {r.ci_high:.2f}]'
            display.append(row)
        parts.append(f'<h3>{regime}</h3><div class="scroll">' + pd.DataFrame(display).to_html(index=False, border=0) + '</div>')
    parts += ['<p>FLAURA/FedCognis noise results are untested in the five-block study. The original five methods cover ten scenarios; v0.5/v0.6 cover only the four scenarios above. Historical recovery and v0.3 component tests do not establish v0.6 recovery or ablations.</p>',
        '<h3>Key paired differences</h3><p>v0.6 minus comparator. Positive accuracy differences and negative damage differences favour v0.6. Each interval is calculated from five paired differences.</p>']
    specs = [('non-IID', V5, 'label_flip', 'attack_active_signed_damage_pp'), ('non-IID', V5, 'label_flip', 'overall_mean_5_24_pp'),
             ('IID', V5, 'gradual_stripe', 'affected_honest_event_pp'), ('non-IID', 'RFA', 'gradual_stripe', 'affected_honest_event_pp'),
             ('IID', 'RFA', 'gradual_stripe', 'affected_honest_event_pp'), ('non-IID', 'RFA', 'label_flip', 'attack_active_signed_damage_pp'),
             ('non-IID', 'RFA', 'label_flip', 'overall_mean_5_24_pp')]
    rows = []
    for regime, comp, scenario, metric in specs:
        r = effects[(effects.candidate == V6) & (effects.comparator == comp) & (effects.regime == regime) & (effects.scenario == scenario) & (effects.metric == metric)].iloc[0]
        name = next(c[2] for c in CORE if c[0] == scenario and c[1] == metric)
        rows.append({'Regime': regime, 'Comparator': LABEL[comp], 'Measurement': name,
                     'Difference (pp)': f'{r["mean"]:+.3f}', 'Nominal 95% interval': f'[{r.ci_low:+.3f}, {r.ci_high:+.3f}]'})
    parts.append('<div class="scroll">' + pd.DataFrame(rows).to_html(index=False, border=0) + '</div>')
    parts += ['<h2 id="evidence">Figures for the report and presentation</h2>']
    for f in FIGURES:
        parts.append('<figure><h3>' + html.escape(f['title']) + '</h3><a href="' + f['png'] + '"><img src="' + f['png'] + '" alt="' + html.escape(f['title']) + '"></a><figcaption>' + html.escape(f['caption']) + ' <a href="' + f['pdf'] + '">Vector PDF</a></figcaption></figure>')
    parts += ['<h2 id="targets">Targets and unchanged decisions</h2><p><strong>45/51 follow-up checks passed</strong>: original practical targets 8/10, IID retention 21/25, non-IID focus 16/16. The original v0.6 pilot remains failed at 25/27. Follow-up does not retroactively change that decision.</p>',
              '<p>The six follow-up failures are IID mean stripe accuracy versus FedAvg, non-IID mean noise accuracy versus FedAvg, and IID stripe retention versus v0.5 in blocks 1, 2, 4 and 5. These are predeclared engineering margins, not statistical superiority tests.</p>',
              '<p><a href="tables/v06_followup_gates.csv">All 51 checks</a> · <a href="tables/v06_failed_targets.csv">All six failures</a> · <a href="decision_history.json">Original decisions</a></p>',
              '<details><summary>Best measured baseline in each main outcome (descriptive only)</summary><p>Missing scenarios cannot compete. Selecting the largest/smallest observed mean is not proof of superiority. This table does not redefine any frozen target.</p><div class="scroll">' + strict[['regime','measurement','best_measured_baseline','best_baseline_mean','candidate_mean']].round(3).to_html(index=False,border=0) + '</div></details>',
              '<h2 id="definitions">How to read the measurements</h2><ul>',
              '<li><strong>24 rounds per run.</strong> Rounds 1–4 are clean warm-up. Interventions begin at round 5. Five blocks mean five repetitions, not five extra clean rounds.</li>',
              '<li><strong>Accuracy:</strong> correct predictions on reserved development images, weighted to the clients’ legitimate task/class populations. The official test set was not used. Fixed balanced task tables have a different denominator.</li>',
              '<li><strong>Attack damage:</strong> matched control minus attacked accuracy. Read it alongside absolute attacked accuracy, since methods can have different clean performance.</li>',
              '<li><strong>Training loss:</strong> online cross-entropy during local SGD. It can use poisoned labels and does not establish clean predictive quality. Training accuracy was not recorded.</li>',
              '<li><strong>Recovery:</strong> within 1 pp of matched control for three consecutive eligible post-attack rounds. Censored runs have no observed recovery time. Already-within-tolerance and later relapse are reported separately. v0.6 recovery was not tested.</li>',
              '<li><strong>IID versus non-IID:</strong> five random IID partitions and five label-skew partitions (alpha 0.5), with matched training/evaluation inputs within each block. Keep the regimes separate.</li>',
              '<li><strong>Reference data:</strong> FLTrust uses 300 clean server images. FLAURA uses 100 auxiliary images plus earlier frozen calibration. FedReTrust v0.6 uses zero server reference images.</li>',
              '<li><strong>Costs:</strong> concurrent fresh-process timings are descriptive. Process RSS includes framework and data. v0.6’s additional history tensor payload is 32.17 MiB for this model/cohort, excluding object and temporary overhead.</li></ul>',
              '<p><a href="metric_dictionary.json">Full metric dictionary</a> · <a href="protocol_and_scenarios.json">Exact protocol, scenarios and partitions</a> · <a href="v06_exact_settings.json">v0.6 settings</a> · <a href="final_method_and_claims.json">Selection and claim boundaries</a></p>',
              '<h3>What the project can claim</h3><p>The controlled experiments demonstrate that protection and honest adaptation can conflict, and that a reference-free prototype changes that tradeoff. v0.6 improves non-IID protection over v0.5 but sacrifices IID stripe adaptation. It is not best on every outcome. This evidence does not establish novelty, malicious intent, broad concept-drift handling, production readiness, adaptive-attack protection, secure aggregation or differential privacy.</p>',
              '<h2 id="files">Complete results and provenance</h2><p>All saved scalar outcomes, five-block values, tail/final results, training/evaluation loss, class results, resource costs, recovery and historical component findings are accessible below. Large original client/round diagnostics and checkpoints remain indexed in place.</p>',
              '<p><a href="../../../code/34_fedretrust_final_results.ipynb">Executed notebook 34</a> · <a href="source_index.json">Frozen source index</a> · <a href="package_manifest.json">Package hashes</a> · <a href="completion_verification.json">Final review</a></p><ul>']
    for name, info in TABLE_INFO.items():
        parts.append('<li><a href="' + info['path'] + '">' + html.escape(name.replace('_', ' ')) + '</a> (' + str(info['rows']) + ' rows). ' + html.escape(info['description']) + '</li>')
    parts += ['</ul><h2>Next agreed step</h2><p>Step 4 in a new chat: experiment questions. Use <strong>GPT-6 Astra — High</strong>. Then use <strong>GPT-6.1 Sol — High</strong> for code cleanup and reproducibility (steps 5–6). <a href="NEXT_CHAT_1.txt">Copy the prepared handoff</a>.</p>',
              '<p>Report deadline: <strong>1 November 2026, 11:59 pm Perth</strong>. Complete report checks and submission after step 7; do not wait for the presentation work. Presentation booking is already confirmed. This package is research evidence, not the cleaned submission repository or final report.</p></main></html>']
    (OUT / 'START_HERE.html').write_text('\n'.join(parts))


if __name__ == '__main__':
    build()
