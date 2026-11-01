"""Frozen offline analysis for the 800-run replication; no training decisions."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from experiment_core import file_sha256, write_json
from research_runs import ROOT
from research_analysis import extract_run, cross_method_pair, contribution_records, cost_record, interval
from research_study import task_series
from scenario_metrics import longer_scenarios, matched_reference_name, model_recovery

MAIN = 'FedReTrust-Direction-loose'
PRIMARY_SPECS = [
    ('clean', 'overall_mean_5_24_pp', 'FedAvg', -1., '>='),
    ('abrupt_shot_noise', 'affected_honest_event_pp', 'FedAvg', -1., '>='),
    ('gradual_stripe', 'affected_honest_event_pp', 'FedAvg', -1., '>='),
    ('label_flip', 'attack_active_signed_damage_pp', 'RFA', 1., '<='),
    ('label_flip', 'attack_active_signed_damage_pp', 'FedAvg', -2., '<=')]


def reference_workload(values):
    """FLTrust records training visits separately from auxiliary forward counts."""
    root = int(values.get('root_example_visits', 0))
    return (int(values.get('reference_forward_examples', 0)) + root,
            int(values.get('reference_backward_examples', 0)) + root,
            int(values.get('root_optimizer_steps', 0)))


def summarize(values, blocks):
    if list(sorted(blocks)) != [1, 2, 3, 4, 5]:
        raise ValueError('Exactly one observation from each declared block is required.')
    return interval(values, expected_n=5)


def paired_tables(metrics):
    table = pd.DataFrame(metrics)
    identity = ['method', 'regime', 'block', 'scenario', 'metric']
    if table.duplicated(identity).any():
        raise ValueError('Duplicate block measurement.')
    means = []
    for keys, group in table.groupby(['panel', 'method', 'regime', 'scenario', 'metric']):
        means.append({**dict(zip(['panel', 'method', 'regime', 'scenario', 'metric'], keys)),
                      **summarize(group.value, group.block)})
    effects = []
    # The second contrast isolates directional final weighting against capped RFA.
    for target, comparators in [(MAIN, sorted(set(table.method) - {MAIN})),
        ('FedReTrust-v03-no-restoration', ['FedReTrust-v03-capped-RFA'])]:
        for other in comparators:
            paired = table[table.method == target].merge(table[table.method == other],
                on=['regime', 'block', 'seed', 'scenario', 'metric'], suffixes=('_target', '_other'), validate='one_to_one')
            for keys, group in paired.groupby(['regime', 'scenario', 'metric']):
                group = group.sort_values('block')
                delta = group.value_target - group.value_other
                effects.append({'target': target, 'comparator': other,
                    **dict(zip(['regime', 'scenario', 'metric'], keys)),
                    'blocks': group.block.tolist(), 'paired_differences_pp': delta.tolist(),
                    **summarize(delta, group.block)})
    return means, effects


def practical_checks(effects):
    lookup = {(e['target'], e['comparator'], e['regime'], e['scenario'], e['metric']): e for e in effects}
    rows = []
    for regime in ['IID', 'non-IID']:
        for scenario, metric, other, bound, direction in PRIMARY_SPECS:
            e = lookup[(MAIN, other, regime, scenario, metric)]
            point = e['mean'] >= bound if direction == '>=' else e['mean'] <= bound
            rows.append({**e, 'bound_pp': bound, 'direction': direction,
                'point_estimate_meets_margin': bool(point),
                'interval_crosses_or_touches_margin': bool(e['ci_low'] <= bound <= e['ci_high']),
                'interpretation': 'Nominal descriptive interval, not simultaneous table-wide significance.'})
    return rows


def zero_tolerance_checks(means):
    frame = pd.DataFrame(means)
    specs = [('clean', 'overall_mean_5_24_pp', True), ('gradual_stripe', 'affected_honest_event_pp', True),
             ('label_flip', 'attack_active_signed_damage_pp', False)]
    rows = []
    for regime in ['IID', 'non-IID']:
        for scenario, metric, higher in specs:
            group = frame[(frame.regime == regime) & (frame.scenario == scenario) & (frame.metric == metric)]
            candidate = float(group[group.method == MAIN]['mean'].iloc[0])
            baselines = group[(group.method != MAIN) & group.panel.isin(['primary', 'recent_methods'])]
            best = baselines.sort_values(['mean', 'method'], ascending=[not higher, True]).iloc[0]
            rows.append({'regime': regime, 'scenario': scenario, 'metric': metric,
                'candidate_mean': candidate, 'best_baseline': best.method, 'best_baseline_mean': float(best['mean']),
                'point_target_met': bool(candidate >= best['mean'] if higher else candidate <= best['mean']),
                'scope': 'Descriptive best of newly compared baselines; no selection or superiority inference.'})
    return rows


def analyse(study):
    from validation_replication import verify, records, jobs, key
    study = Path(study); freeze = verify(study); all_records = records(study)
    if len(all_records) != 800 or {key(r['job']) for r in all_records} != {key(j) for j in jobs()}:
        raise ValueError('Full inference requires all 800 declared runs; no complete-case substitution.')
    indexed = {key(r['job']): r for r in all_records}
    metrics, curves, episodes, recovery, costs = [], [], [], [], []
    contributions, restorations, reductions, diagnostic_rows, round_rows = [], [], [], [], []
    matches, prefixes, first_updates, max_error = 0, 0, 0, 0.
    out = study / 'analysis'; out.mkdir(exist_ok=True)
    for name in ['local_training.csv', 'raw_task_metrics.csv']:
        if (out / name).exists(): (out / name).unlink()
    workload = {'client_updates': 0, 'optimizer_steps': 0, 'train_forward_examples': 0,
                'development_task_forward_examples': 0, 'checkpoint_replay_forward_examples': 0,
                'reference_forward_examples': 0, 'reference_backward_examples': 0, 'reference_optimizer_steps': 0}
    for number, r in enumerate(all_records):
        j = r['job']; path = ROOT / r['run_path']
        ident = {n: j[n] for n in ('panel', 'method', 'regime', 'block', 'seed', 'scenario')}
        ident['job_id'] = r['job_id']
        ref = indexed[(j['method'], j['regime'], j['block'], matched_reference_name(j['scenario']))]
        base = indexed[('FedAvg', j['regime'], j['block'], j['scenario'])]
        cross_method_pair(r, base); matches += 1
        scenario = longer_scenarios()[j['scenario']]
        a, b, c = extract_run(r, ref if scenario.events else None)
        metrics.extend({**v, **ident} for v in a); curves.extend({**v, **ident} for v in b)
        episodes.extend({**v, **ident} for v in c)
        costs.append({**cost_record(r), **ident, 'artifact_bytes': r['artifact_bytes']})
        local = pd.read_csv(path / 'client_metrics.csv'); tasks = pd.read_csv(path / 'task_metrics.csv')
        if len(local) != 240 or not local.optimizer_steps.eq(8).all() or not local.examples.eq(512).all():
            raise AssertionError('Local workload changed.')
        if not np.isfinite(local.online_train_loss).all(): raise AssertionError('Nonfinite training loss.')
        if len(tasks) != 75 or not tasks.examples.eq(1000).all(): raise AssertionError('Evaluation workload changed.')
        for frame, name in [(local, 'local_training.csv'), (tasks, 'raw_task_metrics.csv')]:
            frame.assign(**ident).to_csv(out / name, index=False, mode='w' if number == 0 else 'a', header=number == 0)
        old = pd.read_csv(ROOT / base['run_path'] / 'client_metrics.csv')
        cols = ['client', 'submitted_state_sha256', 'broadcast_state_sha256', 'shuffle_seed', 'optimizer_steps', 'online_train_loss']
        pd.testing.assert_frame_equal(local[local['round'] == 1][cols].reset_index(drop=True),
            old[old['round'] == 1][cols].reset_index(drop=True), check_exact=True)
        first_updates += 10
        if scenario.events:
            a, b, c = contribution_records(r, ref)
            contributions.extend({**v, **ident} for v in a); restorations.extend({**v, **ident} for v in b)
            reductions.extend({**v, **ident} for v in c)
            h = pd.read_csv(path / 'history.csv'); hc = pd.read_csv(ROOT / ref['run_path'] / 'history.csv')
            if h[h['round'] <= 4].global_state_sha256.tolist() != hc[hc['round'] <= 4].global_state_sha256.tolist():
                raise AssertionError('Pre-event global states changed.')
            tc = pd.read_csv(ROOT / ref['run_path'] / 'task_metrics.csv')
            pd.testing.assert_frame_equal(tasks[tasks['round'] <= 4].reset_index(drop=True), tc[tc['round'] <= 4].reset_index(drop=True), check_exact=True)
            prefixes += 1
        attacks = [e for e in scenario.events if e.intent == 'malicious']
        if attacks:
            windows = attacks[0].windows
            obs, ctl = task_series(r), task_series(ref, scenario_name=j['scenario'])
            active = sorted({t for lo, hi in windows for t in range(lo, hi + 1)})
            for name, series in [('attack_active_accuracy_pp', obs), ('matched_reference_attack_window_accuracy_pp', ctl)]:
                metrics.append({**ident, 'phase': 'replication', 'metric': name, 'value': float(100 * np.mean([series[t] for t in active]))})
            for episode, (lo, hi) in enumerate(windows):
                if hi >= 24: continue
                next_attack = windows[episode + 1][0] if episode + 1 < len(windows) else None
                for group, clients in [('all_clients', None), ('affected_attackers', list(attacks[0].clients))]:
                    observed = {t: v for t, v in task_series(r, client_ids=clients).items() if t > 0}
                    control = {t: v for t, v in task_series(ref, scenario_name=j['scenario'], client_ids=clients).items() if t > 0}
                    result = model_recovery(control, observed, hi, next_attack=next_attack)
                    confirmed = result['confirmation_round']
                    relapses = [] if confirmed is None else [t for t in range(confirmed + 1, result['last_eligible_round'] + 1) if observed[t] < control[t] - .01 - 1e-12]
                    recovery.append({**ident, 'group': group, 'episode': episode + 1, **result, 'relapse_rounds_after_confirmation': relapses})
        info = json.loads((path / 'run.json').read_text())
        access = json.loads((path / 'validation_access.json').read_text())
        if info['official_test_used'] or access['official_test_used'] or access['defence_uses_evaluation_data']:
            raise AssertionError('Information boundary violated.')
        diag = json.loads((path / 'aggregation_diagnostics.json').read_text())
        if len(diag) != 24: raise AssertionError('Missing aggregation round.')
        ref_forward, ref_backward, ref_steps = 0, 0, 0
        for row in diag:
            values = row['values']; measure = values['contribution_measurements']
            max_error = max(max_error, measure['reconstruction_max_abs_error'])
            forward, backward, steps = reference_workload(values)
            ref_forward += forward; ref_backward += backward; ref_steps += steps
            round_rows.append({**ident, 'round': row['round'],
                **{k: v for k, v in values.items() if k not in ('client_metrics', 'contribution_measurements')},
                'reference_forward_examples': forward, 'reference_backward_examples': backward, 'reference_optimizer_steps': steps,
                'reconstruction_max_abs_error': measure['reconstruction_max_abs_error']})
            for cid, item in values['client_metrics'].items():
                diagnostic_rows.append({**ident, 'round': row['round'], 'client': int(cid), **item,
                    'bounded_rfa_fallback': values.get('bounded_rfa_fallback'),
                    'clipping_enabled': values.get('clipping_enabled'), 'always_use_rfa': values.get('always_use_rfa')})
        if access['server_reference_examples'] == 0 and (ref_forward or ref_backward):
            raise AssertionError('Reference-free method used server data.')
        workload['client_updates'] += len(local); workload['optimizer_steps'] += int(local.optimizer_steps.sum())
        workload['train_forward_examples'] += int(local.examples.sum())
        workload['development_task_forward_examples'] += int(tasks.examples.sum())
        workload['checkpoint_replay_forward_examples'] += 6000
        workload['reference_forward_examples'] += ref_forward; workload['reference_backward_examples'] += ref_backward
        workload['reference_optimizer_steps'] += ref_steps
        costs[-1].update(reference_forward_examples=ref_forward, reference_backward_examples=ref_backward,
                         reference_optimizer_steps=ref_steps)
    if (matches, prefixes, first_updates) != (800, 680, 8000) or max_error > 1e-6:
        raise AssertionError('Incomplete or inaccurate matched-round audit.')
    means, effects = paired_tables(metrics); checks = practical_checks(effects)
    tables = {'block_metrics': metrics, 'accuracy_curves': curves, 'model_episodes': episodes,
        'recovery': recovery, 'costs': costs, 'contributions': contributions,
        'contribution_restoration': restorations, 'contribution_reduction': reductions,
        'client_diagnostics': diagnostic_rows, 'round_diagnostics': round_rows, 'means': means, 'paired_effects': effects,
        'practical_criteria': checks, 'zero_tolerance_targets': zero_tolerance_checks(means)}
    for name, rows in tables.items(): pd.DataFrame(rows).to_csv(out / (name + '.csv'), index=False)
    summary = {'status': 'complete', 'runs': 800, 'matched_input_pairs': matches,
        'exact_pre_event_trajectories': prefixes, 'exact_first_round_updates': first_updates,
        'audited_rounds': 19200, 'maximum_reconstruction_error': max_error,
        'joint_practical_point_target_met': all(r['point_estimate_meets_margin'] for r in checks),
        'practical_checks': checks, 'independent_blocks_per_regime': 5, 'workload': workload,
        'execution_freeze_sha256': file_sha256(study / 'execution_freeze.json'),
        'analysis_source_sha256': file_sha256(__file__),
        'interpretation': 'Prospective descriptive replication on known MNIST. No table-wide significance, independent unseen dataset, or established novelty claim.'}
    write_json(out / 'summary.json', summary)
    write_json(study / 'analysis_manifest.json', {str(p.relative_to(study)): file_sha256(p) for p in out.iterdir() if p.is_file()})
    return summary
