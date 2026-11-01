"""Five-block exploratory follow-up; the failed two-block pilot stays failed."""
from pathlib import Path
import json
import math

import numpy as np
import pandas as pd

from experiment_core import file_sha256, write_json
from research_runs import ROOT
from research_analysis import extract_run, cross_method_pair, cost_record
from scenario_metrics import longer_scenarios
from v05_pilot_analysis import ACCURACY, HONEST, DAMAGE, CORE_METRICS

BLOCKS = (1, 2, 3, 4, 5)
T_DF4 = 2.7764451051977987
SUBSETS = {'pilot_blocks_1_2': (1, 2), 'additional_blocks_3_5': (3, 4, 5)}


def describe(values, blocks, expected=BLOCKS, interval=True):
    """Intervals use five independent paired blocks, never rounds or clients."""
    if tuple(blocks) != tuple(expected):
        raise ValueError('Exactly the declared blocks in order are required.')
    values = np.asarray(values, dtype=float)
    if values.shape != (len(expected),) or not np.isfinite(values).all():
        raise ValueError('Expected one finite measurement per declared block.')
    if interval and tuple(expected) != BLOCKS:
        raise ValueError('Nominal df4 intervals require all five blocks.')
    mean = float(values.mean())
    sd = float(values.std(ddof=1))
    result = {'n': len(values), **{f'block_{b}': float(v) for b, v in zip(blocks, values)},
        'mean': mean, 'sd': sd, 'min': float(values.min()), 'max': float(values.max()),
        'range': float(np.ptp(values))}
    if interval:
        half = T_DF4 * sd / math.sqrt(5)
        result.update({'ci_low': mean - half, 'ci_high': mean + half,
                       'interval': 'nominal two-sided 95% t, df4, exploratory'})
    return result


def validate_metric_grid(metrics, methods):
    table = pd.DataFrame(metrics)
    keys = ['method', 'regime', 'block', 'scenario', 'metric']
    if table.empty or table.duplicated(keys).any():
        raise ValueError('Missing metrics or duplicate block measurement.')
    if set(table.method) != set(methods) or set(table.regime) != {'IID', 'non-IID'}:
        raise ValueError('Missing or unexpected comparison method/regime.')
    for _, group in table.groupby(['method', 'regime', 'scenario', 'metric']):
        if tuple(sorted(group.block)) != BLOCKS:
            raise ValueError('Incomplete five-block metric group.')
    return table


def summarize(metrics, expected=BLOCKS, interval=True):
    table = pd.DataFrame(metrics)
    if table.duplicated(['method', 'regime', 'block', 'scenario', 'metric']).any():
        raise ValueError('Duplicate block measurement.')
    output = []
    names = ['method', 'regime', 'scenario', 'metric']
    for identity, group in table.groupby(names):
        group = group.sort_values('block')
        output.append({**dict(zip(names, identity)),
            **describe(group.value, group.block.tolist(), expected, interval)})
    return output


def contrasts(metrics, candidate, comparators, expected=BLOCKS, interval=True):
    table = pd.DataFrame(metrics)
    result = []
    merge_keys = ['regime', 'block', 'seed', 'scenario', 'metric']
    target = table[table.method == candidate]
    if target.empty:
        raise ValueError('Missing candidate metrics.')
    for other in comparators:
        reference = table[table.method == other]
        # Outer matching detects a missing block or a changed seed on either side.
        paired = target.merge(reference, on=merge_keys, suffixes=('_candidate', '_other'),
            validate='one_to_one', how='outer', indicator=True)
        if paired.empty or not paired['_merge'].eq('both').all():
            raise ValueError('Incomplete or mismatched paired comparison.')
        for identity, group in paired.groupby(['regime', 'scenario', 'metric']):
            group = group.sort_values('block')
            differences = group.value_candidate - group.value_other
            result.append({'candidate': candidate, 'comparator': other,
                **dict(zip(['regime', 'scenario', 'metric'], identity)),
                **describe(differences, group.block.tolist(), expected, interval)})
    return result


def evaluate_gates(effects, old_method):
    """Reassess unchanged margins; these do not revise the pilot decision."""
    lookup = {(e['comparator'], e['regime'], e['scenario'], e['metric']): e for e in effects}
    if len(lookup) != len(effects):
        raise ValueError('Duplicate paired effect.')
    rows = []
    def add(group, name, other, regime, scenario, metric, bound, direction):
        effect = lookup[(other, regime, scenario, metric)]
        if effect['n'] != 5 or any(f'block_{b}' not in effect for b in BLOCKS):
            raise ValueError('Five-block margin assessment needs five paired values.')
        comparison = (lambda x: x >= bound) if direction == '>=' else (lambda x: x <= bound) if direction == '<=' else (lambda x: x > bound)
        rows.append({**effect, 'assessment_scope': 'five_block_exploratory_followup',
            'gate_group': group, 'criterion': name, 'bound_pp': bound, 'direction': direction,
            'point_pass': bool(comparison(effect['mean'])),
            **{f'block_{b}_pass': bool(comparison(effect[f'block_{b}'])) for b in BLOCKS},
            'interval_crosses_or_touches_margin': bool(effect['ci_low'] <= bound <= effect['ci_high'])})
    for regime in ('IID', 'non-IID'):
        for scenario, metric in CORE_METRICS[:3]:
            add('original_practical_target', 'honest_accuracy_within_1pp_FedAvg',
                'FedAvg', regime, scenario, metric, -1., '>=')
        add('original_practical_target', 'label_damage_within_1pp_RFA',
            'RFA', regime, 'label_flip', DAMAGE, 1., '<=')
        add('original_practical_target', 'label_damage_at_least_2pp_below_FedAvg',
            'FedAvg', regime, 'label_flip', DAMAGE, -2., '<=')
    for scenario, metric in CORE_METRICS[:3]:
        add('focused_fix', 'IID_honest_retention_within_half_pp_v03',
            old_method, 'IID', scenario, metric, -.5, '>=')
    add('focused_fix', 'IID_label_damage_retention_within_half_pp_v03',
        old_method, 'IID', 'label_flip', DAMAGE, .5, '<=')
    add('focused_fix', 'nonIID_label_damage_improves_at_least_1pp_v03',
        old_method, 'non-IID', 'label_flip', DAMAGE, -1., '<=')
    add('focused_fix', 'nonIID_absolute_attack_accuracy_improves_v03',
        old_method, 'non-IID', 'label_flip', ACCURACY, 0., '>')
    target = [r for r in rows if r['gate_group'] == 'original_practical_target']
    focused = [r for r in rows if r['gate_group'] == 'focused_fix']
    honest = [r for r in target if r['criterion'] == 'honest_accuracy_within_1pp_FedAvg']
    decision = {
        'assessment_scope': 'five_block_exploratory_followup_not_retroactive_pilot',
        'original_practical_targets_passed': sum(r['point_pass'] for r in target),
        'original_practical_targets_total': len(target),
        'focused_fix_checks_passed': sum(r['point_pass'] for r in focused),
        'focused_fix_checks_total': len(focused),
        'five_block_focused_and_honest_margins_met': all(r['point_pass'] for r in honest + focused),
        'five_block_all_point_targets_met': all(r['point_pass'] for r in rows),
        'automatic_continuation_authorized': False,
        'interpretation': 'The unchanged margins are reassessed descriptively over five paired development blocks. This follow-up was separately authorized after a failed pilot; its results cannot retroactively pass that pilot or establish independent confirmation.',
        'uncertainty': 'Per-regime paired block differences, SD and nominal two-sided 95% t intervals with df4. These exploratory intervals assume approximately normal independent block differences; they can be unstable at n=5 and have no simultaneous table-wide coverage.',
        'selection_limit': 'All five original block outcomes informed development and blocks 1–2 were already used in the pilot. Blocks 3–5 provide additional candidate trajectories on known development conditions, not unseen confirmation.',
        'failure_rule': 'All 160 trajectories and all five matched blocks are required; no failed-block omissions or replacements.',
        'secondary_diagnostics': 'Onset drops, largest active one-round declines and active-window minima describe fragility; they are exploratory diagnostics, not additional pass/fail targets.'}
    return rows, decision


def fragility(series):
    """Drops are positive for harm; active transitions include onset 4→5."""
    if not all(r in series for r in range(4, 25)):
        raise ValueError('Fragility diagnostics require rounds 4 through 24.')
    if not np.isfinite([series[r] for r in range(4, 25)]).all():
        raise ValueError('Nonfinite accuracy trajectory.')
    drops = {r: max(0., float(series[r - 1] - series[r])) for r in range(5, 25)}
    worst = max(drops, key=drops.get)
    minimum = min(range(5, 25), key=lambda r: series[r])
    return {'round4_accuracy_pp': float(series[4]), 'round5_accuracy_pp': float(series[5]),
        'onset_drop_pp': float(series[4] - series[5]),
        'maximum_active_one_round_drop_pp': drops[worst],
        'maximum_drop_end_round': worst, 'minimum_active_accuracy_pp': float(series[minimum]),
        'minimum_accuracy_round': minimum,
        'transition_scope': 'r-1 to r for r=5..24; includes onset 4->5; negative changes clipped to zero for maximum drop',
        'assessment_scope': 'exploratory_secondary_not_a_success_criterion'}


def validate_failed_pilot_decision(decision):
    for key in ('advancement_point_criterion_met', 'complete_pilot_point_targets_met',
                'automatic_continuation_authorized'):
        if decision.get(key) is not False:
            raise ValueError('Expected the preserved failed pilot and no automatic continuation.')


def analyse(study):
    from v05_followup import verify, records, reused_records, key, MAIN, COMPARATORS, stamp, pilot
    from validation_replication import MAIN as OLD_METHOD
    study = Path(study)
    frozen = verify(study)
    new = records(study)
    if len(new) != 24 or {key(r['job']) for r in new} != {key(j) for j in frozen['jobs']}:
        raise ValueError('All 24 follow-up runs are required; no complete-case substitution.')
    all_records = new + reused_records(study)
    indexed = {key(r['job']): r for r in all_records}
    expected = {(m, r, b, s) for m in (MAIN, *COMPARATORS) for r in ('IID', 'non-IID')
        for b in BLOCKS for s in ('clean', 'abrupt_shot_noise', 'gradual_stripe', 'label_flip')}
    if len(all_records) != 160 or set(indexed) != expected or len(indexed) != 160:
        raise ValueError('Unexpected combined 160-trajectory comparison grid.')
    old_decision_path = pilot.STUDY / 'analysis/decision.json'
    if file_sha256(old_decision_path) != frozen['prior_pilot_decision_sha256']:
        raise ValueError('Original failed-pilot decision changed.')
    old_decision = json.loads(old_decision_path.read_text())
    validate_failed_pilot_decision(old_decision)
    metrics, curves, costs, diagnostics, local_rows, fragility_rows = [], [], [], [], [], []
    matched, first_updates, pre_event = 0, 0, 0
    for result in all_records:
        job = result['job']; path = ROOT / result['run_path']
        ident = {k: job[k] for k in ('method', 'regime', 'block', 'seed', 'scenario')}
        ident.update({'job_id': result['job_id'], 'run_origin': 'new_followup' if job['method'] == MAIN and job['block'] >= 3 else 'reused_pilot' if job['method'] == MAIN else 'reused_original'})
        own_clean = indexed[(job['method'], job['regime'], job['block'], 'clean')]
        fedavg = indexed[('FedAvg', job['regime'], job['block'], job['scenario'])]
        cross_method_pair(result, fedavg); matched += 1
        scenario = longer_scenarios()[job['scenario']]
        extracted, curve, _ = extract_run(result, own_clean if scenario.events else None)
        metrics.extend({**row, **ident} for row in extracted)
        curves.extend({**row, **ident} for row in curve)
        costs.append({**cost_record(result), **ident, 'artifact_bytes': result['artifact_bytes']})
        local = pd.read_csv(path / 'client_metrics.csv'); tasks = pd.read_csv(path / 'task_metrics.csv')
        if len(local) != 240 or not local.optimizer_steps.eq(8).all() or not local.examples.eq(512).all():
            raise ValueError('Local workload changed.')
        if not np.isfinite(local.online_train_loss).all():
            raise ValueError('Nonfinite local training loss.')
        if len(tasks) != 75 or not tasks.examples.eq(1000).all():
            raise ValueError('Evaluation workload changed.')
        old_local = pd.read_csv(ROOT / fedavg['run_path'] / 'client_metrics.csv')
        columns = ['client', 'submitted_state_sha256', 'broadcast_state_sha256', 'shuffle_seed', 'optimizer_steps', 'online_train_loss']
        pd.testing.assert_frame_equal(local[local['round'] == 1][columns].reset_index(drop=True),
            old_local[old_local['round'] == 1][columns].reset_index(drop=True), check_exact=True)
        first_updates += 10
        local_rows.extend({**row, **ident} for row in local.to_dict('records'))
        if scenario.events:
            clean_path = ROOT / own_clean['run_path']
            history = pd.read_csv(path / 'history.csv'); clean_history = pd.read_csv(clean_path / 'history.csv')
            if history[history['round'] <= 4].global_state_sha256.tolist() != clean_history[clean_history['round'] <= 4].global_state_sha256.tolist():
                raise ValueError('Own-method pre-event states differ.')
            clean_tasks = pd.read_csv(clean_path / 'task_metrics.csv')
            pd.testing.assert_frame_equal(tasks[tasks['round'] <= 4].reset_index(drop=True),
                clean_tasks[clean_tasks['round'] <= 4].reset_index(drop=True), check_exact=True)
            pre_event += 1
        access = json.loads((path / 'validation_access.json').read_text())
        if access['official_test_used'] or access['defence_uses_evaluation_data'] or access['server_reference_examples'] != 0:
            raise ValueError('Unexpected reference/evaluation access.')
        raw = json.loads((path / 'aggregation_diagnostics.json').read_text())
        if len(raw) != 24:
            raise ValueError('Missing aggregation diagnostics.')
        diagnostics.extend({**ident, 'round': row['round'],
            'values_json': json.dumps(row['values'], sort_keys=True, allow_nan=False)} for row in raw)
        if job['scenario'] == 'label_flip':
            series = {row['round']: row['accuracy_pp'] for row in curve if row['group'] == 'all_clients'}
            onset = next(row['values'] for row in raw if row['round'] == 5)
            q = [v.get('restored_weight') for v in onset.get('client_metrics', {}).values()]
            all_one = all(float(v) == 1. for v in q) if len(q) == 10 and all(v is not None for v in q) else None
            fragility_rows.append({**ident, **fragility(series), 'round5_all_restored_weights_one': all_one})
    validate_metric_grid(metrics, (MAIN, *COMPARATORS))
    means = summarize(metrics); effects = contrasts(metrics, MAIN, COMPARATORS)
    gates, decision = evaluate_gates(effects, OLD_METHOD)
    subset_means, subset_effects = [], []
    for label, blocks in SUBSETS.items():
        subset = [r for r in metrics if r['block'] in blocks]
        subset_means.extend({'subset': label, **r} for r in summarize(subset, blocks, False))
        subset_effects.extend({'subset': label, **r} for r in contrasts(subset, MAIN, COMPARATORS, blocks, False))
    decision.update({'created_at': stamp(), 'new_runs': 24, 'reused_runs': 136, 'total_runs': 160,
        'reused_pilot_candidate_runs': 16, 'reused_original_comparator_runs': 120,
        'matched_input_pairs': matched, 'exact_first_round_updates': first_updates,
        'exact_own_method_pre_event_trajectories': pre_event,
        'source_freeze_sha256': file_sha256(study / 'execution_freeze.json'),
        'prior_pilot_decision_sha256': frozen['prior_pilot_decision_sha256'],
        'prior_pilot_decision': old_decision,
        'pilot_advancement_point_criterion_met': old_decision['advancement_point_criterion_met'],
        'pilot_complete_point_targets_met': old_decision['complete_pilot_point_targets_met']})
    out = study / 'analysis'; out.mkdir(exist_ok=True)
    for name, rows in [('block_metrics', metrics), ('means', means), ('paired_effects', effects),
        ('gates', gates), ('accuracy_curves', curves), ('costs', costs), ('local_training', local_rows),
        ('round_diagnostics', diagnostics), ('subset_means', subset_means),
        ('subset_paired_effects', subset_effects), ('label_flip_fragility', fragility_rows)]:
        pd.DataFrame(rows).to_csv(out / (name + '.csv'), index=False)
    for name, rows in [('core_comparison', means), ('core_subset_comparison', subset_means)]:
        pd.DataFrame([r for r in rows if (r['scenario'], r['metric']) in CORE_METRICS]).to_csv(out / (name + '.csv'), index=False)
    write_json(out / 'decision.json', decision)
    write_json(study / 'analysis_manifest.json', {'created_at': stamp(),
        'execution_freeze_sha256': file_sha256(study / 'execution_freeze.json'),
        'artifact_sha256': {str(p.relative_to(study)): file_sha256(p) for p in sorted(out.iterdir()) if p.is_file()}})
    return decision
