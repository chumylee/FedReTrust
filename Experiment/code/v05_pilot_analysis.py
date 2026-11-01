"""Descriptive two-block analysis for the fixed, separate v0.5 pilot."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

from experiment_core import file_sha256, write_json
from research_runs import ROOT
from research_analysis import extract_run, cross_method_pair, cost_record
from scenario_metrics import longer_scenarios

ACCURACY = 'overall_mean_5_24_pp'
HONEST = 'affected_honest_event_pp'
DAMAGE = 'attack_active_signed_damage_pp'
CORE_METRICS = [('clean', ACCURACY), ('abrupt_shot_noise', HONEST),
                ('gradual_stripe', HONEST), ('label_flip', DAMAGE),
                ('label_flip', ACCURACY)]


def describe(values, blocks):
    """Two observed blocks are not a five-block confidence interval."""
    if list(blocks) != [1, 2]:
        raise ValueError('Exactly the two registered blocks in order are required.')
    values = np.asarray(values, dtype=float)
    if values.shape != (2,) or not np.isfinite(values).all():
        raise ValueError('Expected two finite measurements.')
    return {'n': 2, 'block_1': float(values[0]), 'block_2': float(values[1]),
            'mean': float(values.mean()), 'min': float(values.min()), 'max': float(values.max()),
            'range': float(np.ptp(values))}


def summarize(metrics):
    table = pd.DataFrame(metrics)
    if table.duplicated(['method', 'regime', 'block', 'scenario', 'metric']).any():
        raise ValueError('Duplicate block metric.')
    output = []
    keys = ['method', 'regime', 'scenario', 'metric']
    for identity, group in table.groupby(keys):
        group = group.sort_values('block')
        output.append({**dict(zip(keys, identity)), **describe(group.value, group.block.tolist())})
    return output


def contrasts(metrics, candidate, comparators):
    table = pd.DataFrame(metrics)
    result = []
    for other in comparators:
        paired = table[table.method == candidate].merge(table[table.method == other],
            on=['regime', 'block', 'seed', 'scenario', 'metric'], suffixes=('_candidate', '_other'),
            validate='one_to_one')
        for identity, group in paired.groupby(['regime', 'scenario', 'metric']):
            group = group.sort_values('block')
            differences = group.value_candidate - group.value_other
            result.append({'candidate': candidate, 'comparator': other,
                **dict(zip(['regime', 'scenario', 'metric'], identity)),
                **describe(differences, group.block.tolist())})
    return result


def evaluate_gates(effects, old_method):
    lookup = {(e['comparator'], e['regime'], e['scenario'], e['metric']): e for e in effects}
    rows = []
    def add(group, name, other, regime, scenario, metric, bound, direction):
        effect = lookup[(other, regime, scenario, metric)]
        comparison = (lambda x: x >= bound) if direction == '>=' else (lambda x: x <= bound) if direction == '<=' else (lambda x: x > bound)
        rows.append({**effect, 'gate_group': group, 'criterion': name, 'bound_pp': bound,
            'direction': direction, 'point_pass': bool(comparison(effect['mean'])),
            'block_1_pass': bool(comparison(effect['block_1'])),
            'block_2_pass': bool(comparison(effect['block_2']))})
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
    advancement = all(r['point_pass'] for r in honest + focused)
    complete = all(r['point_pass'] for r in rows)
    decision = {'original_practical_targets_passed': sum(r['point_pass'] for r in target),
        'original_practical_targets_total': len(target),
        'focused_fix_checks_passed': sum(r['point_pass'] for r in focused),
        'focused_fix_checks_total': len(focused),
        'advancement_point_criterion_met': advancement,
        'complete_pilot_point_targets_met': complete,
        'interpretation': ('All declared pilot point targets passed; a larger, separately declared development evaluation may be considered.' if complete else
            'Focused improvement and retention checks passed, but at least one original practical target failed; consider a larger development check without calling the problem solved.' if advancement else
            'The focused pilot advancement checks failed. Preserve the candidate and all results; no automatic follow-up or parameter changes.'),
        'automatic_continuation_authorized': False,
        'uncertainty': 'Two paired blocks per regime. Both block values and ranges are reported; no inferential superiority, generalization or novelty claim.',
        'block_selection_limit': 'Registered blocks 1 and 2 only. Three observed original blocks, including block 5, are outside this screen; their prior outcomes already informed development.',
        'failure_rule': 'Complete coverage is required. No omitted failures, substituted seeds, or successful-block-only summaries.'}
    return rows, decision


def analyse(study):
    from v05_pilot import verify, records, reused_records, key, MAIN, COMPARATORS, stamp
    from validation_replication import MAIN as OLD_METHOD
    study = Path(study)
    frozen = verify(study)
    candidate_records = records(study)
    if len(candidate_records) != 16 or {key(r['job']) for r in candidate_records} != {key(j) for j in frozen['jobs']}:
        raise ValueError('All sixteen candidate runs are required; no partial inference.')
    all_records = candidate_records + reused_records(study)
    indexed = {key(r['job']): r for r in all_records}
    if len(indexed) != 64:
        raise ValueError('Unexpected complete comparison grid.')
    metrics, curves, costs, diagnostics, local_rows = [], [], [], [], []
    matched, first_updates, pre_event = 0, 0, 0
    for result in all_records:
        job = result['job']
        path = ROOT / result['run_path']
        ident = {k: job[k] for k in ('method', 'regime', 'block', 'seed', 'scenario')}
        ident['job_id'] = result['job_id']
        own_clean = indexed[(job['method'], job['regime'], job['block'], 'clean')]
        fedavg = indexed[('FedAvg', job['regime'], job['block'], job['scenario'])]
        cross_method_pair(result, fedavg)
        matched += 1
        scenario = longer_scenarios()[job['scenario']]
        extracted, curve, _ = extract_run(result, own_clean if scenario.events else None)
        metrics.extend({**row, **ident} for row in extracted)
        curves.extend({**row, **ident} for row in curve)
        costs.append({**cost_record(result), **ident, 'artifact_bytes': result['artifact_bytes']})
        local = pd.read_csv(path / 'client_metrics.csv')
        tasks = pd.read_csv(path / 'task_metrics.csv')
        if len(local) != 240 or not local.optimizer_steps.eq(8).all() or not local.examples.eq(512).all():
            raise ValueError('Local training workload changed.')
        if not np.isfinite(local.online_train_loss).all():
            raise ValueError('Nonfinite local training loss.')
        if len(tasks) != 75 or not tasks.examples.eq(1000).all():
            raise ValueError('Development evaluation workload changed.')
        old_local = pd.read_csv(ROOT / fedavg['run_path'] / 'client_metrics.csv')
        columns = ['client', 'submitted_state_sha256', 'broadcast_state_sha256', 'shuffle_seed',
                   'optimizer_steps', 'online_train_loss']
        pd.testing.assert_frame_equal(local[local['round'] == 1][columns].reset_index(drop=True),
            old_local[old_local['round'] == 1][columns].reset_index(drop=True), check_exact=True)
        first_updates += 10
        local_rows.extend({**row, **ident} for row in local.to_dict('records'))
        if scenario.events:
            clean_path = ROOT / own_clean['run_path']
            history = pd.read_csv(path / 'history.csv')
            clean_history = pd.read_csv(clean_path / 'history.csv')
            if history[history['round'] <= 4].global_state_sha256.tolist() != clean_history[clean_history['round'] <= 4].global_state_sha256.tolist():
                raise ValueError('Own-method pre-event states differ from clean control.')
            clean_tasks = pd.read_csv(clean_path / 'task_metrics.csv')
            pd.testing.assert_frame_equal(tasks[tasks['round'] <= 4].reset_index(drop=True),
                clean_tasks[clean_tasks['round'] <= 4].reset_index(drop=True), check_exact=True)
            pre_event += 1
        access = json.loads((path / 'validation_access.json').read_text())
        if access['official_test_used'] or access['defence_uses_evaluation_data'] or access['server_reference_examples'] != 0:
            raise ValueError('Unexpected reference or evaluation-data access.')
        raw = json.loads((path / 'aggregation_diagnostics.json').read_text())
        if len(raw) != 24:
            raise ValueError('Missing aggregation diagnostics.')
        diagnostics.extend({**ident, 'round': row['round'],
            'values_json': json.dumps(row['values'], sort_keys=True, allow_nan=False)} for row in raw)
    means = summarize(metrics)
    effects = contrasts(metrics, MAIN, COMPARATORS)
    gates, decision = evaluate_gates(effects, OLD_METHOD)
    decision.update({'created_at': stamp(), 'new_runs': 16, 'reused_runs': 48,
        'matched_input_pairs': matched, 'exact_first_round_updates': first_updates,
        'exact_own_method_pre_event_trajectories': pre_event,
        'source_freeze_sha256': file_sha256(study / 'execution_freeze.json')})
    out = study / 'analysis'
    out.mkdir(exist_ok=True)
    for name, rows in [('block_metrics', metrics), ('means', means), ('paired_effects', effects),
                       ('gates', gates), ('accuracy_curves', curves), ('costs', costs),
                       ('local_training', local_rows), ('round_diagnostics', diagnostics)]:
        pd.DataFrame(rows).to_csv(out / (name + '.csv'), index=False)
    comparison = [row for row in means if (row['scenario'], row['metric']) in CORE_METRICS]
    pd.DataFrame(comparison).to_csv(out / 'core_comparison.csv', index=False)
    write_json(out / 'decision.json', decision)
    write_json(study / 'analysis_manifest.json', {'created_at': stamp(),
        'execution_freeze_sha256': file_sha256(study / 'execution_freeze.json'),
        'artifact_sha256': {str(p.relative_to(study)): file_sha256(p) for p in sorted(out.iterdir()) if p.is_file()}})
    return decision
