"""Offline, selected-stress-case v0.6 evaluation; no fresh-confirmation claim."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

from experiment_core import file_sha256, write_json
from research_runs import ROOT
from research_analysis import extract_run, cross_method_pair, cost_record
from research_study import task_series
from scenario_metrics import longer_scenarios
from v05_followup_analysis import summarize, contrasts, fragility
from v05_pilot_analysis import ACCURACY, HONEST, DAMAGE, CORE_METRICS

BLOCKS = (3, 5)
ONSET_DAMAGE = 'attack_onset_signed_damage_pp'
ONSET_ACCURACY = 'overall_round5_pp'
ONSET_DROP = 'onset_raw_drop_pp'
ONSET_METRICS = (ONSET_DAMAGE, ONSET_ACCURACY, ONSET_DROP)
ALL_CORE = [*CORE_METRICS, *(('label_flip', m) for m in ONSET_METRICS)]
PREVIOUS = 'FedReTrust-v05'


def onset_metrics(attack_series, clean_series):
    """Both trajectories use percentage-point units and the same task weights."""
    if 5 not in clean_series or not np.isfinite(clean_series[5]):
        raise ValueError('A finite own-clean round5 control is required.')
    values = fragility(attack_series)
    return {**values, 'own_clean_round5_accuracy_pp': float(clean_series[5]),
        ONSET_DAMAGE: float(clean_series[5] - attack_series[5]),
        ONSET_ACCURACY: float(attack_series[5]), ONSET_DROP: values['onset_drop_pp']}


def evaluate_gates(effects):
    lookup = {(e['comparator'], e['regime'], e['scenario'], e['metric']): e for e in effects}
    if len(lookup) != len(effects):
        raise ValueError('Duplicate effect rows.')
    rows = []
    def add(group, criterion, comparator, regime, scenario, metric, bound, direction, block=None):
        effect = lookup[(comparator, regime, scenario, metric)]
        if effect['n'] != 2 or any(f'block_{b}' not in effect for b in BLOCKS):
            raise ValueError('Both selected blocks3 and5 are required.')
        if any(k in effect for k in ('ci_low', 'ci_high')):
            raise ValueError('This n=2 selected screen does not use confidence intervals.')
        value = float(effect['mean'] if block is None else effect[f'block_{block}'])
        if not np.isfinite(value):
            raise ValueError('Nonfinite gate value.')
        comparison = (lambda x: x >= bound) if direction == '>=' else (lambda x: x <= bound) if direction == '<=' else (lambda x: x > bound)
        rows.append({**effect, 'gate_group': group, 'criterion': criterion,
            'gate_block': block, 'assessment_unit': 'two_selected_block_mean' if block is None else f'block_{block}',
            'checked_difference_pp': value, 'bound_pp': bound, 'direction': direction,
            'point_pass': bool(comparison(value)),
            **{f'block_{b}_meets_same_bound': bool(comparison(effect[f'block_{b}'])) for b in BLOCKS}})
    for regime in ('IID', 'non-IID'):
        for scenario, metric in CORE_METRICS[:3]:
            add('original_practical_target', 'honest_accuracy_within_1pp_FedAvg', 'FedAvg', regime, scenario, metric, -1., '>=')
        add('original_practical_target', 'label_damage_within_1pp_RFA', 'RFA', regime, 'label_flip', DAMAGE, 1., '<=')
        add('original_practical_target', 'label_damage_at_least_2pp_below_FedAvg', 'FedAvg', regime, 'label_flip', DAMAGE, -2., '<=')
    for block in BLOCKS:
        for scenario, metric in [*CORE_METRICS[:3], ('label_flip', ACCURACY)]:
            add('IID_retention', 'IID_accuracy_within_half_pp_v05', PREVIOUS, 'IID', scenario, metric, -.5, '>=', block)
        add('IID_retention', 'IID_damage_within_half_pp_v05', PREVIOUS, 'IID', 'label_flip', DAMAGE, .5, '<=', block)
    add('nonIID_focus', 'nonIID_mean_damage_improves_at_least_1pp_v05', PREVIOUS, 'non-IID', 'label_flip', DAMAGE, -1., '<=')
    for block in BLOCKS:
        add('nonIID_focus', 'nonIID_absolute_attack_mean_improves_v05', PREVIOUS, 'non-IID', 'label_flip', ACCURACY, 0., '>', block)
    for block, gain in ((3, 0.), (5, 5.)):
        add('nonIID_focus', 'nonIID_onset_damage_reduction_v05', PREVIOUS, 'non-IID', 'label_flip', ONSET_DAMAGE, -gain, '<=', block)
        add('nonIID_focus', 'nonIID_onset_absolute_accuracy_gain_v05', PREVIOUS, 'non-IID', 'label_flip', ONSET_ACCURACY, gain, '>=', block)
    groups = {name: [r for r in rows if r['gate_group'] == name]
              for name in ('original_practical_target', 'IID_retention', 'nonIID_focus')}
    decision = {'assessment_scope': 'selected_known_stress_case_development',
        'complete_selected_screen_point_targets_met': all(r['point_pass'] for r in rows),
        'IID_retention_point_criteria_met': all(r['point_pass'] for r in groups['IID_retention']),
        'nonIID_focused_point_criteria_met': all(r['point_pass'] for r in groups['nonIID_focus']),
        'original_practical_targets_passed': sum(r['point_pass'] for r in groups['original_practical_target']),
        'original_practical_targets_total': len(groups['original_practical_target']),
        'IID_retention_checks_passed': sum(r['point_pass'] for r in groups['IID_retention']),
        'IID_retention_checks_total': len(groups['IID_retention']),
        'nonIID_focused_checks_passed': sum(r['point_pass'] for r in groups['nonIID_focus']),
        'nonIID_focused_checks_total': len(groups['nonIID_focus']),
        'total_checks_passed': sum(r['point_pass'] for r in rows), 'total_checks': len(rows),
        'automatic_continuation_authorized': False,
        'interpretation': 'All point criteria refer only to two deliberately selected, previously inspected stress blocks. Partial improvements do not constitute complete balance or independent confirmation; prior failed v0.5 decisions remain unchanged.',
        'uncertainty': 'Both block values, paired differences and observed ranges; no n=2 confidence intervals, significance tests, representative-population or established-novelty claims.',
        'coefficient_scope': 'V0.6 is affine in current raw deltas plus client history. Returned current coefficients alone do not reconstruct its update; full contributions and history residual are audited separately.',
        'onset_scope': 'Round5 own-clean-relative damage and absolute accuracy are the fixed onset checks. Raw4-to5 declines, active minima and later bursts are descriptive secondary diagnostics.'}
    return rows, decision


def audit_affine_round(values):
    if values.get('coefficient_convention') != 'affine_current_raw_deltas_plus_client_history':
        raise ValueError('The v0.6 affine accounting contract is missing.')
    measurements = values['contribution_measurements']
    if measurements['gram_matrix_convention'] != 'full_applied_current_plus_history_contributions':
        raise ValueError('The full Gram matrix must include historical contributions.')
    grams = [np.asarray(measurements[k], dtype=float) for k in
        ('gram_matrix', 'current_gram_matrix', 'historical_gram_matrix', 'current_history_cross_gram_matrix')]
    if any(g.shape != (10, 10) or not np.isfinite(g).all() for g in grams):
        raise ValueError('Invalid affine contribution Gram matrices.')
    full, current, history, cross = grams
    np.testing.assert_allclose(full, current + history + cross + cross.T, rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(full.sum(), measurements['aggregate_l2_norm']**2, rtol=1e-9, atol=1e-10)
    metrics = values['client_metrics']
    if len(metrics) != 10 or set(measurements['client_order']) != set(range(10)):
        raise ValueError('Expected ten distinct client contributions.')
    for i, client in enumerate(measurements['client_order']):
        row = metrics[str(client)]
        if row['applied_contribution_scope'] != 'full_current_plus_client_history':
            raise ValueError('Applied norm must cover current plus historical contribution.')
        np.testing.assert_allclose(full[i, i], row['applied_contribution_l2_norm']**2, rtol=1e-9, atol=1e-10)
        np.testing.assert_allclose(row['current_delta_coefficient'],
            row['filtered_effective_coefficient'] * row['innovation_factor'], rtol=1e-12, atol=1e-14)
    for field in ('affine_reconstruction_max_abs_error', 'reconstruction_max_abs_error'):
        if not np.isfinite(measurements[field]) or measurements[field] < 0:
            raise ValueError('Invalid reconstruction error.')
    return {'innovation_radius': values['innovation_radius'],
        'median_innovation_l2_norm': values['median_innovation_l2_norm'],
        'zero_median_innovation_bypass': values['zero_median_innovation_bypass'],
        'clipped_clients': sum(v['innovation_factor'] < 1. for v in metrics.values()),
        'history_initialized_clients': sum(v['history_initialized_before_round'] for v in metrics.values()),
        'history_clients': values['history_clients'], 'history_tensor_bytes': values['history_tensor_bytes'],
        **{k: measurements[k] for k in ('historical_term_present', 'historical_aggregate_l2_norm',
            'current_aggregate_l2_norm', 'aggregate_l2_norm', 'effective_coefficient_mass',
            'filtered_effective_coefficient_mass', 'historical_coefficient_mass',
            'affine_reconstruction_max_abs_error', 'reconstruction_max_abs_error')}}


def analyse(study):
    from v06_pilot import verify, records, reused_records, key, MAIN, COMPARATORS, stamp, prior
    study = Path(study); frozen = verify(study)
    new = records(study)
    if len(new) != 16 or {key(r['job']) for r in new} != {key(j) for j in frozen['jobs']}:
        raise ValueError('All sixteen candidate runs are required.')
    all_records = new + reused_records(study); indexed = {key(r['job']): r for r in all_records}
    expected = {(m, r, b, s) for m in (MAIN, *COMPARATORS) for r in ('IID', 'non-IID')
        for b in BLOCKS for s in ('clean', 'abrupt_shot_noise', 'gradual_stripe', 'label_flip')}
    if len(all_records) != 80 or set(indexed) != expected or len(indexed) != 80:
        raise ValueError('Unexpected eighty-trajectory grid.')
    metrics, curves, costs, diagnostics, locals_, onset_rows, activation, clients = [], [], [], [], [], [], [], []
    matched, first_updates, pre_event = 0, 0, 0
    for result in all_records:
        job = result['job']; path = ROOT / result['run_path']
        ident = {k: job[k] for k in ('method', 'regime', 'block', 'seed', 'scenario')}
        ident.update({'job_id': result['job_id'], 'run_origin': 'new_v06' if job['method'] == MAIN else 'reused_v05' if job['method'] == PREVIOUS else 'reused_original'})
        clean = indexed[(job['method'], job['regime'], job['block'], 'clean')]
        fedavg = indexed[('FedAvg', job['regime'], job['block'], job['scenario'])]
        cross_method_pair(result, fedavg); matched += 1
        scenario = longer_scenarios()[job['scenario']]
        extracted, curve, _ = extract_run(result, clean if scenario.events else None)
        metrics.extend({**row, **ident} for row in extracted); curves.extend({**row, **ident} for row in curve)
        costs.append({**cost_record(result), **ident, 'artifact_bytes': result['artifact_bytes']})
        local = pd.read_csv(path / 'client_metrics.csv'); tasks = pd.read_csv(path / 'task_metrics.csv')
        if len(local) != 240 or not local.optimizer_steps.eq(8).all() or not local.examples.eq(512).all() or not np.isfinite(local.online_train_loss).all():
            raise ValueError('Local training workload or loss invalid.')
        if len(tasks) != 75 or not tasks.examples.eq(1000).all():
            raise ValueError('Evaluation workload changed.')
        old_local = pd.read_csv(ROOT / fedavg['run_path'] / 'client_metrics.csv')
        cols = ['client', 'submitted_state_sha256', 'broadcast_state_sha256', 'shuffle_seed', 'optimizer_steps', 'online_train_loss']
        pd.testing.assert_frame_equal(local[local['round'] == 1][cols].reset_index(drop=True),
            old_local[old_local['round'] == 1][cols].reset_index(drop=True), check_exact=True)
        first_updates += 10; locals_.extend({**row, **ident} for row in local.to_dict('records'))
        if scenario.events:
            clean_path = ROOT / clean['run_path']
            history = pd.read_csv(path / 'history.csv'); clean_history = pd.read_csv(clean_path / 'history.csv')
            if history[history['round'] <= 4].global_state_sha256.tolist() != clean_history[clean_history['round'] <= 4].global_state_sha256.tolist():
                raise ValueError('Own-method pre-event trajectory changed.')
            clean_tasks = pd.read_csv(clean_path / 'task_metrics.csv')
            pd.testing.assert_frame_equal(tasks[tasks['round'] <= 4].reset_index(drop=True), clean_tasks[clean_tasks['round'] <= 4].reset_index(drop=True), check_exact=True)
            pre_event += 1
        access = json.loads((path / 'validation_access.json').read_text())
        if access['official_test_used'] or access['defence_uses_evaluation_data'] or access['server_reference_examples'] != 0:
            raise ValueError('Unexpected trusted or evaluation-data access.')
        raw = json.loads((path / 'aggregation_diagnostics.json').read_text())
        if len(raw) != 24:
            raise ValueError('Missing aggregation diagnostics.')
        for row in raw:
            diagnostics.append({**ident, 'round': row['round'], 'values_json': json.dumps(row['values'], sort_keys=True, allow_nan=False)})
            if job['method'] == MAIN:
                activation.append({**ident, 'round': row['round'], **audit_affine_round(row['values'])})
                clients.extend({**ident, 'round': row['round'], 'client': int(cid), **v}
                    for cid, v in row['values']['client_metrics'].items())
        if job['scenario'] == 'label_flip':
            observed = {row['round']: row['accuracy_pp'] for row in curve if row['group'] == 'all_clients'}
            control = {r: 100 * v for r, v in task_series(clean, scenario_name='label_flip').items()}
            onset = onset_metrics(observed, control)
            onset_rows.append({**ident, **onset})
            metrics.extend({**ident, 'metric': name, 'value': onset[name]} for name in ONSET_METRICS)
    means = summarize(metrics, expected=BLOCKS, interval=False)
    effects = contrasts(metrics, MAIN, COMPARATORS, expected=BLOCKS, interval=False)
    gates, decision = evaluate_gates(effects)
    prior_pilot = json.loads((prior.pilot.STUDY / 'analysis/decision.json').read_text())
    prior_followup = json.loads((prior.STUDY / 'analysis/decision.json').read_text())
    decision.update({'created_at': stamp(), 'new_runs': 16, 'reused_runs': 64, 'total_runs': 80,
        'matched_input_pairs': matched, 'exact_first_round_updates': first_updates,
        'exact_own_method_pre_event_trajectories': pre_event, 'audited_affine_rounds': len(activation),
        'maximum_affine_reconstruction_error': max(r['affine_reconstruction_max_abs_error'] for r in activation),
        'maximum_model_reconstruction_error': max(r['reconstruction_max_abs_error'] for r in activation),
        'source_freeze_sha256': file_sha256(study / 'execution_freeze.json'),
        'prior_pilot_decision': prior_pilot, 'prior_followup_decision': prior_followup,
        'prior_pilot_decision_sha256': frozen['prior_pilot_decision_sha256'],
        'prior_followup_decision_sha256': frozen['prior_followup_decision_sha256']})
    out = study / 'analysis'; out.mkdir(exist_ok=True)
    for name, rows in [('block_metrics', metrics), ('means', means), ('paired_effects', effects),
        ('gates', gates), ('accuracy_curves', curves), ('costs', costs), ('local_training', locals_),
        ('round_diagnostics', diagnostics), ('onset_fragility', onset_rows),
        ('history_activation', activation), ('candidate_client_diagnostics', clients),
        ('onset_paired_effects', [r for r in effects if r['metric'] in ONSET_METRICS]),
        ('core_comparison', [r for r in means if (r['scenario'], r['metric']) in ALL_CORE])]:
        pd.DataFrame(rows).to_csv(out / (name + '.csv'), index=False)
    write_json(out / 'decision.json', decision)
    write_json(study / 'analysis_manifest.json', {'created_at': stamp(),
        'execution_freeze_sha256': file_sha256(study / 'execution_freeze.json'),
        'artifact_sha256': {str(p.relative_to(study)): file_sha256(p) for p in sorted(out.iterdir()) if p.is_file()}})
    return decision
