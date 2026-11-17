"""Resume-safe FLAURA and no-DP Byz-Clip21 publication extension."""
import argparse
from contextlib import redirect_stdout
import fcntl
import io
import json
import math
import os
import platform
import sys
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from time import monotonic

import numpy as np
import torch

import study
import table2_metrics as metrics
from extended_methods import (ByzClip21State, ByzConfig, CalibrationObserver,
                              PublicationFLAURA, auxiliary_reference, full_gradient)

METHODS = ('FLAURA-reconstruction', 'Byz-Clip21-SGD2M-noDP')
HERE = study.HERE
BASE = HERE / 'results' / 'full'
TUNING_ROUNDS = 24
TUNING_GRID = tuple(ByzConfig(learning_rate=lr, tau=tau)
                    for lr in (.01, .1, 1., 10.) for tau in (.1, 1.))


class Tee:
    def __init__(self, stream, file):
        self.stream, self.file = stream, file

    def write(self, text):
        self.stream.write(text)
        self.file.write(text)
        self.file.flush()

    def flush(self):
        self.stream.flush()
        self.file.flush()


def make_data(dataset, loaded, seed, device):
    data = study.har_plan(loaded, seed) if dataset == 'har' else study.image_plan(dataset, loaded, seed)
    study.to_device(data, device)
    return data


def verify_manifest(path, value):
    value = json.loads(json.dumps(value))
    if path.exists() and json.loads(path.read_text()) != value:
        raise ValueError(f'Existing settings differ: {path}. Use a new output directory.')
    study.write_json(path, value)


def byz_job(data, condition, rounds, device, options, evaluate_round):
    start = monotonic()
    torch.manual_seed(data['plan']['seed'])
    model = data['model']().to(device)
    initial = study.state_sha256(study.clone_state(model))
    state = ByzClip21State(data['clients'], sum(p.numel() for p in model.parameters()), options)
    rows, diagnostics, failure = [], [], None
    try:
        for round_id in range(rounds + 1):
            fraction = min(1., max(0., (round_id - study.ONSET + 1) / study.RAMP))
            if round_id:
                # Algorithm 1 line 2: apply the previous robust gradient first.
                # g^0=0: first model update is a documented no-op, not a warm-up.
                offset = 0
                with torch.no_grad():
                    for parameter in model.parameters():
                        length = parameter.numel()
                        update = state.aggregate[offset:offset + length].reshape_as(parameter)
                        parameter.add_(update.to(parameter.device, parameter.dtype),
                                       alpha=-options.learning_rate)
                        offset += length
                gradients = []
                for cid in state.clients:
                    clean, shifted, labels, order = data['clients'][cid]
                    x, y = clean, labels
                    if round_id >= study.ONSET and cid in data['affected']:
                        if condition == 'label_flip':
                            y = data['permutation'][labels]
                        elif condition == 'honest_drift':
                            x = clean.clone()
                            selected = order[:math.floor(fraction * len(labels))]
                            x[selected] = shifted[selected]
                    gradients.append(full_gradient(model, x, y))
                diagnostics.append({'round': round_id, **state.observe(torch.stack(gradients))})
            rows.append(evaluate_round(model, round_id, fraction))
            if round_id and (round_id % 8 == 0 or round_id == rounds):
                print(f"    Round {round_id}/{rounds}: {rows[-1]['accuracy_pct']:.2f}% accuracy", flush=True)
    except FloatingPointError as error:
        failure = {'round': round_id, 'message': str(error)}
    return rows, {
        'initial_model_sha256': initial, 'status': 'diverged' if failure else 'completed',
        'failure': failure, 'seconds': monotonic() - start,
        'settings': {'config': asdict(options), 'privacy_noise': 0,
                     'paper': 'https://arxiv.org/abs/2603.23472v2',
                     'implementation': 'Algorithm_1_reconstruction_no_DP',
                     'gradient': 'full_local_mean_at_fixed_shared_model',
                     'sample_passes_per_client_per_round': 1,
                     'server_aggregation': 'NNM_then_coordinate_median_even_midpoint',
                     'fault_bound': state.faulty_bound,
                     'fault_bound_rule': 'floor(client_count/3)_fixed_in_every_condition',
                     'initial_buffers': 'zero_first_update_no_op',
                     'label_poisoning': 'poisoned_labels_follow_same_gradient_recursion',
                     'training_protocol_differs_from_local_SGD': True},
        'diagnostics': diagnostics,
    }


def calibrate(dataset, loaded, device, directory, smoke):
    directory.mkdir(parents=True, exist_ok=True)
    chosen = directory / 'selected.json'
    if chosen.exists():
        result = json.loads(chosen.read_text())
        for name, digest in result['evidence_sha256'].items():
            if study.file_sha256(directory / name) != digest:
                raise ValueError('Calibration evidence changed.')
        return result
    # Separate development seed. No official test evaluation or attack/drift selection.
    seed = study.SEEDS[dataset][0] + 50000
    data = make_data(dataset, loaded, seed, device)
    study.write_json(directory / 'plan.json', data['plan'])
    reference = auxiliary_reference(data)
    observer = CalibrationObserver(data['model'], reference, device)
    study.METHODS['_geometry_calibration'] = lambda: observer
    print(f'Calibrating FLAURA geometry: {dataset}, development seed {seed}', flush=True)
    # The shared runner expects a score field. Geometry calibration does not
    # evaluate accuracy: suppress its placeholder progress line, not real scores.
    with redirect_stdout(io.StringIO()):
        _, record = study.run_job(data, '_geometry_calibration', 'clean', 6, device,
                                  evaluate_round=lambda model, r, f: {
                                      'round': r, 'accuracy_pct': 0., 'drift_pct': None})
    print('    Six geometry-calibration rounds completed; accuracy was not evaluated.', flush=True)
    if record['status'] != 'completed':
        raise RuntimeError('Clean FLAURA scale calibration diverged.')
    sigma = max(float(np.median(observer.distances)), 1e-12)
    study.write_json(directory / 'flaura_geometry.json', {
        'distances': observer.distances, 'sigma_g': sigma, 'reference': reference.metadata,
        'rule': 'median_all_client_distances_six_clean_rounds',
        'official_test_evaluated': False, 'accuracy_used': False,
    })
    root_x, root_y = data['root']

    @torch.inference_mode()
    def development_evaluate(model, round_id, fraction):
        model.eval()
        logits = torch.cat([model(x) for x in root_x.split(256)])
        if not torch.isfinite(logits).all():
            raise FloatingPointError('Non-finite development logits.')
        return {'round': round_id,
                'accuracy_pct': 100 * float((logits.argmax(1) == root_y).float().mean()),
                'loss': float(torch.nn.functional.cross_entropy(logits, root_y))}

    grid = (ByzConfig(),) if smoke else TUNING_GRID
    candidates = []
    for index, options in enumerate(grid):
        filename = directory / f'byz_candidate_{index}.json'
        if filename.exists():
            candidate = json.loads(filename.read_text())
            if candidate['config'] != asdict(options):
                raise ValueError('Development candidate settings differ.')
        else:
            print(f'Byz-Clip21 development {dataset} [{index + 1}/{len(grid)}]: '
                  f'lr={options.learning_rate}, tau={options.tau}', flush=True)
            rows, record = byz_job(data, 'clean', 6 if smoke else TUNING_ROUNDS,
                                   device, options, development_evaluate)
            candidate = {'config': asdict(options), 'record': record, 'rows': rows,
                         'validation_loss': rows[-1]['loss'] if record['status'] == 'completed' else None}
            study.write_json(filename, candidate)
        candidates.append(candidate)
    viable = [c for c in candidates if c['validation_loss'] is not None]
    if not viable:
        raise RuntimeError('All Byz-Clip21 development candidates diverged; full experiment stopped.')
    best = min(viable, key=lambda c: c['validation_loss'])
    selected = {'development_seed': seed, 'flaura_sigma_g': sigma, 'byz_config': best['config'],
                'byz_selection': 'lowest_final_reserved_root_cross_entropy_clean_development_only',
                'byz_validation_loss': best['validation_loss'],
                'scope': 'smoke_settings' if smoke else 'frozen_before_comparison',
                'evidence_sha256': {p.name: study.file_sha256(p) for p in directory.glob('*.json')}}
    study.write_json(chosen, selected)
    return selected


def run_one(data, method, condition, rounds, device, path, selected, original):
    done = path / 'result.json'
    if done.exists():
        record = json.loads(done.read_text())
        for name, digest in record['files'].items():
            if study.file_sha256(path / name) != digest:
                raise ValueError(f'Saved result changed: {path / name}')
        if record['initial_model_sha256'] != original['initial_model_sha256']:
            raise ValueError('Saved initial model mismatch.')
        print('    Already recorded; verified saved files.', flush=True)
        return record
    counts = {'unchanged': [], 'shifted': []}

    def evaluator(model, r, fraction):
        return metrics.evaluate_round(data, condition, counts, model, r, fraction)

    if method == METHODS[0]:
        aggregator = PublicationFLAURA(data['model'], auxiliary_reference(data), device,
                                      selected['flaura_sigma_g'])
        study.METHODS[method] = lambda: aggregator
        rows, record = study.run_job(data, method, condition, rounds, device, evaluate_round=evaluator)
        diagnostics = aggregator.diagnostics
    else:
        rows, record = byz_job(data, condition, rounds, device, ByzConfig(**selected['byz_config']), evaluator)
        diagnostics = record.pop('diagnostics')
    if record['initial_model_sha256'] != original['initial_model_sha256']:
        raise ValueError('New baseline initial model differs from existing results.')
    names = {'accuracy': 'drift_pct' if condition == 'honest_drift' else 'accuracy_pct',
             'macro_precision': 'macro_precision_pct', 'macro_recall': 'macro_recall_pct'}
    record['metrics'] = {name: mean(row[key] for row in rows[study.ONSET:])
                         if record['status'] == 'completed' else None for name, key in names.items()}
    path.mkdir(parents=True, exist_ok=True)
    study.write_csv(path / 'per_round.csv', rows)
    study.write_json(path / 'diagnostics.json', diagnostics)
    np.savez_compressed(path / 'confusions.npz', **{key: np.stack(value) for key, value in counts.items() if value})
    record['files'] = {name: study.file_sha256(path / name)
                       for name in ('per_round.csv', 'diagnostics.json', 'confusions.npz')}
    study.write_json(done, record)
    return record


def summarize(output, records, planned_seeds):
    study.write_csv(output / 'results_by_seed.csv', records)
    table = []
    for dataset in planned_seeds:
        for method in METHODS:
            for condition in study.CONDITIONS:
                group = [r for r in records if (r['dataset'], r['method'], r['condition'])
                         == (dataset, method, condition)]
                good = [r for r in group if r['status'] == 'completed']
                row = {'dataset': dataset, 'method': method, 'condition': condition,
                       'completed': len(good), 'recorded': len(group),
                       'planned': len(planned_seeds[dataset]),
                       'diverged': sum(r['status'] == 'diverged' for r in group)}
                row.update({m + '_pct': mean(r[m + '_pct'] for r in good) if good else None
                            for m in metrics.METRICS})
                table.append(row)
    study.write_csv(output / 'results_table.csv', table)


def run(args, output):
    original = json.loads((BASE / 'manifest.json').read_text())
    backend = original['device'] if args.device == 'auto' else args.device
    if backend != original['device']:
        raise ValueError('Use the original backend for this paired extension.')
    if backend == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('Mac GPU unavailable. Run from Terminal with normal GPU access.')
    if str(torch.__version__) != original['torch'] or np.__version__ != original['numpy']:
        raise ValueError('Use the original PyTorch and NumPy versions.')
    # Historical runner revisions only changed recording; all training/data/methods checked.
    for name, digest in original['code_sha256'].items():
        if name in ('run.py', 'study.py'):
            continue
        path = HERE / name if name == 'data.py' else study.CODE / name
        if study.file_sha256(path) != digest:
            raise ValueError(f'Original experiment source changed: {name}')
    device = torch.device(backend)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    if backend != 'mps':
        torch.use_deterministic_algorithms(True)
    rounds = 6 if args.smoke else original['rounds']
    seeds = {d: v[:1] if args.smoke else v for d, v in original['seeds'].items()}
    sources = [HERE / name for name in ('extended_methods.py', 'extended_study.py',
                                       'study.py', 'data.py', 'table2_metrics.py')]
    sources += [study.CODE / name for name in ('flaura.py', 'rfa.py', 'fltrust.py',
                                               'experiment_core.py', 'scenario_utils.py', 'non_iid_partitions.py')]
    manifest = {'source_sha256': {str(p.relative_to(HERE.parent)): study.file_sha256(p) for p in sources},
                'original_manifest_sha256': study.file_sha256(BASE / 'manifest.json'),
                'device': backend, 'torch': str(torch.__version__), 'numpy': np.__version__,
                'machine': platform.machine(), 'rounds': rounds, 'report_rounds': [study.ONSET, rounds],
                'seeds': seeds, 'methods': METHODS, 'conditions': study.CONDITIONS,
                'development_rounds': 6 if args.smoke else TUNING_ROUNDS,
                'development_grid': [asdict(c) for c in ((ByzConfig(),) if args.smoke else TUNING_GRID)],
                'scope': 'implementation_smoke_only' if args.smoke else 'paired_extension_with_declared_protocol_differences',
                'byz_budget': 'one_full_local_gradient_pass_and_one_communication_per_round; not_local_SGD',
                'byz_tuning_privilege': 'separate_clean_development_seed_and_reserved_labeled_root',
                'flaura_privilege': 'reserved_auxiliary_inputs_and_clean_development_geometry_calibration',
                'selection': 'no_official_test_or_attack_or_drift_outcomes_used_for_settings'}
    verify_manifest(output / 'manifest.json', manifest)
    for path in sources:
        target = output / 'source_snapshot' / path.relative_to(HERE.parent)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_bytes(path.read_bytes())
        elif study.file_sha256(target) != study.file_sha256(path):
            raise ValueError('Source snapshot changed.')
    total = sum(map(len, seeds.values())) * len(METHODS) * len(study.CONDITIONS)
    print(f'{total} comparison jobs, {rounds} rounds, device={backend}. Results: {output}', flush=True)
    records = []
    for dataset, dataset_seeds in seeds.items():
        loaded = study.load_har() if dataset == 'har' else study.load_images(dataset)
        selected = calibrate(dataset, loaded, device, output / 'development' / dataset, args.smoke)
        print(f'Frozen settings for {dataset}: {selected["byz_config"]}; '
              f'FLAURA sigma={selected["flaura_sigma_g"]:.6g}', flush=True)
        for seed in dataset_seeds:
            data = make_data(dataset, loaded, seed, device)
            plan = json.loads((BASE / 'plans' / dataset / f'{seed}.json').read_text())
            if data['plan'] != plan:
                raise ValueError('Data plan differs from original experiment.')
            study.write_json(output / 'plans' / dataset / f'{seed}.json', plan)
            for method in METHODS:
                for condition in study.CONDITIONS:
                    print(f'[{len(records) + 1}/{total}] {dataset} seed {seed} {method} {condition}', flush=True)
                    reference = BASE / 'runs' / dataset / str(seed) / 'FedReTrust-v06' / condition / 'result.json'
                    record = run_one(data, method, condition, rounds, device,
                                     output / 'runs' / dataset / str(seed) / method / condition,
                                     selected, json.loads(reference.read_text()))
                    row = {'dataset': dataset, 'seed': seed, 'method': method, 'condition': condition,
                           'status': record['status'], 'seconds': record['seconds'],
                           'failure_round': (record['failure'] or {}).get('round')}
                    row.update({name + '_pct': value for name, value in record['metrics'].items()})
                    records.append(row)
                    summarize(output, records, seeds)
                    study.write_json(output / 'progress.json', {'recorded': len(records), 'planned': total,
                                                              'last_job': row})
    study.write_json(output / 'completed.json', {'recorded': len(records), 'planned': total,
                                               'diverged': sum(r['status'] == 'diverged' for r in records)})
    print(f'Complete. Results: {output / "results_table.csv"}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('auto', 'mps', 'cuda', 'cpu'), default='auto')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    output = HERE / 'results' / ('extended_smoke' if args.smoke else 'extended_baselines')
    output.mkdir(parents=True, exist_ok=True)
    with (output / '.run.lock').open('a+b') as lock, (output / 'run.log').open('a', buffering=1) as log:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        stdout, stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = Tee(stdout, log), Tee(stderr, log)
        try:
            run(args, output)
        finally:
            sys.stdout, sys.stderr = stdout, stderr


if __name__ == '__main__':
    main()
