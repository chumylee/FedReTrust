"""Replay the 60 publication clean jobs, saving all three classification metrics."""
import csv
import fcntl
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from time import monotonic

import numpy as np
import torch

import study
import table2_metrics as metrics

BASE = study.HERE / 'results/full'
OUTPUT = study.HERE / 'results/clean_metrics'
METHODS = metrics.METHODS


def verify_json(path, value):
    value = json.loads(json.dumps(value))
    if path.exists() and json.loads(path.read_text()) != value:
        raise ValueError(f'Saved settings differ: {path}')
    study.write_json(path, value)


def summarize(records, seeds):
    study.write_csv(OUTPUT / 'results_by_seed.csv', records)
    table = []
    for dataset, repeats in seeds.items():
        for method in METHODS:
            jobs = [r for r in records if (r['dataset'], r['method']) == (dataset, method)]
            good = [r for r in jobs if r['status'] == 'completed']
            row = {'dataset': dataset, 'method': method, 'condition': 'clean',
                   'completed': len(good), 'recorded': len(jobs), 'planned': len(repeats),
                   'diverged': sum(r['status'] == 'diverged' for r in jobs)}
            row.update({m + '_pct': mean(r[m + '_pct'] for r in good) if good else None
                        for m in metrics.METRICS})
            table.append(row)
    study.write_csv(OUTPUT / 'results_table.csv', table)


def compare_original(path, original_path, record, old):
    if record['initial_model_sha256'] != old['initial_model_sha256']:
        raise ValueError(f'Initial model differs: {path}')
    if record['settings'] != old['settings']:
        raise ValueError(f'Aggregation settings differ: {path}')
    if study.file_sha256(original_path / 'per_round.csv') != old['measurements_sha256']:
        raise ValueError(f'Original measurements changed: {original_path}')
    with (original_path / 'per_round.csv').open() as handle:
        original_rows = {int(r['round']): float(r['accuracy_pct']) for r in csv.DictReader(handle)}
    with (path / 'per_round.csv').open() as handle:
        new_rows = {int(r['round']): float(r['accuracy_pct']) for r in csv.DictReader(handle)}
    shared = original_rows.keys() & new_rows.keys()
    return {'original_status': old.get('status', 'completed'),
            'original_mean_pct': old['mean_pct'],
            'accuracy_mean_change_pp': record['mean_pct'] - old['mean_pct']
            if record['mean_pct'] is not None and old['mean_pct'] is not None else None,
            'accuracy_curve_max_change_pp': max(abs(new_rows[r] - original_rows[r]) for r in shared),
            'same_recorded_rounds': original_rows.keys() == new_rows.keys()}


def replay():
    start = monotonic()
    original_path = BASE / 'manifest.json'
    original = json.loads(original_path.read_text())
    if original['device'] != 'mps' or not torch.backends.mps.is_available():
        raise RuntimeError('Apple GPU unavailable. Run with normal GPU access.')
    if str(torch.__version__) != original['torch'] or np.__version__ != original['numpy']:
        raise ValueError('Use the original publication Python environment.')
    if original['rounds'] != 48 or original['report_rounds'] != [study.ONSET, 48]:
        raise ValueError('Unexpected original training or reporting rounds.')
    if set(original['seeds']) != {'fashion_mnist', 'emnist_balanced', 'har'}:
        raise ValueError('Unexpected original datasets.')
    if any(len(seeds) != 5 for seeds in original['seeds'].values()):
        raise ValueError('Expected five seeds per dataset.')
    for name, digest in original['code_sha256'].items():
        if name in ('run.py', 'study.py'):
            continue
        source = study.HERE / name if name == 'data.py' else study.CODE / name
        if study.file_sha256(source) != digest:
            raise ValueError(f'Original experiment source changed: {source}')
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    device = torch.device('mps')
    rounds = original['rounds']
    seeds = original['seeds']
    total = sum(map(len, seeds.values())) * len(METHODS)
    sources = [Path(__file__), Path(metrics.__file__), Path(study.__file__), study.HERE / 'data.py']
    sources += [study.CODE / name for name in original['code_sha256']
                if name not in ('data.py', 'run.py', 'study.py')]
    manifest = {'jobs': total, 'rounds': rounds, 'report_rounds': original['report_rounds'],
                'device': str(device), 'torch': str(torch.__version__), 'numpy': np.__version__,
                'seeds': seeds, 'methods': METHODS, 'conditions': ['clean'],
                'local_epochs': original['local_epochs'], 'batch_size': original['batch_size'],
                'momentum': original['momentum'], 'metrics': metrics.METRICS,
                'macro_zero_division': 0,
                'source_manifest_sha256': study.file_sha256(original_path),
                'source_sha256': {str(p.relative_to(study.HERE.parent)): study.file_sha256(p)
                                  for p in sources}}
    verify_json(OUTPUT / 'manifest.json', manifest)
    for source in sources:
        target = OUTPUT / 'source_snapshot' / source.relative_to(study.HERE.parent)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_bytes(source.read_bytes())
        elif study.file_sha256(target) != study.file_sha256(source):
            raise ValueError(f'Source snapshot differs: {target}')
    records = []

    def progress(status, current=None):
        study.write_json(OUTPUT / 'progress.json', {
            'status': status, 'recorded': len(records), 'planned': total,
            'completed': sum(r['status'] == 'completed' for r in records),
            'diverged': sum(r['status'] == 'diverged' for r in records),
            'session_elapsed_seconds': monotonic() - start,
            'updated_at_utc': datetime.now(timezone.utc).isoformat(),
            'current_job': current, 'last_job': records[-1] if records else None})

    print(f'Training on {device}. {total} CLEAN jobs, {rounds} rounds each. Results: {OUTPUT}', flush=True)
    progress('running')
    try:
        for dataset, dataset_seeds in seeds.items():
            loaded = study.load_har() if dataset == 'har' else study.load_images(dataset)
            for seed in dataset_seeds:
                data = study.har_plan(loaded, seed) if dataset == 'har' else study.image_plan(dataset, loaded, seed)
                saved_plan = json.loads((BASE / 'plans' / dataset / f'{seed}.json').read_text())
                if data['plan'] != saved_plan:
                    raise ValueError(f'Original data plan differs: {dataset}, {seed}')
                verify_json(OUTPUT / 'plans' / dataset / f'{seed}.json', saved_plan)
                study.to_device(data, device)
                for method in METHODS:
                    relative = Path('runs') / dataset / str(seed) / method / 'clean'
                    old = json.loads((BASE / relative / 'result.json').read_text())
                    job = {'dataset': dataset, 'seed': seed, 'method': method, 'condition': 'clean'}
                    progress('running', job)
                    print(f'[{len(records)+1}/{total}] {dataset} seed {seed} {method} clean', flush=True)
                    record = metrics.run_one(data, method, 'clean', rounds, device, OUTPUT / relative, old)
                    row = {**job, 'status': record['status'], 'seconds': record['seconds'],
                           'failure_round': (record['failure'] or {}).get('round')}
                    row.update({name + '_pct': value for name, value in record['metrics'].items()})
                    row.update(compare_original(OUTPUT / relative, BASE / relative, record, old))
                    records.append(row)
                    summarize(records, seeds)
                    progress('running')
                    if record['status'] == 'diverged':
                        print(f"    Diverged at round {row['failure_round']}; excluded from means. Continuing.", flush=True)
                    else:
                        print('    Saved ' + ', '.join(f'{name}={value:.2f}%' for name, value
                                                       in record['metrics'].items()), flush=True)
                    print(f"    Progress: {len(records)}/{total}; original accuracy curve max difference: "
                          f"{row['accuracy_curve_max_change_pp']:.8f} pp", flush=True)
        study.write_json(OUTPUT / 'completed.json', {
            'recorded': len(records), 'planned': total,
            'completed': sum(r['status'] == 'completed' for r in records),
            'diverged': sum(r['status'] == 'diverged' for r in records)})
        progress('complete')
        print(f"Complete. Results: {OUTPUT / 'results_table.csv'}", flush=True)
    except BaseException:
        progress('stopped')
        raise


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / '.run.lock').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        replay()


if __name__ == '__main__':
    main()
