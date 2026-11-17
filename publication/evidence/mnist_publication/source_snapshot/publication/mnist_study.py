"""Run MNIST with the publication protocol and save all three evaluation metrics."""
import argparse
import fcntl
import json
import os
from pathlib import Path
from statistics import mean

import numpy as np
import torch
from torchvision.datasets import MNIST
from torchvision.datasets.utils import check_integrity

import study
import table2_metrics as metrics
from data import read_idx

SEEDS = tuple(range(1711201, 1711206))
METHODS = metrics.METHODS


def verify_json(path, value):
    value = json.loads(json.dumps(value))
    if path.exists() and json.loads(path.read_text()) != value:
        raise ValueError(f'Saved settings differ: {path}')
    study.write_json(path, value)


def summarize(output, records, repeats):
    study.write_csv(output / 'results_by_seed.csv', records)
    table = []
    for method in METHODS:
        for condition in study.CONDITIONS:
            jobs = [r for r in records if r['method'] == method and r['condition'] == condition]
            good = [r for r in jobs if r['status'] == 'completed']
            row = {'dataset': 'mnist', 'method': method, 'condition': condition,
                   'completed': len(good), 'recorded': len(jobs), 'planned': repeats,
                   'diverged': sum(r['status'] == 'diverged' for r in jobs)}
            row.update({m + '_pct': mean(r[m + '_pct'] for r in good) if good else None
                        for m in metrics.METRICS})
            table.append(row)
    study.write_csv(output / 'results_table.csv', table)


def run(args, output):
    original = json.loads((study.HERE / 'results/full/manifest.json').read_text())
    if not torch.backends.mps.is_available():
        raise RuntimeError('Apple GPU unavailable. Run with normal GPU access.')
    if str(torch.__version__) != original['torch'] or np.__version__ != original['numpy']:
        raise ValueError('Use the publication Python environment.')
    for name, digest in original['code_sha256'].items():
        if name in ('run.py', 'study.py'):
            continue
        source = study.HERE / name if name == 'data.py' else study.CODE / name
        if study.file_sha256(source) != digest:
            raise ValueError(f'Original experiment source changed: {source}')
    data_dir = study.HERE / 'data/mnist'
    for filename, md5 in MNIST.resources:
        if not check_integrity(data_dir / filename, md5):
            raise ValueError(f'Missing or invalid official MNIST archive: {filename}')
    loaded = tuple(read_idx(data_dir / f'{prefix}-images-idx3-ubyte.gz',
                            data_dir / f'{prefix}-labels-idx1-ubyte.gz', False)
                   for prefix in ('train', 't10k'))
    rounds = 6 if args.smoke else 48
    seeds = SEEDS[:1] if args.smoke else SEEDS
    total = len(seeds) * len(METHODS) * len(study.CONDITIONS)
    device = torch.device('mps')
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    sources = [Path(__file__), study.HERE / 'table2_metrics.py', study.HERE / 'study.py',
               study.HERE / 'data.py']
    sources += [study.CODE / name for name in original['code_sha256']
                if name not in ('data.py', 'run.py', 'study.py')]
    manifest = {'dataset': 'mnist', 'rounds': rounds, 'report_rounds': [study.ONSET, rounds],
                'seeds': list(seeds), 'methods': METHODS, 'conditions': study.CONDITIONS,
                'jobs': total, 'device': str(device), 'torch': str(torch.__version__),
                'numpy': np.__version__, 'local_epochs': 1, 'batch_size': 64, 'momentum': .9,
                'learning_rate': .01, 'clients': 10, 'samples_per_client': 512,
                'dirichlet_alpha': .5, 'affected_clients': [0, 1, 2],
                'root_size': 300, 'evaluation_size': 1000, 'drift_ramp_rounds': study.RAMP,
                'evaluation_selection_seed': 1111199,
                'metrics': metrics.METRICS, 'macro_zero_division': 0,
                'parent_manifest_sha256': study.file_sha256(study.HERE / 'results/full/manifest.json'),
                'data_sha256': {name: study.file_sha256(data_dir / name) for name, _ in MNIST.resources},
                'source_sha256': {str(p.relative_to(study.HERE.parent)): study.file_sha256(p)
                                  for p in sources}}
    verify_json(output / 'manifest.json', manifest)
    for source in sources:
        target = output / 'source_snapshot' / source.relative_to(study.HERE.parent)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_bytes(source.read_bytes())
        elif study.file_sha256(target) != study.file_sha256(source):
            raise ValueError(f'Source snapshot differs: {target}')
    print(f'Training on {device}. {total} MNIST jobs, {rounds} rounds. Results: {output}', flush=True)
    records = []
    for seed in seeds:
        data = study.image_plan('mnist', loaded, seed)
        verify_json(output / 'plans/mnist' / f'{seed}.json', data['plan'])
        study.to_device(data, device)
        torch.manual_seed(seed)
        expected = {'initial_model_sha256': study.state_sha256(study.clone_state(data['model']().to(device)))}
        for method in METHODS:
            for condition in study.CONDITIONS:
                path = output / 'runs/mnist' / str(seed) / method / condition
                print(f'[{len(records)+1}/{total}] mnist seed {seed} {method} {condition}', flush=True)
                record = metrics.run_one(data, method, condition, rounds, device, path, expected)
                if record['initial_model_sha256'] != expected['initial_model_sha256']:
                    raise ValueError(f'Initial model differs: {path}')
                row = {'dataset': 'mnist', 'seed': seed, 'method': method, 'condition': condition,
                       'status': record['status'], 'seconds': record['seconds'],
                       'failure_round': (record['failure'] or {}).get('round')}
                row.update({name + '_pct': value for name, value in record['metrics'].items()})
                records.append(row)
                summarize(output, records, len(seeds))
                study.write_json(output / 'progress.json', {'recorded': len(records), 'planned': total,
                                                           'last_job': row})
                print(f"    Saved {record['status']}: {record['metrics']}", flush=True)
    study.write_json(output / 'completed.json', {'recorded': len(records), 'planned': total,
                                               'diverged': sum(r['status'] == 'diverged' for r in records)})
    print('MNIST complete.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    output = study.HERE / 'results' / ('mnist_smoke' if args.smoke else 'mnist_publication')
    output.mkdir(parents=True, exist_ok=True)
    with (output / '.run.lock').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args, output)


if __name__ == '__main__':
    main()
