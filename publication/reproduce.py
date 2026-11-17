"""Validate saved publication evidence and rebuild the paper's tables (no training)."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, stdev

HERE = Path(__file__).resolve().parent
METHODS = ('RFA', 'FLTrust', 'Viceroy', 'FedReTrust-v06')
DATASETS = ('mnist', 'fashion_mnist', 'emnist_balanced', 'har')
CONDITIONS = ('clean', 'label_flip', 'honest_drift')
METRICS = ('accuracy', 'macro_precision', 'macro_recall')
LABELS = {'mnist': 'MNIST', 'fashion_mnist': 'Fashion-MNIST',
          'emnist_balanced': 'EMNIST Balanced', 'har': 'Smartphone HAR'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def read_csv(path):
    with path.open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def check_confusions(path, rows, condition, plan):
    import numpy as np
    with np.load(path / 'confusions.npz', allow_pickle=False) as archive:
        clean = archive['unchanged']
        expected = {'unchanged', 'shifted'} if condition == 'honest_drift' else {'unchanged'}
        require(set(archive.files) == expected, f'Unexpected confusion arrays: {path}')
        require(len(clean) == len(rows), f'Confusion round count differs: {path}')
        require(np.issubdtype(clean.dtype, np.integer) and (clean >= 0).all(), f'Invalid counts: {path}')
        require((clean.sum(axis=(1, 2)) == plan['test_size']).all(), f'Test sizes differ: {path}')
        if condition == 'honest_drift':
            shifted = archive['shifted']
            require(shifted.shape == clean.shape and (shifted >= 0).all(), f'Invalid drift counts: {path}')
            require(np.array_equal(clean.sum(axis=2), shifted.sum(axis=2)), f'Drift labels differ: {path}')
        for i, row in enumerate(rows):
            matrix = clean[i].astype(float)
            clean_accuracy = 100 * matrix.trace() / matrix.sum()
            require(abs(clean_accuracy - float(row['accuracy_pct'])) < 1e-8, f'Clean accuracy differs: {path}')
            if condition == 'honest_drift':
                fraction = min(1., max(0., (int(row['round']) - 4) / 8))
                matrix = ((1 - fraction) * matrix + fraction * shifted[i]) / matrix.sum(axis=1)[:, None]
                matrix *= np.asarray(plan['class_profile_weights'])[:, None]
            correct = matrix.diagonal()
            predicted, actual = matrix.sum(axis=0), matrix.sum(axis=1)
            values = (100 * correct.sum() / matrix.sum(),
                      100 * np.divide(correct, predicted, out=np.zeros_like(correct), where=predicted > 0).mean(),
                      100 * np.divide(correct, actual, out=np.zeros_like(correct), where=actual > 0).mean())
            keys = ('drift_pct' if condition == 'honest_drift' else 'accuracy_pct',
                    'macro_precision_pct', 'macro_recall_pct')
            require(all(abs(v - float(row[k])) < 1e-8 for v, k in zip(values, keys)), f'Confusion metrics differ: {path}')


def rebuild(evidence, output, deep=False):
    artifact_path = HERE / 'artifact_manifest.json'
    if evidence == HERE / 'evidence' and artifact_path.exists():
        for name, digest in read_json(artifact_path)['files_sha256'].items():
            source = (HERE.parent / name).resolve()
            require(source.is_relative_to(HERE.parent), f'Artifact path leaves repository: {name}')
            require(sha(source) == digest, f'Publication artifact changed: {name}')
    original = read_json(evidence / 'full/manifest.json')
    mnist = read_json(evidence / 'mnist_publication/manifest.json')
    seeds = {'mnist': mnist['seeds'], **original['seeds']}
    original_sha = sha(evidence / 'full/manifest.json')
    for folder, key in (('table2_metrics', 'source_manifest_sha256'),
                        ('clean_metrics', 'source_manifest_sha256'),
                        ('mnist_publication', 'parent_manifest_sha256')):
        manifest = read_json(evidence / folder / 'manifest.json')
        require(manifest[key] == original_sha, f'Original manifest differs: {folder}')
        require(manifest['rounds'] == 48 and manifest['report_rounds'] == [5, 48], f'Protocol differs: {folder}')
        for name, digest in manifest.get('source_sha256', {}).items():
            require(sha(evidence / folder / 'source_snapshot' / name) == digest, f'Source snapshot changed: {folder}/{name}')
    records, table, differences = [], [], []
    initial_models = {}
    for dataset in DATASETS:
        require(len(seeds[dataset]) == 5, f'Expected five seeds: {dataset}')
        for method in METHODS:
            for condition in CONDITIONS:
                folder = ('mnist_publication' if dataset == 'mnist' else
                          'clean_metrics' if condition == 'clean' else 'table2_metrics')
                for seed in seeds[dataset]:
                    relative = Path(folder) / 'runs' / dataset / str(seed) / method / condition
                    path = evidence / relative
                    record = read_json(path / 'result.json')
                    require(set(record['files']) == {'per_round.csv', 'confusions.npz'}, f'Missing measurements: {path}')
                    for name, digest in record['files'].items():
                        require(sha(path / name) == digest, f'Output checksum differs: {path / name}')
                    key = dataset, seed
                    initial = record['initial_model_sha256']
                    require(initial_models.setdefault(key, initial) == initial, f'Initial models differ: {key}')
                    rows = read_csv(path / 'per_round.csv')
                    status = record['status']
                    require(status in ('completed', 'diverged'), f'Invalid status: {path}')
                    require([int(r['round']) for r in rows] == list(range(len(rows))), f'Round sequence differs: {path}')
                    if status == 'completed':
                        require(len(rows) == 49 and record['failure'] is None, f'Incomplete run: {path}')
                    else:
                        require(record['failure']['round'] == len(rows), f'Failure round differs: {path}')
                    keys = ('drift_pct' if condition == 'honest_drift' else 'accuracy_pct',
                            'macro_precision_pct', 'macro_recall_pct')
                    metrics = {}
                    for metric, column in zip(METRICS, keys):
                        values = [float(r[column]) for r in rows]
                        require(all(math.isfinite(v) and 0 <= v <= 100 for v in values), f'Invalid score: {path}')
                        value = mean(values[5:]) if status == 'completed' else None
                        saved = record['metrics'][metric]
                        require(saved is None if value is None else saved is not None and abs(value - saved) < 1e-8,
                                f'Round mean differs: {path}/{metric}')
                        metrics[metric + '_pct'] = value
                    plan_base = 'mnist_publication' if dataset == 'mnist' else 'full'
                    plan = read_json(evidence / plan_base / 'plans' / dataset / f'{seed}.json')
                    if dataset != 'mnist':
                        old_path = evidence / 'full/runs' / dataset / str(seed) / method / condition
                        old = read_json(old_path / 'result.json')
                        require(old['initial_model_sha256'] == initial and old['settings'] == record['settings'],
                                f'Original settings differ: {path}')
                        require(old.get('status', 'completed') == status, f'Original status differs: {path}')
                        require(sha(old_path / 'per_round.csv') == old['measurements_sha256'], f'Original hash differs: {path}')
                        old_rows = read_csv(old_path / 'per_round.csv')
                        require(len(old_rows) == len(rows), f'Original rounds differ: {path}')
                        require(all(abs(float(a[keys[0]]) - float(b[keys[0]])) < 1e-8
                                    for a, b in zip(rows, old_rows)), f'Original accuracy curve differs: {path}')
                    if deep:
                        check_confusions(path, rows, condition, plan)
                    records.append({'dataset': dataset, 'seed': seed, 'method': method,
                                    'condition': condition, 'status': status,
                                    'failure_round': (record['failure'] or {}).get('round'),
                                    **metrics, 'source': str(relative)})
                jobs = [r for r in records if (r['dataset'], r['method'], r['condition']) == (dataset, method, condition)]
                good = [r for r in jobs if r['status'] == 'completed']
                table.append({'dataset': dataset, 'method': method, 'condition': condition,
                              'completed': len(good), 'planned': 5, 'diverged': 5 - len(good),
                              **{m + '_pct': mean(r[m + '_pct'] for r in good) if good else None for m in METRICS}})
    index = {(r['dataset'], r['seed'], r['method'], r['condition']): r for r in records}
    for dataset in DATASETS:
        for method in METHODS[:-1]:
            for condition in CONDITIONS:
                for metric in METRICS:
                    paired = []
                    for seed in seeds[dataset]:
                        a, b = (index[dataset, seed, m, condition] for m in ('FedReTrust-v06', method))
                        if a['status'] == b['status'] == 'completed':
                            paired.append(a[metric + '_pct'] - b[metric + '_pct'])
                    centre = mean(paired) if paired else None
                    half = 2.7764451052 * stdev(paired) / math.sqrt(5) if len(paired) == 5 else None
                    differences.append({'dataset': dataset, 'comparator': method, 'condition': condition,
                                        'metric': metric, 'paired_completed': len(paired), 'planned': 5,
                                        'FedReTrust_minus_comparator_pp': centre,
                                        'nominal_95_ci_low_pp': centre - half if half is not None else None,
                                        'nominal_95_ci_high_pp': centre + half if half is not None else None})
    require(len(records) == 240, 'Expected 240 recorded jobs.')
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / 'results_table.csv', table)
    write_csv(output / 'results_by_seed.csv', records)
    write_csv(output / 'paired_differences.csv', differences)
    indexed = {(r['dataset'], r['method'], r['condition']): r for r in table}
    headings = ['Method', 'Metric'] + [f'{LABELS[d]} {c}' for d in DATASETS for c in ('C', 'P', 'H')]
    markdown = ['# Publication results', '', 'Mean percentages over rounds 5–48, then over completed repeats.', '',
                '| ' + ' | '.join(headings) + ' |', '| ' + ' | '.join(['---'] * len(headings)) + ' |']
    for method in METHODS:
        for metric, label in zip(METRICS, ('Accuracy', 'Precision', 'Recall')):
            cells = ['FedReTrust' if method == 'FedReTrust-v06' else 'Viceroy (adapted)' if method == 'Viceroy' else method, label]
            for dataset in DATASETS:
                for condition in CONDITIONS:
                    row = indexed[dataset, method, condition]
                    value = row[metric + '_pct']
                    best = max(indexed[dataset, m, condition][metric + '_pct'] for m in METHODS)
                    cell = f'{value:.2f}'
                    if value == best:
                        cell = '**' + cell + '**'
                    if row['completed'] < row['planned']:
                        cell += '†'
                    cells.append(cell)
            markdown.append('| ' + ' | '.join(cells) + ' |')
    markdown += ['', 'C: clean training; P: label poisoning; H: honest drift. Precision and recall use macro averaging.',
                 '† Four completed repeats; all other entries use five. Diverged jobs remain in the raw evidence.',
                 'Under clean training and poisoning, macro recall equals accuracy on the three image datasets because their test sets are class-balanced.', '']
    (output / 'results_table.md').write_text('\n'.join(markdown), encoding='utf-8')
    failures = [r for r in records if r['status'] == 'diverged']
    report = {'status': 'passed', 'recorded': len(records), 'completed': len(records) - len(failures),
              'diverged': len(failures), 'confusions_recalculated': deep,
              'hashes_and_round_means_checked': True, 'original_accuracy_curves_match': True,
              'source_manifests': {name: sha(evidence / name / 'manifest.json') for name in
                                   ('full', 'mnist_publication', 'table2_metrics', 'clean_metrics')},
              'failures': failures}
    (output / 'validation.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(f"Validated {len(records)} jobs: {len(records) - len(failures)} completed, {len(failures)} diverged.")
    print(f'Table: {output / "results_table.md"}')
    print(f'CSV: {output / "results_table.csv"}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, default=HERE / 'evidence')
    parser.add_argument('--output', type=Path, default=HERE.parent / 'generated/publication')
    parser.add_argument('--check-confusions', action='store_true', help='Independently recalculate metrics from confusion counts; requires NumPy.')
    args = parser.parse_args()
    evidence, output = args.evidence.resolve(), args.output.resolve()
    require(output != evidence and evidence not in output.parents, 'Write derived results outside the saved evidence.')
    rebuild(evidence, output, args.check_confusions)


if __name__ == '__main__':
    main()
