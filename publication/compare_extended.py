"""Build side-by-side tables from saved baseline and extension evidence."""
import csv
import json
from pathlib import Path
from statistics import mean, stdev
import math

import study
import table2_metrics as metrics

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / 'results' / 'extended_baselines'
OLD = HERE / 'results' / 'table2_metrics'
METHODS = (*metrics.METHODS, 'FLAURA-reconstruction', 'Byz-Clip21-SGD2M-noDP')


def read_csv(path):
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))


def main():
    base_manifest = HERE / 'results' / 'full' / 'manifest.json'
    base_hash = study.file_sha256(base_manifest)
    old_manifest = json.loads((OLD / 'manifest.json').read_text())
    new_manifest = json.loads((OUTPUT / 'manifest.json').read_text())
    if old_manifest['source_manifest_sha256'] != base_hash or new_manifest['original_manifest_sha256'] != base_hash:
        raise ValueError('Original and new experiments have different source manifests.')
    if old_manifest['report_rounds'] != new_manifest['report_rounds']:
        raise ValueError('Reporting windows differ.')
    records = read_csv(OLD / 'results_by_seed.csv') + read_csv(OUTPUT / 'results_by_seed.csv')
    index = {(r['dataset'], int(r['seed']), r['method'], r['condition']): r for r in records}
    if len(index) != len(records):
        raise ValueError('Duplicate seed records.')
    table, differences = [], []
    for dataset, seeds in new_manifest['seeds'].items():
        for method in METHODS:
            row = {'dataset': dataset, 'method': method}
            for condition, label in (('label_flip', 'poisoning'), ('honest_drift', 'honest_drift')):
                jobs = [index[dataset, seed, method, condition] for seed in seeds
                        if (dataset, seed, method, condition) in index]
                good = [j for j in jobs if j['status'] == 'completed']
                row[label + '_completed'] = len(good)
                row[label + '_recorded'] = len(jobs)
                row[label + '_planned'] = len(seeds)
                for metric in metrics.METRICS:
                    key = metric + '_pct'
                    row[label + '_' + key] = mean(float(j[key]) for j in good) if good else None
                    if method == 'FedReTrust-v06':
                        continue
                    paired = []
                    for job in good:
                        proposed = index.get((dataset, int(job['seed']), 'FedReTrust-v06', condition))
                        if proposed and proposed['status'] == 'completed':
                            paired.append(float(proposed[key]) - float(job[key]))
                    n = len(paired)
                    estimate = mean(paired) if paired else None
                    critical = {2: 12.7062047364, 3: 4.3026527297, 4: 3.1824463053, 5: 2.7764451052}
                    half = critical[n] * stdev(paired) / math.sqrt(n) if n in critical else None
                    differences.append({'dataset': dataset, 'comparator': method, 'condition': condition,
                                        'metric': metric, 'paired_completed': n, 'planned': len(seeds),
                                        'FedReTrust_minus_comparator_pp': estimate,
                                        'nominal_95_ci_low_pp': estimate - half if half is not None else None,
                                        'nominal_95_ci_high_pp': estimate + half if half is not None else None})
            table.append(row)
    study.write_csv(OUTPUT / 'comparison_side_by_side.csv', table)
    study.write_csv(OUTPUT / 'paired_differences.csv', differences)
    for row in table:
        parts = []
        for label in ('poisoning', 'honest_drift'):
            values = [row[label + '_' + metric + '_pct'] for metric in metrics.METRICS]
            numbers = ' / '.join('--' if v is None else f'{v:.2f}' for v in values)
            parts.append(f'{label}: {numbers} (n={row[label + "_completed"]}/{row[label + "_planned"]})')
        print(f'{row["dataset"]:16} {row["method"]:25} ' + ' | '.join(parts))
    print('Columns: accuracy / macro precision / macro recall. Means use completed jobs only; inspect n.')
    print('Byz-Clip21 uses a gradient protocol; see EXTENDED_BASELINES.md before interpreting comparisons.')


if __name__ == '__main__':
    main()
