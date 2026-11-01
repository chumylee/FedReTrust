"""Recalculate five-block summaries and paired intervals without ML dependencies.

Input: frozen per-block measurements. Output: a new generated/reproduction-NAME
directory. This does not regenerate measurements from raw client/round logs.
"""
import argparse
from collections import defaultdict
import csv
import html
import json
import math
import os
import statistics

from common import ROOT, RESULTS, TABLES, new_output, write_json

T4 = 2.7764451051977987
V3, V5, V6 = 'FedReTrust-Direction-loose', 'FedReTrust-v05', 'FedReTrust-v06'
METHODS = ['FedAvg', 'RFA', 'FLTrust', 'Viceroy', V3, V5, V6,
           'FLAURA-reconstruction', 'FLAURA-sigma-double',
           'FedCognis-aggregation', 'FedCognis-squared-distance']
LABELS = dict(zip(METHODS, ['FedAvg', 'RFA', 'FLTrust', 'Viceroy',
    'FedReTrust v0.3', 'FedReTrust v0.5', 'FedReTrust v0.6',
    'FLAURA reconstruction', 'FLAURA double sigma', 'FedCognis aggregation',
    'FedCognis squared distance']))
CORE = [('clean', 'overall_mean_5_24_pp', 'Clean accuracy (%)', 'higher'),
        ('abrupt_shot_noise', 'affected_honest_event_pp', 'Honest noise accuracy (%)', 'higher'),
        ('gradual_stripe', 'affected_honest_event_pp', 'Honest stripe accuracy (%)', 'higher'),
        ('label_flip', 'overall_mean_5_24_pp', 'Attacked accuracy (%)', 'higher'),
        ('label_flip', 'attack_active_signed_damage_pp', 'Label-flip damage (pp)', 'lower')]
STAT_FIELDS = ['n', *[f'block_{i}' for i in range(1, 6)], 'mean', 'sd', 'min', 'max', 'ci_low', 'ci_high']


def read(name):
    with (TABLES / f'{name}.csv').open(newline='') as handle:
        return list(csv.DictReader(handle))


def stats(values):
    if set(values) != {1, 2, 3, 4, 5} or not all(math.isfinite(v) for v in values.values()):
        raise ValueError('Exactly five finite block values are required.')
    xs = [values[i] for i in range(1, 6)]
    mean, sd = statistics.mean(xs), statistics.stdev(xs)
    half = T4 * sd / math.sqrt(5)
    return {'n': 5, **{f'block_{i}': v for i, v in values.items()}, 'mean': mean, 'sd': sd,
            'min': min(xs), 'max': max(xs), 'ci_low': mean - half, 'ci_high': mean + half}


def calculate(rows):
    grouped = defaultdict(dict)
    for row in rows:
        key = tuple(row[k] for k in ('method', 'regime', 'scenario', 'metric'))
        block, seed, value = int(row['block']), int(row['seed']), float(row['value'])
        if block in grouped[key]:
            raise ValueError('Duplicate block measurement: ' + str(key))
        grouped[key][block] = (seed, value)
    means = []
    for key, block_values in sorted(grouped.items()):
        means.append(dict(zip(('method', 'regime', 'scenario', 'metric'), key)) |
                     stats({b: v[1] for b, v in block_values.items()}))
    effects = []
    for candidate, comparators in [(V6, [m for m in METHODS if m != V6]), (V5, ['FedAvg', 'RFA', V3])]:
        for comparator in comparators:
            candidate_cells = {k[1:] for k in grouped if k[0] == candidate}
            other_cells = {k[1:] for k in grouped if k[0] == comparator}
            for cell in sorted(candidate_cells & other_cells):
                a, b = grouped[(candidate, *cell)], grouped[(comparator, *cell)]
                if set(a) != set(b) or any(a[i][0] != b[i][0] for i in a):
                    raise ValueError('Paired blocks/seeds differ: ' + str(cell))
                effects.append({'candidate': candidate, 'comparator': comparator,
                    **dict(zip(('regime', 'scenario', 'metric'), cell)),
                    **stats({i: a[i][1] - b[i][1] for i in a})})
    lookup = {tuple(r[k] for k in ('method', 'regime', 'scenario', 'metric')): r for r in means}
    core = []
    for regime in ('IID', 'non-IID'):
        for method in METHODS:
            for scenario, metric, measurement, direction in CORE:
                row = {'regime': regime, 'method': method, 'display_name': LABELS[method],
                       'measurement': measurement, 'direction': direction, 'scenario': scenario, 'metric': metric}
                saved = lookup.get((method, regime, scenario, metric))
                row['status'] = 'measured' if saved else 'not_tested'
                row.update({k: saved[k] if saved else 'NA' for k in STAT_FIELDS})
                core.append(row)
    return means, effects, core


def compare(actual, expected, keys):
    def index(rows):
        result = {tuple(str(r[k]) for k in keys): r for r in rows}
        if len(result) != len(rows):
            raise ValueError('Duplicate reference or recalculated rows.')
        return result
    aa, bb = index(actual), index(expected)
    if aa.keys() != bb.keys():
        raise ValueError('Recalculated table coverage differs from frozen reference.')
    maximum = 0.0
    for key, row in aa.items():
        ref = bb[key]
        if 'status' in ref and row['status'] != ref['status']:
            raise ValueError('Missingness differs: ' + str(key))
        for field in STAT_FIELDS:
            a, b = row[field], ref[field]
            if str(a) in {'NA', ''} and b in {'NA', ''}:
                continue
            error = abs(float(a) - float(b))
            if not math.isfinite(error) or error > 1e-9:
                raise ValueError(f'Numeric mismatch {key}, {field}: {a} versus {b}')
            maximum = max(maximum, error)
    return {'rows': len(actual), 'max_absolute_error': maximum, 'absolute_tolerance': 1e-9}


def write_csv(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def report(path, core):
    chunks = ['<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">',
        '<title>FedReTrust v0.6 — saved results</title><style>body{font:16px/1.55 system-ui;margin:32px auto;padding:0 20px;max-width:1250px;color:#233047}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:10px;border-bottom:1px solid #ddd;text-align:right}th:first-child,td:first-child{text-align:left}th{background:#edf0f5}.scroll{overflow:auto}img{width:100%}strong{color:#623c94}</style>',
        '<h1>FedReTrust v0.6: saved development results</h1>',
        '<p>Recalculated from frozen block measurements; no model training or test-set evaluation. '
        'Five paired development blocks, rounds 5–24. Cells show mean [nominal 95% interval]. '
        'Honest noise/stripe scores concern affected honest clients. Accuracy is percent; damage is percentage points.</p>',
        '<p><strong>v0.6 improves non-IID attack protection over v0.5, with an IID stripe-adaptation cost.</strong> '
        'RFA retains better mean attack protection. The 45/51 follow-up checks indicate partial success; '
        'the original 25/27 pilot remains failed. All blocks were known during development.</p>']
    for regime in ('IID', 'non-IID'):
        chunks.append(f'<h2>{regime}</h2><div class="scroll"><table><tr><th>Method</th>')
        chunks.extend(f'<th>{html.escape(spec[2])}</th>' for spec in CORE)
        chunks.append('</tr>')
        for method in METHODS:
            chunks.append(f'<tr><td>{html.escape(LABELS[method])}</td>')
            for _, _, measurement, _ in CORE:
                row = next(r for r in core if (r['regime'], r['method'], r['measurement']) == (regime, method, measurement))
                value = 'Not tested' if row['status'] == 'not_tested' else f"{row['mean']:.2f} [{row['ci_low']:.2f}, {row['ci_high']:.2f}]"
                chunks.append(f'<td>{value}</td>')
            chunks.append('</tr>')
        chunks.append('</table></div>')
    chunks.append('<h2>Paired comparisons</h2><p>Intervals for method differences use matched block differences. '
                  'Do not infer a difference from overlap of the separate intervals above. '
                  'IID and non-IID are never pooled; five blocks give limited, exploratory uncertainty estimates.</p>')
    for name in ('paired_v06', 'core_trajectories'):
        relative = os.path.relpath(RESULTS / 'figures' / f'{name}.png', path.parent)
        chunks.append(f'<img alt="{name.replace("_", " ")}" src="{html.escape(relative)}">')
    chunks.append('<p>These figures are copied frozen artifacts, not regenerated images. '
                  'v0.6 recovery and component ablations were not tested. No universal superiority or established novelty claim.</p></html>')
    path.write_text('\n'.join(chunks))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', default='results', help='Unique output label; existing outputs are never overwritten.')
    args = parser.parse_args()
    means, effects, core = calculate(read('block_metrics'))
    checks = {
        'means_all_metrics': compare(means, read('means_all_metrics'), ('method', 'regime', 'scenario', 'metric')),
        'paired_effects': compare(effects, read('paired_effects'), ('candidate', 'comparator', 'regime', 'scenario', 'metric')),
        'core_results': compare(core, read('core_results'), ('regime', 'method', 'measurement')),
    }
    runs, gates = read('run_index'), read('v06_followup_gates')
    identities = {(r['method'], r['regime'], r['block'], r['scenario']) for r in runs}
    passed = sum(r['point_pass'] == 'True' for r in gates)
    if len(runs) != 880 or len(identities) != 880 or len(gates) != 51 or passed != 45:
        raise ValueError('Frozen run count or gate history differs.')
    out = new_output('reproduction', args.name)
    for name, rows in [('means_all_metrics', means), ('paired_effects', effects), ('core_results', core)]:
        write_csv(out / f'{name}.csv', rows)
    report(out / 'results.html', core)
    summary = {'status': 'matched_frozen_tables', 'distinct_runs': 880, 'checks': checks,
        'followup_checks_passed': passed, 'followup_checks_total': 51,
        'scope': 'Recalculation from saved block metrics, not raw-log extraction, checkpoint replay or new training.',
        'output': str(out.relative_to(ROOT))}
    write_json(out / 'summary.json', summary)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
