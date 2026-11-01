"""Six-run second-seed follow-up of the previously evaluated doubled scale.

The original FLAURA implementation, calibration and study remain frozen.
This is a development replication after first-seed selection, not a holdout test.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
import argparse
import fcntl
import json
import platform
import resource
import shutil
import subprocess
import sys
import time
import numpy as np
import pandas as pd
from experiment_core import TrainConfig, load_data, run_experiment, file_sha256, write_json
from research_runs import ROOT, PARTITIONS, canonical_hash
from research_analysis import extract_run, cross_method_pair, contribution_records, cost_record
from scenario_metrics import ClasswiseEvaluator, ContributionRecorder, longer_scenarios
from directional_study import checked_result as checked_artifacts
from flaura import FLAURAConfig, FLAURAAggregator, build_auxiliary_reference
from flaura_study import (STUDY as PARENT, MAIN, CONTROL, BASELINES, CORE, SEEDS,
                          hashes as parent_hashes, verify as verify_parent, job, key, stamp)

STUDY = ROOT / 'outputs/flaura/20261030_sigma_double_seed2'
TARGET = 'FLAURA-sigma-double'
METHODS = (*BASELINES, MAIN, CONTROL, TARGET)


def jobs():
    return [job(TARGET, g, s, SEEDS[1]) for g in PARTITIONS for s in CORE]


def hashes():
    return {**parent_hashes(), 'flaura_second_seed.py': file_sha256(__file__)}


def config():
    scale = json.loads((PARENT / 'scale_freeze.json').read_text())['sigma_g']
    return FLAURAConfig(sigma_g=2 * scale, gamma=.8)


def prepare(study):
    study.mkdir(parents=True, exist_ok=True)
    if (study / 'protocol.json').exists():
        verify(study)
        return
    verify_parent(PARENT)
    assert json.loads((PARENT / 'status.json').read_text())['status'] == 'complete'
    parent = json.loads((PARENT / 'result_manifest.json').read_text())
    reused = [r for r in parent['records'] if r['job']['method'] in METHODS and r['job']['scenario'] in CORE]
    expected = {(m, g, s, seed) for m in METHODS for g in PARTITIONS for s in CORE for seed in SEEDS}
    expected -= {key(j) for j in jobs()}
    assert len(reused) == 126 and {key(r['job']) for r in reused} == expected
    for r in reused:
        checked_artifacts(r)
        if r['job']['method'] == TARGET:
            info = json.loads((ROOT / r['run_path'] / 'run.json').read_text())
            assert info['method']['settings']['config'] == asdict(config())
    write_json(study / 'reused_manifest.json', {'records': reused})
    # Capture all earlier study artifacts, plus its executed notebook; do not rewrite them.
    prior_paths = [p for p in PARENT.rglob('*') if p.is_file() and p.suffix != '.lock']
    prior_paths += [ROOT / 'code/26_flaura_comparison.ipynb']
    protocol = {
        'created_at': stamp(), 'status': 'declared_before_second_seed_training',
        'authorization': 'User: alright, do the second seed. Six runs discussed immediately before authorization.',
        'source_sha256': hashes(), 'parent_study': str(PARENT.relative_to(ROOT)),
        'parent_artifact_sha256': {str(p.relative_to(ROOT)): file_sha256(p) for p in prior_paths},
        'reused_manifest_sha256': file_sha256(study / 'reused_manifest.json'),
        'jobs': jobs(), 'config': asdict(config()), 'new_runs': 6, 'reused_runs': 126,
        'selection_context': 'Doubled scale chosen for follow-up after seeing first-seed sensitivity results. Second seed was already used for other methods. This is development evidence, not a pristine held-out or confirmatory test.',
        'primary_question': 'Does doubling the scale improve honest-stripe accuracy and reduce own-clean label-flip damage versus main FLAURA on seed 20261015 in both regimes?',
        'reporting': 'Report each seed separately and the two-seed means; clean and absolute attacked accuracy accompany damage. Keep all eight earlier baselines, main and no-filter control. No significance claim or client-round pseudoreplication.',
        'unchanged': 'Same FLAURA code, alpha=.5 beta=.5 gamma=.8, fixed sigma=74.37725291648866, 100 reserved auxiliary images, calibration, partitions, affected IDs, CNN, local SGD, 10 clients, 512 images/client, 24 rounds and development evaluation.',
        'access': 'Auxiliary labels only select balanced images; no server backpropagation. Development results are reporting-only; no official-test access or new calibration.',
        'stopping': 'Complete exactly six new runs regardless of outcome. No further parameter search, fresh seeds, FedCognis or FedReTrust changes in this follow-up.',
    }
    write_json(study / 'protocol.json', protocol)
    for name in hashes():
        dest = study / 'source_snapshot' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / 'code' / name, dest)
    print(json.dumps({'prepared': str(study), 'new_runs': 6, 'sigma_g': config().sigma_g}), flush=True)


def verify(study):
    p = json.loads((study / 'protocol.json').read_text())
    assert p['source_sha256'] == hashes() and p['config'] == asdict(config()) and p['jobs'] == jobs()
    assert p['reused_manifest_sha256'] == file_sha256(study / 'reused_manifest.json')
    for name, digest in p['parent_artifact_sha256'].items():
        assert file_sha256(ROOT / name) == digest, 'Changed parent artifact: ' + name
    for name, digest in p['source_sha256'].items():
        assert file_sha256(study / 'source_snapshot' / name) == digest
    verify_parent(PARENT)
    return p


def records(study):
    rows = [json.loads(p.read_text()) for p in sorted((study / 'jobs').glob('*/result.json'))]
    for r in rows:
        checked_artifacts(r)
        assert r['source_sha256'] == hashes() and r['job'] in jobs()
    return rows


def worker(j, study):
    verify(study)
    assert j in jobs(), 'Undeclared training job'
    jid = canonical_hash({'job': j, 'sources': hashes(), 'protocol': file_sha256(study / 'protocol.json')})[:20]
    directory = study / 'jobs' / jid
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / 'worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (directory / 'result.json').exists():
            r = json.loads((directory / 'result.json').read_text()); checked_artifacts(r); return r
        if (directory / 'failure.json').exists():
            raise ValueError('Investigate preserved failure before retrying.')
        write_json(directory / 'request.json', {'job': j, 'source_sha256': hashes()})
        started = time.perf_counter()
        try:
            cfg = TrainConfig(partition_id=PARTITIONS[j['regime']], development_seed=j['seed'], rounds=24,
                              max_train_per_client=512, max_validation=1000, num_threads=4)
            data = load_data(ROOT, cfg); reference = build_auxiliary_reference(data)
            agg = FLAURAAggregator(reference, config(), name=TARGET)
            run = run_experiment(data, cfg, longer_scenarios()[j['scenario']], ContributionRecorder(agg),
                                 notebook_code_sha256=jid, verbose=False, evaluator=ClasswiseEvaluator())
            counts = {i: np.bincount(data.pool.targets[ix].numpy(), minlength=10).tolist()
                      for i, ix in data.manifest['indices']['clients'].items()}
            write_json(run / 'population_class_counts.json', counts)
            write_json(run / 'flaura_access.json', {'job': j, 'reference': reference.metadata,
                'official_test_used': False, 'client_local_training_unchanged': True,
                'scale_freeze_sha256': file_sha256(PARENT / 'scale_freeze.json'),
                'auxiliary_labels_in_aggregation': False, 'server_backpropagation': False})
            names = ('run.json', 'selection.json', 'summary.json', 'history.csv', 'task_metrics.csv',
                     'client_metrics.csv', 'aggregation_diagnostics.json', 'population_class_counts.json',
                     'flaura_access.json', 'initial_global_model_state.pt', 'final_global_model_state.pt',
                     'status.json', 'truth/scenario_truth.csv')
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            r = {'job_id': jid, 'job': j, 'status': 'complete', 'run_path': str(run.relative_to(ROOT)),
                 'source_sha256': hashes(), 'artifact_sha256': {n: file_sha256(run / n) for n in names},
                 'worker_wall_seconds': time.perf_counter() - started,
                 'process_lifetime_peak_rss_bytes': int(rss if platform.system() == 'Darwin' else rss * 1024)}
            write_json(directory / 'result.json', r)
            return r
        except BaseException as exc:
            write_json(directory / 'failure.json', {'job': j, 'error': repr(exc), 'at': stamp()})
            raise


def analyse(study):
    verify(study)
    new = records(study)
    assert len(new) == 6 and {key(r['job']) for r in new} == {key(j) for j in jobs()}
    reused = json.loads((study / 'reused_manifest.json').read_text())['records']
    all_records = reused + new
    idx = {key(r['job']): r for r in all_records}
    assert len(idx) == len(all_records) == 132
    metrics, curves, episodes, costs, training, tasks, trust, contributions = [], [], [], [], [], [], [], []
    pairs, prefixes = [], []
    for r in all_records:
        checked_artifacts(r)
        for name, digest in r['source_sha256'].items():
            assert file_sha256(ROOT / 'code' / name) == digest
        j = r['job']; m, g, s, seed = key(j); path = ROOT / r['run_path']
        baseline = idx['FedAvg', g, s, seed]; ref = idx[m, g, 'clean', seed]
        cross_method_pair(r, baseline); pairs.append({'job_id': r['job_id'], 'fedavg_job_id': baseline['job_id']})
        a, b, c = extract_run(r, ref if s != 'clean' else None)
        metrics.extend(a); curves.extend(b); episodes.extend(c); costs.append(cost_record(r))
        ident = {'method': m, 'regime': g, 'scenario': s, 'seed': seed, 'job_id': r['job_id']}
        local = pd.read_csv(path / 'client_metrics.csv'); task = pd.read_csv(path / 'task_metrics.csv')
        training.extend({**row, **ident} for row in local.to_dict('records'))
        tasks.extend({**row, **ident} for row in task.to_dict('records'))
        if s != 'clean':
            h = pd.read_csv(path / 'history.csv'); hc = pd.read_csv(ROOT / ref['run_path'] / 'history.csv')
            assert h[h['round'] <= 4].global_state_sha256.tolist() == hc[hc['round'] <= 4].global_state_sha256.tolist()
            tc = pd.read_csv(ROOT / ref['run_path'] / 'task_metrics.csv')
            pd.testing.assert_frame_equal(task[task['round'] <= 4].reset_index(drop=True), tc[tc['round'] <= 4].reset_index(drop=True), check_exact=True)
            prefixes.append({'job_id': r['job_id'], 'clean_job_id': ref['job_id']})
            contribution, _, _ = contribution_records(r, ref); contributions.extend(contribution)
        if s == 'label_flip':
            value = next(x['value'] for x in a if x['metric'] == 'overall_mean_5_24_pp')
            metrics.append({**ident, 'phase': 'development', 'metric': 'attack_active_accuracy_pp', 'value': value})
        if m in (MAIN, CONTROL, TARGET):
            diagnostics = json.loads((path / 'aggregation_diagnostics.json').read_text())
            for row in diagnostics:
                for cid, values in row['values']['client_metrics'].items():
                    trust.append({**ident, 'round': row['round'], 'client': int(cid),
                        'coefficient': values['applied_delta_coefficient'], 'excluded': values['excluded'],
                        'trust': values['trust'], 'global_score': values['global_score'],
                        'local_score': values['local_score'], 'threshold': row['values']['threshold']})
    table = pd.DataFrame(metrics)
    means = table.groupby(['method', 'regime', 'scenario', 'metric']).value.agg(['mean', 'count', 'std']).reset_index()
    assert means['count'].eq(2).all()
    target = table[table.method == TARGET]; effects = []
    for comparator in METHODS:
        if comparator == TARGET: continue
        other = table[table.method == comparator]
        joined = target.merge(other, on=['regime', 'scenario', 'seed', 'metric'], suffixes=('_target', '_comparator'), validate='one_to_one')
        effects.extend({'comparator': comparator, **{k: row[k] for k in ['regime', 'scenario', 'seed', 'metric']},
            'target_value': row['value_target'], 'comparator_value': row['value_comparator'],
            'difference_pp': row['value_target'] - row['value_comparator']} for row in joined.to_dict('records'))
    audits = []; errors = []; median_iterations = []; first_round = 0
    for r in new:
        j = r['job']; path = ROOT / r['run_path']; m, g, s, seed = key(j)
        info = json.loads((path / 'run.json').read_text()); access = json.loads((path / 'flaura_access.json').read_text())
        first = idx[TARGET, g, s, SEEDS[0]]
        first_info = json.loads((ROOT / first['run_path'] / 'run.json').read_text())
        assert info['method'] == first_info['method'], 'Scoring or reference changed between seeds'
        assert info['method']['settings']['config'] == asdict(config())
        assert not info['official_test_used'] and not access['official_test_used']
        assert not access['server_backpropagation'] and not access['auxiliary_labels_in_aggregation']
        assert access['scale_freeze_sha256'] == file_sha256(PARENT / 'scale_freeze.json')
        indices = set(access['reference']['original_indices'])
        part = json.loads((ROOT / 'data/partitions' / PARTITIONS[g] / 'indices.json').read_text())['indices']
        forbidden = set(part['development_validation']) | {i for values in part['clients'].values() for i in values}
        assert len(indices) == 100 and not indices & forbidden and indices <= set(part['server_reference'])
        local = pd.read_csv(path / 'client_metrics.csv')
        assert len(local) == 240 and local.optimizer_steps.eq(8).all() and local.examples.eq(512).all()
        assert np.isfinite(local.online_train_loss).all()
        old = pd.read_csv(ROOT / idx['FedAvg', g, s, seed]['run_path'] / 'client_metrics.csv')
        columns = ['client', 'submitted_state_sha256', 'broadcast_state_sha256', 'shuffle_seed', 'optimizer_steps', 'online_train_loss']
        pd.testing.assert_frame_equal(local[local['round'] == 1][columns].reset_index(drop=True), old[old['round'] == 1][columns].reset_index(drop=True), check_exact=True)
        first_round += 10
        diagnostics = json.loads((path / 'aggregation_diagnostics.json').read_text())
        assert len(diagnostics) == 24
        for row in diagnostics:
            v = row['values']; clients = v['client_metrics']; weights = [x['applied_delta_coefficient'] for x in clients.values()]
            assert all(w >= 0 for w in weights) and np.isclose(sum(weights), 1)
            assert all(x['applied_delta_coefficient'] == 0 for x in clients.values() if x['excluded'])
            assert v['reference_forward_examples'] == 1000 and v['reference_backward_examples'] == 0
            errors.append(v['contribution_measurements']['reconstruction_max_abs_error'])
            median_iterations.append(v['median_solver']['iterations_used'])
            assert v['median_solver']['termination'] == 'relative_objective_tolerance'
        audits.append({'job_id': r['job_id'], 'client_updates': 240, 'optimizer_steps': 1920,
            'train_forward_examples': 122880, 'reference_forward_examples': 24000,
            'development_task_forward_examples': 75000, 'checkpoint_replay_forward_examples': 6000})
    assert len(pairs) == 132 and len(prefixes) == 88 and first_round == 60
    assert len(errors) == 144 and max(errors) <= 1e-6
    out = study / 'analysis'; out.mkdir(exist_ok=True)
    for name, rows in {'seed_metrics': metrics, 'accuracy_curves': curves, 'model_episodes': episodes,
        'costs': costs, 'local_training': training, 'raw_task_metrics': tasks, 'trust': trust,
        'contributions': contributions, 'paired_effects': effects}.items():
        pd.DataFrame(rows).to_csv(out / (name + '.csv'), index=False)
    means.to_csv(out / 'means.csv', index=False)
    summary = {'status': 'passed', 'new_runs': 6, 'reused_runs': 126, 'comparison_runs': 132,
        'matching_checks': pairs, 'pre_event_checks': prefixes, 'exact_first_round_updates': first_round,
        'audited_new_rounds': len(errors), 'maximum_reconstruction_error': max(errors),
        'maximum_median_iterations': max(median_iterations), 'source_sha256': file_sha256(__file__),
        'interpretation': 'Selected development follow-up; report each seed and means, no significance or universal method claim.'}
    write_json(out / 'summary.json', summary)
    write_json(study / 'result_manifest.json', {'new_records': new, 'records': all_records})
    write_json(study / 'access_audit.json', {'official_test_used': False, 'server_auxiliary_images': 100,
        'new_calibrations': 0, 'new_runs': 6, 'runs': audits,
        'totals': {k: sum(a[k] for a in audits) for k in audits[0] if k != 'job_id'}})


def run(study):
    verify(study)
    with (study / 'coordinator.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write_json(study / 'status.json', {'status': 'running', 'at': stamp()})
        requests = study / 'requests'; requests.mkdir(exist_ok=True)
        def launch(j):
            ident = canonical_hash(j)[:20]; path = requests / (ident + '.json'); write_json(path, j)
            with (requests / (ident + '.log')).open('w') as out:
                subprocess.run([sys.executable, str(Path(__file__).resolve()), 'worker', '--study', str(study), '--job', str(path)], stdout=out, stderr=subprocess.STDOUT, check=True)
            return j
        try:
            completed = {key(r['job']) for r in records(study)}
            with ThreadPoolExecutor(max_workers=2) as pool:
                for f in as_completed([pool.submit(launch, j) for j in jobs() if key(j) not in completed]):
                    j = f.result(); completed.add(key(j))
                    progress = {'completed': len(completed), 'total': 6, 'job': j, 'at': stamp()}
                    write_json(study / 'progress.json', progress); print(json.dumps(progress), flush=True)
            analyse(study)
            write_json(study / 'status.json', {'status': 'complete', 'new_runs': 6, 'at': stamp()})
        except BaseException as exc:
            write_json(study / 'status.json', {'status': 'stopped_on_error', 'error': repr(exc), 'at': stamp()})
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=('prepare', 'run', 'worker', 'analyse'))
    parser.add_argument('--study', type=Path, default=STUDY); parser.add_argument('--job', type=Path)
    args = parser.parse_args()
    if args.action == 'worker': worker(json.loads(args.job.read_text()), args.study)
    else: globals()[args.action](args.study)
