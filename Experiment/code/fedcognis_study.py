"""Frozen, reference-free FedCognis aggregation adaptation comparison."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
import argparse, fcntl, json, platform, resource, shutil, subprocess, sys, time
import numpy as np
from experiment_core import TrainConfig, load_data, run_experiment, file_sha256, write_json
from research_runs import ROOT, PARTITIONS, canonical_hash
from scenario_metrics import ClasswiseEvaluator, ContributionRecorder, longer_scenarios
from directional_study import checked_result as checked_artifacts
from flaura_second_seed import STUDY as PARENT, TARGET as FLAURA_DOUBLE, hashes as parent_hashes, verify as verify_parent
from flaura_study import STUDY as FLAURA_STUDY, MAIN as FLAURA_MAIN, BASELINES as ORIGINAL_BASELINES, SEEDS, CORE, REPEAT, SCENARIOS, stamp
from fedcognis import FedCognisConfig, FedCognisAggregator

STUDY = ROOT / 'outputs/fedcognis/20261030_v1'
MAIN = 'FedCognis-aggregation'
CONTROL = 'FedCognis-no-memory'
SQUARED = 'FedCognis-squared-distance'
BASELINES = (*ORIGINAL_BASELINES, FLAURA_MAIN, FLAURA_DOUBLE)
SETTINGS = {
    MAIN: FedCognisConfig(), CONTROL: FedCognisConfig(alpha=0),
    SQUARED: FedCognisConfig(distance_power=2),
    'FedCognis-beta-0.1': FedCognisConfig(beta=.1),
    'FedCognis-beta-0.7': FedCognisConfig(beta=.7),
}


def hashes():
    return {**parent_hashes(), **{n: file_sha256(ROOT / 'code' / n) for n in
        ('fedcognis.py', 'fedcognis_study.py', 'fedcognis_analysis.py', 'tests/test_fedcognis.py')}}


def key(j): return j['method'], j['regime'], j['scenario'], j['seed']


def job(m, g, s, seed):
    return {'method': m, 'regime': g, 'scenario': s, 'seed': seed, 'phase': 'development',
            'relative_rate': 1., 'root_steps': 8, 'rounds': 24, 'train_images': 512, 'num_threads': 4,
            'fedcognis_stage': 'main' if m == MAIN else 'control' if m == CONTROL else 'interpretation' if m == SQUARED else 'sensitivity'}


def jobs():
    out = [job(MAIN, g, s, seed) for seed in SEEDS for g in PARTITIONS for s in (SCENARIOS if seed == SEEDS[0] else REPEAT)]
    out += [job(CONTROL, g, s, seed) for seed in SEEDS for g in PARTITIONS for s in (*CORE, 'attack_stops')]
    out += [job(SQUARED, g, s, seed) for seed in SEEDS for g in PARTITIONS for s in CORE]
    out += [job(m, g, s, SEEDS[0]) for m in SETTINGS if m.startswith('FedCognis-beta-') for g in PARTITIONS for s in CORE]
    assert len(out) == 70 and len({key(j) for j in out}) == 70
    return out


def prepare(study):
    study.mkdir(parents=True, exist_ok=True)
    if (study / 'protocol.json').exists(): verify(study); return
    verify_parent(PARENT)
    assert json.loads((PARENT / 'status.json').read_text())['status'] == 'complete'
    first = json.loads((FLAURA_STUDY / 'result_manifest.json').read_text())
    second = json.loads((PARENT / 'result_manifest.json').read_text())
    reused = [r for r in first['records'] if r['job']['method'] in (*ORIGINAL_BASELINES, FLAURA_MAIN)]
    reused += [r for r in second['records'] if r['job']['method'] == FLAURA_DOUBLE]
    context = first['context_records']
    assert len(reused) == 282 and len({key(r['job']) for r in reused}) == 282 and len(context) == 72
    for r in reused + context:
        checked_artifacts(r)
        for name, h in r['source_sha256'].items():
            assert file_sha256(ROOT / 'code' / name) == h
    write_json(study / 'reused_manifest.json', {'records': reused, 'context_records': context})
    resources = ROOT / 'resources/fedcognis_reference'
    paths = [p for p in PARENT.rglob('*') if p.is_file() and p.suffix != '.lock']
    paths += [ROOT / 'code/26_flaura_comparison.ipynb', ROOT / 'code/27_flaura_second_seed.ipynb']
    protocol = {
        'created_at': stamp(), 'status': 'declared_before_training',
        'authorization': 'User approved implementing FedCognis aggregation and matched clean/drift/attack/recovery comparison, with explicit reconstruction choices.',
        'source_sha256': hashes(), 'parent_artifact_sha256': {str(p.relative_to(ROOT)): file_sha256(p) for p in paths},
        'resource_sha256': {str(p.relative_to(ROOT)): file_sha256(p) for p in resources.iterdir() if p.is_file()},
        'reused_manifest_sha256': file_sha256(study / 'reused_manifest.json'),
        'partition_sha256': {g: file_sha256(ROOT / 'data/partitions' / part / 'indices.json') for g, part in PARTITIONS.items()},
        'jobs': jobs(), 'settings': {m: asdict(c) for m, c in SETTINGS.items()},
        'new_runs': 70, 'main_runs': 30, 'memory_control_runs': 16, 'squared_distance_runs': 12, 'penalty_sensitivity_runs': 12,
        'reused_primary_runs': 282, 'reused_context_runs': 72,
        'method_status': 'Documented Eq15/Eq20 trust-aggregation adaptation. No public author code located. Not full FedCognis, SALSTM, QSA or original IIoT benchmark reproduction.',
        'paper': 'https://www.techscience.com/cmc/v85n1/63554/html',
        'specification': json.loads((resources / 'source_manifest.json').read_text())['specification_notes'],
        'rule': 'd_i=norm(local_i-current_broadcast,2); e_i=exp(-beta*d_i**power); q_i=alpha*q_i+(1-alpha)*e_i, initially1; a_i=n_i*q_i/sum(n_j*q_j); global += sum(a_i*delta_i). Log-space arithmetic is mathematically equivalent and avoids underflow.',
        'information_boundary': 'Aggregator receives submitted parameters, counts, current broadcast and round only; no data, labels, reference model training, attack truth, validation metrics or clean calibration.',
        'authentication': 'All submitted identities, including compromised participants, are assumed authenticated. No cryptographic implementation or timing claim. No simulator attacker indicator is used as signature validity.',
        'reference_images': 0, 'calibration_runs': 0, 'hard_trust_filter': False,
        'scope': 'Main all10 scenarios in seed20261014, repeat5 in seed20261015. No-memory clean/stripe/label/stopped attacks in both seeds. Squared-distance interpretation clean/stripe/label in both seeds. Beta .1/.7 alternatives core scenarios in first seed.',
        'training': 'Unchanged MNIST CNN, local SGD lr.01 momentum.9, 10 clients, 512 images/client, 1 epoch, batch64, 24 rounds; 1000 development images/task; four CPU threads and two workers.',
        'measurements': 'Clean/affected honest/unchanged honest/absolute attacked accuracy, own-control damage, all scenarios, final24 and tail21-24, local and task losses, actual applied influence, memory/recovery, overhead.',
        'recovery': 'Within own matched control minus1pp for3 consecutive post-attack rounds; preserve onset, confirmation, already-within-criterion, next-attack and horizon censoring.',
        'statistics': 'Two reused development seeds, fixed partitions and affected client IDs; first-seed penalty sensitivities. Descriptive paired comparisons, no formal significance or client-round pseudoreplication.',
        'selection': 'Every setting and outcome retained. Main is fixed before training and is not replaced by the best sensitivity result. No parameter search after seeing outcomes.',
        'stopping': 'Complete all70 runs regardless of outcome; no extra training, new FedReTrust or official-test access.',
    }
    write_json(study / 'protocol.json', protocol)
    for name in hashes():
        dest = study / 'source_snapshot' / name; dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / 'code' / name, dest)
    print(json.dumps({'prepared': str(study), 'new_runs': 70, 'reused_primary': 282, 'context': 72}), flush=True)


def verify(study):
    p = json.loads((study / 'protocol.json').read_text())
    assert p['source_sha256'] == hashes() and p['jobs'] == jobs()
    assert p['settings'] == {m: asdict(c) for m, c in SETTINGS.items()}
    assert p['reused_manifest_sha256'] == file_sha256(study / 'reused_manifest.json')
    for name, h in {**p['parent_artifact_sha256'], **p['resource_sha256']}.items():
        assert file_sha256(ROOT / name) == h, 'Frozen artifact changed: ' + name
    for g, h in p['partition_sha256'].items():
        assert file_sha256(ROOT / 'data/partitions' / PARTITIONS[g] / 'indices.json') == h
    for name, h in p['source_sha256'].items():
        assert file_sha256(study / 'source_snapshot' / name) == h
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
    assert j in jobs(), 'Undeclared job'
    jid = canonical_hash({'job': j, 'sources': hashes(), 'protocol': file_sha256(study / 'protocol.json')})[:20]
    directory = study / 'jobs' / jid; directory.mkdir(parents=True, exist_ok=True)
    with (directory / 'worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (directory / 'result.json').exists():
            r = json.loads((directory / 'result.json').read_text()); checked_artifacts(r); return r
        if (directory / 'failure.json').exists(): raise ValueError('Investigate preserved failure before retry.')
        write_json(directory / 'request.json', {'job': j, 'source_sha256': hashes()})
        started = time.perf_counter()
        try:
            cfg = TrainConfig(partition_id=PARTITIONS[j['regime']], development_seed=j['seed'], rounds=24,
                              max_train_per_client=512, max_validation=1000, num_threads=4)
            data = load_data(ROOT, cfg)
            base = FedCognisAggregator(SETTINGS[j['method']], name=j['method'])
            run = run_experiment(data, cfg, longer_scenarios()[j['scenario']], ContributionRecorder(base),
                                 notebook_code_sha256=jid, verbose=False, evaluator=ClasswiseEvaluator())
            counts = {i: np.bincount(data.pool.targets[ix].numpy(), minlength=10).tolist() for i, ix in data.manifest['indices']['clients'].items()}
            write_json(run / 'population_class_counts.json', counts)
            write_json(run / 'fedcognis_access.json', {'job': j, 'official_test_used': False,
                'server_reference_examples': 0, 'calibration_runs': 0, 'reference_forward_examples': 0,
                'reference_backward_examples': 0, 'defence_inputs': ['submitted_parameters', 'sample_counts', 'broadcast', 'round'],
                'client_local_training_unchanged': True, 'authentication': 'valid_participant_assumption_not_cryptographic_measurement'})
            names = ('run.json', 'selection.json', 'summary.json', 'history.csv', 'task_metrics.csv', 'client_metrics.csv',
                     'aggregation_diagnostics.json', 'population_class_counts.json', 'fedcognis_access.json',
                     'initial_global_model_state.pt', 'final_global_model_state.pt', 'status.json', 'truth/scenario_truth.csv')
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            r = {'job_id': jid, 'job': j, 'status': 'complete', 'run_path': str(run.relative_to(ROOT)), 'source_sha256': hashes(),
                 'artifact_sha256': {n: file_sha256(run / n) for n in names}, 'worker_wall_seconds': time.perf_counter() - started,
                 'process_lifetime_peak_rss_bytes': int(rss if platform.system() == 'Darwin' else rss * 1024)}
            write_json(directory / 'result.json', r)
            return r
        except BaseException as exc:
            write_json(directory / 'failure.json', {'job': j, 'error': repr(exc), 'at': stamp()}); raise


def run(study):
    from fedcognis_analysis import analyse
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
            have = {key(r['job']) for r in records(study)}
            with ThreadPoolExecutor(max_workers=2) as pool:
                for future in as_completed([pool.submit(launch, j) for j in jobs() if key(j) not in have]):
                    j = future.result(); have.add(key(j))
                    p = {'completed': len(have), 'total': 70, 'job': j, 'at': stamp()}
                    write_json(study / 'progress.json', p); print(json.dumps(p), flush=True)
            new = records(study); reused = json.loads((study / 'reused_manifest.json').read_text())
            write_json(study / 'result_manifest.json', {'new_records': new, 'records': new + reused['records'], 'context_records': reused['context_records'], 'at': stamp()})
            analyse(study)
            write_json(study / 'status.json', {'status': 'complete', 'new_runs': 70, 'at': stamp()})
        except BaseException as exc:
            write_json(study / 'status.json', {'status': 'stopped_on_error', 'error': repr(exc), 'at': stamp()}); raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('action', choices=('prepare', 'run', 'worker', 'analyse'))
    p.add_argument('--study', type=Path, default=STUDY); p.add_argument('--job', type=Path); args = p.parse_args()
    if args.action == 'worker': worker(json.loads(args.job.read_text()), args.study)
    elif args.action == 'analyse':
        from fedcognis_analysis import analyse
        analyse(args.study)
    else: globals()[args.action](args.study)
