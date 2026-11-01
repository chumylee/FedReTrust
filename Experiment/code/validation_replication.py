"""Prepare, verify and resume the fixed v0.3 replication; no implicit training."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import argparse
import fcntl
import importlib.metadata
import json
import platform
import resource
import shutil
import subprocess
import sys
import time
import numpy as np
from experiment_core import TrainConfig, load_data, run_experiment, file_sha256, write_json
from research_runs import ROOT, PARTITIONS, canonical_hash, make_aggregator
from directional_study import checked_result
from fedretrust_directional import DirectionalConfig, DirectionalFedReTrust
from fedcognis import FedCognisConfig, FedCognisAggregator
from flaura import FLAURAConfig, FLAURAAggregator, build_auxiliary_reference
from scenario_metrics import ContributionRecorder, ClasswiseEvaluator, longer_scenarios
from scenario_utils import Scenario, Event
from validation_controls import CONTROLS, make_control
from validation_partitions import prepare as prepare_partitions

PLAN = ROOT / 'outputs/validation_plan/20261030_v03_v1'
STUDY = ROOT / 'outputs/validation_replication/20261030_v03_v1'
MAIN = 'FedReTrust-Direction-loose'
NEW_SOURCES = ('validation_controls.py', 'validation_partitions.py', 'validation_replication.py',
               'validation_analysis.py', 'validation_preflight.py', 'tests/test_validation.py')
ARTIFACTS = ('run.json', 'selection.json', 'summary.json', 'history.csv', 'task_metrics.csv',
    'client_metrics.csv', 'aggregation_diagnostics.json', 'population_class_counts.json',
    'validation_access.json', 'initial_global_model_state.pt', 'final_global_model_state.pt',
    'status.json', 'truth/scenario_truth.csv')


def stamp(): return datetime.now(timezone.utc).isoformat()


def plan():
    p = json.loads((PLAN / 'protocol.json').read_text())
    guard = json.loads((PLAN / 'verification.json').read_text())
    if file_sha256(PLAN / 'protocol.json') != guard['protocol_sha256']:
        raise ValueError('Declared scientific protocol changed.')
    if file_sha256(PLAN / 'mechanism_audit.json') != p['mechanism_audit_sha256']:
        raise ValueError('Mechanism audit changed.')
    for name, digest in p['source_sha256'].items():
        if file_sha256(ROOT / 'code' / name) != digest or file_sha256(PLAN / 'source_snapshot' / name) != digest:
            raise ValueError('Frozen existing source changed: ' + name)
    if p['candidate_config'] != asdict(DirectionalConfig()):
        raise ValueError('Candidate parameters changed.')
    return p


def source_hashes():
    return {**plan()['source_sha256'], **{n: file_sha256(ROOT / 'code' / n) for n in NEW_SOURCES}}


def environment():
    return {'python': platform.python_version(), 'executable': str(Path(sys.executable).resolve()),
        'platform': platform.platform(), 'machine': platform.machine(),
        'packages': {n: importlib.metadata.version(n) for n in ('numpy', 'pandas', 'torch', 'torchvision', 'matplotlib', 'nbformat')}}


def jobs():
    return [{**j, 'phase': 'replication', 'rounds': 24, 'train_images': 512, 'evaluation_images': 1000,
             'relative_rate': 1., 'root_steps': 8, 'num_threads': 4} for j in plan()['jobs']]


def key(j): return j['method'], j['regime'], j['block'], j['scenario']


def job_id(j, freeze_sha256):
    return canonical_hash({'job': j, 'execution_freeze_sha256': freeze_sha256})[:24]


def method_names():
    p = plan()
    return [m for group in ('primary', 'recent_methods', 'component_controls') for m in p['methods'][group]]


def make_method(method, data, cfg):
    p = plan()
    if method in ('FedAvg', 'RFA', 'FLTrust', 'Viceroy'):
        return make_aggregator(method, data, cfg, relative_rate=1., root_steps=8)
    if method == MAIN:
        base = DirectionalFedReTrust(DirectionalConfig(**p['candidate_config']))
    elif method in CONTROLS:
        base = make_control(method)
    elif method.startswith('FLAURA-') and method in p['methods']['recent_methods']:
        base = FLAURAAggregator(build_auxiliary_reference(data), FLAURAConfig(**p['baseline_settings'][method]), name=method)
    elif method.startswith('FedCognis-') and method in p['methods']['recent_methods']:
        base = FedCognisAggregator(FedCognisConfig(**p['baseline_settings'][method]), name=method)
    else:
        raise ValueError('Unregistered method.')
    return ContributionRecorder(base)


def prepare(study=STUDY):
    study = Path(study); p = plan(); study.mkdir(parents=True, exist_ok=True)
    if (study / 'execution_freeze.json').exists(): return verify(study)
    parts = prepare_partitions(ROOT, PLAN / 'protocol.json', study)
    write_json(study / 'status.json', {'status': 'partitions_prepared_not_frozen', 'at': stamp(),
                                     'planned_runs': 800, 'completed_new_runs': 0})
    return parts


def freeze(study=STUDY):
    study = Path(study)
    if (study / 'execution_freeze.json').exists(): return verify(study)
    p = plan(); sources = source_hashes()
    pre = json.loads((study / 'preflight.json').read_text())
    if pre['status'] != 'passed' or pre['source_sha256'] != sources:
        raise ValueError('A passing preflight for these exact sources is required.')
    for name, digest in pre['artifact_sha256'].items():
        if file_sha256(ROOT / name) != digest: raise ValueError('Preflight evidence changed.')
    parts = json.loads((study / 'partitions.json').read_text())
    if parts['protocol_sha256'] != file_sha256(PLAN / 'protocol.json') or parts['generator_sha256'] != sources['validation_partitions.py']:
        raise ValueError('Partition preparation used different code or protocol.')
    if {(r['block'], r['regime']) for r in parts['partitions']} != {(i, g) for i in range(1, 6) for g in ('IID', 'non-IID')}:
        raise ValueError('Incomplete partition grid.')
    for r in parts['partitions']:
        if file_sha256(ROOT / r['path']) != r['sha256']: raise ValueError('Partition changed.')
    fresh = {b['training_seed'] for b in p['blocks']}
    for path in (ROOT / 'outputs').rglob('result.json'):
        if path.parent.parent.name == 'jobs':
            if json.loads(path.read_text()).get('job', {}).get('seed') in fresh:
                raise ValueError('A declared fresh seed has already trained: ' + str(path))
    # Pin every existing study artifact captured by the plan, along with its text.
    evidence = json.loads((PLAN / 'mechanism_audit.json').read_text())['evidence_sha256']
    for name, digest in evidence.items():
        if file_sha256(ROOT / name) != digest: raise ValueError('Prior evidence changed.')
    report = ROOT / 'reports/FedReTrust_v03_Validation_Protocol.md'
    if file_sha256(report) != json.loads((PLAN / 'verification.json').read_text())['document_sha256']:
        raise ValueError('Human-readable protocol changed.')
    captured = {str(path.relative_to(ROOT)): file_sha256(path) for path in PLAN.rglob('*') if path.is_file()}
    captured[str(report.relative_to(ROOT))] = file_sha256(report)
    captured.update(evidence)
    for name, digest in sources.items():
        dest = study / 'source_snapshot' / name; dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / 'code' / name, dest)
    value = {'status': 'frozen_before_replication_training', 'created_at': stamp(),
        'protocol_sha256': file_sha256(PLAN / 'protocol.json'), 'source_sha256': sources,
        'design_artifact_sha256': captured, 'environment': environment(), 'jobs': jobs(),
        'partitions': parts['partitions'], 'partitions_inventory_sha256': file_sha256(study / 'partitions.json'),
        'preflight_sha256': file_sha256(study / 'preflight.json'),
        'official_test_access': False, 'new_training_at_freeze': 0, 'workers': 2,
        'interpretation': 'Frozen replication on known data; not unseen-dataset confirmation.'}
    write_json(study / 'execution_freeze.json', value)
    write_json(study / 'freeze_digest.json', {'sha256': file_sha256(study / 'execution_freeze.json')})
    write_json(study / 'status.json', {'status': 'ready_for_execution', 'at': stamp(),
        'planned_runs': 800, 'completed_new_runs': 0})
    verify(study)
    return value


def verify(study=STUDY):
    study = Path(study); path = study / 'execution_freeze.json'
    if not path.is_file(): raise ValueError('Scientific training requires a completed execution freeze.')
    f = json.loads(path.read_text())
    if file_sha256(path) != json.loads((study / 'freeze_digest.json').read_text())['sha256']:
        raise ValueError('Execution freeze changed.')
    if f['source_sha256'] != source_hashes() or f['jobs'] != jobs() or f['environment'] != environment():
        raise ValueError('Frozen sources, job grid or environment changed.')
    for name, digest in f['source_sha256'].items():
        if file_sha256(study / 'source_snapshot' / name) != digest: raise ValueError('Snapshot changed.')
    for name, digest in f['design_artifact_sha256'].items():
        if file_sha256(ROOT / name) != digest: raise ValueError('Design evidence changed: ' + name)
    for r in f['partitions']:
        if file_sha256(ROOT / r['path']) != r['sha256']: raise ValueError('Frozen partition changed.')
    if file_sha256(study / 'partitions.json') != f['partitions_inventory_sha256'] or file_sha256(study / 'preflight.json') != f['preflight_sha256']:
        raise ValueError('Preparation evidence changed.')
    return f


def read_result(path, study):
    r = json.loads(Path(path).read_text()); checked_result(r)
    f = json.loads((Path(study) / 'execution_freeze.json').read_text())
    if r['job'] not in f['jobs'] or r['source_sha256'] != f['source_sha256']:
        raise ValueError('Unregistered result or source mismatch.')
    if r['job_id'] != job_id(r['job'], file_sha256(Path(study) / 'execution_freeze.json')):
        raise ValueError('Wrong run identity.')
    return r


def records(study=STUDY):
    return [read_result(p, study) for p in sorted((Path(study) / 'jobs').glob('*/result.json'))]


def execute_run(j, partition_id, source_digest, sources):
    """Shared execution after caller validates either fixed preflight or frozen job."""
    started = time.perf_counter()
    cfg = TrainConfig(partition_id=partition_id, development_seed=j['seed'], rounds=j['rounds'],
        max_train_per_client=j['train_images'], max_validation=j['evaluation_images'], num_threads=4)
    data = load_data(ROOT, cfg)
    method = make_method(j['method'], data, cfg)
    if j['phase'] == 'preflight':
        scenario = Scenario(j['scenario']) if j['scenario'] == 'clean' else Scenario('label_flip',
            (Event('label_flip', (0, 1, 2), ((5, 6),), intent='malicious'),))
    else:
        scenario = longer_scenarios()[j['scenario']]
    run = run_experiment(data, cfg, scenario, method, notebook_code_sha256=source_digest,
                         verbose=False, evaluator=ClasswiseEvaluator())
    counts = {i: np.bincount(data.pool.targets[ix].numpy(), minlength=10).tolist()
              for i, ix in data.manifest['indices']['clients'].items()}
    write_json(run / 'population_class_counts.json', counts)
    refs = 300 if j['method'] == 'FLTrust' else 100 if j['method'].startswith('FLAURA-') else 0
    write_json(run / 'validation_access.json', {'official_test_used': False,
        'defence_uses_evaluation_data': False, 'server_reference_examples': refs,
        'aggregation_call_inputs': ['submitted_parameters_and_counts', 'broadcast', 'round'],
        'additional_constructor_access': 'reserved_root_training_reference' if refs == 300 else 'reserved_auxiliary_images' if refs else 'scalar_configuration_only',
        'phase': j['phase'], 'partition_id': partition_id, 'calibration_runs': 0})
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {'job_id': source_digest, 'job': j, 'status': 'complete', 'run_path': str(run.relative_to(ROOT)),
        'source_sha256': sources, 'artifact_sha256': {n: file_sha256(run / n) for n in ARTIFACTS},
        'artifact_bytes': sum((run / n).stat().st_size for n in ARTIFACTS),
        'worker_wall_seconds': time.perf_counter() - started,
        'process_lifetime_peak_rss_bytes': int(rss if platform.system() == 'Darwin' else rss * 1024)}


def worker(j, study=STUDY):
    study = Path(study); f = verify(study)
    if j not in f['jobs']: raise ValueError('Undeclared job; no training started.')
    ident = job_id(j, file_sha256(study / 'execution_freeze.json'))
    folder = study / 'jobs' / ident; folder.mkdir(parents=True, exist_ok=True)
    with (folder / 'worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (folder / 'result.json').exists(): return read_result(folder / 'result.json', study)
        if (folder / 'failure.json').exists(): raise ValueError('Preserved failure requires a documented retry.')
        part = next(r for r in f['partitions'] if (r['block'], r['regime']) == (j['block'], j['regime']))
        write_json(folder / 'request.json', {'job': j, 'execution_freeze_sha256': file_sha256(study / 'execution_freeze.json')})
        try:
            result = execute_run(j, part['partition_id'], ident, f['source_sha256'])
            write_json(folder / 'result.json', result); return result
        except BaseException as exc:
            write_json(folder / 'failure.json', {'job': j, 'error': repr(exc), 'at': stamp(),
                'run_path': getattr(exc, 'run_output_path', None)})
            raise


def run(study=STUDY, panel='all'):
    from validation_analysis import analyse
    study = Path(study); f = verify(study)
    if panel not in ('all', 'primary', 'recent_methods', 'component_controls'): raise ValueError('Unknown panel.')
    with (study / 'coordinator.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        have = {key(r['job']) for r in records(study)}
        pending = [j for j in f['jobs'] if key(j) not in have and (panel == 'all' or j['panel'] == panel)]
        requests = study / 'requests'; requests.mkdir(exist_ok=True)
        def launch(j):
            ident = job_id(j, file_sha256(study / 'execution_freeze.json'))
            request = requests / (ident + '.json'); write_json(request, j)
            log = requests / (ident + '_' + str(time.time_ns()) + '.log')
            with log.open('x') as out:
                subprocess.run([sys.executable, str(Path(__file__).resolve()), 'worker', '--study', str(study), '--job', str(request)],
                               stdout=out, stderr=subprocess.STDOUT, check=True)
            return j
        write_json(study / 'status.json', {'status': 'running', 'panel': panel, 'completed_new_runs': len(have), 'at': stamp()})
        try:
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(launch, j) for j in pending]
                try:
                    for future in as_completed(futures):
                        j = future.result(); have.add(key(j))
                        progress = {'completed': len(have), 'total': 800, 'job': j, 'at': stamp()}
                        write_json(study / 'progress.json', progress); print(json.dumps(progress), flush=True)
                except BaseException:
                    for future in futures: future.cancel()
                    raise
            completed = records(study)
            write_json(study / 'result_manifest.json', {'records': completed, 'execution_freeze_sha256': file_sha256(study / 'execution_freeze.json')})
            if len(completed) == 800: analyse(study)
            write_json(study / 'status.json', {'status': 'complete' if len(completed) == 800 else 'stage_complete',
                'completed_new_runs': len(completed), 'planned_runs': 800, 'at': stamp()})
        except BaseException as exc:
            write_json(study / 'status.json', {'status': 'stopped_on_error', 'error': repr(exc), 'at': stamp()})
            raise


def retry(study, ident, reason):
    study = Path(study); f = verify(study)
    if not reason or not reason.strip(): raise ValueError('A concrete retry reason is required.')
    valid = {job_id(j, file_sha256(study / 'execution_freeze.json')) for j in f['jobs']}
    if ident not in valid: raise ValueError('Unknown job ID.')
    folder = study / 'jobs' / ident
    with (folder / 'worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (folder / 'result.json').exists(): raise ValueError('Cannot retry a complete job.')
        failure = folder / 'failure.json'
        if not failure.exists(): raise ValueError('No failed attempt to retry.')
        history = folder / 'failures'; history.mkdir(exist_ok=True)
        tag = str(time.time_ns())
        write_json(history / (tag + '_decision.json'), {'reason': reason, 'at': stamp(), 'failure_sha256': file_sha256(failure)})
        failure.rename(history / (tag + '_failure.json'))
    return {'retry_registered': ident, 'next_action': 'Resume the same frozen run or panel.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'preflight', 'freeze', 'verify', 'worker', 'run', 'analyse', 'status', 'retry'])
    parser.add_argument('--study', type=Path, default=STUDY); parser.add_argument('--job', type=Path)
    parser.add_argument('--panel', default='all'); parser.add_argument('--job-id'); parser.add_argument('--reason')
    args = parser.parse_args()
    if args.action == 'worker': result = worker(json.loads(args.job.read_text()), args.study)
    elif args.action == 'run': result = run(args.study, args.panel)
    elif args.action == 'preflight':
        from validation_preflight import preflight
        result = preflight(args.study)
    elif args.action == 'analyse':
        from validation_analysis import analyse
        result = analyse(args.study)
    elif args.action == 'status': result = json.loads((args.study / 'status.json').read_text())
    elif args.action == 'retry': result = retry(args.study, args.job_id, args.reason)
    else: result = globals()[args.action](args.study)
    print(json.dumps(result if args.action in ('status', 'retry') else {'action': args.action, 'completed': True}, indent=2))
