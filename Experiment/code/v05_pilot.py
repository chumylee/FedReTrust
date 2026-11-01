"""Separate, bounded v0.5 development pilot; never changes the v0.3 study.

Freeze is explicit. Sixteen new candidate runs reuse 48 exactly matched runs.
Existing blocks are development evidence, not fresh confirmation for v0.5.
"""
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
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
from experiment_core import TrainConfig, load_data, run_experiment, file_sha256, write_json
from research_runs import ROOT, canonical_hash
from directional_study import checked_result
from scenario_metrics import ContributionRecorder, ClasswiseEvaluator, longer_scenarios
import validation_replication as original

STUDY = ROOT / 'outputs/reference_free/20261031_v05'
MAIN = 'FedReTrust-v05'
BLOCKS = (1, 2)
REGIMES = ('IID', 'non-IID')
SCENARIOS = ('clean', 'abrupt_shot_noise', 'gradual_stripe', 'label_flip')
COMPARATORS = ('FedAvg', 'RFA', original.MAIN)
REQUIRED_NEW_SOURCES = ('fedretrust_v05.py', 'tests/test_fedretrust_v05.py',
                        'v05_pilot.py', 'v05_pilot_analysis.py')


def stamp():
    return datetime.now(timezone.utc).isoformat()


def key(job):
    return job['method'], job['regime'], job['block'], job['scenario']


def jobs():
    registered = {key(j): j for j in original.jobs()}
    return [{**registered[(original.MAIN, regime, block, scenario)],
             'method': MAIN, 'phase': 'v05_development', 'panel': 'candidate'}
            for block in BLOCKS for regime in REGIMES for scenario in SCENARIOS]


def job_id(job, digest):
    return canonical_hash({'job': job, 'v05_pilot_freeze_sha256': digest})[:24]


def candidate():
    from fedretrust_v05 import FedReTrustV05
    method = FedReTrustV05()
    if method.name != MAIN:
        raise ValueError('Unexpected candidate name.')
    if method.settings.get('trusted_reference_examples') != 0:
        raise ValueError('The candidate must declare zero trusted server examples.')
    return method


def source_hashes(extra_sources=()):
    """The exact map a passing preflight must record before freezing."""
    names = tuple(dict.fromkeys((*REQUIRED_NEW_SOURCES, *extra_sources)))
    for name in names:
        if Path(name).is_absolute() or '..' in Path(name).parts:
            raise ValueError('Source paths must stay within Experiment/code.')
    return {**original.verify(original.STUDY)['source_sha256'],
            **{name: file_sha256(ROOT / 'code' / name) for name in names}}


def selected_reused():
    """Validate only the selected complete records, without changing originals."""
    wanted = {(method, regime, block, scenario) for method in COMPARATORS
              for regime in REGIMES for block in BLOCKS for scenario in SCENARIOS}
    found = {}
    for path in sorted((original.STUDY / 'jobs').glob('*/result.json')):
        raw = json.loads(path.read_text())
        ident = key(raw['job'])
        if ident in wanted:
            if ident in found:
                raise ValueError('Duplicate original comparison record.')
            record = original.read_result(path, original.STUDY)
            found[ident] = {'result_path': str(path.relative_to(ROOT)),
                            'result_sha256': file_sha256(path), 'record': record}
    if set(found) != wanted:
        raise ValueError('Missing exact matched reused runs.')
    return [found[k] for k in sorted(found)]


def protocol():
    return {
        'candidate': MAIN, 'blocks': list(BLOCKS), 'regimes': list(REGIMES),
        'mechanism': {'equation': 'm=sum_i(base_i*q_i); coefficient_i=(base_i*q_i+(1-m)*rfa_weight_i)*clip_scale_i',
            'change_from_v03': 'Replace base_i*q_i/m renormalization with redistribution of rejected mass through the existing capped-RFA weights. Preserve v0.3 norm cap, anchor, directional evidence, restoration and degenerate fallback; add no hyperparameters.',
            'rationale': 'Test whether robust redistribution of rejected mass limits poisoning without discarding the accepted honest updates.',
            'known_blindspot': 'If all q_i=1, m=1 and the candidate exactly reduces to the existing sample-weighted clipped update. It cannot identify harmful contributions receiving full current and restored evidence.',
            'causal_scope': 'The matched v0.3 comparator isolates the changed final weighting formula on each method own trajectory; it does not separately identify interactions with restoration.'},
        'scenarios': list(SCENARIOS), 'comparators': list(COMPARATORS),
        'new_runs': 16, 'reused_runs': 48, 'jobs': jobs(),
        'selection': 'Blocks 1 and 2 by existing registered order, chosen before candidate training; no outcome-based substitution. The previous difficult block 5 is not screened, and all five original outcomes are already development evidence.',
        'evidence_scope': 'Exploratory development. The prior five-block outcomes informed v0.5; neither this pilot nor unused blocks are fresh confirmation.',
        'measurement': 'Means of rounds 5 through 24. Honest drift uses affected clients legitimate tasks. Label damage is each method own clean-control accuracy minus its attacked accuracy.',
        'uncertainty': 'Exactly two paired blocks per regime: show both values, paired differences, means and ranges. No significance test or five-block confidence interval.',
        'complete_success': 'Original ten practical point targets: each regime clean/noise/stripe within 1pp of FedAvg; label damage no more than RFA+1pp and at least 2pp below FedAvg. Complete pilot target success additionally requires focused-fix advancement and IID retention.',
        'advancement': 'Each regime clean/noise/stripe means at least FedAvg-1pp. IID clean/noise/stripe at least v0.3-0.5pp and damage at most v0.3+0.5pp. Non-IID label damage at least 1pp below v0.3 AND absolute attacked accuracy strictly above v0.3. These point checks justify considering a larger development check; they do not establish full success or superiority.',
        'stopping': 'Exactly sixteen new runs. No parameter sweeps, replacement blocks, extra scenarios or automatic follow-up experiment. Preserve failure evidence; retry only the same job with a recorded diagnosis.',
        'data_access': 'Candidate construction receives scalar configuration only; aggregate receives submitted parameters/counts/IDs, broadcast and round. No root images, client validation reports, evaluation metrics, scenario truth or known attacker identities. Offline reporting can inspect simulator truth.',
        'privacy_scope': 'Server sees individual model updates and stable client identifiers; zero reference images does not provide secure aggregation, differential privacy or a Sybil defence.',
        'limits': 'Four scenarios do not establish recovery, recurring-attack, mixed-drift-and-attack or cross-dataset performance. A v0.3 comparison isolates only the frozen code change, not a new general causal account of every component.',
    }


def freeze(study=STUDY, extra_sources=()):
    """Call only after implementation review; creates no training runs."""
    study = Path(study)
    study.mkdir(parents=True, exist_ok=True)
    with (study / 'preparation.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (study / 'execution_freeze.json').exists():
            return verify(study)
        old = original.verify(original.STUDY)
        if len(old['source_sha256']) != 54:
            raise ValueError('Expected the 54-source original freeze.')
        if json.loads((original.STUDY / 'status.json').read_text())['status'] != 'complete':
            raise ValueError('Original study is not complete.')
        names = tuple(dict.fromkeys((*REQUIRED_NEW_SOURCES, *extra_sources)))
        sources = source_hashes(extra_sources)
        preflight = json.loads((study / 'preflight.json').read_text())
        if preflight.get('status') != 'passed' or preflight.get('source_sha256') != sources:
            raise ValueError('A passing preflight for these exact frozen sources is required.')
        reused = selected_reused()
        write_json(study / 'protocol.json', protocol())
        write_json(study / 'reused_manifest.json', {'records': reused,
            'original_execution_freeze_sha256': file_sha256(original.STUDY / 'execution_freeze.json')})
        for name, digest in sources.items():
            destination = study / 'source_snapshot' / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / 'code' / name, destination)
            if file_sha256(destination) != digest:
                raise ValueError('Source snapshot mismatch.')
        frozen = {'schema_version': 1, 'created_at': stamp(), 'jobs': jobs(),
            'candidate_settings': candidate().settings, 'source_sha256': sources,
            'new_sources': list(names), 'environment': original.environment(),
            'original_execution_freeze_sha256': file_sha256(original.STUDY / 'execution_freeze.json'),
            'protocol_sha256': file_sha256(study / 'protocol.json'),
            'reused_manifest_sha256': file_sha256(study / 'reused_manifest.json'),
            'preflight_sha256': file_sha256(study / 'preflight.json'),
            'partitions': [p for p in old['partitions'] if p['block'] in BLOCKS]}
        write_json(study / 'execution_freeze.json', frozen)
        write_json(study / 'freeze_digest.json', {'sha256': file_sha256(study / 'execution_freeze.json')})
        write_json(study / 'status.json', {'status': 'frozen_not_started', 'completed_new_runs': 0,
            'planned_new_runs': 16, 'reused_runs': 48, 'at': stamp()})
    return verify(study)


def verify(study=STUDY, check_reused=True):
    study = Path(study)
    old = original.verify(original.STUDY)
    frozen = json.loads((study / 'execution_freeze.json').read_text())
    if file_sha256(study / 'execution_freeze.json') != json.loads((study / 'freeze_digest.json').read_text())['sha256']:
        raise ValueError('Pilot freeze changed.')
    if file_sha256(original.STUDY / 'execution_freeze.json') != frozen['original_execution_freeze_sha256']:
        raise ValueError('Original freeze changed.')
    if frozen['jobs'] != jobs() or frozen['environment'] != original.environment():
        raise ValueError('Pilot job grid or environment changed.')
    if frozen['candidate_settings'] != candidate().settings:
        raise ValueError('Candidate settings changed.')
    if frozen['partitions'] != [p for p in old['partitions'] if p['block'] in BLOCKS]:
        raise ValueError('Matched partitions changed.')
    for name, digest in frozen['source_sha256'].items():
        if file_sha256(ROOT / 'code' / name) != digest or file_sha256(study / 'source_snapshot' / name) != digest:
            raise ValueError('Frozen source changed: ' + name)
    if any(frozen['source_sha256'].get(n) != h for n, h in old['source_sha256'].items()):
        raise ValueError('Original sources were not preserved in pilot freeze.')
    for name in ('protocol', 'reused_manifest', 'preflight'):
        if file_sha256(study / (name + '.json')) != frozen[name + '_sha256']:
            raise ValueError('Frozen pilot evidence changed: ' + name)
    preflight = json.loads((study / 'preflight.json').read_text())
    if preflight.get('status') != 'passed' or preflight.get('source_sha256') != frozen['source_sha256']:
        raise ValueError('Frozen preflight did not pass for these sources.')
    if json.loads((study / 'protocol.json').read_text()) != protocol():
        raise ValueError('Pilot protocol and implementation disagree.')
    if check_reused:
        reused_records(study)
    return frozen


def reused_records(study=STUDY):
    entries = json.loads((Path(study) / 'reused_manifest.json').read_text())['records']
    expected = {(m, r, b, s) for m in COMPARATORS for r in REGIMES for b in BLOCKS for s in SCENARIOS}
    found = set()
    for entry in entries:
        path = ROOT / entry['result_path']
        if file_sha256(path) != entry['result_sha256'] or json.loads(path.read_text()) != entry['record']:
            raise ValueError('Reused result changed.')
        record = original.read_result(path, original.STUDY)
        if key(record['job']) in found:
            raise ValueError('Duplicate reused result.')
        found.add(key(record['job']))
    if found != expected or len(entries) != 48:
        raise ValueError('Unexpected reused comparison grid.')
    return [e['record'] for e in entries]


def read_result(path, study=STUDY):
    study = Path(study)
    result = json.loads(Path(path).read_text())
    checked_result(result)
    frozen = json.loads((study / 'execution_freeze.json').read_text())
    if result['job'] not in frozen['jobs'] or result['source_sha256'] != frozen['source_sha256']:
        raise ValueError('Unregistered candidate result.')
    if result['job_id'] != job_id(result['job'], file_sha256(study / 'execution_freeze.json')):
        raise ValueError('Candidate result identity mismatch.')
    if set(result['artifact_sha256']) != set(original.ARTIFACTS):
        raise ValueError('Candidate artifact coverage changed.')
    return result


def records(study=STUDY):
    result = [read_result(p, study) for p in sorted((Path(study) / 'jobs').glob('*/result.json'))]
    if len({key(r['job']) for r in result}) != len(result):
        raise ValueError('Duplicate candidate result.')
    return result


def execute_run(job, partition_id, digest, sources):
    started = time.perf_counter()
    cfg = TrainConfig(partition_id=partition_id, development_seed=job['seed'], rounds=job['rounds'],
        max_train_per_client=job['train_images'], max_validation=job['evaluation_images'], num_threads=4)
    data = load_data(ROOT, cfg)
    run_path = run_experiment(data, cfg, longer_scenarios()[job['scenario']],
        ContributionRecorder(candidate()), notebook_code_sha256=digest, verbose=False,
        evaluator=ClasswiseEvaluator())
    counts = {i: np.bincount(data.pool.targets[ix].numpy(), minlength=10).tolist()
              for i, ix in data.manifest['indices']['clients'].items()}
    write_json(run_path / 'population_class_counts.json', counts)
    write_json(run_path / 'validation_access.json', {'official_test_used': False,
        'defence_uses_evaluation_data': False, 'server_reference_examples': 0,
        'aggregation_call_inputs': ['submitted_parameters_and_counts', 'broadcast', 'round'],
        'additional_constructor_access': 'scalar_configuration_only', 'phase': job['phase'],
        'partition_id': partition_id, 'calibration_runs': 0})
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {'job_id': digest, 'job': job, 'status': 'complete',
        'run_path': str(run_path.relative_to(ROOT)), 'source_sha256': sources,
        'artifact_sha256': {n: file_sha256(run_path / n) for n in original.ARTIFACTS},
        'artifact_bytes': sum((run_path / n).stat().st_size for n in original.ARTIFACTS),
        'worker_wall_seconds': time.perf_counter() - started,
        'process_lifetime_peak_rss_bytes': int(rss if platform.system() == 'Darwin' else rss * 1024)}


def worker(job, study=STUDY):
    study = Path(study)
    frozen = verify(study)
    if job not in frozen['jobs']:
        raise ValueError('Undeclared job; no training started.')
    ident = job_id(job, file_sha256(study / 'execution_freeze.json'))
    folder = study / 'jobs' / ident
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / 'worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (folder / 'result.json').exists():
            return read_result(folder / 'result.json', study)
        if (folder / 'failure.json').exists():
            raise ValueError('Preserved failure requires a documented same-job retry.')
        part = next(p for p in frozen['partitions'] if (p['regime'], p['block']) == (job['regime'], job['block']))
        write_json(folder / 'request.json', {'job': job,
            'execution_freeze_sha256': file_sha256(study / 'execution_freeze.json')})
        try:
            result = execute_run(job, part['partition_id'], ident, frozen['source_sha256'])
            write_json(folder / 'result.json', result)
            return result
        except BaseException as exc:
            write_json(folder / 'failure.json', {'job': job, 'error': repr(exc), 'at': stamp(),
                'run_path': getattr(exc, 'run_output_path', None)})
            raise


def run(study=STUDY):
    study = Path(study)
    frozen = verify(study)
    with (study / 'coordinator.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        have = {key(r['job']) for r in records(study)}
        pending = [j for j in frozen['jobs'] if key(j) not in have]
        digest = file_sha256(study / 'execution_freeze.json')
        if any((study / 'jobs' / job_id(j, digest) / 'failure.json').exists() for j in pending):
            raise ValueError('A preserved scientific failure requires investigation and a documented retry.')
        requests = study / 'requests'
        requests.mkdir(exist_ok=True)
        def launch(job):
            ident = job_id(job, digest)
            request = requests / (ident + '.json')
            write_json(request, job)
            with (requests / (ident + '_' + str(time.time_ns()) + '.log')).open('x') as log:
                subprocess.run([sys.executable, str(Path(__file__).resolve()), 'worker',
                    '--study', str(study), '--job', str(request)], stdout=log,
                    stderr=subprocess.STDOUT, check=True)
            return job
        write_json(study / 'status.json', {'status': 'running', 'completed_new_runs': len(have),
            'planned_new_runs': 16, 'reused_runs': 48, 'at': stamp()})
        try:
            iterator = iter(pending)
            with ThreadPoolExecutor(max_workers=2) as pool:
                active = {pool.submit(launch, j) for j in (next(iterator, None), next(iterator, None)) if j is not None}
                while active:
                    done, active = wait(active, return_when=FIRST_COMPLETED)
                    # Inspect every completed future before scheduling more work.
                    completed = [f.result() for f in done]
                    for job in completed:
                        have.add(key(job))
                        progress = {'completed': len(have), 'total': 16, 'job': job, 'at': stamp()}
                        write_json(study / 'progress.json', progress)
                        print(json.dumps(progress), flush=True)
                    for _ in completed:
                        job = next(iterator, None)
                        if job is not None:
                            active.add(pool.submit(launch, job))
            completed = records(study)
            if {key(r['job']) for r in completed} != {key(j) for j in frozen['jobs']}:
                raise ValueError('Incomplete candidate grid.')
            write_json(study / 'result_manifest.json', {'records': completed,
                'execution_freeze_sha256': digest})
            from v05_pilot_analysis import analyse
            result = analyse(study)
            write_json(study / 'status.json', {'status': 'complete', 'completed_new_runs': 16,
                'reused_runs': 48, 'planned_new_runs': 16, 'at': stamp()})
            return result
        except BaseException as exc:
            write_json(study / 'status.json', {'status': 'stopped_on_error', 'error': repr(exc),
                'completed_new_runs': len(records(study)), 'at': stamp()})
            raise


def retry(study, ident, reason):
    study = Path(study)
    frozen = verify(study)
    if not reason or not reason.strip():
        raise ValueError('A concrete investigation and retry reason is required.')
    if ident not in {job_id(j, file_sha256(study / 'execution_freeze.json')) for j in frozen['jobs']}:
        raise ValueError('Unknown job ID.')
    folder = study / 'jobs' / ident
    with (folder / 'worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (folder / 'result.json').exists():
            raise ValueError('Cannot retry a completed job.')
        failure = folder / 'failure.json'
        if not failure.exists():
            raise ValueError('No preserved failure to retry.')
        history = folder / 'failures'
        history.mkdir(exist_ok=True)
        tag = str(time.time_ns())
        write_json(history / (tag + '_decision.json'), {'reason': reason, 'at': stamp(),
            'failure_sha256': file_sha256(failure)})
        failure.rename(history / (tag + '_failure.json'))
    return {'retry_registered': ident, 'next_action': 'Resume the identical frozen pilot.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('freeze', 'verify', 'run', 'worker', 'analyse', 'status', 'retry'))
    parser.add_argument('--study', type=Path, default=STUDY)
    parser.add_argument('--extra-source', action='append', default=[])
    parser.add_argument('--job', type=Path)
    parser.add_argument('--job-id')
    parser.add_argument('--reason')
    args = parser.parse_args()
    if args.action == 'freeze':
        result = freeze(args.study, args.extra_source)
    elif args.action == 'worker':
        result = worker(json.loads(args.job.read_text()), args.study)
    elif args.action == 'analyse':
        from v05_pilot_analysis import analyse
        result = analyse(args.study)
    elif args.action == 'status':
        result = json.loads((args.study / 'status.json').read_text())
    elif args.action == 'retry':
        result = retry(args.study, args.job_id, args.reason)
    else:
        result = globals()[args.action](args.study)
    print(json.dumps(result if args.action in ('status', 'retry', 'analyse') else
                     {'action': args.action, 'completed': True}, indent=2))
