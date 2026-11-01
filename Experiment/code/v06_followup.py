"""Unchanged v0.6 follow-up:24 new runs complete five known development blocks."""
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
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
from scenario_metrics import ClasswiseEvaluator, longer_scenarios
import validation_replication as original
import v05_followup as prior
import v06_pilot as pilot

STUDY = ROOT / 'outputs/reference_free/20261031_v06_followup'
MAIN = 'FedReTrust-v06'
BLOCKS = (1, 2, 4)
ALL_BLOCKS = (1, 2, 3, 4, 5)
REGIMES = prior.REGIMES
SCENARIOS = prior.SCENARIOS
COMPARATORS = ('FedReTrust-v05', original.MAIN, 'RFA', 'FedAvg')
REQUIRED_NEW_SOURCES = ('v06_followup.py', 'v06_followup_analysis.py',
    'tests/test_v06_followup.py', 'tests/test_v06_followup_analysis.py')
stamp = prior.stamp
key = prior.key


def jobs():
    registered = {key(j): j for j in original.jobs()}
    return [{**registered[(original.MAIN, regime, block, scenario)],
             'method': MAIN, 'phase': 'v06_five_block_followup', 'panel': 'candidate'}
            for block in BLOCKS for regime in REGIMES for scenario in SCENARIOS]


def job_id(job, digest):
    return canonical_hash({'job': job, 'v06_followup_freeze_sha256': digest})[:24]


def candidate():
    from fedretrust_v06 import FedReTrustV06
    method = FedReTrustV06()
    if method.name != MAIN or method.settings.get('trusted_reference_examples') != 0:
        raise ValueError('The fixed v0.6 candidate must use zero reference examples.')
    return method


def recorded_method():
    from v06_recording import InnovationContributionRecorder
    return InnovationContributionRecorder(candidate())


def source_hashes(extra_sources=()):
    previous = pilot.verify(pilot.STUDY, check_reused=False)
    if len(previous['source_sha256']) != 69:
        raise ValueError('Expected the 69 immutable prior scientific sources.')
    names = tuple(dict.fromkeys((*REQUIRED_NEW_SOURCES, *extra_sources)))
    for name in names:
        if Path(name).is_absolute() or '..' in Path(name).parts or name in previous['source_sha256']:
            raise ValueError('New source paths must be new files within Experiment/code.')
    return {**previous['source_sha256'], **{n: file_sha256(ROOT / 'code' / n) for n in names}}


def prior_evidence_hashes():
    hashes = pilot.prior_evidence_hashes()
    location = pilot.STUDY
    if json.loads((location / 'status.json').read_text())['status'] != 'complete':
        raise ValueError('The v0.6 pilot must be complete.')
    review = json.loads((location / 'completion_verification.json').read_text())
    decision = json.loads((location / 'analysis/decision.json').read_text())
    digest = file_sha256(location / 'execution_freeze.json')
    manifest = json.loads((location / 'analysis_manifest.json').read_text())
    if review.get('status') != 'complete_verified' or review['decision'] != decision:
        raise ValueError('Missing or changed reviewed pilot decision.')
    if decision['complete_selected_screen_point_targets_met'] is not False or decision['automatic_continuation_authorized'] is not False:
        raise ValueError('The failed pilot remains failed; this follow-up is separate.')
    if review['execution_freeze_sha256'] != digest or manifest['execution_freeze_sha256'] != digest or review['analysis_manifest_sha256'] != file_sha256(location / 'analysis_manifest.json'):
        raise ValueError('Pilot review provenance mismatch.')
    names = ('execution_freeze.json','freeze_digest.json','protocol.json','reused_manifest.json',
        'result_manifest.json','analysis_manifest.json','analysis/decision.json','completion_verification.json','status.json')
    for name in names:
        hashes[str((location / name).relative_to(ROOT))] = file_sha256(location / name)
    for name, expected in manifest['artifact_sha256'].items():
        if Path(name).is_absolute() or '..' in Path(name).parts:
            raise ValueError('Invalid evidence path.')
        if file_sha256(location / name) != expected:
            raise ValueError('Pilot analysis artifact changed.')
        hashes[str((location / name).relative_to(ROOT))] = expected
    return hashes


def _expected_reused():
    return ({(m,r,b,s) for m in COMPARATORS for r in REGIMES for b in ALL_BLOCKS for s in SCENARIOS}
        | {(MAIN,r,b,s) for r in REGIMES for b in pilot.BLOCKS for s in SCENARIOS})


def origins():
    return {'original': (original.STUDY, original.read_result),
        'v05_pilot': (prior.pilot.STUDY, prior.pilot.read_result),
        'v05_followup': (prior.STUDY, prior.read_result),
        'v06_pilot': (pilot.STUDY, pilot.read_result)}


def selected_reused():
    wanted, found = _expected_reused(), {}
    for origin, (location, reader) in origins().items():
        for path in sorted((location / 'jobs').glob('*/result.json')):
            raw = json.loads(path.read_text()); identity = key(raw['job'])
            if identity not in wanted: continue
            expected_origin = 'v06_pilot' if identity[0] == MAIN else ('v05_pilot' if identity[2] in (1,2) else 'v05_followup') if identity[0] == prior.MAIN else 'original'
            if origin != expected_origin: continue
            if identity in found: raise ValueError('Duplicate comparison result.')
            record = reader(path, location)
            found[identity] = {'origin': origin, 'result_path': str(path.relative_to(ROOT)),
                'result_sha256': file_sha256(path), 'record': record}
    if set(found) != wanted or len(found) != 176:
        raise ValueError('Missing five-block reused comparisons.')
    return [found[k] for k in sorted(found)]


def protocol():
    base = pilot.protocol()
    return {**base, 'blocks': list(ALL_BLOCKS), 'new_blocks': list(BLOCKS), 'jobs': jobs(),
        'new_runs':24, 'reused_runs':176, 'combined_runs':200,
        'selection':'Unchanged v0.6 adds blocks1,2,4 to preserved pilot blocks3,5. All five original blocks informed development; these are known conditions, not independent confirmation.',
        'evidence_scope':'Five-block exploratory comparison; preserve the failed two-block pilot and all prior versions.',
        'uncertainty':'Five paired block values, mean, SD, range and nominal two-sided95% t intervals(df4). Intervals assume independent approximately normal block differences, are unstable at n=5 and have no simultaneous coverage; all blocks are development data.',
        'IID_retention':'Apply the existing half-point margins separately to all five IID blocks: clean/noise/stripe/attacked accuracy>=v0.5-0.5pp and damage<=v0.5+0.5pp.',
        'nonIID_focus':'Five-block mean damage reduction>=1pp versusv0.5; absolute attacked mean accuracy improves in each block. Block5 onset damage and absolute onset accuracy improve>=5pp; both onset measures do not worsen in each of blocks1,2,3,4.',
        'criteria_interpretation':'51 descriptive point checks:10 unchanged original mean targets,25 per-block IID retention checks,16 nonIID checks. This extends coverage before new outcomes, preserves existing margins, and does not revise the original27-check pilot outcome.',
        'prior_evidence':'All past freezes and decisions remain immutable. This separately authorized follow-up does not retroactively pass the pilot.',
        'stopping':'Exactly24 new jobs. No new version, tuning, substituted seeds, extra scenarios, or automatic further training. Preserve failures and investigate before documented same-job retry.',
        'automatic_continuation_authorized':False}


def freeze(study=STUDY, extra_sources=()):
    study = Path(study); study.mkdir(parents=True, exist_ok=True)
    with (study / 'preparation.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (study / 'execution_freeze.json').exists():
            return verify(study)
        old = original.verify(original.STUDY); sources = source_hashes(extra_sources)
        evidence = prior_evidence_hashes()
        preflight = json.loads((study / 'preflight.json').read_text())
        if preflight.get('status') != 'passed' or preflight.get('source_sha256') != sources:
            raise ValueError('Passing preflight for these exact source hashes is required.')
        write_json(study / 'protocol.json', protocol())
        write_json(study / 'reused_manifest.json', {'records': selected_reused(),
            'v06_pilot_execution_freeze_sha256': file_sha256(pilot.STUDY / 'execution_freeze.json'),
            'original_execution_freeze_sha256': file_sha256(original.STUDY / 'execution_freeze.json'),
            'prior_followup_execution_freeze_sha256': file_sha256(prior.STUDY / 'execution_freeze.json')})
        for name, digest in sources.items():
            destination = study / 'source_snapshot' / name; destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / 'code' / name, destination)
            if file_sha256(destination) != digest:
                raise ValueError('Source snapshot mismatch.')
        frozen = {'schema_version': 1, 'created_at': stamp(), 'jobs': jobs(),
            'v06_pilot_execution_freeze_sha256': file_sha256(pilot.STUDY / 'execution_freeze.json'),
            'v06_pilot_decision_sha256': file_sha256(pilot.STUDY / 'analysis/decision.json'),
            'candidate_settings': candidate().settings, 'recorder_settings': recorded_method().settings,
            'source_sha256': sources, 'new_sources': list(dict.fromkeys((*REQUIRED_NEW_SOURCES, *extra_sources))),
            'environment': original.environment(), 'prior_evidence_sha256': evidence,
            'original_execution_freeze_sha256': file_sha256(original.STUDY / 'execution_freeze.json'),
            'prior_followup_execution_freeze_sha256': file_sha256(prior.STUDY / 'execution_freeze.json'),
            'prior_pilot_decision_sha256': file_sha256(prior.pilot.STUDY / 'analysis/decision.json'),
            'prior_followup_decision_sha256': file_sha256(prior.STUDY / 'analysis/decision.json'),
            'protocol_sha256': file_sha256(study / 'protocol.json'),
            'reused_manifest_sha256': file_sha256(study / 'reused_manifest.json'),
            'preflight_sha256': file_sha256(study / 'preflight.json'),
            'partitions': [p for p in old['partitions'] if p['block'] in ALL_BLOCKS]}
        write_json(study / 'execution_freeze.json', frozen)
        write_json(study / 'freeze_digest.json', {'sha256': file_sha256(study / 'execution_freeze.json')})
        write_json(study / 'status.json', {'status': 'frozen_not_started', 'completed_new_runs': 0,
            'planned_new_runs': 24, 'reused_runs': 176, 'combined_runs': 200, 'at': stamp()})
    return verify(study)


def verify(study=STUDY, check_reused=True):
    study = Path(study); previous = pilot.verify(pilot.STUDY, check_reused=False)
    old = original.verify(original.STUDY); frozen = json.loads((study / 'execution_freeze.json').read_text())
    if file_sha256(study / 'execution_freeze.json') != json.loads((study / 'freeze_digest.json').read_text())['sha256']:
        raise ValueError('V0.6 freeze changed.')
    checks = {'v06_pilot_execution_freeze_sha256': pilot.STUDY / 'execution_freeze.json',
        'v06_pilot_decision_sha256': pilot.STUDY / 'analysis/decision.json',
        'original_execution_freeze_sha256': original.STUDY / 'execution_freeze.json',
        'prior_followup_execution_freeze_sha256': prior.STUDY / 'execution_freeze.json',
        'prior_pilot_decision_sha256': prior.pilot.STUDY / 'analysis/decision.json',
        'prior_followup_decision_sha256': prior.STUDY / 'analysis/decision.json'}
    for name, path in checks.items():
        if file_sha256(path) != frozen[name]:
            raise ValueError('Pinned prior evidence changed.')
    if frozen['prior_evidence_sha256'] != prior_evidence_hashes():
        raise ValueError('Prior scientific evidence changed.')
    if frozen['jobs'] != jobs() or frozen['environment'] != original.environment():
        raise ValueError('Frozen job grid or environment changed.')
    if frozen['candidate_settings'] != candidate().settings or frozen['recorder_settings'] != recorded_method().settings:
        raise ValueError('Candidate or recorder settings changed.')
    if frozen['partitions'] != [p for p in old['partitions'] if p['block'] in ALL_BLOCKS]:
        raise ValueError('Matched partitions changed.')
    if set(frozen['source_sha256']) != set(previous['source_sha256']) | set(frozen['new_sources']) or not set(REQUIRED_NEW_SOURCES).issubset(frozen['new_sources']):
        raise ValueError('Frozen source coverage changed.')
    if any(frozen['source_sha256'].get(n) != h for n, h in previous['source_sha256'].items()):
        raise ValueError('The69 prior scientific sources were not preserved.')
    for name, digest in frozen['source_sha256'].items():
        if file_sha256(ROOT / 'code' / name) != digest or file_sha256(study / 'source_snapshot' / name) != digest:
            raise ValueError('Frozen source changed: ' + name)
    for name in ('protocol', 'reused_manifest', 'preflight'):
        if file_sha256(study / (name + '.json')) != frozen[name + '_sha256']:
            raise ValueError('Frozen v0.6 evidence changed: ' + name)
    preflight = json.loads((study / 'preflight.json').read_text())
    if preflight.get('status') != 'passed' or preflight.get('source_sha256') != frozen['source_sha256']:
        raise ValueError('Preflight did not pass for the frozen sources.')
    if json.loads((study / 'protocol.json').read_text()) != protocol():
        raise ValueError('Protocol and implementation disagree.')
    if check_reused:
        reused_records(study)
    return frozen


def reused_records(study=STUDY):
    manifest = json.loads((Path(study) / 'reused_manifest.json').read_text())
    for field, location in [('original_execution_freeze_sha256',original.STUDY),('prior_followup_execution_freeze_sha256',prior.STUDY),('v06_pilot_execution_freeze_sha256',pilot.STUDY)]:
        if manifest[field] != file_sha256(location / 'execution_freeze.json'):
            raise ValueError('Reused manifest provenance changed.')
    found, output = set(), []
    for entry in manifest['records']:
        path = ROOT / entry['result_path']; identity = key(entry['record']['job'])
        if file_sha256(path) != entry['result_sha256'] or json.loads(path.read_text()) != entry['record']:
            raise ValueError('Reused result changed.')
        expected_origin = 'v06_pilot' if identity[0] == MAIN else ('v05_pilot' if identity[2] in (1,2) else 'v05_followup') if identity[0] == prior.MAIN else 'original'
        if entry['origin'] != expected_origin: raise ValueError('Wrong result origin.')
        location, reader = origins()[expected_origin]
        if not path.resolve().is_relative_to((location / 'jobs').resolve()):
            raise ValueError('Invalid reused path.')
        record = reader(path,location)
        if identity in found: raise ValueError('Duplicate result.')
        found.add(identity); output.append(record)
    if found != _expected_reused() or len(output) != 176:
        raise ValueError('Unexpected reused grid.')
    return output


# Reuse the exact frozen pilot training helper and affine recorder.
execute_run = pilot.execute_run


def read_result(path, study=STUDY):
    study = Path(study)
    result = json.loads(Path(path).read_text())
    checked_result(result)
    frozen = json.loads((study / 'execution_freeze.json').read_text())
    if result['job'] not in frozen['jobs'] or result['source_sha256'] != frozen['source_sha256']:
        raise ValueError('Unregistered v0.6 follow-up result.')
    if result['job_id'] != job_id(result['job'], file_sha256(study / 'execution_freeze.json')):
        raise ValueError('V0.6 follow-up result identity mismatch.')
    if set(result['artifact_sha256']) != set(original.ARTIFACTS):
        raise ValueError('V0.6 follow-up artifact coverage changed.')
    return result


def records(study=STUDY):
    output = [read_result(p, study) for p in sorted((Path(study) / 'jobs').glob('*/result.json'))]
    if len({key(r['job']) for r in output}) != len(output):
        raise ValueError('Duplicate v0.6 follow-up result.')
    return output


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
            'planned_new_runs': 24, 'reused_runs': 176, 'combined_runs': 200, 'at': stamp()})
        try:
            iterator = iter(pending)
            with ThreadPoolExecutor(max_workers=2) as pool:
                active = {pool.submit(launch, j) for j in (next(iterator, None), next(iterator, None)) if j is not None}
                while active:
                    done, active = wait(active, return_when=FIRST_COMPLETED)
                    completed, failures = [], []
                    for future in done:
                        try:
                            completed.append(future.result())
                        except BaseException as exc:
                            failures.append(exc)
                    for job in completed:
                        have.add(key(job))
                        progress = {'completed': len(have), 'total': 24, 'job': job, 'at': stamp()}
                        write_json(study / 'progress.json', progress)
                        print(json.dumps(progress), flush=True)
                    if failures:
                        # Running workers may finish; do not discard their
                        # results or errors and do not schedule another job.
                        for future in active:
                            try:
                                future.result()
                            except BaseException as exc:
                                failures.append(exc)
                        raise RuntimeError('V0.6 follow-up workers failed: ' + '; '.join(repr(e) for e in failures))
                    for _ in completed:
                        job = next(iterator, None)
                        if job is not None:
                            active.add(pool.submit(launch, job))
            completed = records(study)
            if {key(r['job']) for r in completed} != {key(j) for j in frozen['jobs']}:
                raise ValueError('Incomplete v0.6 pilot grid.')
            write_json(study / 'result_manifest.json', {'records': completed, 'execution_freeze_sha256': digest})
            from v06_followup_analysis import analyse
            result = analyse(study)
            write_json(study / 'status.json', {'status': 'complete', 'completed_new_runs': 24,
                'reused_runs': 176, 'combined_runs': 200, 'planned_new_runs': 24, 'at': stamp()})
            return result
        except BaseException as exc:
            write_json(study / 'status.json', {'status': 'stopped_on_error', 'error': repr(exc),
                'completed_new_runs': len(records(study)), 'planned_new_runs': 24, 'at': stamp()})
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
            'failure_sha256': file_sha256(failure), 'job_id': ident,
            'execution_freeze_sha256': file_sha256(study / 'execution_freeze.json')})
        failure.rename(history / (tag + '_failure.json'))
    return {'retry_registered': ident, 'next_action': 'Resume the identical frozen v0.6 pilot.'}


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
        from v06_followup_analysis import analyse
        result = analyse(args.study)
    elif args.action == 'status':
        result = json.loads((args.study / 'status.json').read_text())
    elif args.action == 'retry':
        result = retry(args.study, args.job_id, args.reason)
    else:
        result = globals()[args.action](args.study)
    print(json.dumps(result if args.action in ('status', 'retry', 'analyse') else
                     {'action': args.action, 'completed': True}, indent=2))
