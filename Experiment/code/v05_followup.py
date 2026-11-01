"""User-authorized, unchanged-v0.5 follow-up after a failed advancement gate.

Exactly 24 new runs complete the already known five-block development panel.
The failed 16-run pilot and all earlier freezes remain immutable evidence.
"""
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path
import argparse
import fcntl
import json
import shutil
import subprocess
import sys
import time

from experiment_core import file_sha256, write_json
from research_runs import ROOT, canonical_hash
from directional_study import checked_result
import validation_replication as original
import v05_pilot as pilot

STUDY = ROOT / 'outputs/reference_free/20261031_v05_followup'
MAIN = pilot.MAIN
BLOCKS = (3, 4, 5)
ALL_BLOCKS = (1, 2, 3, 4, 5)
REGIMES = pilot.REGIMES
SCENARIOS = pilot.SCENARIOS
COMPARATORS = pilot.COMPARATORS
REQUIRED_NEW_SOURCES = ('v05_followup.py', 'v05_followup_analysis.py',
                        'tests/test_v05_followup.py', 'tests/test_v05_followup_analysis.py')
stamp = pilot.stamp
key = pilot.key


def jobs():
    registered = {key(j): j for j in original.jobs()}
    return [{**registered[(original.MAIN, regime, block, scenario)],
             'method': MAIN, 'phase': 'v05_exploratory_followup', 'panel': 'candidate'}
            for block in BLOCKS for regime in REGIMES for scenario in SCENARIOS]


def job_id(job, digest):
    return canonical_hash({'job': job, 'v05_followup_freeze_sha256': digest})[:24]


def source_hashes(extra_sources=()):
    """Required exact source map for the preflight; prior 58 files are fixed."""
    previous = pilot.verify(pilot.STUDY)
    if len(previous['source_sha256']) != 58:
        raise ValueError('Expected the prior 54 original plus 4 pilot scientific sources.')
    names = tuple(dict.fromkeys((*REQUIRED_NEW_SOURCES, *extra_sources)))
    for name in names:
        if Path(name).is_absolute() or '..' in Path(name).parts:
            raise ValueError('Source paths must stay within Experiment/code.')
        if name in previous['source_sha256']:
            raise ValueError('Additional sources must not replace a prior frozen source.')
    return {**previous['source_sha256'],
            **{name: file_sha256(ROOT / 'code' / name) for name in names}}


def prior_evidence_hashes():
    """Verify and pin the pilot decision, completed analysis and provenance."""
    if json.loads((pilot.STUDY / 'status.json').read_text())['status'] != 'complete':
        raise ValueError('The prior pilot is not complete.')
    completion = json.loads((pilot.STUDY / 'completion_verification.json').read_text())
    if completion.get('status') != 'complete_verified':
        raise ValueError('The pilot completion review is missing.')
    decision = json.loads((pilot.STUDY / 'analysis/decision.json').read_text())
    if (decision.get('advancement_point_criterion_met') is not False or
            decision.get('automatic_continuation_authorized') is not False):
        raise ValueError('Preserve the actual failed pilot advancement decision.')
    manifest = json.loads((pilot.STUDY / 'analysis_manifest.json').read_text())
    digest = file_sha256(pilot.STUDY / 'execution_freeze.json')
    if manifest['execution_freeze_sha256'] != digest or completion['execution_freeze_sha256'] != digest:
        raise ValueError('Pilot analysis/completion provenance mismatch.')
    if completion['analysis_manifest_sha256'] != file_sha256(pilot.STUDY / 'analysis_manifest.json'):
        raise ValueError('Pilot reviewed analysis manifest changed.')
    if completion['decision'] != decision:
        raise ValueError('Reviewed pilot decision changed.')
    names = ('execution_freeze.json', 'freeze_digest.json', 'protocol.json', 'reused_manifest.json',
             'result_manifest.json', 'analysis_manifest.json', 'analysis/decision.json',
             'completion_verification.json', 'status.json')
    hashes = {str((pilot.STUDY / n).relative_to(ROOT)): file_sha256(pilot.STUDY / n) for n in names}
    for name, expected in manifest['artifact_sha256'].items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Invalid prior analysis artifact path.')
        path = pilot.STUDY / relative
        if file_sha256(path) != expected:
            raise ValueError('Prior pilot analysis artifact changed: ' + name)
        hashes[str(path.relative_to(ROOT))] = expected
    return hashes


def _expected_reused():
    return ({(m, r, b, s) for m in COMPARATORS for r in REGIMES for b in ALL_BLOCKS for s in SCENARIOS}
            | {(MAIN, r, b, s) for r in REGIMES for b in pilot.BLOCKS for s in SCENARIOS})


def selected_reused():
    """Check 120 original baselines plus all 16 completed candidate pilot runs."""
    wanted = _expected_reused()
    found = {}
    for source, location, reader in (('original', original.STUDY, original.read_result),
                                     ('pilot', pilot.STUDY, pilot.read_result)):
        for path in sorted((location / 'jobs').glob('*/result.json')):
            raw = json.loads(path.read_text())
            identity = key(raw['job'])
            if identity not in wanted:
                continue
            if (source == 'original' and identity[0] not in COMPARATORS) or (
                    source == 'pilot' and identity[0] != MAIN):
                continue
            if identity in found:
                raise ValueError('Duplicate reused comparison record.')
            record = reader(path, location)
            found[identity] = {'origin': source, 'result_path': str(path.relative_to(ROOT)),
                               'result_sha256': file_sha256(path), 'record': record}
    if set(found) != wanted or len(found) != 136:
        raise ValueError('Missing exact matched reused runs.')
    return [found[k] for k in sorted(found)]


def protocol():
    return {
        'candidate': MAIN, 'new_blocks': list(BLOCKS), 'combined_blocks': list(ALL_BLOCKS),
        'regimes': list(REGIMES), 'scenarios': list(SCENARIOS), 'comparators': list(COMPARATORS),
        'new_runs': 24, 'reused_candidate_pilot_runs': 16, 'reused_original_baseline_runs': 120,
        'reused_runs': 136, 'combined_runs': 160, 'jobs': jobs(),
        'authorization': 'Explicit user authorization for this separate 24-run exploratory follow-up after viewing the completed pilot. This is an exception to the pilot stopping rule, not a retroactive pilot pass or automatic advancement.',
        'prior_pilot_advancement_met': False, 'automatic_continuation_authorized': False,
        'mechanism': pilot.protocol()['mechanism'],
        'candidate_change': 'None: the exact previously frozen v0.5 implementation and settings are reused.',
        'selection': 'All remaining registered blocks 3, 4 and 5 in original order, all four original pilot scenarios, both regimes. No outcome-based substitutions or seed changes.',
        'evidence_scope': 'Exploratory development on known MNIST data and previously inspected original block outcomes. Adding the remaining blocks is broader development evidence, not untouched confirmatory validation.',
        'measurement': pilot.protocol()['measurement'],
        'uncertainty': 'Keep IID and non-IID separate. Report all five paired block values, paired differences, means, standard deviations and nominal two-sided 95% paired-t confidence intervals with four degrees of freedom. Small n=5, known development evidence and multiple comparisons prohibit claims of universal superiority or established novelty.',
        'criteria': 'Report the unchanged original practical margins and focused-fix checks over the combined five-block panel as descriptive checks. Preserve the two-block pilot decision separately. No automatic further study is authorized by any follow-up result.',
        'execution': {'workers': 2, 'threads_per_worker': 4, 'training_helper': 'unchanged v05_pilot.execute_run'},
        'stopping': 'Exactly 24 new runs. Preserve completed jobs and failure evidence. A scientific failure stops scheduling; investigate before a documented same-job retry. No parameter search, replacement seeds, extra scenarios, or further experiments.',
        'data_access': pilot.protocol()['data_access'], 'privacy_scope': pilot.protocol()['privacy_scope'],
        'limits': pilot.protocol()['limits'],
    }


def freeze(study=STUDY, extra_sources=()):
    """Explicit preparation only; does not run training."""
    study = Path(study)
    study.mkdir(parents=True, exist_ok=True)
    with (study / 'preparation.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (study / 'execution_freeze.json').exists():
            return verify(study)
        previous = pilot.verify(pilot.STUDY)
        old = original.verify(original.STUDY)
        sources = source_hashes(extra_sources)
        prior = prior_evidence_hashes()
        preflight = json.loads((study / 'preflight.json').read_text())
        if preflight.get('status') != 'passed' or preflight.get('source_sha256') != sources:
            raise ValueError('A passing preflight for these exact source hashes is required.')
        if pilot.candidate().settings != previous['candidate_settings']:
            raise ValueError('The follow-up must reuse the unchanged frozen candidate.')
        reused = selected_reused()
        write_json(study / 'protocol.json', protocol())
        write_json(study / 'reused_manifest.json', {'records': reused,
            'original_execution_freeze_sha256': file_sha256(original.STUDY / 'execution_freeze.json'),
            'prior_pilot_execution_freeze_sha256': file_sha256(pilot.STUDY / 'execution_freeze.json')})
        for name, digest in sources.items():
            destination = study / 'source_snapshot' / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / 'code' / name, destination)
            if file_sha256(destination) != digest:
                raise ValueError('Follow-up source snapshot mismatch.')
        frozen = {'schema_version': 1, 'created_at': stamp(), 'jobs': jobs(),
            'candidate_settings': pilot.candidate().settings, 'source_sha256': sources,
            'new_sources': list(dict.fromkeys((*REQUIRED_NEW_SOURCES, *extra_sources))),
            'environment': original.environment(), 'prior_evidence_sha256': prior,
            'original_execution_freeze_sha256': file_sha256(original.STUDY / 'execution_freeze.json'),
            'prior_pilot_execution_freeze_sha256': file_sha256(pilot.STUDY / 'execution_freeze.json'),
            'prior_pilot_decision_sha256': file_sha256(pilot.STUDY / 'analysis/decision.json'),
            'prior_pilot_analysis_manifest_sha256': file_sha256(pilot.STUDY / 'analysis_manifest.json'),
            'prior_pilot_completion_verification_sha256': file_sha256(pilot.STUDY / 'completion_verification.json'),
            'protocol_sha256': file_sha256(study / 'protocol.json'),
            'reused_manifest_sha256': file_sha256(study / 'reused_manifest.json'),
            'preflight_sha256': file_sha256(study / 'preflight.json'),
            'partitions': [p for p in old['partitions'] if p['block'] in ALL_BLOCKS]}
        write_json(study / 'execution_freeze.json', frozen)
        write_json(study / 'freeze_digest.json', {'sha256': file_sha256(study / 'execution_freeze.json')})
        write_json(study / 'status.json', {'status': 'frozen_not_started', 'completed_new_runs': 0,
            'planned_new_runs': 24, 'reused_runs': 136, 'combined_runs': 160, 'at': stamp()})
    return verify(study)


def verify(study=STUDY, check_reused=True):
    study = Path(study)
    previous = pilot.verify(pilot.STUDY)
    old = original.verify(original.STUDY)
    frozen = json.loads((study / 'execution_freeze.json').read_text())
    if file_sha256(study / 'execution_freeze.json') != json.loads((study / 'freeze_digest.json').read_text())['sha256']:
        raise ValueError('Follow-up freeze changed.')
    checks = {'original_execution_freeze_sha256': original.STUDY / 'execution_freeze.json',
              'prior_pilot_execution_freeze_sha256': pilot.STUDY / 'execution_freeze.json',
              'prior_pilot_decision_sha256': pilot.STUDY / 'analysis/decision.json',
              'prior_pilot_analysis_manifest_sha256': pilot.STUDY / 'analysis_manifest.json',
              'prior_pilot_completion_verification_sha256': pilot.STUDY / 'completion_verification.json'}
    for field, path in checks.items():
        if file_sha256(path) != frozen[field]:
            raise ValueError('Pinned prior evidence changed: ' + field)
    if frozen['prior_evidence_sha256'] != prior_evidence_hashes():
        raise ValueError('Prior pilot evidence changed.')
    if frozen['jobs'] != jobs() or frozen['environment'] != original.environment():
        raise ValueError('Follow-up job grid or environment changed.')
    if (frozen['candidate_settings'] != previous['candidate_settings'] or
            frozen['candidate_settings'] != pilot.candidate().settings):
        raise ValueError('Candidate settings changed.')
    if frozen['partitions'] != [p for p in old['partitions'] if p['block'] in ALL_BLOCKS]:
        raise ValueError('Matched partitions changed.')
    if set(frozen['source_sha256']) != set(previous['source_sha256']) | set(frozen['new_sources']):
        raise ValueError('Follow-up source coverage changed.')
    if not set(REQUIRED_NEW_SOURCES).issubset(frozen['new_sources']):
        raise ValueError('Required follow-up sources are missing.')
    if any(frozen['source_sha256'].get(n) != h for n, h in previous['source_sha256'].items()):
        raise ValueError('Prior 58 scientific sources were not preserved.')
    for name, digest in frozen['source_sha256'].items():
        if file_sha256(ROOT / 'code' / name) != digest or file_sha256(study / 'source_snapshot' / name) != digest:
            raise ValueError('Frozen source changed: ' + name)
    for name in ('protocol', 'reused_manifest', 'preflight'):
        if file_sha256(study / (name + '.json')) != frozen[name + '_sha256']:
            raise ValueError('Frozen follow-up evidence changed: ' + name)
    preflight = json.loads((study / 'preflight.json').read_text())
    if preflight.get('status') != 'passed' or preflight.get('source_sha256') != frozen['source_sha256']:
        raise ValueError('Frozen preflight did not pass for these sources.')
    if json.loads((study / 'protocol.json').read_text()) != protocol():
        raise ValueError('Follow-up protocol and implementation disagree.')
    if check_reused:
        reused_records(study)
    return frozen


def reused_records(study=STUDY):
    manifest = json.loads((Path(study) / 'reused_manifest.json').read_text())
    if (manifest['original_execution_freeze_sha256'] != file_sha256(original.STUDY / 'execution_freeze.json') or
            manifest['prior_pilot_execution_freeze_sha256'] != file_sha256(pilot.STUDY / 'execution_freeze.json')):
        raise ValueError('Reused manifest provenance changed.')
    entries, found, output = manifest['records'], set(), []
    for entry in entries:
        path = ROOT / entry['result_path']
        if file_sha256(path) != entry['result_sha256'] or json.loads(path.read_text()) != entry['record']:
            raise ValueError('Reused result changed.')
        identity = key(entry['record']['job'])
        origin = 'pilot' if identity[0] == MAIN else 'original'
        location, reader = (pilot.STUDY, pilot.read_result) if origin == 'pilot' else (original.STUDY, original.read_result)
        if entry['origin'] != origin or not path.resolve().is_relative_to((location / 'jobs').resolve()):
            raise ValueError('Invalid reused result origin.')
        record = reader(path, location)
        if identity in found:
            raise ValueError('Duplicate reused result.')
        found.add(identity)
        output.append(record)
    if found != _expected_reused() or len(entries) != 136:
        raise ValueError('Unexpected reused comparison grid.')
    return output


def read_result(path, study=STUDY):
    study = Path(study)
    result = json.loads(Path(path).read_text())
    checked_result(result)
    frozen = json.loads((study / 'execution_freeze.json').read_text())
    if result['job'] not in frozen['jobs'] or result['source_sha256'] != frozen['source_sha256']:
        raise ValueError('Unregistered follow-up result.')
    if result['job_id'] != job_id(result['job'], file_sha256(study / 'execution_freeze.json')):
        raise ValueError('Follow-up result identity mismatch.')
    if set(result['artifact_sha256']) != set(original.ARTIFACTS):
        raise ValueError('Follow-up artifact coverage changed.')
    return result


def records(study=STUDY):
    output = [read_result(p, study) for p in sorted((Path(study) / 'jobs').glob('*/result.json'))]
    if len({key(r['job']) for r in output}) != len(output):
        raise ValueError('Duplicate follow-up result.')
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
            result = pilot.execute_run(job, part['partition_id'], ident, frozen['source_sha256'])
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
            'planned_new_runs': 24, 'reused_runs': 136, 'combined_runs': 160, 'at': stamp()})
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
                        raise RuntimeError('Follow-up workers failed: ' + '; '.join(repr(e) for e in failures))
                    for _ in completed:
                        job = next(iterator, None)
                        if job is not None:
                            active.add(pool.submit(launch, job))
            completed = records(study)
            if {key(r['job']) for r in completed} != {key(j) for j in frozen['jobs']}:
                raise ValueError('Incomplete follow-up grid.')
            write_json(study / 'result_manifest.json', {'records': completed, 'execution_freeze_sha256': digest})
            from v05_followup_analysis import analyse
            result = analyse(study)
            write_json(study / 'status.json', {'status': 'complete', 'completed_new_runs': 24,
                'reused_runs': 136, 'combined_runs': 160, 'planned_new_runs': 24, 'at': stamp()})
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
    return {'retry_registered': ident, 'next_action': 'Resume the identical frozen follow-up.'}


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
        from v05_followup_analysis import analyse
        result = analyse(args.study)
    elif args.action == 'status':
        result = json.loads((args.study / 'status.json').read_text())
    elif args.action == 'retry':
        result = retry(args.study, args.job_id, args.reason)
    else:
        result = globals()[args.action](args.study)
    print(json.dumps(result if args.action in ('status', 'retry', 'analyse') else
                     {'action': args.action, 'completed': True}, indent=2))
