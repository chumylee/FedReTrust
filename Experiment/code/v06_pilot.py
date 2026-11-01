"""Fixed v0.6 screen on selected, already known development stress cases."""
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

STUDY = ROOT / 'outputs/reference_free/20261031_v06'
MAIN = 'FedReTrust-v06'
BLOCKS = (3, 5)
REGIMES = prior.REGIMES
SCENARIOS = prior.SCENARIOS
COMPARATORS = ('FedReTrust-v05', original.MAIN, 'RFA', 'FedAvg')
REQUIRED_NEW_SOURCES = ('fedretrust_v06.py', 'v06_recording.py', 'tests/test_fedretrust_v06.py',
    'v06_pilot.py', 'v06_pilot_analysis.py', 'tests/test_v06_pilot.py', 'tests/test_v06_pilot_analysis.py')
stamp = prior.stamp
key = prior.key


def jobs():
    registered = {key(j): j for j in original.jobs()}
    return [{**registered[(original.MAIN, regime, block, scenario)],
             'method': MAIN, 'phase': 'v06_selected_stress_development', 'panel': 'candidate'}
            for block in BLOCKS for regime in REGIMES for scenario in SCENARIOS]


def job_id(job, digest):
    return canonical_hash({'job': job, 'v06_pilot_freeze_sha256': digest})[:24]


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
    previous = prior.verify(prior.STUDY, check_reused=False)
    if len(previous['source_sha256']) != 62:
        raise ValueError('Expected the 62 immutable prior scientific sources.')
    names = tuple(dict.fromkeys((*REQUIRED_NEW_SOURCES, *extra_sources)))
    for name in names:
        if Path(name).is_absolute() or '..' in Path(name).parts or name in previous['source_sha256']:
            raise ValueError('New source paths must be new files within Experiment/code.')
    return {**previous['source_sha256'], **{n: file_sha256(ROOT / 'code' / n) for n in names}}


def prior_evidence_hashes():
    hashes = prior.prior_evidence_hashes()
    if json.loads((prior.STUDY / 'status.json').read_text())['status'] != 'complete':
        raise ValueError('The v0.5 follow-up is not complete.')
    completion = json.loads((prior.STUDY / 'completion_verification.json').read_text())
    decision = json.loads((prior.STUDY / 'analysis/decision.json').read_text())
    if completion.get('status') != 'complete_verified' or completion['decision'] != decision:
        raise ValueError('The v0.5 follow-up review changed or is missing.')
    if decision.get('five_block_all_point_targets_met') is not False or decision.get('pilot_advancement_point_criterion_met') is not False:
        raise ValueError('Preserve the actual prior failed point targets and pilot gate.')
    manifest = json.loads((prior.STUDY / 'analysis_manifest.json').read_text())
    if manifest['execution_freeze_sha256'] != file_sha256(prior.STUDY / 'execution_freeze.json') or completion['analysis_manifest_sha256'] != file_sha256(prior.STUDY / 'analysis_manifest.json'):
        raise ValueError('Prior follow-up analysis provenance changed.')
    names = ('execution_freeze.json', 'freeze_digest.json', 'protocol.json', 'reused_manifest.json',
        'result_manifest.json', 'analysis_manifest.json', 'analysis/decision.json', 'completion_verification.json', 'status.json')
    hashes.update({str((prior.STUDY / n).relative_to(ROOT)): file_sha256(prior.STUDY / n) for n in names})
    for name, expected in manifest['artifact_sha256'].items():
        if Path(name).is_absolute() or '..' in Path(name).parts:
            raise ValueError('Invalid prior artifact path.')
        path = prior.STUDY / name
        if file_sha256(path) != expected:
            raise ValueError('Prior analysis artifact changed.')
        hashes[str(path.relative_to(ROOT))] = expected
    return hashes


def _expected_reused():
    return {(m, r, b, s) for m in COMPARATORS for r in REGIMES for b in BLOCKS for s in SCENARIOS}


def selected_reused():
    wanted, found = _expected_reused(), {}
    for origin, location, reader in (('original', original.STUDY, original.read_result),
                                     ('v05_followup', prior.STUDY, prior.read_result)):
        for path in sorted((location / 'jobs').glob('*/result.json')):
            raw = json.loads(path.read_text()); identity = key(raw['job'])
            if identity not in wanted:
                continue
            if (origin == 'original') != (identity[0] != prior.MAIN):
                continue
            if identity in found:
                raise ValueError('Duplicate comparison record.')
            record = reader(path, location)
            found[identity] = {'origin': origin, 'result_path': str(path.relative_to(ROOT)),
                'result_sha256': file_sha256(path), 'record': record}
    if set(found) != wanted or len(found) != 64:
        raise ValueError('Missing matched reused stress-case runs.')
    return [found[k] for k in sorted(found)]


def protocol():
    return {'candidate': MAIN, 'blocks': list(BLOCKS), 'regimes': list(REGIMES),
        'scenarios': list(SCENARIOS), 'comparators': list(COMPARATORS), 'jobs': jobs(),
        'new_runs': 16, 'reused_runs': 64, 'combined_runs': 80,
        'mechanism': {'history_initialization': 'Per-client vector h=0; stable authenticated client IDs; no scenario-dependent or regime-dependent rule.',
            'filter': 'innovation=raw_delta-h; radius=2*current lower-median innovation norm; y=h+clip_by_radius(innovation); an effectively zero median bypasses clipping and is logged.',
            'aggregate': 'Apply the unchanged v0.5 rule to double-precision filtered deltas y.',
            'history_update': 'Set each participating h to y only after successful aggregation; preserve absent-client history; reset all state between runs.',
            'coefficient_audit': 'Current raw coefficients and retained-history residual are audited separately. History is not falsely attributed to current submitted updates.',
            'fixed_choice': 'One fixed candidate, radius multiplier2; no threshold or variant sweep.'},
        'selection': 'Blocks3 and5 were deliberately chosen from already inspected development results: heterogeneity/honest-noise weakness and all-trust-one attack-onset collapse. They are selected stress cases, not representative random replication or fresh confirmation.',
        'evidence_scope': 'Exploratory development on known MNIST conditions. Keep both regimes separate; the same candidate rule can affect IID as well as non-IID.',
        'measurement': 'Round5-24 means; honest drift uses affected clients legitimate task. Mean and round5 attack damage use each method own matched clean trajectory. Also report absolute attacked accuracy, raw4-to5 drop, minimum active accuracy and largest active one-round drop.',
        'uncertainty': 'Report both selected block values, paired differences, means and observed ranges. No n=2 confidence intervals, significance, representative-population or novelty claim.',
        'practical_targets': 'Original10 mean checks: each regime clean/noise/stripe>=FedAvg-1pp; persistent label damage<=RFA+1pp and<=FedAvg-2pp.',
        'IID_retention': 'For each selected IID block, clean/noise/stripe and absolute attacked accuracy>=v0.5-0.5pp; mean-window attack damage<=v0.5+0.5pp.',
        'nonIID_focus': 'Selected-block mean label damage improves at least1pp versusv0.5; absolute mean-window attacked accuracy improves strictly in both blocks. Block5 own-clean-relative round5 damage decreases at least5pp AND absolute round5 attacked accuracy improves at least5pp; block3 does not worsen either onset metric.',
        'criteria_interpretation': 'Complete selected-screen point success requires every practical, IID-retention and focused check. Partial progress is reported separately. All margins, including5pp onset hurdle, are declared development choices rather than established standards.',
        'execution': {'workers': 2, 'threads_per_worker': 4, 'engine': 'unchanged TrainConfig/load_data/run_experiment and ClasswiseEvaluator; new InnovationContributionRecorder audits current plus history contributions'},
        'data_access': 'No trusted server images, evaluation metrics, client validation reports, scenario labels, attacker identities or regime flag enter the defence. Simulator truth and evaluation data are offline reporting only.',
        'privacy_scope': 'Individual submitted updates, counts and stable IDs remain visible to the server; per-client history adds state. No secure aggregation, differential privacy or Sybil protection is implemented.',
        'prior_evidence': 'The failed v0.5 pilot gate and follow-up targets remain immutable; this is a separately authorized new candidate, never a retroactive pass.',
        'stopping': 'Exactly16 new jobs. No automatic continuation, outcome-based replacements, parameter tuning, extra scenarios or experiments. Preserve errors; investigate before documented identical-job retry.',
        'automatic_continuation_authorized': False,
        'limits': 'No recovery/recurring-attack, mixed-event, partial-participation, new-dataset or adaptive-adversary conclusions. Filtering honest changes is a risk tested explicitly by noise/stripe jobs.'}


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
            'original_execution_freeze_sha256': file_sha256(original.STUDY / 'execution_freeze.json'),
            'prior_followup_execution_freeze_sha256': file_sha256(prior.STUDY / 'execution_freeze.json')})
        for name, digest in sources.items():
            destination = study / 'source_snapshot' / name; destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / 'code' / name, destination)
            if file_sha256(destination) != digest:
                raise ValueError('Source snapshot mismatch.')
        frozen = {'schema_version': 1, 'created_at': stamp(), 'jobs': jobs(),
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
            'partitions': [p for p in old['partitions'] if p['block'] in BLOCKS]}
        write_json(study / 'execution_freeze.json', frozen)
        write_json(study / 'freeze_digest.json', {'sha256': file_sha256(study / 'execution_freeze.json')})
        write_json(study / 'status.json', {'status': 'frozen_not_started', 'completed_new_runs': 0,
            'planned_new_runs': 16, 'reused_runs': 64, 'combined_runs': 80, 'at': stamp()})
    return verify(study)


def verify(study=STUDY, check_reused=True):
    study = Path(study); previous = prior.verify(prior.STUDY, check_reused=False)
    old = original.verify(original.STUDY); frozen = json.loads((study / 'execution_freeze.json').read_text())
    if file_sha256(study / 'execution_freeze.json') != json.loads((study / 'freeze_digest.json').read_text())['sha256']:
        raise ValueError('V0.6 freeze changed.')
    checks = {'original_execution_freeze_sha256': original.STUDY / 'execution_freeze.json',
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
    if frozen['partitions'] != [p for p in old['partitions'] if p['block'] in BLOCKS]:
        raise ValueError('Matched partitions changed.')
    if set(frozen['source_sha256']) != set(previous['source_sha256']) | set(frozen['new_sources']) or not set(REQUIRED_NEW_SOURCES).issubset(frozen['new_sources']):
        raise ValueError('Frozen source coverage changed.')
    if any(frozen['source_sha256'].get(n) != h for n, h in previous['source_sha256'].items()):
        raise ValueError('The62 prior scientific sources were not preserved.')
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
    for field, location in [('original_execution_freeze_sha256', original.STUDY), ('prior_followup_execution_freeze_sha256', prior.STUDY)]:
        if manifest[field] != file_sha256(location / 'execution_freeze.json'):
            raise ValueError('Reused manifest provenance changed.')
    entries, found, output = manifest['records'], set(), []
    for entry in entries:
        path = ROOT / entry['result_path']; identity = key(entry['record']['job'])
        if file_sha256(path) != entry['result_sha256'] or json.loads(path.read_text()) != entry['record']:
            raise ValueError('Reused result changed.')
        origin = 'v05_followup' if identity[0] == prior.MAIN else 'original'
        location, reader = (prior.STUDY, prior.read_result) if origin == 'v05_followup' else (original.STUDY, original.read_result)
        if entry['origin'] != origin or not path.resolve().is_relative_to((location / 'jobs').resolve()):
            raise ValueError('Invalid reused result origin.')
        record = reader(path, location)
        if identity in found:
            raise ValueError('Duplicate reused result.')
        found.add(identity); output.append(record)
    if found != _expected_reused() or len(entries) != 64:
        raise ValueError('Unexpected reused comparison grid.')
    return output


def execute_run(job, partition_id, digest, sources):
    started = time.perf_counter()
    cfg = TrainConfig(partition_id=partition_id, development_seed=job['seed'], rounds=job['rounds'],
        max_train_per_client=job['train_images'], max_validation=job['evaluation_images'], num_threads=4)
    data = load_data(ROOT, cfg)
    run_path = run_experiment(data, cfg, longer_scenarios()[job['scenario']], recorded_method(),
        notebook_code_sha256=digest, verbose=False, evaluator=ClasswiseEvaluator())
    counts = {i: np.bincount(data.pool.targets[ix].numpy(), minlength=10).tolist()
              for i, ix in data.manifest['indices']['clients'].items()}
    write_json(run_path / 'population_class_counts.json', counts)
    write_json(run_path / 'validation_access.json', {'official_test_used': False,
        'defence_uses_evaluation_data': False, 'server_reference_examples': 0,
        'aggregation_call_inputs': ['submitted_parameters_and_counts', 'broadcast', 'round'],
        'additional_constructor_access': 'scalar_configuration_only', 'phase': job['phase'],
        'partition_id': partition_id, 'calibration_runs': 0})
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {'job_id': digest, 'job': job, 'status': 'complete', 'run_path': str(run_path.relative_to(ROOT)),
        'source_sha256': sources, 'artifact_sha256': {n: file_sha256(run_path / n) for n in original.ARTIFACTS},
        'artifact_bytes': sum((run_path / n).stat().st_size for n in original.ARTIFACTS),
        'worker_wall_seconds': time.perf_counter() - started,
        'process_lifetime_peak_rss_bytes': int(rss if platform.system() == 'Darwin' else rss * 1024)}


def read_result(path, study=STUDY):
    study = Path(study)
    result = json.loads(Path(path).read_text())
    checked_result(result)
    frozen = json.loads((study / 'execution_freeze.json').read_text())
    if result['job'] not in frozen['jobs'] or result['source_sha256'] != frozen['source_sha256']:
        raise ValueError('Unregistered v0.6 pilot result.')
    if result['job_id'] != job_id(result['job'], file_sha256(study / 'execution_freeze.json')):
        raise ValueError('V0.6 pilot result identity mismatch.')
    if set(result['artifact_sha256']) != set(original.ARTIFACTS):
        raise ValueError('V0.6 pilot artifact coverage changed.')
    return result


def records(study=STUDY):
    output = [read_result(p, study) for p in sorted((Path(study) / 'jobs').glob('*/result.json'))]
    if len({key(r['job']) for r in output}) != len(output):
        raise ValueError('Duplicate v0.6 pilot result.')
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
            'planned_new_runs': 16, 'reused_runs': 64, 'combined_runs': 80, 'at': stamp()})
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
                        progress = {'completed': len(have), 'total': 16, 'job': job, 'at': stamp()}
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
                        raise RuntimeError('V0.6 pilot workers failed: ' + '; '.join(repr(e) for e in failures))
                    for _ in completed:
                        job = next(iterator, None)
                        if job is not None:
                            active.add(pool.submit(launch, job))
            completed = records(study)
            if {key(r['job']) for r in completed} != {key(j) for j in frozen['jobs']}:
                raise ValueError('Incomplete v0.6 pilot grid.')
            write_json(study / 'result_manifest.json', {'records': completed, 'execution_freeze_sha256': digest})
            from v06_pilot_analysis import analyse
            result = analyse(study)
            write_json(study / 'status.json', {'status': 'complete', 'completed_new_runs': 16,
                'reused_runs': 64, 'combined_runs': 80, 'planned_new_runs': 16, 'at': stamp()})
            return result
        except BaseException as exc:
            write_json(study / 'status.json', {'status': 'stopped_on_error', 'error': repr(exc),
                'completed_new_runs': len(records(study)), 'planned_new_runs': 16, 'at': stamp()})
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
        from v06_pilot_analysis import analyse
        result = analyse(args.study)
    elif args.action == 'status':
        result = json.loads((args.study / 'status.json').read_text())
    elif args.action == 'retry':
        result = retry(args.study, args.job_id, args.reason)
    else:
        result = globals()[args.action](args.study)
    print(json.dumps(result if args.action in ('status', 'retry', 'analyse') else
                     {'action': args.action, 'completed': True}, indent=2))
