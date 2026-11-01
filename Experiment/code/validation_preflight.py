"""Synthetic checks and short old-seed integration runs before execution freeze."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import argparse
import json
import re
import subprocess
import sys
import time
import pandas as pd
from experiment_core import file_sha256, write_json
from research_runs import ROOT, PARTITIONS, canonical_hash
from research_analysis import cross_method_pair
from directional_study import checked_result
from validation_replication import STUDY, MAIN, plan, source_hashes, method_names, execute_run, stamp
from validation_analysis import reference_workload


def preflight_jobs():
    return [{'method': m, 'regime': g, 'scenario': s, 'seed': 20261014,
        'phase': 'preflight', 'panel': 'preflight', 'block': 0, 'rounds': 6,
        'train_images': 64, 'evaluation_images': 100, 'relative_rate': 1., 'root_steps': 8,
        'num_threads': 4} for g in ['IID', 'non-IID'] for m in method_names() for s in ['clean', 'label_flip']]


def worker(request, directory):
    j = json.loads(Path(request).read_text()); directory = Path(directory)
    if j not in preflight_jobs(): raise ValueError('Preflight permits only fixed old-seed checks.')
    context = json.loads((directory / 'context.json').read_text())
    sources = source_hashes()
    if sources != context['source_sha256']: raise ValueError('Sources changed during preflight.')
    ident = canonical_hash({'job': j, 'context_sha256': file_sha256(directory / 'context.json')})[:24]
    folder = directory / 'jobs' / ident; folder.mkdir(parents=True, exist_ok=True)
    write_json(folder / 'request.json', j)
    try:
        result = execute_run(j, PARTITIONS[j['regime']], ident, sources)
        write_json(folder / 'result.json', result)
    except BaseException as exc:
        write_json(folder / 'failure.json', {'error': repr(exc), 'run_path': getattr(exc, 'run_output_path', None)})
        raise


def preflight(study=STUDY):
    study = Path(study)
    if (study / 'execution_freeze.json').exists():
        raise ValueError('Do not replace preflight evidence after freezing.')
    if not (study / 'partitions.json').exists(): raise ValueError('Prepare partitions first.')
    sources = source_hashes()
    directory = study / 'readiness' / ('attempt_' + str(time.time_ns())); directory.mkdir(parents=True)
    write_json(directory / 'context.json', {'at': stamp(), 'source_sha256': sources,
        'scope': 'Integration checks only: old development seed20261014, original two partitions,6rounds,64training images/client,100evaluation images/task. Not part of800replication runs.'})
    write_json(study / 'preflight.json', {'status': 'running', 'source_sha256': sources, 'attempt': str(directory.relative_to(ROOT))})
    try:
        with (directory / 'unit_tests.log').open('w') as log:
            subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', str(ROOT / 'code/tests'), '-v'], stdout=log, stderr=subprocess.STDOUT, check=True)
        tests = (directory / 'unit_tests.log').read_text(); match = re.search(r'Ran (\d+) tests', tests)
        if not match or not tests.rstrip().endswith('OK'): raise ValueError('Preflight tests did not pass.')
        requests = directory / 'requests'; requests.mkdir()
        def launch(j):
            ident = canonical_hash(j)[:20]; request = requests / (ident + '.json'); write_json(request, j)
            with (requests / (ident + '.log')).open('w') as log:
                subprocess.run([sys.executable, str(Path(__file__).resolve()), 'worker', '--request', str(request), '--directory', str(directory)],
                               stdout=log, stderr=subprocess.STDOUT, check=True)
            return j
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(launch, j) for j in preflight_jobs()]
            try:
                for i, future in enumerate(as_completed(futures), 1):
                    j = future.result(); print(json.dumps({'preflight_completed': i, 'total': 48, 'method': j['method'], 'regime': j['regime'], 'scenario': j['scenario']}), flush=True)
            except BaseException:
                for future in futures: future.cancel()
                raise
        records = [checked_result(json.loads(p.read_text())) for p in sorted((directory / 'jobs').glob('*/result.json'))]
        indexed = {(r['job']['method'], r['job']['regime'], r['job']['scenario']): r for r in records}
        if len(records) != 48 or len(indexed) != 48: raise AssertionError('Incomplete preflight grid.')
        first_updates, prefix_count, max_error = 0, 0, 0.
        for r in records:
            j = r['job']; path = ROOT / r['run_path']
            if j not in preflight_jobs() or j['seed'] != 20261014: raise AssertionError('Unexpected preflight job.')
            base = indexed[('FedAvg', j['regime'], j['scenario'])]
            cross_method_pair(r, base)
            local = pd.read_csv(path / 'client_metrics.csv'); old = pd.read_csv(ROOT / base['run_path'] / 'client_metrics.csv')
            if len(local) != 60 or not local.examples.eq(64).all() or not local.optimizer_steps.eq(1).all():
                raise AssertionError('Preflight training workload changed.')
            cols = ['client', 'submitted_state_sha256', 'broadcast_state_sha256', 'shuffle_seed', 'optimizer_steps', 'online_train_loss']
            pd.testing.assert_frame_equal(local[local['round'] == 1][cols].reset_index(drop=True), old[old['round'] == 1][cols].reset_index(drop=True), check_exact=True)
            first_updates += 10
            info = json.loads((path / 'run.json').read_text())
            if info['official_test_used']: raise AssertionError('Preflight used official test data.')
            if j['scenario'] != 'clean':
                ref = ROOT / indexed[(j['method'], j['regime'], 'clean')]['run_path']
                h, hc = pd.read_csv(path / 'history.csv'), pd.read_csv(ref / 'history.csv')
                if h[h['round'] <= 4].global_state_sha256.tolist() != hc[hc['round'] <= 4].global_state_sha256.tolist():
                    raise AssertionError('Pre-event model divergence.')
                t, tc = pd.read_csv(path / 'task_metrics.csv'), pd.read_csv(ref / 'task_metrics.csv')
                pd.testing.assert_frame_equal(t[t['round'] <= 4].reset_index(drop=True), tc[tc['round'] <= 4].reset_index(drop=True), check_exact=True)
                prefix_count += 1
            for row in json.loads((path / 'aggregation_diagnostics.json').read_text()):
                v = row['values']; max_error = max(max_error, v['contribution_measurements']['reconstruction_max_abs_error'])
                forward, backward, steps = reference_workload(v)
                if j['method'] not in ('FLTrust', 'FLAURA-reconstruction', 'FLAURA-sigma-double'):
                    if forward or backward or steps:
                        raise AssertionError('Reference-free method performed reference evaluations.')
                elif j['method'] == 'FLTrust':
                    batch = info['method']['settings']['base_settings']['root_batch_size']
                    if (forward, backward, steps) != (8 * batch, 8 * batch, 8):
                        raise AssertionError('Root training was not counted correctly.')
                elif (forward, backward, steps) != (1000, 0, 0):
                    raise AssertionError('Auxiliary representation work changed.')
                if j['method'] == MAIN:
                    setting = info['method']['settings']
                    for k, expected in plan()['candidate_config'].items():
                        if setting[k] != expected: raise AssertionError('Candidate parameter changed.')
        if (first_updates, prefix_count) != (480, 24) or max_error > 1e-6:
            raise AssertionError('Incomplete preflight audit.')
        if source_hashes() != sources: raise AssertionError('Sources changed during preflight.')
        write_json(directory / 'result_manifest.json', {'records': records})
        artifacts = {str(p.relative_to(ROOT)): file_sha256(p) for p in directory.rglob('*') if p.is_file()}
        for r in records:
            artifacts.update({str((ROOT / r['run_path'] / name).relative_to(ROOT)): digest for name, digest in r['artifact_sha256'].items()})
        result = {'status': 'passed', 'source_sha256': sources, 'attempt': str(directory.relative_to(ROOT)),
            'unit_tests_passed': int(match.group(1)), 'old_seed_integration_runs': 48,
            'integration_methods': method_names(), 'exact_first_round_client_updates': first_updates,
            'exact_pre_event_pairs': prefix_count, 'matched_cross_method_pairs': 48,
            'audited_integration_rounds': 288, 'maximum_reconstruction_error': max_error,
            'reference_training_and_auxiliary_costs_checked': True,
            'fresh_replication_seed_runs': 0, 'official_test_used': False, 'artifact_sha256': artifacts}
        write_json(study / 'preflight.json', result)
        return result
    except BaseException as exc:
        write_json(study / 'preflight.json', {'status': 'failed', 'source_sha256': sources,
            'attempt': str(directory.relative_to(ROOT)), 'error': repr(exc)})
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('action', choices=['worker', 'run'])
    p.add_argument('--request', type=Path); p.add_argument('--directory', type=Path)
    p.add_argument('--study', type=Path, default=STUDY); args = p.parse_args()
    if args.action == 'worker': worker(args.request, args.directory)
    else: preflight(args.study)
