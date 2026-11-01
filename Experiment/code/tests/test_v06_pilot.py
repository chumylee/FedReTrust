"""Fixed stress-grid, immutable-source, locking and retry boundaries."""
import copy
import fcntl
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import v06_pilot as pilot


class ProtocolTests(unittest.TestCase):
    def test_exact_stress_grid_preserves_all_registered_training_inputs(self):
        jobs = pilot.jobs(); old = {pilot.key(j): j for j in pilot.original.jobs()}
        self.assertEqual(len(jobs), 16)
        self.assertEqual(len({pilot.key(j) for j in jobs}), 16)
        self.assertEqual([j['block'] for j in jobs], [3] * 8 + [5] * 8)
        for job in jobs:
            reference = old[(pilot.original.MAIN, job['regime'], job['block'], job['scenario'])]
            self.assertEqual({k: v for k, v in job.items() if k not in ('method', 'phase', 'panel')},
                {k: v for k, v in reference.items() if k not in ('method', 'phase', 'panel')})
            self.assertEqual(job['method'], 'FedReTrust-v06')
            self.assertEqual(job['phase'], 'v06_selected_stress_development')
            self.assertEqual(job['num_threads'], 4)

    def test_reused_and_new_grid_is_disjoint_exact_80_trajectories(self):
        reused = pilot._expected_reused(); new = {pilot.key(j) for j in pilot.jobs()}
        expected = {(m, r, b, s) for m in (*pilot.COMPARATORS, pilot.MAIN)
            for r in pilot.REGIMES for b in pilot.BLOCKS for s in pilot.SCENARIOS}
        self.assertEqual(len(reused), 64)
        self.assertFalse(reused & new)
        self.assertEqual(reused | new, expected)
        self.assertEqual(len(expected), 80)

    def test_protocol_declares_selection_no_ci_and_no_automatic_continuation(self):
        protocol = pilot.protocol()
        self.assertIn('deliberately chosen', protocol['selection'])
        self.assertIn('not representative', protocol['selection'])
        self.assertIn('No n=2 confidence intervals', protocol['uncertainty'])
        self.assertIn('History is not falsely attributed', protocol['mechanism']['coefficient_audit'])
        self.assertFalse(protocol['automatic_continuation_authorized'])
        self.assertEqual(protocol['execution']['workers'], 2)
        self.assertEqual(protocol['execution']['threads_per_worker'], 4)

    def test_source_map_preserves_62_and_disallows_replacement_or_path_escape(self):
        old = {'source_sha256': {f'old_{i}.py': f'hash{i}' for i in range(62)}}
        with patch.object(pilot.prior, 'verify', return_value=old), patch.object(pilot, 'file_sha256', return_value='new'):
            result = pilot.source_hashes()
            self.assertEqual(len(result), 62 + len(pilot.REQUIRED_NEW_SOURCES))
            self.assertTrue(all(result[k] == v for k, v in old['source_sha256'].items()))
            for name in ('/tmp/escape.py', '../escape.py', 'old_0.py'):
                with self.assertRaises(ValueError):
                    pilot.source_hashes((name,))
        old['source_sha256'].pop('old_0.py')
        with patch.object(pilot.prior, 'verify', return_value=old), self.assertRaises(ValueError):
            pilot.source_hashes()

    def test_candidate_identity_is_distinct_from_prior_even_with_same_input(self):
        job = pilot.jobs()[0]
        self.assertNotEqual(pilot.job_id(job, 'same'), pilot.prior.job_id(job, 'same'))
        self.assertNotEqual(pilot.job_id(job, 'a'), pilot.job_id(job, 'b'))


class ExecutionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.study = Path(self.temp.name)
        self.job = pilot.jobs()[0]
        self.frozen = {'jobs': [self.job], 'source_sha256': {'saved.py': 'saved'},
            'partitions': [{'regime': self.job['regime'], 'block': self.job['block'], 'partition_id': 'fixed'}]}
        patch.object(pilot, 'verify', return_value=self.frozen).start()
        patch.object(pilot, 'file_sha256', return_value='fixed-digest').start()
        self.ident = pilot.job_id(self.job, 'fixed-digest'); self.folder = self.study / 'jobs' / self.ident

    def tearDown(self):
        patch.stopall(); self.temp.cleanup()

    def test_worker_executes_declared_method_once_and_skips_complete_result(self):
        result = {'job': self.job, 'job_id': self.ident, 'status': 'complete'}
        with patch.object(pilot, 'execute_run', return_value=result) as execute:
            self.assertEqual(pilot.worker(self.job, self.study), result)
            execute.assert_called_once_with(self.job, 'fixed', self.ident, self.frozen['source_sha256'])
            with patch.object(pilot, 'read_result', return_value=result):
                self.assertEqual(pilot.worker(self.job, self.study), result)
            self.assertEqual(execute.call_count, 1)

    def test_changed_seed_is_rejected_before_training(self):
        changed = copy.deepcopy(self.job); changed['seed'] += 1
        with patch.object(pilot, 'execute_run') as execute, self.assertRaises(ValueError):
            pilot.worker(changed, self.study)
        execute.assert_not_called()

    def test_error_is_preserved_and_same_job_retry_requires_reason(self):
        with patch.object(pilot, 'execute_run', side_effect=RuntimeError('scientific failure')) as execute:
            with self.assertRaises(RuntimeError):
                pilot.worker(self.job, self.study)
            failure = self.folder / 'failure.json'; content = failure.read_bytes()
            with self.assertRaises(ValueError):
                pilot.worker(self.job, self.study)
            self.assertEqual(execute.call_count, 1)
            with self.assertRaises(ValueError):
                pilot.retry(self.study, self.ident, ' ')
            self.assertEqual(failure.read_bytes(), content)
            pilot.retry(self.study, self.ident, 'Investigated preserved exception; retry identical job.')
            archived = list((self.folder / 'failures').glob('*_failure.json'))
            self.assertEqual(len(archived), 1); self.assertEqual(archived[0].read_bytes(), content)
            self.assertFalse(failure.exists())

    def test_worker_and_coordinator_locks_prevent_duplicate_training(self):
        self.folder.mkdir(parents=True)
        with (self.folder / 'worker.lock').open('a') as lock, patch.object(pilot, 'execute_run') as execute:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaises(BlockingIOError):
                pilot.worker(self.job, self.study)
            execute.assert_not_called()
        with (self.study / 'coordinator.lock').open('a') as lock, patch.object(pilot.subprocess, 'run') as launch:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaises(BlockingIOError):
                pilot.run(self.study)
            launch.assert_not_called()

    def test_completed_grid_builds_analysis_without_retraining(self):
        self.frozen['jobs'] = pilot.jobs()
        saved = [{'job': j, 'status': 'complete'} for j in self.frozen['jobs']]
        analysis = types.ModuleType('v06_pilot_analysis'); analysis.analyse = lambda study: {'done': True}
        with patch.object(pilot, 'records', return_value=saved), patch.object(pilot.subprocess, 'run') as launch, patch.dict(sys.modules, {'v06_pilot_analysis': analysis}):
            self.assertEqual(pilot.run(self.study), {'done': True}); launch.assert_not_called()
        status = json.loads((self.study / 'status.json').read_text())
        self.assertEqual(status['status'], 'complete'); self.assertEqual(status['completed_new_runs'], 16)
        self.assertEqual(status['combined_runs'], 80)

    def test_preserved_failure_blocks_coordinator(self):
        self.folder.mkdir(parents=True); (self.folder / 'failure.json').write_text('{}')
        with patch.object(pilot, 'records', return_value=[]), patch.object(pilot.subprocess, 'run') as launch, self.assertRaises(ValueError):
            pilot.run(self.study)
        launch.assert_not_called()


if __name__ == '__main__':
    unittest.main()
