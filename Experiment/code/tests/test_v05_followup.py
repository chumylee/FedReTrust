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
import v05_followup as followup


class FollowupProtocolChecks(unittest.TestCase):
    def test_exact_remaining_grid_preserves_registered_training_inputs(self):
        jobs = followup.jobs()
        old = {followup.key(j): j for j in followup.original.jobs()}
        self.assertEqual(len(jobs), 24)
        self.assertEqual(len({followup.key(j) for j in jobs}), 24)
        self.assertEqual([j['block'] for j in jobs], [3] * 8 + [4] * 8 + [5] * 8)
        self.assertEqual({j['num_threads'] for j in jobs}, {4})
        for job in jobs:
            previous = old[(followup.original.MAIN, job['regime'], job['block'], job['scenario'])]
            self.assertEqual({k: v for k, v in job.items() if k not in ('method', 'phase', 'panel')},
                             {k: v for k, v in previous.items() if k not in ('method', 'phase', 'panel')})
            self.assertEqual(job['method'], followup.MAIN)
            self.assertEqual(job['phase'], 'v05_exploratory_followup')

    def test_reused_and_new_keys_cover_exact_160_panel_without_overlap(self):
        old = followup._expected_reused()
        new = {followup.key(j) for j in followup.jobs()}
        full = {(m, r, b, s) for m in (*followup.COMPARATORS, followup.MAIN)
                for r in followup.REGIMES for b in followup.ALL_BLOCKS for s in followup.SCENARIOS}
        self.assertEqual(len(old), 136)
        self.assertFalse(old & new)
        self.assertEqual(old | new, full)
        self.assertEqual(len(full), 160)

    def test_protocol_preserves_failed_pilot_and_no_automatic_advancement(self):
        p = followup.protocol()
        self.assertFalse(p['prior_pilot_advancement_met'])
        self.assertFalse(p['automatic_continuation_authorized'])
        self.assertIn('exception', p['authorization'])
        self.assertIn('known MNIST', p['evidence_scope'])
        self.assertEqual(p['new_runs'], 24)
        self.assertEqual(p['reused_runs'], 136)
        self.assertEqual(p['combined_runs'], 160)
        self.assertEqual(p['execution']['workers'], 2)
        self.assertEqual(p['execution']['threads_per_worker'], 4)
        self.assertEqual(p['mechanism'], followup.pilot.protocol()['mechanism'])

    def test_source_map_requires_prior_58_and_blocks_source_replacement(self):
        previous = {'source_sha256': {f'old_{i}.py': f'hash{i}' for i in range(58)}}
        with patch.object(followup.pilot, 'verify', return_value=previous), \
                patch.object(followup, 'file_sha256', return_value='new'):
            sources = followup.source_hashes()
            self.assertEqual(len(sources), 62)
            for name, digest in previous['source_sha256'].items():
                self.assertEqual(sources[name], digest)
            for extra in ('/tmp/escape.py', '../escape.py', 'old_0.py'):
                with self.assertRaises(ValueError):
                    followup.source_hashes((extra,))
        previous['source_sha256'].pop('old_0.py')
        with patch.object(followup.pilot, 'verify', return_value=previous):
            with self.assertRaises(ValueError):
                followup.source_hashes()

    def test_followup_identity_cannot_collide_with_pilot_identity(self):
        job = followup.jobs()[0]
        self.assertNotEqual(followup.job_id(job, 'same'), followup.pilot.job_id(job, 'same'))
        self.assertNotEqual(followup.job_id(job, 'first'), followup.job_id(job, 'second'))


class FollowupExecutionChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.study = Path(self.temp.name)
        self.job = followup.jobs()[0]
        self.frozen = {'jobs': [self.job], 'source_sha256': {'saved.py': 'saved-hash'},
                       'partitions': [{'regime': self.job['regime'], 'block': self.job['block'],
                                       'partition_id': 'fixed-partition'}]}
        self.verify = patch.object(followup, 'verify', return_value=self.frozen).start()
        patch.object(followup, 'file_sha256', return_value='fixed-digest').start()
        self.ident = followup.job_id(self.job, 'fixed-digest')
        self.folder = self.study / 'jobs' / self.ident

    def tearDown(self):
        patch.stopall()
        self.temp.cleanup()

    def test_worker_calls_unchanged_helper_and_skips_completed_job(self):
        result = {'job': self.job, 'job_id': self.ident, 'status': 'complete'}
        with patch.object(followup.pilot, 'execute_run', return_value=result) as execute:
            self.assertEqual(followup.worker(self.job, self.study), result)
            execute.assert_called_once_with(self.job, 'fixed-partition', self.ident, self.frozen['source_sha256'])
            with patch.object(followup, 'read_result', return_value=result):
                self.assertEqual(followup.worker(self.job, self.study), result)
            self.assertEqual(execute.call_count, 1)

    def test_worker_rejects_undeclared_job_before_training(self):
        other = copy.deepcopy(self.job)
        other['seed'] += 1
        with patch.object(followup.pilot, 'execute_run') as execute:
            with self.assertRaises(ValueError):
                followup.worker(other, self.study)
            execute.assert_not_called()

    def test_worker_preserves_failure_and_documented_retry_archives_it(self):
        with patch.object(followup.pilot, 'execute_run', side_effect=RuntimeError('scientific failure')) as execute:
            with self.assertRaises(RuntimeError):
                followup.worker(self.job, self.study)
            failure = self.folder / 'failure.json'
            original_failure = failure.read_bytes()
            with self.assertRaises(ValueError):
                followup.worker(self.job, self.study)
            self.assertEqual(execute.call_count, 1)
            self.assertEqual(failure.read_bytes(), original_failure)
            with self.assertRaises(ValueError):
                followup.retry(self.study, self.ident, ' ')
            self.assertTrue(failure.exists())
            result = followup.retry(self.study, self.ident, 'Investigated the preserved exception; retry identical job.')
            self.assertEqual(result['retry_registered'], self.ident)
            archived = list((self.folder / 'failures').glob('*_failure.json'))
            self.assertEqual(len(archived), 1)
            self.assertEqual(archived[0].read_bytes(), original_failure)
            decisions = list((self.folder / 'failures').glob('*_decision.json'))
            self.assertEqual(len(decisions), 1)
            self.assertEqual(json.loads(decisions[0].read_text())['job_id'], self.ident)
            self.assertFalse(failure.exists())

    def test_worker_and_coordinator_locks_prevent_duplicates(self):
        self.folder.mkdir(parents=True)
        with (self.folder / 'worker.lock').open('a') as lock, patch.object(followup.pilot, 'execute_run') as execute:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaises(BlockingIOError):
                followup.worker(self.job, self.study)
            execute.assert_not_called()
        with (self.study / 'coordinator.lock').open('a') as lock, patch.object(followup.subprocess, 'run') as launch:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaises(BlockingIOError):
                followup.run(self.study)
            launch.assert_not_called()

    def test_completed_grid_builds_analysis_without_new_training(self):
        self.frozen['jobs'] = followup.jobs()
        saved = [{'job': job, 'status': 'complete'} for job in self.frozen['jobs']]
        analysis = types.ModuleType('v05_followup_analysis')
        analysis.analyse = lambda study: {'analysis_complete': True}
        with patch.object(followup, 'records', return_value=saved), \
                patch.object(followup.subprocess, 'run') as launch, \
                patch.dict(sys.modules, {'v05_followup_analysis': analysis}):
            self.assertEqual(followup.run(self.study), {'analysis_complete': True})
            launch.assert_not_called()
        status = json.loads((self.study / 'status.json').read_text())
        self.assertEqual(status['status'], 'complete')
        self.assertEqual(status['completed_new_runs'], 24)
        self.assertEqual(len(json.loads((self.study / 'result_manifest.json').read_text())['records']), 24)

    def test_preserved_failure_blocks_coordinator_before_subprocess(self):
        self.folder.mkdir(parents=True)
        (self.folder / 'failure.json').write_text('{}')
        with patch.object(followup, 'records', return_value=[]), \
                patch.object(followup.subprocess, 'run') as launch:
            with self.assertRaises(ValueError):
                followup.run(self.study)
            launch.assert_not_called()


if __name__ == '__main__':
    unittest.main()
