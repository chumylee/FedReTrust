import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiment_core import ClientSubmission, TrainConfig, load_data
from fedretrust_directional import DirectionalFedReTrust, DirectionalConfig
from rfa import RFAAggregator
from scenario_metrics import ContributionRecorder
from validation_controls import CONTROLS, NO_MEMORY, NO_CLIP, CAPPED_RFA, make_control
from validation_partitions import build_manifest, validate_manifest
from validation_analysis import MAIN, PRIMARY_SPECS, summarize, paired_tables, practical_checks, reference_workload
import validation_replication as runner


def clients(values, counts=None):
    counts = counts or [10] * len(values)
    return [ClientSubmission(i, counts[i], {'x': torch.tensor(v, dtype=torch.float64)}) for i, v in enumerate(values)]


class ControlTests(unittest.TestCase):
    def test_no_restoration_matches_existing_scalar_switch(self):
        candidate = DirectionalFedReTrust(DirectionalConfig(use_restoration=False)); control = make_control(NO_MEMORY)
        b = {'x': torch.zeros(2, dtype=torch.float64)}
        for t, values in enumerate([[[1., .1]] * 7 + [[-1., .1]] * 3, [[1., .1]] * 10], 1):
            left, right = candidate.aggregate(clients(values), b, t), control.aggregate(clients(values), b, t)
            torch.testing.assert_close(left.state['x'], right.state['x'], rtol=0, atol=0)
            self.assertEqual(left.coefficients, right.coefficients)
        self.assertEqual(control.weights[9], 1.)

    def test_restoration_control_changes_the_return_participation(self):
        full, instant = DirectionalFedReTrust(), make_control(NO_MEMORY)
        b = {'x': torch.zeros(2, dtype=torch.float64)}
        for a in [full, instant]:
            a.aggregate(clients([[1., 0.]] * 7 + [[-1., 0.]] * 3), b, 1)
            a.aggregate(clients([[1., 0.]] * 10), b, 2)
        self.assertEqual(full.weights[9], .5); self.assertEqual(instant.weights[9], 1.)

    def test_no_clipping_matches_candidate_when_cap_is_inactive(self):
        generator = torch.Generator().manual_seed(400)
        full, raw = DirectionalFedReTrust(), make_control(NO_CLIP)
        b = {'x': torch.zeros(3, dtype=torch.float64)}
        for t in range(1, 15):
            points = torch.randn((10, 3), generator=generator, dtype=torch.float64)
            points /= points.norm(dim=1)[:, None]
            s = clients(points.tolist())
            left, right = full.aggregate(s, b, t), raw.aggregate(s, b, t)
            torch.testing.assert_close(left.state['x'], right.state['x'], rtol=0, atol=0)
            self.assertEqual(left.coefficients, right.coefficients); self.assertEqual(full.weights, raw.weights)

    def test_no_clipping_removes_bound_without_renormalizing(self):
        b = {'x': torch.zeros(1, dtype=torch.float64)}; s = clients([[1.]] * 7 + [[100.]] * 3)
        full = DirectionalFedReTrust().aggregate(s, b, 1)
        raw = make_control(NO_CLIP).aggregate(s, b, 1)
        self.assertAlmostEqual(float(full.state['x'][0]), 2.5)
        self.assertAlmostEqual(float(raw.state['x'][0]), 30.7)
        self.assertTrue(all(v['clip_factor'] == 1 for v in raw.diagnostics['client_metrics'].values()))

    def test_no_clipping_recomputes_raw_anchor(self):
        b = {'x': torch.zeros(2, dtype=torch.float64)}
        points = np.array([[1., 0.]] * 6 + [[.8, .4]] + [[20., 10.], [30., -20.], [15., 30.]])
        centre = points.mean(axis=0)
        for _ in range(3):
            weights = 1 / np.maximum(np.linalg.norm(points - centre, axis=1), 1e-6)
            weights /= weights.sum(); centre = np.sum(weights[:, None] * points, axis=0)
        value = make_control(NO_CLIP).aggregate(clients(points.tolist()), b, 1)
        self.assertAlmostEqual(value.diagnostics['anchor_l2_norm'], float(np.linalg.norm(centre)), places=12)

    def test_capped_rfa_equals_aggregation_of_explicitly_clipped_updates(self):
        b = {'x': torch.zeros(2, dtype=torch.float64)}
        points = torch.tensor([[1., 0.]] * 7 + [[50., 10.], [-25., -30.], [40., 15.]], dtype=torch.float64)
        norms = points.norm(dim=1); scales = torch.minimum(torch.ones(10), 6 * norms.median() / norms)
        expected = RFAAggregator().aggregate(clients((points * scales[:, None]).tolist()), b, 1)
        control = make_control(CAPPED_RFA); actual = control.aggregate(clients(points.tolist()), b, 1)
        torch.testing.assert_close(actual.state['x'], expected.state['x'], rtol=1e-14, atol=1e-14)
        for i in range(10): self.assertAlmostEqual(actual.coefficients[i], expected.coefficients[i] * float(scales[i]), places=14)
        self.assertEqual(control.weights, {})

    def test_all_controls_reconstruct_with_unequal_counts(self):
        b = {'x': torch.zeros(2, dtype=torch.float64)}
        s = clients([[1., .1], [1., -.1], [40., 30.], [-1., 0.]], [10, 20, 5, 3])
        for name in CONTROLS:
            result = ContributionRecorder(make_control(name)).aggregate(s, b, 1)
            self.assertLessEqual(result.diagnostics['contribution_measurements']['reconstruction_max_abs_error'], 1e-12)
            self.assertTrue(all(v >= 0 for v in result.coefficients.values()))

    def test_no_clip_fallback_preserves_memory(self):
        control = make_control(NO_CLIP); control.weights = {0: .25}
        result = control.aggregate(clients([[0., 0.]] * 10), {'x': torch.zeros(2, dtype=torch.float64)}, 1)
        self.assertEqual(control.weights, {0: .25}); self.assertTrue(result.diagnostics['bounded_rfa_fallback'])
        control.reset(); self.assertEqual(control.weights, {}); self.assertEqual(control.last_round, 0)

    def test_controls_do_not_mutate_inputs_or_rng_and_are_permutation_invariant(self):
        b = {'x': torch.zeros(2, dtype=torch.float64)}; s = clients([[1., .1]] * 7 + [[-1., 0.]] * 3)
        before = copy.deepcopy(s); rng = torch.get_rng_state().clone()
        for name in CONTROLS:
            left = make_control(name).aggregate(s, b, 1)
            right = make_control(name).aggregate(list(reversed(s)), b, 1)
            torch.testing.assert_close(left.state['x'], right.state['x'])
            for x, y in zip(before, s): torch.testing.assert_close(x.state['x'], y.state['x'], rtol=0, atol=0)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))

    def test_control_failures_leave_state_unchanged(self):
        b = {'x': torch.zeros(2, dtype=torch.float64)}
        for name in CONTROLS:
            control = make_control(name); s = clients([[1., 0.]] * 10); control.aggregate(s, b, 1)
            old = dict(control.weights)
            with self.assertRaises(ValueError): control.aggregate(s, b, 1)
            s[0].state['x'][0] = float('nan')
            with self.assertRaises((ValueError, FloatingPointError)): control.aggregate(s, b, 2)
            self.assertEqual(control.weights, old); self.assertEqual(control.last_round, 1)


class PartitionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original = json.loads((runner.ROOT / runner.plan()['partition_design']['source']).read_text())
        cls.labels = np.arange(60000) % 10
        cls.block = runner.plan()['blocks'][0]

    def test_iid_matches_declared_permutation_and_preserves_reserved_sets(self):
        original = copy.deepcopy(self.original)
        manifest = build_manifest(original, self.labels, self.block, 'IID', 'test')
        validate_manifest(manifest, original)
        pool = sorted(i for v in original['indices']['clients'].values() for i in v)
        expected = np.random.default_rng(self.block['iid_partition_seed']).permutation(pool)
        self.assertEqual(manifest['indices']['clients']['0'], expected[:5370].tolist())
        self.assertEqual(original, self.original)

    def test_non_iid_reproducible_and_equal_sizes(self):
        one = build_manifest(self.original, self.labels, self.block, 'non-IID', 'test')
        two = build_manifest(self.original, self.labels, self.block, 'non-IID', 'test')
        validate_manifest(one, self.original); self.assertEqual(one, two)
        self.assertTrue(all(len(v) == 5370 for v in one['indices']['clients'].values()))

    def test_different_seeds_change_assignment(self):
        for regime in ['IID', 'non-IID']:
            one = build_manifest(self.original, self.labels, self.block, regime, 'test')
            two = build_manifest(self.original, self.labels, runner.plan()['blocks'][1], regime, 'test')
            self.assertNotEqual(one['indices']['clients'], two['indices']['clients'])

    def test_changed_reserved_data_is_rejected(self):
        bad = copy.deepcopy(self.original); bad['indices']['server_reference'].reverse()
        with self.assertRaises(AssertionError): validate_manifest(bad, self.original)


class AnalysisTests(unittest.TestCase):
    def test_reference_cost_counts_root_training_and_auxiliary_forwards(self):
        self.assertEqual(reference_workload({'root_example_visits': 256, 'root_optimizer_steps': 8}), (256, 256, 8))
        self.assertEqual(reference_workload({'reference_forward_examples': 1000, 'reference_backward_examples': 0}), (1000, 0, 0))
        self.assertEqual(reference_workload({}), (0, 0, 0))

    def test_interval_matches_independent_known_sample(self):
        result = summarize([1., 2., 3., 4., 5.], [1, 2, 3, 4, 5])
        self.assertEqual(result['mean'], 3.)
        self.assertAlmostEqual(result['ci_high'], 3 + 2.7764451051977987 * np.sqrt(.5))

    def test_missing_and_duplicate_blocks_are_rejected(self):
        for blocks in ([1, 2, 3, 4], [1, 2, 3, 4, 4], [1, 2, 3, 4, 5, 6]):
            with self.assertRaises(ValueError): summarize([0.] * len(blocks), blocks)

    def test_pairing_uses_block_not_input_order(self):
        rows = []
        for method, offset in [(MAIN, 5.), (NO_MEMORY, 2.), (CAPPED_RFA, 0.)]:
            for block in [5, 3, 1, 4, 2]:
                rows.append({'panel': 'primary' if method == MAIN else 'component_controls',
                    'method': method, 'regime': 'IID', 'block': block, 'seed': 713000 + block,
                    'scenario': 'clean', 'metric': 'score', 'value': block + offset})
        means, effects = paired_tables(rows)
        target = next(e for e in effects if e['target'] == NO_MEMORY and e['comparator'] == CAPPED_RFA)
        self.assertEqual(target['paired_differences_pp'], [2.] * 5)
        with self.assertRaises(ValueError): paired_tables(rows + [rows[0]])

    def test_practical_boundaries_and_uncertainty(self):
        effects = []
        for regime in ['IID', 'non-IID']:
            for scenario, metric, comparator, bound, direction in PRIMARY_SPECS:
                effects.append({'target': MAIN, 'comparator': comparator, 'regime': regime,
                    'scenario': scenario, 'metric': metric, 'mean': bound, 'ci_low': bound - .2, 'ci_high': bound + .2})
        checks = practical_checks(effects)
        self.assertEqual(len(checks), 10)
        self.assertTrue(all(r['point_estimate_meets_margin'] and r['interval_crosses_or_touches_margin'] for r in checks))


class RunnerTests(unittest.TestCase):
    def test_exact_job_grid_and_matched_controls(self):
        js = runner.jobs(); self.assertEqual(len(js), 800)
        keys = {runner.key(j) for j in js}; self.assertEqual(len(keys), 800)
        for j in js:
            self.assertIn((j['method'], j['regime'], j['block'], 'clean'), keys)
            self.assertIn(('FedAvg', j['regime'], j['block'], j['scenario']), keys)

    def test_no_training_before_freeze(self):
        with tempfile.TemporaryDirectory() as path, patch.object(runner, 'execute_run') as mocked:
            with self.assertRaises(ValueError): runner.worker(runner.jobs()[0], Path(path))
            mocked.assert_not_called()

    def test_plan_tampering_rejected(self):
        with patch.object(runner, 'file_sha256', return_value='changed'):
            with self.assertRaises(ValueError): runner.plan()

    def test_freeze_changes_job_identity(self):
        j = runner.jobs()[0]
        self.assertNotEqual(runner.job_id(j, 'a'), runner.job_id(j, 'b'))
        self.assertNotEqual(runner.job_id(j, 'a'), runner.job_id({**j, 'block': 99}, 'a'))

    def test_unregistered_job_rejected_before_execution(self):
        job = runner.jobs()[0]
        with tempfile.TemporaryDirectory() as path, patch.object(runner, 'verify', return_value={'jobs': [job]}), patch.object(runner, 'execute_run') as execute:
            with self.assertRaises(ValueError): runner.worker({**job, 'seed': 42}, Path(path))
            execute.assert_not_called()

    def test_completed_job_is_reused_without_training(self):
        job = runner.jobs()[0]
        with tempfile.TemporaryDirectory() as path:
            study = Path(path); (study/'execution_freeze.json').write_text('{}')
            ident = runner.job_id(job, runner.file_sha256(study/'execution_freeze.json'))
            folder = study/'jobs'/ident; folder.mkdir(parents=True); (folder/'result.json').write_text('{}')
            with patch.object(runner, 'verify', return_value={'jobs': [job]}), patch.object(runner, 'read_result', return_value={'reused': True}) as read, patch.object(runner, 'execute_run') as execute:
                self.assertEqual(runner.worker(job, study), {'reused': True})
                read.assert_called_once(); execute.assert_not_called()

    def test_failed_attempt_is_preserved_and_not_silently_retried(self):
        job = runner.jobs()[0]
        freeze = {'jobs': [job], 'source_sha256': {}, 'partitions': [{'block': job['block'], 'regime': job['regime'], 'partition_id': 'test'}]}
        with tempfile.TemporaryDirectory() as path:
            study = Path(path); (study/'execution_freeze.json').write_text('{}')
            ident = runner.job_id(job, runner.file_sha256(study/'execution_freeze.json'))
            with patch.object(runner, 'verify', return_value=freeze), patch.object(runner, 'execute_run', side_effect=RuntimeError('synthetic failure')) as execute:
                with self.assertRaises(RuntimeError): runner.worker(job, study)
                failure = study/'jobs'/ident/'failure.json'; original = failure.read_text()
                with self.assertRaises(ValueError): runner.worker(job, study)
                self.assertEqual(failure.read_text(), original); self.assertEqual(execute.call_count, 1)
                result = runner.retry(study, ident, 'Synthetic test of preserved retry history')
                self.assertEqual(result['retry_registered'], ident)
                self.assertFalse(failure.exists())
                archived = list((failure.parent/'failures').glob('*_failure.json'))
                self.assertEqual(len(archived), 1); self.assertEqual(archived[0].read_text(), original)

    def test_reference_free_factory_never_reads_data_or_evaluation(self):
        class Trap:
            def __getattribute__(self, name): raise AssertionError('Data access: ' + name)
        cfg = TrainConfig(development_seed=20261014)
        for name in [MAIN, *CONTROLS, 'FedAvg', 'RFA', 'FedCognis-aggregation', 'FedCognis-squared-distance']:
            self.assertTrue(runner.make_method(name, Trap(), cfg).name.startswith(name))

    def test_reference_factories_use_only_reserved_banks(self):
        cfg = TrainConfig(partition_id=runner.PARTITIONS['IID'], development_seed=20261014, max_train_per_client=64, max_validation=100)
        data = load_data(runner.ROOT, cfg)
        class Guarded:
            def __getattr__(self, name):
                if name not in ('pool', 'manifest', 'manifest_sha256', 'selection'):
                    raise AssertionError('Forbidden constructor access: ' + name)
                return getattr(data, name)
        for name in ['FLTrust', 'FLAURA-reconstruction', 'FLAURA-sigma-double', 'Viceroy']:
            method = runner.make_method(name, Guarded(), cfg)
            self.assertTrue(method.name.startswith(name))
            if name.startswith('FLAURA-'):
                self.assertEqual(method.base.reference.metadata['examples'], 100)
                self.assertEqual(len(method.base.reference.images), 100)

    def test_preflight_registry_excludes_replication_seeds(self):
        from validation_preflight import preflight_jobs
        js = preflight_jobs(); self.assertEqual(len(js), 48)
        self.assertEqual({j['seed'] for j in js}, {20261014})
        self.assertEqual({j['rounds'] for j in js}, {6})


if __name__ == '__main__': unittest.main()
