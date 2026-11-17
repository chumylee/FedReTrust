"""Numerical checks against independent calculations and the old reconstruction."""
import unittest
from unittest.mock import patch

import numpy as np
import torch

import study
from data import HarCNN
from experiment_core import ClientSubmission, MnistCNN, clone_state
from flaura import AuxiliaryReference, FLAURAConfig, FLAURAAggregator
from extended_methods import (ByzClip21State, ByzConfig, PublicationFLAURA,
                              Representations, full_gradient, nnm_coordinate_median,
                              auxiliary_reference)
from extended_study import byz_job, verify_manifest
from pathlib import Path
from tempfile import TemporaryDirectory


class ExtendedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_nnm_and_median_against_numpy(self):
        points = np.array([[0., 1.], [1., 0.], [2., 2.], [100., 100.]])
        mixed = []
        for p in points:
            neighbours = np.argsort(((points - p)**2).sum(1), kind='stable')[:3]
            mixed.append(points[neighbours].mean(0))
        actual = nnm_coordinate_median(torch.tensor(points), 1)
        np.testing.assert_allclose(actual, np.median(mixed, axis=0))

    def test_byz_clipping_history_recurrence(self):
        state = ByzClip21State([0], 2, ByzConfig(beta=.5, beta_hat=.5, tau=1))
        state.observe(torch.tensor([[6., 8.]], dtype=torch.float64))
        torch.testing.assert_close(state.v, torch.tensor([[3., 4.]], dtype=torch.float64))
        torch.testing.assert_close(state.g, torch.tensor([[.3, .4]], dtype=torch.float64))
        state.observe(torch.zeros(1, 2, dtype=torch.float64))
        # innovation (1.2,1.6), clipped (.6,.8), half added to history.
        torch.testing.assert_close(state.g, torch.tensor([[.6, .8]], dtype=torch.float64))
        torch.testing.assert_close(state.g, state.m)
        torch.testing.assert_close(state.aggregate, state.g[0])

    def test_full_gradient_matches_one_batch_and_does_not_update_model(self):
        torch.manual_seed(30)
        model = torch.nn.Linear(3, 2)
        x, y = torch.randn(131, 3), torch.arange(131) % 2
        before = clone_state(model)
        expected_loss = torch.nn.functional.cross_entropy(model(x), y)
        expected = torch.cat([g.flatten() for g in torch.autograd.grad(expected_loss, model.parameters())])
        actual = full_gradient(model, x, y)
        torch.testing.assert_close(actual.float(), expected, atol=1e-7, rtol=1e-5)
        for key, value in before.items():
            torch.testing.assert_close(value, model.state_dict()[key])

    def test_penultimate_features_exclude_output_head_for_all_models(self):
        for factory, shape, width in ((MnistCNN, (3, 1, 28, 28), 128),
                                      (lambda: MnistCNN(47), (3, 1, 28, 28), 128),
                                      (HarCNN, (3, 6, 128), 32)):
            model = factory()
            a, b = clone_state(model), clone_state(model)
            list(b.values())[-1].add_(100)
            reference = AuxiliaryReference(torch.randn(shape), {})
            rng = torch.random.get_rng_state().clone()
            extractor = Representations(factory, reference, torch.device('cpu'))
            actual = extractor([ClientSubmission(0, 1, a), ClientSubmission(1, 1, b)])
            self.assertEqual(actual.shape, (2, 3, width))
            torch.testing.assert_close(actual[0], actual[1], rtol=0, atol=0)
            self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))

    def test_flaura_matches_existing_reconstruction(self):
        reference = AuxiliaryReference(torch.zeros(2, 1, 28, 28), {})
        old = FLAURAAggregator(reference, FLAURAConfig(sigma_g=2))
        new = PublicationFLAURA(MnistCNN, reference, torch.device('cpu'), 2)
        broadcast = {'w': torch.tensor([0.])}
        submissions = [ClientSubmission(i, 1, {'w': torch.tensor([float(i)])}) for i in range(4)]
        features = torch.tensor([[[0.], [1.]], [[.5], [.8]], [[1.], [.9]], [[9.], [12.]]])
        for round_id in (1, 2):
            with patch('flaura.extract_representations', return_value=features), patch.object(new, 'extract', return_value=features):
                a = old.aggregate(submissions, broadcast, round_id)
                b = new.aggregate(submissions, broadcast, round_id)
            torch.testing.assert_close(a.state['w'], b.state['w'], rtol=0, atol=0)
            self.assertEqual(a.coefficients, b.coefficients)
            self.assertEqual(old.trust, new.trust)

    def test_auxiliary_boundaries_and_no_labels(self):
        data = {'root': (torch.randn(120, 2), torch.arange(120) % 6), 'classes': 6,
                'plan': {'root_indices': list(range(120)), 'client_indices': {'0': [200, 201]}}}
        ref = auxiliary_reference(data)
        self.assertEqual(len(ref.images), 96)
        self.assertFalse(hasattr(ref, 'labels'))
        data['plan']['client_indices']['0'].append(0)
        with self.assertRaises(ValueError):
            auxiliary_reference(data)

    def test_byz_round_order_preserves_initial_model_and_first_noop(self):
        data = {'plan': {'seed': 42}, 'model': lambda: torch.nn.Linear(2, 2),
                'clients': {0: (torch.eye(2), torch.eye(2), torch.tensor([0, 1]), torch.arange(2))},
                'affected': (0,), 'permutation': torch.tensor([1, 0])}
        seen = []
        def evaluate(model, r, fraction):
            seen.append(clone_state(model))
            return {'round': r, 'accuracy_pct': 0.}
        _, record = byz_job(data, 'clean', 2, torch.device('cpu'), ByzConfig(), evaluate)
        self.assertEqual(record['status'], 'completed')
        self.assertEqual(study.state_sha256(seen[0]), study.state_sha256(seen[1]))
        self.assertNotEqual(study.state_sha256(seen[1]), study.state_sha256(seen[2]))

    def test_resume_rejects_modified_settings(self):
        with TemporaryDirectory() as temp:
            path = Path(temp) / 'manifest.json'
            verify_manifest(path, {'rounds': 48})
            verify_manifest(path, {'rounds': 48})
            with self.assertRaises(ValueError):
                verify_manifest(path, {'rounds': 24})


if __name__ == '__main__':
    unittest.main()
