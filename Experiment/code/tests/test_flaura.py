"""Independent numerical and information-boundary checks for FLAURA."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import math
import sys
import unittest
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiment_core import ClientSubmission, MnistCNN, clone_state, state_sha256
from flaura import (FLAURAConfig, AuxiliaryReference, build_auxiliary_reference,
    pairwise_mmd, adaptive_neighbours, trust_weights, representation_center,
    extract_representations, weighted_update, FLAURAAggregator)
from scenario_metrics import ContributionRecorder


class FLAURATests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)

    def test_mmd_matches_independent_kernel_sum_not_squared_distance(self):
        x = np.array([[[0.], [2.]], [[3.], [4.]], [[0.], [2.]]])
        actual, _ = pairwise_mmd(torch.tensor(x), bandwidth_squared=2.)
        kernel = lambda a, b: math.exp(-float(np.sum((a-b)**2))/4.)
        expected = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                within1 = sum(kernel(a, b) for a in x[i] for b in x[i])/4
                within2 = sum(kernel(a, b) for a in x[j] for b in x[j])/4
                cross = sum(kernel(a, b) for a in x[i] for b in x[j])/4
                expected[i,j] = math.sqrt(max(0., within1+within2-2*cross))
        np.testing.assert_allclose(actual.numpy(), expected, atol=1e-12)

    def test_mmd_identical_representations_and_kernel_bandwidth(self):
        x = torch.ones(3, 4, 2)
        actual, info = pairwise_mmd(x)
        self.assertTrue(torch.equal(actual, torch.zeros(3,3,dtype=torch.float64)))
        self.assertEqual(info['bandwidth_squared'], 1.)
        _, info = pairwise_mmd(torch.tensor([[[0.],[1.]],[[2.],[3.]]]))
        self.assertEqual(info['bandwidth_squared'], 1.)  # lower median of 1,1,1,4,4,9

    def test_kernel_is_invariant_to_common_translation_and_client_order(self):
        r = torch.tensor([[[0.,1.],[2.,3.]],[[4.,3.],[2.,1.]],[[5.,0.],[3.,2.]]])
        a, _ = pairwise_mmd(r)
        b, _ = pairwise_mmd(r[[2,0,1]] + 7.)
        torch.testing.assert_close(b, a[[2,0,1]][:,[2,0,1]], atol=1e-12, rtol=0)

    def test_joint_geometric_median_rejects_single_large_outlier(self):
        _, distance, info = representation_center(torch.tensor([[[0.]],[[0.]],[[100.]]]), FLAURAConfig())
        self.assertLess(float(distance[0]), 1e-5)
        self.assertGreater(float(distance[2]), 99.99)
        self.assertLessEqual(info['final_objective'], info['initial_objective'])

    def test_knee_boundary_ties_and_flat_profiles(self):
        d = torch.tensor([[0.,1.,1.,9.],[1.,0.,1.,9.],[1.,1.,0.,9.],[9.,9.,9.,0.]])
        n = adaptive_neighbours(d, [0,1,2,3])
        self.assertEqual(n[0], [1,2])
        self.assertEqual(n[3], [0,1,2])
        self.assertEqual(adaptive_neighbours(torch.zeros(1,1), [7]), [[]])

    def test_algorithm1_trust_threshold_and_weight_arithmetic(self):
        o = FLAURAConfig(alpha=.25,beta=.5,gamma=.8)
        instant, updated, threshold, retained, weight = trust_weights([.2,.8,1.],[.6,.4,1.],[.4,.6,1.],o)
        np.testing.assert_allclose(instant.numpy(), [.5,.5,1.])
        np.testing.assert_allclose(updated.numpy(), [.45,.55,1.])
        self.assertAlmostEqual(threshold, .8*2/3)
        self.assertEqual(retained.tolist(), [False,True,True])
        np.testing.assert_allclose(weight.numpy(), [0.,.55/1.55,1/1.55])

    def test_excluded_clients_can_recover_with_fresh_scores(self):
        o = FLAURAConfig(alpha=1.,beta=.5,gamma=.8)
        _, t, _, keep, _ = trust_weights([0.,1.],[1.,1.],[1.,1.],o)
        self.assertEqual(keep.tolist(), [False,True])
        _, _, _, keep, _ = trust_weights([1.,1.],[1.,1.],t,o)
        self.assertEqual(keep.tolist(), [True,True])

    def test_no_filter_control_and_zero_mass_policy(self):
        _, _, _, keep, w = trust_weights([.1,1.],[.1,1.],[1.,1.],FLAURAConfig(beta=0,gamma=0))
        self.assertTrue(keep.all());self.assertGreater(float(w[0]),0.)
        _, _, _, _, w = trust_weights([0.,0.],[0.,0.],[1.,1.],FLAURAConfig(beta=0))
        self.assertEqual(float(w.sum()),0.)

    def test_update_uses_trust_only_and_preserves_inputs(self):
        b = {'w':torch.tensor([10.,20.])}
        s = [ClientSubmission(1,1,{'w':torch.tensor([12.,18.])}),ClientSubmission(2,1000,{'w':torch.tensor([18.,22.])})]
        hashes = [state_sha256(b)]+[state_sha256(x.state) for x in s]
        actual = weighted_update(s,b,torch.tensor([.25,.75],dtype=torch.float64))
        torch.testing.assert_close(actual['w'],torch.tensor([16.5,21.]))
        self.assertEqual(hashes,[state_sha256(b)]+[state_sha256(x.state) for x in s])

    def test_auxiliary_boundary_and_copy(self):
        pool = SimpleNamespace(data=torch.zeros(320,28,28,dtype=torch.uint8), targets=torch.arange(320)%10)
        data = SimpleNamespace(pool=pool, manifest_sha256='test', manifest={
            'indices':{'server_reference':list(range(300)), 'clients':{'0':list(range(300,310))},
                       'development_validation':list(range(310,320))},'normalization':{'mean':[.1307],'std':[.3081]}})
        ref = build_auxiliary_reference(data)
        self.assertEqual(len(ref.images),100)
        self.assertEqual(set(ref.metadata['original_indices']),set(range(100)))
        self.assertFalse(hasattr(ref,'targets'))
        previous = ref.images.clone();pool.data.fill_(255)
        self.assertTrue(torch.equal(ref.images,previous))
        data.manifest['indices']['clients']['0'].append(0)
        with self.assertRaises(ValueError):build_auxiliary_reference(data)

    def test_plr_excludes_logits_and_preserves_rng(self):
        model = MnistCNN();a=clone_state(model);b=clone_state(model)
        b['classifier.3.weight'].add_(100)
        images=torch.randn(3,1,28,28)
        rng=torch.random.get_rng_state().clone()
        r=extract_representations([ClientSubmission(0,1,a),ClientSubmission(1,1,b)],images)
        self.assertEqual(tuple(r.shape),(2,3,128))
        self.assertTrue(torch.equal(r[0],r[1]))
        self.assertTrue(torch.equal(rng,torch.random.get_rng_state()))

    def test_aggregator_recording_reset_and_absent_memory(self):
        ref=AuxiliaryReference(torch.zeros(1,1,28,28),{'examples':1})
        agg=FLAURAAggregator(ref,FLAURAConfig())
        b={'w':torch.tensor([0.])}
        items=[ClientSubmission(i,1,{'w':torch.tensor([float(i)])}) for i in range(3)]
        fake=torch.tensor([[[0.]],[[1.]],[[10.]]],dtype=torch.float64)
        with patch('flaura.extract_representations',return_value=fake):
            result=ContributionRecorder(agg).aggregate(items,b,1)
            self.assertAlmostEqual(sum(result.coefficients.values()),1.)
            before=agg.trust[2]
        with patch('flaura.extract_representations',return_value=fake[:2]):
            agg.aggregate(items[:2],b,3)
        self.assertEqual(agg.trust[2],before)
        with self.assertRaises(ValueError):agg.aggregate(items,b,3)
        agg.reset();self.assertEqual(agg.trust,{})

    def test_validation(self):
        for kwargs in [{'sigma_g':0},{'gamma':1.1},{'beta':-1},{'alpha':float('nan')},{'median_iterations':0}]:
            with self.assertRaises(ValueError):FLAURAConfig(**kwargs)
        with self.assertRaises(ValueError):pairwise_mmd(torch.full((2,2,2),float('nan')))


if __name__=='__main__':unittest.main()
