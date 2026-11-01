"""Independent mixing, coefficient-audit, and invariance checks for NNM-RFA."""
from dataclasses import replace
from pathlib import Path
import copy
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiment_core import ClientSubmission, fedavg, state_sha256
from nnm_rfa import NNMRFAAggregator, NNMRFAConfig, nearest_neighbor_mixing, nnm_rfa_rule
from rfa import RFAConfig, smoothed_weiszfeld
from scenario_metrics import ContributionRecorder


def submissions(points, counts=None, origin=None, ids=None, dtype=torch.float64):
    points = torch.as_tensor(points, dtype=dtype)
    origin = torch.zeros(points.shape[1], dtype=dtype) if origin is None else torch.tensor(origin, dtype=dtype)
    # Exercise reconstruction across two differently shaped parameters.
    broadcast = {"a": origin[:1].clone(), "b": origin[1:].reshape(1, -1).clone()}
    counts = [512] * len(points) if counts is None else counts
    ids = list(range(len(points))) if ids is None else ids
    clients = [ClientSubmission(cid, count, {"a": (point + origin)[:1].clone(),
               "b": (point + origin)[1:].reshape(1, -1).clone()})
               for cid, count, point in zip(ids, counts, points)]
    return clients, broadcast


def flat(state):
    return torch.cat([tensor.flatten() for tensor in state.values()])


class NNMRFATests(unittest.TestCase):
    def test_official_documentation_example(self):
        points = torch.tensor([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]], dtype=torch.float64)
        mixed, matrix, neighbors = nearest_neighbor_mixing(points, [0, 1, 2], 1)
        torch.testing.assert_close(mixed, torch.tensor([[2.5, 3.5, 4.5], [2.5, 3.5, 4.5],
                                                       [5.5, 6.5, 7.5]], dtype=torch.float64))
        torch.testing.assert_close(matrix.sum(1), torch.ones(3, dtype=torch.float64))
        self.assertEqual(neighbors, {"0": [0, 1], "1": [1, 0], "2": [2, 1]})

    def test_f_zero_matches_fedavg_for_equal_counts(self):
        clients, broadcast = submissions([[1., -2.], [2., 4.], [-6., 5.], [8., 9.]])
        got = nnm_rfa_rule(clients, broadcast, NNMRFAConfig(f=0))
        expected = fedavg([c.state for c in clients], [c.num_examples for c in clients])
        torch.testing.assert_close(flat(got.state), flat(expected), rtol=0, atol=1e-14)
        for coefficient in got.coefficients.values(): self.assertAlmostEqual(coefficient, .25)

    def test_f_zero_unequal_counts_still_uses_arithmetic_neighbor_means(self):
        clients, broadcast = submissions([[0., 0.], [9., 6.], [3., -3.]], counts=[50, 1, 2])
        got = nnm_rfa_rule(clients, broadcast, NNMRFAConfig(f=0))
        torch.testing.assert_close(flat(got.state), torch.tensor([4., 1.], dtype=torch.float64))
        for coefficient in got.coefficients.values(): self.assertAlmostEqual(coefficient, 1/3)
        self.assertAlmostEqual(got.diagnostics["solver"]["base_weights"][0], 50/53)

    def test_manual_mixing_then_original_three_step_solver_and_coefficient_expansion(self):
        points = torch.tensor([[0., 0.], [1., 0.], [3., 0.], [8., 0.], [20., 0.]], dtype=torch.float64)
        counts = [2, 3, 5, 7, 11]
        # Four nearest include self. All but the last exclude client 4; the last excludes 0.
        manual_matrix = torch.tensor([[.25, .25, .25, .25, 0.]] * 4
                                     + [[0., .25, .25, .25, .25]], dtype=torch.float64)
        manual_mixtures = torch.stack([points[:4].mean(0)] * 4 + [points[1:].mean(0)])
        expected, weights, solver = smoothed_weiszfeld(manual_mixtures, counts, RFAConfig())
        clients, broadcast = submissions(points, counts=counts)
        got = nnm_rfa_rule(clients, broadcast, NNMRFAConfig(f=1))
        torch.testing.assert_close(torch.tensor(got.diagnostics["mixing_matrix"], dtype=torch.float64), manual_matrix)
        torch.testing.assert_close(flat(got.state), expected, rtol=0, atol=1e-14)
        torch.testing.assert_close(torch.tensor(list(got.coefficients.values()), dtype=torch.float64), manual_matrix.T @ weights)
        self.assertEqual(got.diagnostics["solver"]["iterations_used"], 3)
        self.assertEqual(got.diagnostics["solver"], solver)

    def test_recorder_reconstructs_original_client_contributions_in_both_dtypes(self):
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                clients, broadcast = submissions([[0., 2., 1.], [.2, 1., -1.], [.5, -2., 3.],
                                                   [3., 4., 5.], [-8., 7., 9.]],
                                                  origin=[4., -3., 2.], dtype=dtype)
                got = ContributionRecorder(NNMRFAAggregator(NNMRFAConfig(f=1))).aggregate(clients, broadcast, 1)
                actual = flat(got.state)
                deltas = torch.stack([flat(c.state).double() - flat(broadcast).double() for c in clients])
                coeff = torch.tensor([got.coefficients[c.client_id] for c in clients], dtype=torch.float64)
                expected = (flat(broadcast).double() + (coeff[:, None] * deltas).sum(0)).to(dtype)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                self.assertEqual(got.diagnostics["contribution_measurements"]["reconstruction_max_abs_error"], 0.)
                self.assertAlmostEqual(sum(got.coefficients.values()), 1.)

    def test_input_permutation_and_broadcast_translation_equivariance(self):
        points = [[0., 0.], [1., 2.], [3., -1.], [-2., 2.], [5., 4.]]
        clients, broadcast = submissions(points, counts=[2, 3, 5, 7, 11], ids=[40, 10, 30, 20, 50])
        shifted, origin = submissions(points, counts=[2, 3, 5, 7, 11], origin=[16., -8.], ids=[40, 10, 30, 20, 50])
        agg = NNMRFAAggregator(NNMRFAConfig(f=1))
        a = agg.aggregate(clients, broadcast, 1)
        b = agg.aggregate([clients[i] for i in [4, 2, 0, 3, 1]], broadcast, 1)
        c = agg.aggregate(shifted, origin, 100)
        self.assertEqual(a.coefficients, b.coefficients)
        self.assertEqual(a.diagnostics, b.diagnostics)
        torch.testing.assert_close(flat(a.state), flat(b.state), rtol=0, atol=0)
        torch.testing.assert_close(flat(c.state) - flat(origin), flat(a.state), rtol=0, atol=2e-15)
        self.assertEqual(a.coefficients, c.coefficients)

    def test_distance_ties_choose_client_ids_but_always_retain_self(self):
        # All identical points; a high-ID anchor must remain in its own mixture.
        points = torch.zeros((5, 2), dtype=torch.float64)
        _, matrix, neighbors = nearest_neighbor_mixing(points, [50, 40, 30, 20, 10], 2)
        self.assertEqual(neighbors["50"], [50, 10, 20])
        self.assertEqual(neighbors["10"], [10, 20, 30])
        self.assertTrue((matrix.diagonal() == 1/3).all())
        # Nonidentical vectors at an equal distance use the same ID tie rule.
        _, _, neighbors = nearest_neighbor_mixing(torch.tensor([[0., 0.], [1., 0.], [-1., 0.]],
                                                   dtype=torch.float64), [90, 20, 10], 1)
        self.assertEqual(neighbors["90"], [90, 10])

    def test_identical_updates_and_zero_geometry_are_finite_and_preserved(self):
        for point in ([0., 0.], [2., -1.]):
            clients, broadcast = submissions([point] * 10, origin=[8., 16.])
            got = NNMRFAAggregator().aggregate(clients, broadcast, 1)
            torch.testing.assert_close(flat(got.state), flat(clients[0].state), rtol=0, atol=4e-15)
            self.assertTrue(all(torch.isfinite(v).all() for v in got.state.values()))
            self.assertAlmostEqual(sum(got.coefficients.values()), 1.)
            self.assertEqual(got.diagnostics["neighborhood_size"], 7)
            self.assertEqual(got.diagnostics["reference_forward_examples"], 0)

    def test_one_client_with_zero_budget(self):
        clients, broadcast = submissions([[2., -3.]], origin=[10., 5.])
        got = nnm_rfa_rule(clients, broadcast, NNMRFAConfig(f=0))
        torch.testing.assert_close(flat(got.state), flat(clients[0].state), rtol=0, atol=0)
        self.assertEqual(got.coefficients, {0: 1.})

    def test_declared_three_outliers_have_less_influence_than_fedavg_in_known_geometry(self):
        clients, broadcast = submissions([[1., 0.]] * 7 + [[1000., 0.]] * 3)
        got = NNMRFAAggregator().aggregate(clients, broadcast, 1)
        average = fedavg([c.state for c in clients], [c.num_examples for c in clients])
        # This is one synthetic regression check, not a general robustness claim.
        self.assertLess(abs(float(flat(got.state)[0]) - 1.), 20.)
        self.assertGreater(abs(float(flat(average)[0]) - 1.), 299.)
        self.assertLess(sum(got.coefficients[i] for i in [7, 8, 9]), .02)

    def test_stateless_and_no_input_or_rng_mutation(self):
        clients, broadcast = submissions([[float(i), float(i % 3)] for i in range(10)])
        hashes = [state_sha256(broadcast), *[state_sha256(c.state) for c in clients]]
        rng = torch.random.get_rng_state().clone()
        agg = NNMRFAAggregator()
        a, b = agg.aggregate(clients, broadcast, 1), agg.aggregate(clients, broadcast, 24)
        self.assertEqual(hashes, [state_sha256(broadcast), *[state_sha256(c.state) for c in clients]])
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        self.assertEqual(a.coefficients, b.coefficients)
        self.assertEqual(a.diagnostics, b.diagnostics)
        self.assertEqual(agg.settings["f"], 3)
        self.assertEqual(agg.settings["rfa"], {"max_iterations": 3, "smoothing": 1e-6, "relative_tolerance": 1e-6})

    def test_invalid_budget_and_insufficient_participation_rejected(self):
        for f in (-1, True, 1.5, "3", None):
            with self.subTest(f=f), self.assertRaises(ValueError): NNMRFAConfig(f=f)
        for invalid in ({"f": 3}, RFAConfig()):
            with self.assertRaises(TypeError): NNMRFAAggregator(invalid)
        for n, f in ((6, 3), (5, 3), (2, 1), (1, 1)):
            clients, broadcast = submissions([[0., 0.]] * n)
            with self.subTest(n=n, f=f), self.assertRaises(ValueError):
                nnm_rfa_rule(clients, broadcast, NNMRFAConfig(f=f))

    def test_invalid_counts_ids_and_models_rejected(self):
        clients, broadcast = submissions([[0., 0.], [1., 2.], [2., -1.]])
        for count in (0, -1, True, 1.5):
            invalid = [replace(clients[0], num_examples=count), *clients[1:]]
            with self.subTest(count=count), self.assertRaises(ValueError):
                nnm_rfa_rule(invalid, broadcast, NNMRFAConfig(f=1))
        for invalid in ([], [clients[0]] * 3,
                        [replace(clients[0], client_id="0"), *clients[1:]],
                        [replace(clients[0], client_id=True), *clients[1:]]):
            with self.assertRaises(ValueError): nnm_rfa_rule(invalid, broadcast, NNMRFAConfig(f=0))
        for problem in ("key", "shape", "dtype", "nan", "inf", "broadcast_nan", "empty_broadcast"):
            altered, origin = copy.deepcopy(clients), copy.deepcopy(broadcast)
            if problem == "key": altered[0].state.pop("a")
            elif problem == "shape": altered[0].state["a"] = torch.zeros(2, dtype=torch.float64)
            elif problem == "dtype": altered[0].state["a"] = altered[0].state["a"].float()
            elif problem in {"nan", "inf"}: altered[0].state["a"][0] = float(problem)
            elif problem == "broadcast_nan": origin["a"][0] = float("nan")
            else: origin = {}
            with self.subTest(problem=problem), self.assertRaises((ValueError, FloatingPointError)):
                nnm_rfa_rule(altered, origin, NNMRFAConfig(f=1))


if __name__ == "__main__":
    unittest.main()
