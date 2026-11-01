"""Numerical, reference-source and integration checks for the RFA rule."""
from dataclasses import replace
from pathlib import Path
import ast
import json
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiment_core import ClientSubmission, file_sha256, state_sha256
from rfa import RFAAggregator, RFAConfig, rfa_rule, smoothed_weiszfeld


def submissions(points, counts=None, origin=None):
    points = np.asarray(points, dtype=np.float64)
    origin = np.zeros(points.shape[1]) if origin is None else np.asarray(origin, dtype=np.float64)
    broadcast = {"a": torch.tensor(origin[:1]), "b": torch.tensor(origin[1:])}
    clients = [ClientSubmission(i, int(n), {"a": torch.tensor((point + origin)[:1]),
               "b": torch.tensor((point + origin)[1:])})
               for i, (point, n) in enumerate(zip(points, counts if counts is not None else [1] * len(points)))]
    return clients, broadcast


def flat(state):
    return torch.cat([value.flatten() for value in state.values()])


def pinned_numpy_reference():
    """Load only the four inspected pure NumPy methods, never the TF module."""
    root = Path(__file__).resolve().parents[2] / "resources/rfa_reference"
    manifest = json.loads((root / "source_manifest.json").read_text())
    for name, item in manifest["files"].items():
        if file_sha256(root / name) != item["sha256"]:
            raise AssertionError(f"Pinned reference fingerprint changed: {name}")
    tree = ast.parse((root / "model.py").read_text())
    server = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ServerModel")
    names = {"weighted_average_oracle", "geometric_median_update", "l2dist", "geometric_median_objective"}
    methods = [n for n in server.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(methods) == len(names)
    isolated = ast.Module(body=[ast.ClassDef(name="ServerModel", bases=[], keywords=[], body=methods,
                                           decorator_list=[])], type_ignores=[])
    namespace = {"np": np}
    exec(compile(ast.fix_missing_locations(isolated), str(root / "model.py"), "exec"), namespace)
    return namespace["ServerModel"]


class RFATests(unittest.TestCase):
    def test_independent_numpy_calculation_with_unequal_weights_and_smoothing(self):
        points = np.array([[0., 0.], [0.2, 0.3], [3., -2.]])
        alpha = np.array([2., 3., 1.]); alpha /= alpha.sum()
        estimate = (alpha[:, None] * points).sum(axis=0)
        for _ in range(3):
            distances = np.sqrt(((points - estimate) ** 2).sum(axis=1))
            weights = alpha / np.maximum(0.5, distances)
            weights /= weights.sum()
            estimate = (weights[:, None] * points).sum(axis=0)
        got, got_weights, info = smoothed_weiszfeld(torch.tensor(points), [2, 3, 1],
                            RFAConfig(smoothing=0.5, relative_tolerance=0))
        np.testing.assert_allclose(got.numpy(), estimate, rtol=0, atol=1e-14)
        np.testing.assert_allclose(got_weights.numpy(), weights, rtol=0, atol=1e-14)
        self.assertEqual(info["averaging_operations"], 4)

    def test_matches_pinned_author_implementation_for_fixed_iteration_budget(self):
        rng = np.random.default_rng(42)
        points = [list(rng.normal(size=(2, 3))) for _ in range(7)]
        points = [[np.asarray(x, dtype=np.float64) for x in p] for p in points]
        counts = [3, 7, 2, 5, 11, 4, 1]
        options = RFAConfig(relative_tolerance=0)
        reference, calls, _ = pinned_numpy_reference().geometric_median_update(
            points, counts, maxiter=options.max_iterations, eps=options.smoothing, ftol=0)
        matrix = torch.tensor(np.stack([np.concatenate(p) for p in points]))
        got, _, info = smoothed_weiszfeld(matrix, counts, options)
        np.testing.assert_allclose(got.numpy(), np.concatenate(reference), rtol=1e-13, atol=1e-14)
        self.assertEqual(info["averaging_operations"], calls)

    def test_joint_median_on_equilateral_triangle(self):
        points = torch.tensor([[0., 0.], [2., 0.], [1., np.sqrt(3)]], dtype=torch.float64)
        got, _, _ = smoothed_weiszfeld(points, [1, 1, 1])
        torch.testing.assert_close(got, torch.tensor([1., np.sqrt(3)/3], dtype=torch.float64))
        self.assertGreater(float(torch.linalg.vector_norm(got - points.median(dim=0).values)), 0.5)

    def test_far_outlier_has_less_influence_than_arithmetic_mean(self):
        points = torch.tensor([[0., 0.], [0.1, 0.], [0., 0.1], [100., 100.]], dtype=torch.float64)
        got, weights, _ = smoothed_weiszfeld(points, [1]*4, RFAConfig(max_iterations=100))
        self.assertLess(float(torch.linalg.vector_norm(got)), 0.15)
        self.assertGreater(float(torch.linalg.vector_norm(points.mean(dim=0))), 30)
        self.assertLess(float(weights[-1]), 0.001)

    def test_sample_counts_change_the_aggregate(self):
        clients, broadcast = submissions([[0., 0.], [10., 0.]], [9, 1])
        result = rfa_rule(clients, broadcast)
        self.assertGreater(result.coefficients[0], 0.9)
        self.assertLess(float(flat(result.state)[0]), 1.)
        self.assertAlmostEqual(result.diagnostics["client_metrics"]["0"]["base_sample_weight"], .9)

    def test_weighted_majority_matches_known_one_dimensional_median(self):
        got, _, _ = smoothed_weiszfeld(torch.tensor([[0.], [10.]], dtype=torch.float64), [3, 1],
                                      RFAConfig(max_iterations=100, relative_tolerance=1e-12))
        self.assertLess(abs(float(got[0])), 1e-5)

    def test_translation_equivariance_and_model_delta_equivalence(self):
        points = [[0., 0.], [0.2, 0.3], [-0.1, .4], [3., -2.]]
        clients, broadcast = submissions(points, [2, 1, 4, 1])
        translated, origin = submissions(points, [2, 1, 4, 1], [7., -3.])
        left, right = rfa_rule(clients, broadcast), rfa_rule(translated, origin)
        torch.testing.assert_close(flat(right.state) - flat(origin), flat(left.state), rtol=0, atol=2e-15)
        absolute_models = torch.stack([flat(s.state) for s in translated])
        direct, _, _ = smoothed_weiszfeld(absolute_models, [s.num_examples for s in translated])
        torch.testing.assert_close(direct, flat(right.state), rtol=0, atol=2e-15)

    def test_rotation_equivariance(self):
        points = torch.tensor([[0., 0.], [1., 2.], [3., -1.]], dtype=torch.float64)
        rotation = torch.tensor([[0., -1.], [1., 0.]], dtype=torch.float64)
        a, _, _ = smoothed_weiszfeld(points, [1, 2, 3])
        b, _, _ = smoothed_weiszfeld(points @ rotation, [1, 2, 3])
        torch.testing.assert_close(a @ rotation, b, rtol=0, atol=1e-14)

    def test_identical_and_zero_updates_are_finite(self):
        for point in [[0., 0.], [2., -1.]]:
            clients, broadcast = submissions([point]*3, [1, 2, 3], [5., 8.])
            result = rfa_rule(clients, broadcast)
            torch.testing.assert_close(flat(result.state), flat(clients[0].state))
            np.testing.assert_allclose(list(result.coefficients.values()), [1/6, 2/6, 3/6], atol=1e-14)

    def test_single_client_applies_its_update(self):
        clients, broadcast = submissions([[.1, -.2]], origin=[4., 9.])
        result = rfa_rule(clients, broadcast)
        torch.testing.assert_close(flat(result.state), flat(clients[0].state), rtol=0, atol=0)
        self.assertEqual(result.coefficients, {0: 1.0})

    def test_coefficients_reconstruct_joint_state_and_preserve_dtype(self):
        clients, broadcast = submissions([[0., 1.], [.2, .3], [4., -2.]], [2, 3, 1])
        broadcast = {k: v.float() for k, v in broadcast.items()}
        for client in clients: client.state = {k: v.float() for k, v in client.state.items()}
        result = rfa_rule(clients, broadcast)
        expected = sum(result.coefficients[c.client_id] * flat(c.state).double() for c in clients)
        torch.testing.assert_close(flat(result.state), expected.float(), rtol=0, atol=0)
        self.assertAlmostEqual(sum(result.coefficients.values()), 1.)

    def test_smoothed_objective_never_increases(self):
        points = torch.tensor([[0., 0.], [.1, .2], [.3, .1], [10., 10.]], dtype=torch.float64)
        _, _, info = smoothed_weiszfeld(points, [2, 3, 1, 1],
                                      RFAConfig(max_iterations=30, smoothing=.25, relative_tolerance=0))
        values = [row["smoothed_objective"] for row in info["objective_trace"]]
        self.assertTrue(np.all(np.diff(values) <= 1e-14))
        self.assertLess(values[-1], values[0])

    def test_stopping_condition_and_fixed_budget(self):
        points = torch.zeros((3, 2), dtype=torch.float64)
        _, _, early = smoothed_weiszfeld(points, [1]*3)
        _, _, fixed = smoothed_weiszfeld(points, [1]*3, RFAConfig(relative_tolerance=0))
        self.assertEqual(early["iterations_used"], 1)
        self.assertEqual(early["termination"], "relative_objective_tolerance")
        self.assertEqual(fixed["iterations_used"], 3)
        self.assertEqual(fixed["termination"], "iteration_budget")
        self.assertAlmostEqual(early["initial_smoothed_objective"], .5e-6)

    def test_inputs_and_rng_are_unchanged_and_aggregator_has_no_history(self):
        clients, broadcast = submissions([[.2, .3], [.4, .1], [1., 2.]])
        hashes = [state_sha256(broadcast), *[state_sha256(c.state) for c in clients]]
        rng = torch.random.get_rng_state().clone()
        aggregator = RFAAggregator()
        a = aggregator.aggregate(clients, broadcast, 1)
        b = aggregator.aggregate(clients, broadcast, 99)
        self.assertEqual(hashes, [state_sha256(broadcast), *[state_sha256(c.state) for c in clients]])
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        torch.testing.assert_close(flat(a.state), flat(b.state), rtol=0, atol=0)
        self.assertFalse(aggregator.settings["secure_aggregation_implemented"])

    def test_invalid_solver_settings_and_weights_are_rejected(self):
        for settings in [{"max_iterations": 0}, {"max_iterations": True}, {"smoothing": 0},
                         {"smoothing": float('nan')}, {"relative_tolerance": -1}]:
            with self.subTest(settings=settings), self.assertRaises(ValueError): RFAConfig(**settings)
        for weights in [[1], [1, 0], [1, -1], [1, float('inf')]]:
            with self.subTest(weights=weights), self.assertRaises(ValueError):
                smoothed_weiszfeld(torch.zeros((2, 3)), weights)
        with self.assertRaises(FloatingPointError):
            smoothed_weiszfeld(torch.tensor([[float('nan')]]), [1])

    def test_incompatible_or_nonfinite_submissions_are_rejected(self):
        clients, broadcast = submissions([[0., 0.], [.2, .3]])
        for invalid in [[], [clients[0], clients[0]], [replace(clients[0], num_examples=0)],
                        [replace(clients[0], state={"a": torch.zeros(5), "b": torch.zeros(1)})],
                        [replace(clients[0], state={"a": torch.tensor([float('inf')], dtype=torch.float64),
                                                  "b": torch.zeros(1, dtype=torch.float64)})]]:
            with self.subTest(invalid=invalid), self.assertRaises((ValueError, FloatingPointError)):
                rfa_rule(invalid, broadcast)


if __name__ == "__main__":
    unittest.main()
