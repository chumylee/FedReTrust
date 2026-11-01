"""Source parity and stateful protocol checks for Viceroy."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import ast
import json
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiment_core import ClientSubmission, file_sha256, state_sha256
from viceroy import ViceroyAggregator, ViceroyConfig, fedscale


def submissions(points, counts=None, origin=None):
    points = np.asarray(points, dtype=np.float64)
    origin = np.zeros(points.shape[1]) if origin is None else np.asarray(origin, dtype=np.float64)
    broadcast = {"a": torch.tensor(origin[:1]), "b": torch.tensor(origin[1:])}
    clients = [ClientSubmission(i, int(n), {"a": torch.tensor((point + origin)[:1]),
               "b": torch.tensor((point + origin)[1:])}) for i, (point, n) in
               enumerate(zip(points, counts if counts is not None else [1] * len(points)))]
    return clients, broadcast


def flat(state):
    return torch.cat([value.flatten() for value in state.values()])


def numpy_cosine(x):
    # Independent NumPy equivalent of sklearn's cosine_similarity on these
    # finite, ordinary-sized dense vectors (including zero rows).
    norms = np.sqrt(np.sum(x * x, axis=1, keepdims=True))
    normalized = x / np.where(norms == 0, 1, norms)
    return normalized.dot(normalized.T)


def pinned_author_reference():
    """Extract inspected numeric methods; no legacy training module imports.

    NumPy substitutes for JAX arrays, and the independent cosine function
    substitutes for sklearn. This checks the author's arithmetic/state order,
    not compatibility with their original runtime or whole training pipeline.
    """
    root = Path(__file__).resolve().parents[2] / "resources/viceroy_reference"
    manifest = json.loads((root / "source_manifest.json").read_text())
    for name, item in manifest["files"].items():
        assert file_sha256(root / name) == item["sha256"], name
    tree = ast.parse((root / "fl/server/viceroy.py").read_text())
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
    for node in nodes:
        if isinstance(node, ast.FunctionDef): node.decorator_list = []
    class Base:
        def __init__(self, *args): pass
    namespace = {"sys": sys, "np": np, "jnp": np,
                 "smp": SimpleNamespace(cosine_similarity=numpy_cosine),
                 "server": SimpleNamespace(Server=Base),
                 "jax": SimpleNamespace(flatten_util=SimpleNamespace(ravel_pytree=lambda p: (np.asarray(p).ravel(), None)))}
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
                 str(root / "fl/server/viceroy.py"), "exec"), namespace)
    return namespace


class ViceroyTests(unittest.TestCase):
    def test_matches_pinned_author_scale_on_random_and_clustered_vectors(self):
        reference = pinned_author_reference()["scale"]
        rng = np.random.default_rng(45)
        for _ in range(20):
            x = rng.normal(size=(10, 13))
            x[1:4] = x[0] + rng.normal(scale=.02, size=(3, 13))
            np.testing.assert_allclose(fedscale(torch.tensor(x)), reference(x), rtol=1e-11, atol=1e-12)

    def test_twenty_round_state_and_aggregate_match_author_methods(self):
        reference = pinned_author_reference()["Server"](np.zeros(9), None, None, [None] * 6)
        aggregator = ViceroyAggregator()
        rng = np.random.default_rng(65)
        for r in range(1, 21):
            x = rng.normal(size=(6, 9))
            if r % 4 in (0, 1): x[1:3] = x[0] + .01 * x[1:3]
            reference.update(list(x))
            coefficients = reference.scale(list(x))
            clients, broadcast = submissions(x)
            got = aggregator.aggregate(clients, broadcast, r)
            np.testing.assert_allclose(list(got.coefficients.values()), coefficients, rtol=1e-11, atol=1e-12)
            np.testing.assert_allclose(aggregator.histories, reference.histories, rtol=1e-13, atol=1e-14)
            np.testing.assert_allclose(aggregator.reputations, reference.reps, rtol=1e-11, atol=1e-12)
            np.testing.assert_allclose(flat(got.state), np.sum(coefficients[:, None] * x, axis=0), rtol=1e-11, atol=1e-12)

    def test_defaults_and_first_round_follow_author_initialization(self):
        options = ViceroyConfig()
        self.assertEqual(options.omega, sys.float_info.epsilon ** (1 / 56))
        self.assertEqual(options.eta, 1 / 5)
        clients, broadcast = submissions([[1., 0.], [0., 1.], [-1., 0.]])
        agg = ViceroyAggregator()
        result = agg.aggregate(clients, broadcast, 1)
        self.assertFalse(result.diagnostics["reputation_update_applied"])
        torch.testing.assert_close(agg.reputations, torch.ones(3, dtype=torch.float64))
        torch.testing.assert_close(agg.histories, torch.tensor([[1., 0.], [0., 1.], [-1., 0.]], dtype=torch.float64))

    def test_orthogonal_updates_use_raw_sum_not_weighted_mean(self):
        clients, broadcast = submissions([[1., 0.], [0., 1.]], [1, 19], [5., -3.])
        got = ViceroyAggregator().aggregate(clients, broadcast, 1)
        self.assertEqual(got.coefficients, {0: 1., 1: 1.})
        self.assertEqual(got.diagnostics["coefficient_mass"], 2.)
        torch.testing.assert_close(flat(got.state), torch.tensor([6., -2.], dtype=torch.float64))

    def test_duplicate_direction_pair_suppressed_and_distinct_direction_retained(self):
        got = fedscale(torch.tensor([[1., 0.], [2., 0.], [0., 1.]], dtype=torch.float64))
        torch.testing.assert_close(got, torch.tensor([0., 0., 1.], dtype=torch.float64), rtol=0, atol=0)

    def test_identical_nonzero_directions_use_explicit_zero_mass_policy(self):
        clients, broadcast = submissions([[1., 2.], [2., 4.], [3., 6.]], origin=[7., -2.])
        agg = ViceroyAggregator()
        got = agg.aggregate(clients, broadcast, 1)
        self.assertTrue(got.diagnostics["zero_total_weight"])
        torch.testing.assert_close(flat(got.state), flat(broadcast), rtol=0, atol=0)
        self.assertGreater(float(torch.linalg.vector_norm(agg.histories)), 0.)

    def test_zero_updates_have_unit_scales_and_keep_broadcast(self):
        clients, broadcast = submissions([[0., 0.]] * 3, origin=[7., 2.])
        agg = ViceroyAggregator()
        for r in range(1, 4):
            got = agg.aggregate(clients, broadcast, r)
            self.assertEqual(got.coefficients, {0: 1., 1: 1., 2: 1.})
            torch.testing.assert_close(flat(got.state), flat(broadcast), rtol=0, atol=0)

    def test_reputation_uses_old_history_but_weight_uses_new_history(self):
        agg = ViceroyAggregator()
        first = torch.tensor([[1., 0.], [1., 0.], [0., 1.]], dtype=torch.float64)
        second = torch.tensor([[1., 0.], [0., 1.], [-1., 0.]], dtype=torch.float64)
        agg.aggregate(*submissions(first.numpy()), 1)
        got = agg.aggregate(*submissions(second.numpy()), 2)
        expected_reps = (torch.ones(3) + .1 * (1 - 2 * abs(fedscale(first) - fedscale(second)))).clamp(0, 1)
        expected_history = agg.options.omega * first + second
        expected_weights = expected_reps * fedscale(expected_history) + (1 - expected_reps) * fedscale(second)
        torch.testing.assert_close(agg.reputations, expected_reps.double())
        torch.testing.assert_close(agg.histories, expected_history)
        torch.testing.assert_close(torch.tensor(list(got.coefficients.values()), dtype=torch.float64), expected_weights)
        self.assertEqual(float(agg.reputations[0]), .9)

    def test_history_decays_across_zero_observations(self):
        agg = ViceroyAggregator()
        initial = np.array([[1., 0.], [0., 1.], [-1., 0.]])
        agg.aggregate(*submissions(initial), 1)
        for r in range(2, 8): agg.aggregate(*submissions(np.zeros_like(initial)), r)
        np.testing.assert_allclose(agg.histories, initial * agg.options.omega ** 6, rtol=1e-14, atol=1e-15)

    def test_sample_counts_do_not_change_viceroy(self):
        a = ViceroyAggregator().aggregate(*submissions([[1., 0.], [0., 1.]], [1, 1]), 1)
        b = ViceroyAggregator().aggregate(*submissions([[1., 0.], [0., 1.]], [3, 99]), 1)
        torch.testing.assert_close(flat(a.state), flat(b.state), rtol=0, atol=0)

    def test_client_order_does_not_change_identity_linked_history(self):
        a, b = ViceroyAggregator(), ViceroyAggregator()
        rng = np.random.default_rng(13)
        for r in range(1, 4):
            clients, broadcast = submissions(rng.normal(size=(4, 3)))
            left, right = a.aggregate(clients, broadcast, r), b.aggregate(clients[::-1], broadcast, r)
            self.assertEqual(left.coefficients, right.coefficients)
            torch.testing.assert_close(a.histories, b.histories, rtol=0, atol=0)

    def test_reset_removes_all_cross_run_history(self):
        agg = ViceroyAggregator()
        clients, broadcast = submissions([[1., 0.], [0., 1.]])
        first = agg.aggregate(clients, broadcast, 1)
        agg.aggregate(*submissions([[1., 0.], [1., 0.]]), 2)
        agg.reset()
        second = agg.aggregate(clients, broadcast, 1)
        self.assertEqual(first.coefficients, second.coefficients)
        self.assertEqual(first.diagnostics, second.diagnostics)

    def test_snapshot_is_independent_and_captures_client_state(self):
        agg = ViceroyAggregator()
        agg.aggregate(*submissions([[1., 0.], [0., 1.]]), 1)
        saved = agg.snapshot()
        saved["histories"].zero_(); saved["reputations"].zero_()
        self.assertEqual(float(agg.histories.sum()), 2.)
        self.assertEqual(float(agg.reputations.sum()), 2.)
        self.assertEqual(saved["client_ids"], [0, 1])

    def test_inputs_and_rng_are_not_modified(self):
        clients, broadcast = submissions([[1., 0.], [0., 1.]], origin=[7., -2.])
        hashes = [state_sha256(broadcast), *[state_sha256(c.state) for c in clients]]
        rng = torch.random.get_rng_state().clone()
        ViceroyAggregator().aggregate(clients, broadcast, 1)
        self.assertEqual(hashes, [state_sha256(broadcast), *[state_sha256(c.state) for c in clients]])
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))

    def test_round_sequence_membership_and_layout_changes_rejected(self):
        agg = ViceroyAggregator()
        clients, broadcast = submissions([[1., 0.], [0., 1.]])
        with self.assertRaises(ValueError): agg.aggregate(clients, broadcast, 2)
        agg.aggregate(clients, broadcast, 1)
        for r in [1, 3, True]:
            with self.assertRaises(ValueError): agg.aggregate(clients, broadcast, r)
        with self.assertRaises(ValueError): agg.aggregate(clients[:1], broadcast, 2)
        reversed_layout = dict(reversed(list(broadcast.items())))
        with self.assertRaises(ValueError): agg.aggregate(clients, reversed_layout, 2)
        self.assertEqual(agg.last_round, 1)

    def test_single_client_applies_its_delta(self):
        clients, broadcast = submissions([[.2, -.3]], origin=[4., 9.])
        got = ViceroyAggregator().aggregate(clients, broadcast, 1)
        self.assertEqual(got.coefficients, {0: 1.})
        torch.testing.assert_close(flat(got.state), flat(clients[0].state), rtol=0, atol=0)

    def test_nonfinite_and_incompatible_inputs_rejected_without_state_change(self):
        clients, broadcast = submissions([[1., 0.], [0., 1.]])
        agg = ViceroyAggregator()
        invalid = [[], [clients[0], clients[0]], [replace(clients[0], num_examples=0)],
                   [replace(clients[0], state={"a": torch.zeros(3), "b": torch.zeros(1)})],
                   [replace(clients[0], state={"a": torch.tensor([float('nan')], dtype=torch.float64), "b": torch.zeros(1, dtype=torch.float64)})]]
        for group in invalid:
            with self.assertRaises((ValueError, FloatingPointError)): agg.aggregate(group, broadcast, 1)
            self.assertEqual(agg.last_round, 0)

    def test_invalid_options_rejected(self):
        for options in [{"omega": -1}, {"omega": float('nan')}, {"eta": 0}, {"eta": 2},
                        {"numerical_epsilon": 0}, {"numerical_epsilon": float('inf')}]:
            with self.assertRaises(ValueError): ViceroyConfig(**options)


if __name__ == "__main__":
    unittest.main()
