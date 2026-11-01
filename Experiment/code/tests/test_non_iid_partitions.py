"""Check data membership, exact margins, integer rounding and descriptive metrics."""
from dataclasses import replace
from pathlib import Path
import copy
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from non_iid_partitions import (LabelSkewConfig, class_counts, distribution_metrics,
                                fit_margins, prepare_label_skew, round_margins)


class NonIIDPartitionTests(unittest.TestCase):
    def setUp(self):
        self.labels = np.tile(np.arange(4), 100)
        self.clients = {"0": list(range(20, 140)), "1": list(range(140, 260)), "2": list(range(260, 380))}
        self.options = LabelSkewConfig(partition_seed=821, num_classes=4)

    def test_original_pool_is_covered_once_with_exact_client_and_class_totals(self):
        result, detail = prepare_label_skew(self.labels, self.clients, self.options)
        self.assertEqual(sorted(i for group in result.values() for i in group), list(range(20, 380)))
        counts = class_counts(self.labels, result, 4)
        np.testing.assert_array_equal(counts.sum(axis=1), [120, 120, 120])
        np.testing.assert_array_equal(counts.sum(axis=0), [90, 90, 90, 90])
        np.testing.assert_array_equal(counts, detail["integer_counts"])
        self.assertLess(detail["rounding_maximum_cell_error"], 1)

    def test_unequal_sizes_are_retained_without_dropping_or_duplicating_images(self):
        original = {"0": list(range(10, 50)), "1": list(range(50, 160)), "2": list(range(160, 385))}
        result, detail = prepare_label_skew(self.labels, original, self.options)
        self.assertEqual([len(result[str(i)]) for i in range(3)], [40, 110, 225])
        self.assertEqual(sorted(i for group in result.values() for i in group), list(range(10, 385)))
        np.testing.assert_array_equal(np.array(detail["integer_counts"]).sum(axis=0), np.bincount(self.labels[10:385]))

    def test_repeatability_input_order_independence_and_input_immutability(self):
        before_clients, before_labels = copy.deepcopy(self.clients), self.labels.copy()
        a, da = prepare_label_skew(self.labels, self.clients, self.options)
        reordered = {key: list(reversed(value)) for key, value in reversed(list(self.clients.items()))}
        b, db = prepare_label_skew(self.labels, reordered, self.options)
        self.assertEqual(a, b)
        self.assertEqual(da, db)
        self.assertEqual(before_clients, self.clients)
        np.testing.assert_array_equal(before_labels, self.labels)

    def test_different_seed_changes_allocation_and_preserves_pool(self):
        a, da = prepare_label_skew(self.labels, self.clients, self.options)
        b, db = prepare_label_skew(self.labels, self.clients, replace(self.options, partition_seed=822))
        self.assertNotEqual(a, b)
        self.assertNotEqual(da["integer_counts"], db["integer_counts"])
        self.assertEqual(sorted(i for g in a.values() for i in g), sorted(i for g in b.values() for i in g))

    def test_generation_does_not_consume_numpy_global_rng(self):
        np.random.seed(117)
        expected = np.random.random(5)
        np.random.seed(117)
        prepare_label_skew(self.labels, self.clients, self.options)
        np.testing.assert_array_equal(np.random.random(5), expected)

    def test_rounding_can_reroute_an_earlier_assignment(self):
        # Greedy first choices use columns 0 and 1, blocking row 2. A reverse
        # residual edge must move one earlier assignment into column 2.
        table = np.array([[.5, 0, .5], [0, .5, .5], [.5, .5, 0]])
        rounded = round_margins(table, [1, 1, 1], [1, 1, 1])
        np.testing.assert_array_equal(rounded, [[0, 0, 1], [0, 1, 0], [1, 0, 0]])

    def test_many_positive_tables_preserve_margins_and_floor_ceiling_bounds(self):
        rng = np.random.default_rng(224)
        for _ in range(20):
            row_sizes = [37, 51, 62]
            class_sizes = [13, 19, 40, 78]
            fitted, _ = fit_margins(rng.uniform(.01, 1, (3, 4)), row_sizes, class_sizes, self.options)
            counts = round_margins(fitted, row_sizes, class_sizes)
            np.testing.assert_array_equal(counts.sum(axis=1), row_sizes)
            np.testing.assert_array_equal(counts.sum(axis=0), class_sizes)
            self.assertTrue(np.all((counts == np.floor(fitted)) | (counts == np.ceil(fitted))))

    def test_empty_class_and_single_client_remain_well_defined(self):
        result, detail = prepare_label_skew(np.array([0, 0, 1, 1]), {"0": [0, 1, 2, 3]},
                                           replace(self.options, num_classes=3))
        self.assertEqual(sorted(result["0"]), [0, 1, 2, 3])
        self.assertEqual(detail["integer_counts"], [[2, 2, 0]])

    def test_entropy_tv_and_js_match_independent_analytic_examples(self):
        stats = distribution_metrics([[2, 0], [0, 2]])
        np.testing.assert_allclose(stats["entropy_bits"], [0, 0])
        np.testing.assert_allclose(stats["effective_classes"], [1, 1])
        np.testing.assert_allclose(stats["total_variation_from_global"], [.5, .5])
        expected_js = .5 * np.log2(1/.75) + .25 * np.log2(.5/.75) + .25
        np.testing.assert_allclose(stats["js_divergence_from_global_bits"], [expected_js, expected_js])
        uniform = distribution_metrics([[2, 2], [2, 2]])
        np.testing.assert_allclose(uniform["entropy_bits"], [1, 1])
        np.testing.assert_allclose(uniform["js_divergence_from_global_bits"], [0, 0])

    def test_invalid_inputs_and_nonconvergence_fail_explicitly(self):
        for alpha in (0, -1, True, float("nan"), float("inf")):
            with self.assertRaises(ValueError): LabelSkewConfig(alpha=alpha)
        for seed in (-1, True, 1.5):
            with self.assertRaises(ValueError): LabelSkewConfig(partition_seed=seed)
        for clients in ({"0": [1, 1]}, {"1": [1]}, {"0": []}, {"0": [500]}, {"0": [-1]}, {"0": [True]}):
            with self.assertRaises(ValueError): prepare_label_skew(self.labels, clients, self.options)
        with self.assertRaises(ValueError): prepare_label_skew(self.labels.astype(float), self.clients, self.options)
        with self.assertRaises(ValueError): round_margins([[.5, .5]], [2], [1, 1])
        with self.assertRaises(ValueError): round_margins([[1]], [1.5], [1])
        with self.assertRaises(RuntimeError):
            fit_margins([[10., 1.], [1., 1.]], [1, 3], [2, 2], replace(self.options, max_balance_iterations=1))


if __name__ == "__main__":
    unittest.main()
