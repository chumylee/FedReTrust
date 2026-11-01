"""Boundary, source-equivalence and simulator/defence separation checks."""
import ast
from dataclasses import fields
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np
import torch

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
from experiment_core import ClientSubmission, FedAvgAggregator, MnistCNN, TrainConfig, clone_state, file_sha256, state_sha256, train_client
from scenario_utils import Event, Scenario, ScenarioDataset, client_view, gaussian_update_corruption, mnist_c_corruption, scenario_catalog, seed_for, truth_row


class ScenarioTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        rng = np.random.default_rng(19)
        cls.pool = SimpleNamespace(data=torch.from_numpy(rng.integers(0, 256, (20, 28, 28), dtype=np.uint8)),
                                   targets=torch.arange(20) % 10)
        cls.indices = list(range(20))
        cls.norm = {"mean": [0.1307], "std": [0.3081]}

    def view(self, scenario, client=0, round_number=3):
        return client_view(self.pool, self.indices, self.norm, 20261014, scenario, client, round_number)

    def test_recipes_match_pinned_upstream_functions(self):
        path = CODE.parent / "resources/mnist_c_reference/corruptions.py"
        self.assertEqual(file_sha256(path), "fb1b451715d9ba837e1778241bf959242156b8d0c66c9c01a3fa2f70e9bbfa49")
        parsed = ast.parse(path.read_text())
        functions = [n for n in parsed.body if isinstance(n, ast.FunctionDef) and n.name in {"shot_noise", "stripe"}]
        self.assertEqual(len(functions), 2)
        for severity in (1, 3, 5):
            proxy = SimpleNamespace(array=np.array, clip=np.clip, float32=np.float32, random=np.random.default_rng(71))
            namespace = {"np": proxy}
            exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)
            image = self.pool.data[0].numpy()
            np.testing.assert_array_equal(namespace["shot_noise"](image, severity),
                                          mnist_c_corruption(image, "shot_noise", severity, np.random.default_rng(71)))
            np.testing.assert_array_equal(namespace["stripe"](image), mnist_c_corruption(image, "stripe"))

    def test_inclusive_stop_and_recurring_boundaries(self):
        catalog = scenario_catalog()
        stop = catalog["attack_stops"].events[0]
        self.assertEqual([stop.strength(r) for r in range(1, 7)], [0, 0, 1, 1, 0, 0])
        recurring = catalog["recurring_attack"].events[0]
        self.assertEqual([recurring.strength(r) for r in range(1, 7)], [0, 1, 1, 0, 1, 1])
        for scenario in catalog.values():
            scenario.validate(list(range(10)), 6)

    def test_gradual_changes_are_nested_and_exact(self):
        scenario = scenario_catalog()["gradual_stripe"]
        counts, previous = [], set()
        for round_number in range(1, 7):
            view = self.view(scenario, round_number=round_number)
            current = set(np.flatnonzero(view.changed))
            self.assertTrue(previous.issubset(current))
            previous = current
            counts.append(len(current))
        self.assertEqual(counts, [0, 5, 10, 15, 20, 20])

    def test_labels_images_and_unaffected_clients(self):
        data_before, labels_before = self.pool.data.clone(), self.pool.targets.clone()
        catalog = scenario_catalog()
        clean = self.view(catalog["clean"])
        flipped = self.view(catalog["label_flip"])
        drifted = self.view(catalog["abrupt_shot_noise"])
        unaffected = self.view(catalog["abrupt_shot_noise"], client=9)
        before = self.view(catalog["abrupt_shot_noise"], round_number=2)
        after = self.view(catalog["attack_stops"], round_number=5)
        for i in range(20):
            for view in (flipped, unaffected, before, after):
                torch.testing.assert_close(clean[i][0], view[i][0], rtol=0, atol=0)
            self.assertEqual(flipped[i][1], 9 - clean[i][1])
            self.assertEqual(drifted[i][1], clean[i][1])
            torch.testing.assert_close(drifted[i][0], self.view(catalog["abrupt_shot_noise"])[i][0], rtol=0, atol=0)
        self.assertTrue(torch.equal(data_before, self.pool.data))
        self.assertTrue(torch.equal(labels_before, self.pool.targets))

    def test_clean_normalization_matches_torchvision(self):
        from torchvision import transforms
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        view = self.view(scenario_catalog()["clean"])
        for i in range(20):
            torch.testing.assert_close(view[i][0], transform(self.pool.data[i].numpy()), rtol=0, atol=0)

    def test_gaussian_corruption_uses_local_delta_variance(self):
        global_state = {"w": torch.tensor([2., -3., 1., 0.])}
        local = {"w": torch.tensor([3., 1., -1., 0.])}
        original = local["w"].clone()
        result, sigma = gaussian_update_corruption(local, global_state, 9, 2, 4)
        self.assertEqual(sigma, float(torch.tensor([1., 4., -2., 0.], dtype=torch.float64).var(correction=0).sqrt()))
        noise = np.random.default_rng(seed_for(9, "gaussian_update", 4, 2)).normal(0, sigma, 4)
        expected = (original.double() + torch.from_numpy(noise)).float()
        torch.testing.assert_close(result["w"], expected, rtol=0, atol=0)
        self.assertTrue(torch.equal(local["w"], original))
        zero, zero_sigma = gaussian_update_corruption(global_state, global_state, 9, 2, 4)
        self.assertEqual(zero_sigma, 0)
        self.assertTrue(torch.equal(zero["w"], global_state["w"]))

    def test_identical_observations_do_not_reveal_intent(self):
        honest = Scenario("honest", (Event("input_change", (0,), ((1, 1),), corruption="stripe"),))
        malicious = Scenario("malicious", (Event("input_change", (0,), ((1, 1),), corruption="stripe", intent="malicious"),))
        honest_view = self.view(honest, round_number=1)
        malicious_view = self.view(malicious, round_number=1)
        torch.manual_seed(20)
        model = MnistCNN()
        broadcast = clone_state(model)
        config = TrainConfig(rounds=1)
        h_state, _ = train_client(model, honest_view, config, 0, 1, broadcast)
        m_state, _ = train_client(model, malicious_view, config, 0, 1, broadcast)
        self.assertEqual(state_sha256(h_state), state_sha256(m_state))
        self.assertEqual(state_sha256(clone_state(model)), state_sha256(broadcast))
        self.assertFalse(truth_row(honest, 0, 1, honest_view)["attack_active"])
        self.assertTrue(truth_row(malicious, 0, 1, malicious_view)["attack_active"])

    def test_submission_contains_no_oracle_fields(self):
        self.assertEqual([f.name for f in fields(ClientSubmission)], ["client_id", "num_examples", "state"])

    def test_fedavg_unequal_sizes_and_single_client(self):
        small = ClientSubmission(0, 1, {"w": torch.tensor([1., 4.])})
        large = ClientSubmission(1, 3, {"w": torch.tensor([9., 0.])})
        method = FedAvgAggregator()
        result = method.aggregate([small, large], {}, 1)
        torch.testing.assert_close(result.state["w"], torch.tensor([7., 1.]), rtol=0, atol=0)
        self.assertEqual(result.coefficients, {0: .25, 1: .75})
        torch.testing.assert_close(method.aggregate([small], {}, 1).state["w"], small.state["w"], rtol=0, atol=0)
        with self.assertRaises(ValueError):
            method.aggregate([small, small], {}, 1)

    def test_invalid_or_ambiguous_schedules_are_rejected(self):
        with self.assertRaises(ValueError):
            Event("label_flip", (0,), ((0, 2),), intent="malicious")
        event = Event("input_change", (0,), ((1, 3),), corruption="stripe")
        with self.assertRaises(ValueError):
            Scenario("overlap", (event, event)).validate([0], 3)
        with self.assertRaises(ValueError):
            Scenario("outside", (event,)).validate([0], 2)
        with self.assertRaises(ValueError):
            Scenario("../bad").validate([0], 2)
        with self.assertRaises(ValueError):
            ScenarioDataset(self.pool, [0, 0], .1307, .3081, 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
