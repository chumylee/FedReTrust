"""Known-answer, tensor-contract and failure-atomicity checks for the draft."""
import copy
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiment_core import ClientSubmission, FedAvgAggregator
from fedretrust import (FedReTrustAggregator, FedReTrustConfig, TrustState,
                       clipped_geometry, evidence_category, transition)
from scenario_metrics import ContributionRecorder
from baseline_calibration import ServerLearningRate

ROOT = Path(__file__).resolve().parents[2]


class ReferenceStub:
    metadata = {"purpose": "synthetic arithmetic test; not MNIST"}


class ToyAggregator(FedReTrustAggregator):
    def _reference_evidence(self, broadcast, clipped, p, usable, round_number):
        self.probes = [self.options.server_learning_rate*float(p[i])*clipped[i].clone()
                       for i in range(len(p))]
        losses = getattr(self, "forced_losses", [.99]*len(p))
        return torch.ones_like(clipped[0]), 1., losses, {"synthetic_reference": True}


def toy(options=None):
    return ToyAggregator(ReferenceStub(), 1, {i: 10 for i in range(10)}, options)


def submissions(points=None):
    points = torch.eye(10, dtype=torch.float64) if points is None else points
    return [ClientSubmission(i, 10, {"x": points[i].clone()}) for i in range(10)]


class ScalarChecks(unittest.TestCase):
    def test_saved_independent_step5_traces(self):
        records = json.loads((ROOT/"outputs/fedretrust_design/20261011/attempt_0001/worked_traces.json").read_text())
        for name, rows in records["traces"].items():
            with self.subTest(trace=name):
                state = TrustState()
                for row in rows:
                    state, q, detail = transition(state, row["evidence"], row["round"], FedReTrustConfig())
                    self.assertAlmostEqual(state.trust, row["trust"], places=12)
                    self.assertAlmostEqual(q, row["q_applied"], places=12)
                    self.assertEqual(state.label, row["state"])
                    self.assertEqual(len(state.onsets), row["episode_count"])
                    self.assertEqual(detail["new_episode"], row["new_episode"])

    def test_saved_threshold_cases(self):
        records = json.loads((ROOT/"outputs/fedretrust_design/20261011/attempt_0001/worked_traces.json").read_text())
        for kwargs, expected in records["threshold_examples"]:
            with self.subTest(kwargs=kwargs):
                defaults = dict(root=None, peer=None, history=None, clipped=False, usable=True)
                defaults.update(kwargs)
                self.assertEqual(evidence_category(**defaults, options=FedReTrustConfig())[0], expected)

    def test_allowance_liveness_and_run_boundaries(self):
        state, total = TrustState(), 0.
        for t in range(1, 101):
            state, q, _ = transition(state, "B", t, FedReTrustConfig())
            total += q
        self.assertLessEqual(total, 4+1e-12)
        self.assertEqual(state.onsets, ())  # Continuous adversity is not a new episode every round.
        for t in range(101, 115):
            state, q, _ = transition(state, "G", t, FedReTrustConfig())
        self.assertEqual(state.trust, 1)
        self.assertEqual(state.allowance, 4)

    def test_ablation_interventions(self):
        old = TrustState(trust=.1, good=2, onsets=(4, 8))
        full = transition(old, "G", 10, FedReTrustConfig())[0]
        no_repeat = transition(old, "G", 10, FedReTrustConfig(variant="no_repeat_memory"))[0]
        no_recovery = transition(old, "G", 10, FedReTrustConfig(variant="no_recovery"))[0]
        self.assertEqual(full.trust, .1)
        self.assertAlmostEqual(no_repeat.trust, .35)
        self.assertEqual(no_recovery.trust, .1)
        state = TrustState()
        for t in (1, 2):
            state, q, detail = transition(state, "U", t, FedReTrustConfig(variant="ambiguous_as_adverse"))
        self.assertEqual(state.trust, .1)
        self.assertTrue(detail["new_episode"])
        state = TrustState(allowance=0)
        _, q, _ = transition(state, "U", 1, FedReTrustConfig(variant="no_finite_allowance"))
        self.assertEqual(q, .5)

    def test_invalid_configurations(self):
        for change in [{"server_learning_rate":0}, {"clip_multiplier":float("nan")},
                       {"confirmation_rounds":True}, {"episode_window_rounds":0},
                       {"penalty_cap":.8}, {"direction_history_decay":1}, {"variant":"invented"},
                       {"non_supportive_allowance":0}, {"severe_loss_multiplier":1}]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                FedReTrustConfig(**change)

    def test_policy_json_agrees(self):
        p = json.loads((ROOT/"fedretrust_policy_draft.json").read_text())
        cfg = asdict(FedReTrustConfig())
        for key, value in p["illustrative_parameters"].items():
            self.assertEqual(cfg[key], value)


class TensorChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.broadcast = {"x": torch.zeros(10, dtype=torch.float64)}

    def test_fedavg_limit_and_input_immutability(self):
        subs, agg = submissions(), toy()
        before = copy.deepcopy(subs)
        actual = agg.aggregate(subs, self.broadcast, 1)
        expected = FedAvgAggregator().aggregate(subs, self.broadcast, 1)
        torch.testing.assert_close(actual.state["x"], expected.state["x"], rtol=0, atol=0)
        self.assertEqual(actual.coefficients, expected.coefficients)
        self.assertTrue(torch.equal(self.broadcast["x"], torch.zeros(10)))
        for a, b in zip(subs, before):
            self.assertTrue(torch.equal(a.state["x"], b.state["x"]))

    def test_actual_coefficients_and_recorder(self):
        points = torch.eye(10, dtype=torch.float64)
        points[7:] *= 100
        agg = toy()
        result = ContributionRecorder(agg).aggregate(submissions(points), self.broadcast, 1)
        self.assertEqual(result.diagnostics["clip_radius"], 2)
        for i in range(7, 10):
            self.assertAlmostEqual(result.coefficients[i], .002)
        expected = sum((result.coefficients[i]*points[i] for i in range(10)))
        torch.testing.assert_close(result.state["x"], expected, rtol=0, atol=0)
        self.assertLessEqual(float(torch.linalg.vector_norm(expected)), 2)

    def test_zero_radius_no_reference_and_no_passive_recovery(self):
        agg = toy()
        agg.states[0] = TrustState(trust=0, label="Penalised")
        agg._reference_evidence = lambda *args: self.fail("Zero radius must skip reference work.")
        result = agg.aggregate(submissions(torch.zeros(10, 10, dtype=torch.float64)), self.broadcast, 1)
        self.assertTrue(all(v == 0 for v in result.coefficients.values()))
        self.assertEqual(agg.states[0].trust, 0)
        self.assertEqual(result.diagnostics["reference_forward_examples"], 0)

    def test_zero_trust_still_full_share_probed(self):
        agg = toy()
        agg.states[0] = TrustState(trust=0, label="Penalised")
        result = agg.aggregate(submissions(), self.broadcast, 1)
        self.assertEqual(result.coefficients[0], 0)
        torch.testing.assert_close(agg.probes[0], .1*torch.eye(10, dtype=torch.float64)[0])
        result = agg.aggregate(submissions(), self.broadcast, 2)
        self.assertAlmostEqual(agg.states[0].trust, .25)
        self.assertAlmostEqual(result.coefficients[0], .025)

    def test_severe_reference_harm_and_fixed_denominator(self):
        agg = toy()
        agg.forced_losses = [1.03]*10
        result = agg.aggregate(submissions(), self.broadcast, 1)
        self.assertEqual(result.diagnostics["trust_mass"], 0)
        self.assertTrue(torch.equal(result.state["x"], self.broadcast["x"]))
        agg.forced_losses = [1.01]*10  # Aligned but adverse: U.
        for i in agg.states:
            agg.states[i] = TrustState(trust=.1)
        result = agg.aggregate(submissions(), self.broadcast, 2)
        self.assertAlmostEqual(sum(result.coefficients.values()), .1)

    def test_mass_only_redistribution(self):
        full, mass = toy(), toy(FedReTrustConfig(variant="mass_only"))
        full.forced_losses = mass.forced_losses = [1.03]+[.99]*9
        a = full.aggregate(submissions(), self.broadcast, 1)
        b = mass.aggregate(submissions(), self.broadcast, 1)
        self.assertEqual(a.coefficients[0], 0)
        self.assertAlmostEqual(b.coefficients[0], .09)
        self.assertAlmostEqual(sum(a.coefficients.values()), sum(b.coefficients.values()))
        self.assertEqual(b.diagnostics["client_metrics"]["0"]["policy_multiplier"], 0)

    def test_no_history_has_no_persistent_direction_effect(self):
        agg = toy(FedReTrustConfig(variant="no_history"))
        a = agg.aggregate(submissions(), self.broadcast, 1)
        b = agg.aggregate(submissions(), self.broadcast, 2)
        self.assertEqual(a.coefficients, b.coefficients)
        self.assertEqual(float(torch.linalg.vector_norm(agg.direction_memory)), 0)

    def test_clipped_fedavg_needs_no_reference(self):
        agg = FedReTrustAggregator(None, 1, {i:10 for i in range(10)},
                                  FedReTrustConfig(variant="clipped_FedAvg"))
        result = agg.aggregate(submissions(), self.broadcast, 1)
        torch.testing.assert_close(result.state["x"], torch.full((10,), .1, dtype=torch.float64))

    def test_reference_failure_is_atomic(self):
        agg = toy()
        agg.aggregate(submissions(), self.broadcast, 1)
        snapshot = copy.deepcopy(agg.states)
        memory = agg.direction_memory.clone()
        agg.forced_losses = [.99]*9+[float("nan")]
        with self.assertRaises(FloatingPointError):
            agg.aggregate(submissions(), self.broadcast, 2)
        self.assertEqual(agg.states, snapshot)
        self.assertEqual(agg.last_round, 1)
        self.assertTrue(torch.equal(agg.direction_memory, memory))

    def test_bad_input_and_count_spoofing(self):
        for kind in ("count", "membership", "nan", "shape", "dtype"):
            agg, subs = toy(), submissions()
            if kind == "count": subs[0].num_examples = 999
            if kind == "membership": subs.pop()
            if kind == "nan": subs[0].state["x"][0] = float("nan")
            if kind == "shape": subs[0].state["x"] = torch.ones(9, dtype=torch.float64)
            if kind == "dtype": subs[0].state["x"] = subs[0].state["x"].float()
            with self.subTest(kind=kind), self.assertRaises((ValueError, FloatingPointError)):
                agg.aggregate(subs, self.broadcast, 1)
            self.assertEqual(agg.last_round, 0)

    def test_layout_rounds_and_reset(self):
        agg = toy()
        with self.assertRaises(ValueError): agg.aggregate(submissions(), self.broadcast, 2)
        agg.aggregate(submissions(), self.broadcast, 1)
        with self.assertRaises(ValueError): agg.aggregate(submissions(), self.broadcast, 1)
        agg.reset()
        self.assertEqual(agg.last_round, 0)
        self.assertIsNone(agg.direction_memory)
        agg.aggregate(submissions(), self.broadcast, 1)

    def test_internal_server_rate_and_external_double_scaling_rejected(self):
        agg = toy(FedReTrustConfig(server_learning_rate=2))
        result = agg.aggregate(submissions(), self.broadcast, 1)
        self.assertAlmostEqual(result.coefficients[0], .2)
        torch.testing.assert_close(agg.probes[0], .2*torch.eye(10, dtype=torch.float64)[0])
        with self.assertRaises(ValueError): ServerLearningRate(agg, 2)

    def test_saved_population_weights_separate_from_selected_training_counts(self):
        counts = {i: 10*(i+1) for i in range(10)}
        agg = ToyAggregator(ReferenceStub(), 1, counts, submission_counts={i:10 for i in range(10)})
        result = agg.aggregate(submissions(), self.broadcast, 1)
        for i in range(10):
            self.assertAlmostEqual(result.coefficients[i], counts[i]/sum(counts.values()))

    def test_history_is_previous_direction_and_zero_initially(self):
        agg = toy()
        first = agg.aggregate(submissions(), self.broadcast, 1)
        self.assertIsNone(first.diagnostics["client_metrics"]["0"]["history_cosine"])
        points = -torch.eye(10, dtype=torch.float64)
        second = agg.aggregate(submissions(points), self.broadcast, 2)
        self.assertAlmostEqual(second.diagnostics["client_metrics"]["0"]["history_cosine"], -1)
        self.assertAlmostEqual(float(agg.direction_memory[0,0]), -.25)

    def test_zero_majority_clips_every_client_and_skips_probes(self):
        points = torch.eye(10, dtype=torch.float64)
        points[:6] = 0
        agg = toy()
        agg._reference_evidence = lambda *args: self.fail("Median radius is zero.")
        result = agg.aggregate(submissions(points), self.broadcast, 1)
        self.assertEqual(result.diagnostics["clip_radius"], 0)
        self.assertEqual(sum(result.coefficients.values()), 0)

    def test_post_decision_overflow_keeps_state_atomic(self):
        agg = toy(FedReTrustConfig(server_learning_rate=1e308))
        points = torch.eye(10, dtype=torch.float64)*1e100
        with self.assertRaises(FloatingPointError):
            agg.aggregate(submissions(points), self.broadcast, 1)
        self.assertEqual(agg.last_round, 0)
        self.assertIsNone(agg.direction_memory)
        self.assertTrue(all(s.trust == 1 for s in agg.states.values()))


if __name__ == "__main__":
    unittest.main()
