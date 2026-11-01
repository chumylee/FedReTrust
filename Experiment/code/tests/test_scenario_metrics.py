"""Scientific metric checks: known answers, censoring, semantics and integration."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import copy
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baseline_calibration import ServerLearningRate
from experiment_core import (ClientSubmission, FedAvgAggregator, TrainConfig,
                             clone_state, run_experiment, state_sha256)
from fltrust import fltrust_rule
from rfa import RFAAggregator
from scenario_metrics import (ClasswiseEvaluator, ContributionRecorder, PRIMARY_SCENARIOS,
    PERFORMANCE_SCENARIOS, SECONDARY_CORE_SCENARIOS, OBSERVABILITY_CHECK_SCENARIOS,
    classify_contribution_episode, classify_model_episode, coefficient_ratios,
    contribution_norm_ratios, contribution_reduction_event, contribution_restoration_event,
    group_contribution_norm, influence_recovery, legitimate_task_weights,
    longer_scenarios, model_recovery, paired_damage, primary_development_score,
    schedule_rows, suppression_event, sustained_event, validate_matched_runs, weighted_task_score)
from scenario_utils import Scenario, client_view, truth_row
from viceroy import ViceroyAggregator


def series(value, horizon=24):
    return {r: value for r in range(1, horizon + 1)}


def submissions(points):
    origin = {"vector": torch.tensor([4., -2.], dtype=torch.float64)}
    clients = [ClientSubmission(i, i + 1, {"vector": origin["vector"] + torch.tensor(p, dtype=torch.float64)})
               for i, p in enumerate(points)]
    return clients, origin


class FixedRoot:
    name = "FLTrust"
    settings = {}
    options = SimpleNamespace(global_learning_rate=1.)
    def aggregate(self, clients, broadcast, round_number):
        return fltrust_rule(clients, broadcast, torch.tensor([2., 0.], dtype=torch.float64))


class ScenarioMetricTests(unittest.TestCase):
    def test_schedule_boundaries_ramp_and_recovery_opportunities(self):
        catalog = longer_scenarios()
        self.assertEqual(len(catalog), 11)
        self.assertEqual(len(PERFORMANCE_SCENARIOS), 10)
        self.assertEqual(len(schedule_rows()), 2640)
        for scenario in catalog.values():
            self.assertFalse(any(scenario.active(c, r) for c in range(10) for r in range(1, 5)))
        event = catalog["gradual_stripe"].events[0]
        self.assertEqual([event.strength(r) for r in range(4, 14)], [0., .125, .25, .375, .5, .625, .75, .875, 1., 1.])
        self.assertTrue(catalog["attack_stops"].active(0, 10))
        self.assertFalse(catalog["attack_stops"].active(0, 11))
        self.assertEqual([r for r in range(1, 25) if catalog["recurring_attack"].active(0, r)], list(range(5, 9)) + list(range(15, 19)))
        self.assertEqual(24 - 10, 14)
        self.assertEqual(SECONDARY_CORE_SCENARIOS, ("honest_change_stops",))
        self.assertEqual(OBSERVABILITY_CHECK_SCENARIOS, ("ambiguous_malicious_input",))

    def test_temporary_honest_change_returns_to_clean_without_label_change(self):
        rng = np.random.default_rng(820)
        data = torch.from_numpy(rng.integers(0, 256, (40, 28, 28), dtype=np.uint8))
        pool = SimpleNamespace(data=data, targets=torch.arange(40).remainder(10))
        scenario = longer_scenarios()["honest_change_stops"]
        original_data, original_targets = data.clone(), pool.targets.clone()
        views = [client_view(pool, list(range(40)), {"mean": [.13], "std": [.3]}, 42, scenario, 0, r)
                 for r in [4, 5, 10, 11, 24]]
        self.assertEqual([int(v.changed.sum()) for v in views], [0, 40, 40, 0, 0])
        for v in views:
            self.assertFalse(v.flip_labels)
            self.assertEqual([v[i][1] for i in range(40)], [int(y) for y in original_targets])
        for i in range(40):
            torch.testing.assert_close(views[0][i][0], views[3][i][0], rtol=0, atol=0)
            torch.testing.assert_close(views[0][i][0], views[4][i][0], rtol=0, atol=0)
        self.assertTrue(torch.equal(data, original_data))
        self.assertTrue(torch.equal(pool.targets, original_targets))
        counts = np.eye(10)*100
        active = legitimate_task_weights(scenario, 5, counts)
        returned = legitimate_task_weights(scenario, 11, counts)
        self.assertAlmostEqual(active["shot_noise_s3"].sum(), .3)
        self.assertAlmostEqual(returned["shot_noise_s3"].sum(), 0.)
        self.assertAlmostEqual(returned["clean"].sum(), 1.)
        truth = truth_row(scenario, 0, 11, views[3])
        self.assertTrue(truth["honest_drift_client"])
        self.assertEqual(truth["input_changed_examples"], 0)
        stopped = longer_scenarios()["attack_stops"]
        stopped_view = client_view(pool, list(range(40)), {"mean": [.13], "std": [.3]}, 42, stopped, 0, 11)
        attack_truth = truth_row(stopped, 0, 11, stopped_view)
        self.assertTrue(attack_truth["malicious_client"])
        self.assertFalse(attack_truth["attack_active"])

    def test_mixed_control_preserves_drift_and_removes_only_attack(self):
        mixed = longer_scenarios()["mixed_drift_and_attack"]
        control = longer_scenarios()["mixed_drift_only_control"]
        self.assertEqual(tuple(e for e in mixed.events if e.intent == "honest"), control.events)
        for r in range(1, 25):
            for c in range(3, 10):
                self.assertEqual(mixed.active(c, r), control.active(c, r))

    def test_ambiguous_intent_produces_identical_observable_tensors(self):
        pool = SimpleNamespace(data=torch.arange(20 * 28 * 28).reshape(20, 28, 28).remainder(256).byte(),
                               targets=torch.arange(20).remainder(10))
        a, b = longer_scenarios()["abrupt_shot_noise"], longer_scenarios()["ambiguous_malicious_input"]
        for r in [4, 5, 24]:
            left = client_view(pool, list(range(20)), {"mean": [.13], "std": [.3]}, 27, a, 0, r)
            right = client_view(pool, list(range(20)), {"mean": [.13], "std": [.3]}, 27, b, 0, r)
            for i in range(20):
                torch.testing.assert_close(left[i][0], right[i][0], rtol=0, atol=0)
                self.assertEqual(left[i][1], right[i][1])

    def test_legitimate_task_mixture_uses_full_client_digit_frequencies(self):
        counts = np.eye(10) * 100
        scenario = longer_scenarios()["gradual_stripe"]
        weights = legitimate_task_weights(scenario, 8, counts)
        np.testing.assert_allclose(weights["stripe"], [.05] * 3 + [0.] * 7)
        np.testing.assert_allclose(weights["clean"], [.05] * 3 + [.1] * 7)
        changed = legitimate_task_weights(scenario, 12, counts, (0, 1, 2))
        np.testing.assert_allclose(changed["stripe"], [1/3] * 3 + [0.] * 7)
        self.assertAlmostEqual(sum(v.sum() for v in changed.values()), 1.)
        attack = legitimate_task_weights(longer_scenarios()["label_flip"], 12, counts)
        np.testing.assert_allclose(attack["clean"], [.1] * 10)
        self.assertEqual(attack["stripe"].sum(), 0.)

    def test_weighted_score_known_non_iid_result_and_missing_class_rejection(self):
        def row(correct):
            return {**{f"class_examples_{i}": 10 for i in range(10)},
                    **{f"class_correct_{i}": correct for i in range(10)},
                    **{f"class_loss_sum_{i}": 10. for i in range(10)}}
        tasks = {"clean": row(9), "shot_noise_s3": row(5), "stripe": row(4)}
        weights = legitimate_task_weights(longer_scenarios()["abrupt_shot_noise"], 5, np.ones((10, 10)))
        result = weighted_task_score(tasks, weights)
        self.assertAlmostEqual(result["accuracy"], .7 * .9 + .3 * .5)
        self.assertAlmostEqual(result["loss"], 1.)
        tasks["shot_noise_s3"]["class_examples_0"] = 0
        with self.assertRaises(ValueError): weighted_task_score(tasks, weights)
        with self.assertRaises(ValueError): weighted_task_score(tasks, {"clean": np.ones(10)})

    def test_classwise_evaluator_exact_statistics_without_training(self):
        logits = torch.zeros(4, 10)
        logits[0, 0], logits[1, 0], logits[2, 1], logits[3, 2] = 2., 2., 2., 2.
        labels = torch.tensor([0, 1, 1, 2])
        result = ClasswiseEvaluator().evaluate(nn.Identity(), TensorDataset(logits, labels), SimpleNamespace(evaluation_batch_size=2))
        self.assertEqual(result["examples"], 4)
        self.assertEqual(result["accuracy"], .75)
        self.assertEqual(result["class_examples_1"], 2)
        self.assertEqual(result["class_correct_1"], 1)
        self.assertEqual(result["class_examples_9"], 0)
        self.assertAlmostEqual(result["loss"], float(nn.functional.cross_entropy(logits.double(), labels)), places=6)
        self.assertEqual(sum(result[f"class_correct_{i}"] for i in range(10)), 3)

    def test_primary_objective_equal_scenario_round_weights_and_no_missing_runs(self):
        values = {name: {r: i / 10 for r in range(5, 25)} for i, name in enumerate(PRIMARY_SCENARIOS)}
        self.assertAlmostEqual(primary_development_score(values)["accuracy"], .35)
        values["clean"].pop(24)
        with self.assertRaises(ValueError): primary_development_score(values)
        values["clean"][24] = float("nan")
        with self.assertRaises(ValueError): primary_development_score(values)

    def test_recorder_preserves_four_baseline_models_and_applied_coefficients(self):
        clients, broadcast = submissions([[5., 0.], [3., 4.], [-2., .5]])
        before = [state_sha256(broadcast), *[state_sha256(s.state) for s in clients]]
        for factory in [FedAvgAggregator, FixedRoot, RFAAggregator, ViceroyAggregator]:
            for rate in [None, .5, 1., 2.]:
                base = factory() if rate is None else ServerLearningRate(factory(), rate)
                expected = (factory() if rate is None else ServerLearningRate(factory(), rate)).aggregate(clients, broadcast, 1)
                result = ContributionRecorder(base).aggregate(clients, broadcast, 1)
                self.assertEqual(state_sha256(result.state), state_sha256(expected.state))
                stats = result.diagnostics["contribution_measurements"]
                self.assertAlmostEqual(group_contribution_norm(stats, (0, 1, 2)), stats["aggregate_l2_norm"])
                for c in range(3):
                    self.assertAlmostEqual(group_contribution_norm(stats, (c,)), result.diagnostics["client_metrics"][str(c)]["applied_contribution_l2_norm"])
                if rate is not None or factory != FixedRoot:
                    self.assertEqual(result.coefficients, expected.coefficients)
        self.assertEqual(before, [state_sha256(broadcast), *[state_sha256(s.state) for s in clients]])

    def test_fltrust_recorder_converts_native_shares_before_comparison(self):
        clients, broadcast = submissions([[5., 0.], [3., 4.]])
        result = ContributionRecorder(FixedRoot()).aggregate(clients, broadcast, 1)
        self.assertAlmostEqual(result.coefficients[0], .25)
        self.assertAlmostEqual(result.coefficients[1], .15)
        self.assertAlmostEqual(result.diagnostics["client_metrics"]["0"]["recorder_native_coefficient"], .625)

    def test_cancellation_and_projection_are_not_positive_trust(self):
        clients, broadcast = submissions([[1., 0.], [-.5, 0.]])
        result = ContributionRecorder(FedAvgAggregator()).aggregate(clients, broadcast, 1)
        stats = result.diagnostics["contribution_measurements"]
        self.assertAlmostEqual(stats["aggregate_l2_norm"], 0.)
        self.assertEqual(stats["cancellation_ratio"], 0.)
        self.assertTrue(stats["zero_aggregate_from_cancellation"])
        self.assertFalse(stats["zero_contribution_mass"])
        self.assertIsNone(result.diagnostics["client_metrics"]["0"]["aggregate_projection_share"])
        clients, broadcast = submissions([[1., 0.], [-.25, 0.]])
        result = ContributionRecorder(FedAvgAggregator()).aggregate(clients, broadcast, 1)
        self.assertAlmostEqual(result.diagnostics["client_metrics"]["0"]["aggregate_projection_share"], 2.)
        self.assertAlmostEqual(result.diagnostics["client_metrics"]["1"]["aggregate_projection_share"], -1.)

    def test_recorder_history_reset_and_zero_mass(self):
        clients, broadcast = submissions([[1., 0.]] * 3)
        recorder = ContributionRecorder(ViceroyAggregator())
        result = recorder.aggregate(clients, broadcast, 1)
        self.assertEqual(result.diagnostics["contribution_measurements"]["sum_individual_norms"], 0.)
        self.assertIsNone(result.diagnostics["contribution_measurements"]["cancellation_ratio"])
        self.assertTrue(result.diagnostics["contribution_measurements"]["zero_contribution_mass"])
        self.assertFalse(result.diagnostics["contribution_measurements"]["zero_aggregate_from_cancellation"])
        recorder.reset()
        self.assertIsNone(recorder.base.histories)

    def test_recovery_exact_percentage_point_boundary_and_confirmation(self):
        ref, obs = series(.9), series(.87)
        for r in range(11, 25): obs[r] = .89
        result = model_recovery(ref, obs, 10)
        self.assertEqual(result["streak_start_round"], 11)
        self.assertEqual(result["confirmation_round"], 13)
        self.assertEqual(result["rounds_to_streak_start"], 1)
        self.assertEqual(result["rounds_to_confirmation"], 3)
        self.assertFalse(result["criterion_met_at_attack_end"])

    def test_recurrence_breaks_recovery_streak_and_is_right_censored(self):
        ref, obs = series(.9), series(.8)
        obs[13] = obs[14] = .9
        for r in range(15, 25): obs[r] = .9
        result = model_recovery(ref, obs, 8, next_attack=15)
        self.assertTrue(result["right_censored"])
        self.assertEqual(result["censor_reason"], "next_attack")
        self.assertEqual(result["last_eligible_round"], 14)
        final = model_recovery(ref, obs, 18)
        self.assertEqual(final["streak_start_round"], 19)
        self.assertEqual(final["confirmation_round"], 21)

    def test_non_recovery_no_time_is_invented_and_no_post_attack_rounds(self):
        result = model_recovery(series(.9), series(.8), 10)
        self.assertTrue(result["right_censored"])
        self.assertIsNone(result["rounds_to_confirmation"])
        self.assertEqual(result["eligible_rounds"], 14)
        result = model_recovery(series(.9), series(.9), 24)
        self.assertEqual(result["eligible_rounds"], 0)
        self.assertTrue(result["right_censored"])

    def test_model_recovery_does_not_imply_client_influence_recovery(self):
        self.assertFalse(model_recovery(series(.9), series(.9), 10)["right_censored"])
        self.assertTrue(influence_recovery(series(.1), series(.01), 10)["right_censored"])
        recovered = series(.01)
        for r in range(11, 25): recovered[r] = .08
        self.assertEqual(influence_recovery(series(.1), recovered, 10)["confirmation_round"], 13)

    def test_contribution_measurement_is_invariant_to_fltrust_input_magnitude(self):
        broadcast = {"vector": torch.zeros(2, dtype=torch.float64)}
        def recorded(scale):
            clients = [ClientSubmission(0, 1, {"vector":torch.tensor([5.*scale, 0.], dtype=torch.float64)}),
                       ClientSubmission(1, 1, {"vector":torch.tensor([3., 4.], dtype=torch.float64)})]
            result = ContributionRecorder(FixedRoot()).aggregate(clients, broadcast, 1)
            return result, result.diagnostics["client_metrics"]["0"]
        original, m0 = recorded(1)
        enlarged, m1 = recorded(10)
        self.assertEqual(state_sha256(original.state), state_sha256(enlarged.state))
        self.assertAlmostEqual(m1["applied_delta_coefficient"]/m0["applied_delta_coefficient"], .1)
        self.assertAlmostEqual(m1["applied_contribution_l2_norm"]/m0["applied_contribution_l2_norm"], 1.)
        coeff_ref, coeff_obs = series(m0["applied_delta_coefficient"]), series(m1["applied_delta_coefficient"])
        norm_ref, norm_obs = series(m0["applied_contribution_l2_norm"]), series(m1["applied_contribution_l2_norm"])
        self.assertEqual(suppression_event(coeff_ref,coeff_obs,5,24)["confirmation_round"], 7)
        self.assertTrue(contribution_reduction_event(norm_ref,norm_obs,5,24)["right_censored"])
        self.assertEqual(classify_contribution_episode(norm_ref,norm_obs,5,10)["status"],
                         "no_qualifying_impairment_observed")
        self.assertTrue(influence_recovery(coeff_ref,coeff_obs,10)["right_censored"])
        self.assertEqual(contribution_restoration_event(norm_ref,norm_obs,10)["confirmation_round"], 13)

    def test_actual_contribution_reduction_restoration_zero_reference_and_cancellation(self):
        ref, obs = series(.1), series(.1)
        for r in range(5,17): obs[r] = 0.
        for r in range(17,25): obs[r] = .08
        reduction = contribution_reduction_event(ref,obs,5,10)
        self.assertEqual(reduction["confirmation_round"], 7)
        self.assertEqual(reduction["reduced_client_rounds"], 6)
        episode = classify_contribution_episode(ref,obs,5,10)
        self.assertEqual(episode["status"], "post_event_restored")
        self.assertEqual(episode["observed_restoration_delay_rounds"], 9)
        self.assertEqual(episode["post_event_qualifying_streak"]["streak_start_round"], 17)
        ref[6] = ref[8] = 0
        self.assertIsNone(contribution_norm_ratios(ref,obs)[6])
        self.assertTrue(contribution_reduction_event(ref,obs,5,10)["right_censored"])
        self.assertEqual(contribution_reduction_event(ref,obs,5,10)["undefined_rounds"], 2)
        all_zero = series(0.)
        self.assertEqual(contribution_reduction_event(series(.1),all_zero,5,10)["confirmation_round"],7)
        self.assertEqual(classify_contribution_episode(all_zero,all_zero,5,10)["status"],"unassessable")

    def test_recovery_episode_statuses_respect_prior_impairment_and_censoring(self):
        ref, intact = series(.9), series(.9)
        self.assertEqual(classify_model_episode(ref,intact,5,10)["status"],"no_qualifying_impairment_observed")
        impaired = series(.9)
        for r in range(5,11): impaired[r] = .85
        recovered = classify_model_episode(ref,impaired,5,10)
        self.assertEqual(recovered["status"],"post_event_restored")
        self.assertEqual(recovered["post_event_qualifying_streak"]["streak_start_round"],11)
        self.assertEqual(recovered["observed_restoration_delay_rounds"],3)
        early = series(.9)
        early[5] = .85
        self.assertEqual(classify_model_episode(ref,early,5,10)["status"],"within_criterion_at_event_end")
        persistent = series(.85)
        self.assertEqual(classify_model_episode(ref,persistent,5,8,next_event=15)["status"],
                         "restoration_not_observed_before_censoring")
        self.assertEqual(classify_model_episode(ref,persistent,5,8,next_event=15)
                         ["post_event_qualifying_streak"]["censor_reason"],"next_attack")
        delayed = series(.9)
        delayed[12] = .85
        self.assertEqual(classify_model_episode(ref,delayed,5,10)["status"],"delayed_post_event_impairment")
        with self.assertRaises(ValueError): classify_model_episode(ref,intact,0,10)

    def test_suppression_strict_boundary_zero_reference_and_false_suppression_denominator(self):
        ref, obs = series(.1), series(.05)
        self.assertTrue(suppression_event(ref, obs, 5, 10)["right_censored"])
        obs[5] = obs[6] = obs[7] = .049
        result = suppression_event(ref, obs, 5, 10)
        self.assertEqual(result["confirmation_round"], 7)
        self.assertEqual(result["rounds_from_onset_to_confirmation"], 3)
        self.assertEqual(result["defined_client_rounds"], 6)
        self.assertEqual(result["suppressed_client_rounds"], 3)
        ref[6] = 0
        self.assertIsNone(coefficient_ratios(ref, obs)[6])
        result = suppression_event(ref, obs, 5, 10)
        self.assertTrue(result["right_censored"])
        self.assertEqual(result["undefined_rounds"], 1)
        self.assertEqual(result["defined_client_rounds"], 5)

    def test_signed_damage_retains_improvement_and_positive_area_units(self):
        ref, obs = series(.9), series(.9)
        obs.update({5: .8, 6: .95, 7: .9})
        result = paired_damage(ref, obs, ((5, 7),))
        np.testing.assert_allclose(result["damage_pp"], [10., -5., 0.])
        self.assertAlmostEqual(result["positive_damage_area_pp_rounds"], 10.)
        self.assertAlmostEqual(result["mean_signed_damage_pp"], 5/3)
        with self.assertRaises(ValueError): paired_damage(ref, obs, ((5, 7), (7, 9)))

    def test_pair_validation_rejects_changed_seed_method_code_or_wrong_control(self):
        fields = dict(config={"rounds": 24, "seed": 3}, method={"name": "FedAvg"}, partition_sha256="a",
                      selection_sha256="b", initial_state_sha256="c", code_sha256={"x": "d"}, versions={"python": "3"},
                      validation_tasks=["clean"], evaluation_recorder={"name": "classwise"})
        observed = {**fields, "scenario": longer_scenarios()["mixed_drift_and_attack"].to_dict()}
        reference = {**fields, "scenario": longer_scenarios()["mixed_drift_only_control"].to_dict()}
        self.assertTrue(validate_matched_runs(json.loads(json.dumps(observed)), json.loads(json.dumps(reference))))
        for key in ["config", "method", "code_sha256", "selection_sha256"]:
            invalid = copy.deepcopy(reference)
            invalid[key] = "different"
            with self.assertRaises(ValueError): validate_matched_runs(observed, invalid)
        reference["scenario"] = longer_scenarios()["clean"].to_dict()
        with self.assertRaises(ValueError): validate_matched_runs(observed, reference)

    def test_identical_observations_yield_identical_stateful_trajectories(self):
        # A bounded 24-round integration across all ten clients. Local deltas
        # depend on the actual scenario views; only the simulator intent differs.
        rng = np.random.default_rng(82046)
        pool = SimpleNamespace(data=torch.from_numpy(rng.integers(0,256,(40,28,28),dtype=np.uint8)),
                               targets=torch.arange(40).remainder(10))
        honest = longer_scenarios()["abrupt_shot_noise"]
        malicious_intent = longer_scenarios()["ambiguous_malicious_input"]
        for factory in [FedAvgAggregator, FixedRoot, RFAAggregator, ViceroyAggregator]:
            left_method, right_method = factory(), factory()
            left, right = {"vector":torch.tensor([.1,.2],dtype=torch.float64)}, {"vector":torch.tensor([.1,.2],dtype=torch.float64)}
            for round_number in range(1,25):
                submissions_by_intent=[]
                for scenario in [honest,malicious_intent]:
                    submitted=[]
                    for client in range(10):
                        view=client_view(pool,list(range(4*client,4*client+4)),{"mean":[.13],"std":[.3]},321,
                                         scenario,client,round_number)
                        values=[view[i] for i in range(4)]
                        mean=float(torch.stack([x for x,_ in values]).mean())
                        labels=sum(y for _,y in values)
                        delta=torch.tensor([.01+.02*mean,.03+.001*labels],dtype=torch.float64)
                        submitted.append(ClientSubmission(client,4,{"vector":left["vector"]+delta}))
                    submissions_by_intent.append(submitted)
                for one,two in zip(*submissions_by_intent):
                    self.assertEqual(state_sha256(one.state),state_sha256(two.state))
                left_result=left_method.aggregate(submissions_by_intent[0],left,round_number)
                right_result=right_method.aggregate(submissions_by_intent[1],right,round_number)
                self.assertEqual(state_sha256(left_result.state),state_sha256(right_result.state))
                self.assertEqual(left_result.coefficients,right_result.coefficients)
                left,right=left_result.state,right_result.state
            if isinstance(left_method,ViceroyAggregator):
                torch.testing.assert_close(left_method.histories,right_method.histories,rtol=0,atol=0)
                torch.testing.assert_close(left_method.reputations,right_method.reputations,rtol=0,atol=0)

    def test_invalid_missing_data_thresholds_and_flags_fail(self):
        bad = series(.9); bad.pop(12)
        with self.assertRaises(ValueError): model_recovery(bad, series(.8), 10)
        with self.assertRaises(ValueError): model_recovery(series(.9), series(.8), 10, next_attack=9)
        with self.assertRaises(ValueError): model_recovery(series(.9), series(.8), 10, tolerance_pp=-1)
        with self.assertRaises(ValueError): influence_recovery(series(.1), series(.1), 10, minimum_ratio=0)
        with self.assertRaises(ValueError): sustained_event({1: 1}, 1, 1)
        with self.assertRaises(ValueError): sustained_event({1: True, 2: True, 3: True, 4: "invalid"}, 1, 4)
        with self.assertRaises(ValueError): legitimate_task_weights(Scenario("clean"), 1, np.zeros((10, 10)))

    def test_optional_evaluator_and_recorder_integrate_with_common_runner(self):
        # Run actual recording/aggregation/evaluation with deterministic toy
        # client updates. No MNIST training, official test, or benchmark run.
        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(1, 10)
            def forward(self, x): return self.linear(x)
        pool = SimpleNamespace(data=torch.zeros(10, 28, 28, dtype=torch.uint8), targets=torch.arange(10))
        tasks = {"clean": TensorDataset(torch.ones(10, 1), torch.arange(10))}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "resources/mnist_c_reference").mkdir(parents=True)
            (root / "resources/mnist_c_reference/source_manifest.json").write_text("{}")
            config = replace(TrainConfig(), rounds=1)
            data = SimpleNamespace(root=root, pool=pool, manifest={"normalization": {"mean": [.13], "std": [.3]}},
                                   manifest_sha256="toy", selection={"clients": {str(c): [c] for c in range(10)},
                                   "selection_seed": config.development_seed}, validation_tasks=tasks)
            def toy_train(model, view, options, client, round_number, broadcast):
                state = clone_state(model)
                state["linear.bias"][client] += .1
                return state, {"optimizer_steps": 0, "shuffle_seed": 0, "online_train_loss": 0.}
            with patch("experiment_core.MnistCNN", TinyModel), patch("experiment_core.train_client", toy_train):
                plain = run_experiment(data, config, Scenario("toy_check"), FedAvgAggregator(), verbose=False)
                out = run_experiment(data, config, Scenario("toy_check"), ContributionRecorder(FedAvgAggregator()),
                                     verbose=False, evaluator=ClasswiseEvaluator())
            self.assertEqual(json.loads((out / "status.json").read_text())["status"], "complete")
            self.assertIn("class_examples_9", (out / "task_metrics.csv").read_text())
            self.assertIn("evaluation_recorder", json.loads((out / "run.json").read_text()))
            diagnostics = json.loads((out / "aggregation_diagnostics.json").read_text())[0]["values"]
            self.assertEqual(diagnostics["contribution_measurements"]["client_order"], list(range(10)))
            self.assertTrue(json.loads((out / "summary.json").read_text())["checkpoint_logits_identical"])
            self.assertEqual(state_sha256(torch.load(plain / "final_global_model_state.pt", weights_only=True)),
                             state_sha256(torch.load(out / "final_global_model_state.pt", weights_only=True)))
            self.assertNotIn("evaluation_recorder", json.loads((plain / "run.json").read_text()))


if __name__ == "__main__": unittest.main()
