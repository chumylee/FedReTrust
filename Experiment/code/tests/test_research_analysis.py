"""Known-answer tests for inference and evaluation-only task matching."""
from pathlib import Path
import sys
import unittest
import json
from unittest.mock import patch

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import research_analysis as analysis


def record(scenario="honest_change_stops",method="FedReTrust",seed=1):
    return {"job_id":"synthetic","status":"complete","job":{"phase":"final","regime":"IID",
            "seed":seed,"method":method,"scenario":scenario,"rounds":24}}


class ResearchAnalysisTests(unittest.TestCase):
    def test_paired_interval_known_answer_and_incomplete_seed_guard(self):
        result=analysis.interval([1,2,3,4,5])
        self.assertEqual(result["mean"],3)
        self.assertAlmostEqual(result["sd"],np.sqrt(2.5))
        self.assertAlmostEqual(result["ci_low"],1.036756838522439,places=10)
        incomplete=analysis.interval([100,100,100,100])
        self.assertFalse(incomplete["complete_seed_coverage"])
        self.assertIsNone(incomplete["ci_low"])
        with self.assertRaises(ValueError): analysis.interval([1,2,3,4,5,6])

    def test_reference_is_scored_on_observed_honest_task(self):
        observed=record();reference=record("clean")
        calls=[]
        def task_series(result,scenario_name=None,client_ids=None):
            calls.append((result["job"]["scenario"],scenario_name,client_ids))
            return {r:.8 for r in range(25)}
        with patch.object(analysis,"task_series",side_effect=task_series),patch.object(analysis,"checked_pair"):
            metrics,curves,episodes=analysis.extract_run(observed,reference)
        self.assertIn(("clean","honest_change_stops",None),calls)
        self.assertIn(("clean","honest_change_stops",[0,1,2]),calls)
        self.assertEqual(episodes[0]["status"],"no_qualifying_impairment_observed")
        self.assertNotIn("delayed_post_event_impairment_observed",episodes[0])
        self.assertFalse(episodes[0]["any_post_event_impairment_observed"])
        self.assertAlmostEqual(next(r["value"] for r in metrics if r["metric"]=="affected_honest_return_11_24_pp"),80.)

    def test_recurrence_damage_uses_same_second_window(self):
        observed=record("recurring_attack");reference=record("clean")
        def task_series(result,**kwargs):
            values={r:.9 for r in range(25)}
            if result["job"]["scenario"]=="recurring_attack":
                values.update({r:.7 for r in range(5,9)})
                values.update({r:.85 for r in range(15,19)})
            return values
        with patch.object(analysis,"task_series",side_effect=task_series),patch.object(analysis,"checked_pair"):
            metrics,_,episodes=analysis.extract_run(observed,reference)
        self.assertAlmostEqual(next(r["value"] for r in metrics if r["metric"]=="second_attack_signed_damage_pp"),5.)
        self.assertAlmostEqual(next(r["value"] for r in metrics if r["metric"]=="attack_active_signed_damage_pp"),12.5)
        self.assertEqual(episodes[0]["post_event_qualifying_streak"]["last_eligible_round"],14)
        self.assertEqual(episodes[1]["post_event_qualifying_streak"]["eligible_rounds"],6)

    def test_primary_requires_both_regimes_all_guards_and_coverage(self):
        blank=analysis.assess_primary([],False)
        self.assertFalse(blank["overall_predeclared_criterion_met"])
        rows=[]
        for item in blank["criteria"]:
            value=2. if item["criterion"]=="gain" else 0.
            rows.append({"regime":item["regime"],"scenario":item["scenario"],"metric":item["metric"],
                         "comparator":"Viceroy",**analysis.interval([value]*5)})
        self.assertTrue(analysis.assess_primary(rows,True)["overall_predeclared_criterion_met"])
        self.assertFalse(analysis.assess_primary(rows,False)["overall_predeclared_criterion_met"])
        rows[-1]["ci_high"]=1.01
        self.assertFalse(analysis.assess_primary(rows,True)["overall_predeclared_criterion_met"])

    def test_honest_composite_requires_all_four_and_equal_scenario_weight(self):
        rows=[]
        for i,name in enumerate(analysis.HONEST_SCENARIOS):
            rows.append({"phase":"final","regime":"IID","seed":1,"method":"FedReTrust", "scenario":name,
                         "metric":"affected_honest_event_pp","value":70+10*i})
        self.assertEqual(len(analysis.add_honest_composite(rows[:-1].copy())),3)
        result=analysis.add_honest_composite(rows.copy())
        self.assertEqual(result[-1]["value"],85.)
        self.assertEqual(result[-1]["scenario"],"honest_composite")
        json.dumps(result,allow_nan=False)

    def test_threshold_rates_keep_zero_denominators_undefined_and_are_serializable(self):
        fields={"phase":"final","regime":"IID","seed":1,"method":"FedReTrust","scenario":"attack_stops",
                "group":"attacking","event_start":5,"event_end":10,"maximum_ratio":.5}
        rows=[{**fields,"client":i,"defined_client_rounds":6 if i<2 else 0,"reduced_client_rounds":3 if i<2 else 0} for i in range(3)]
        result=analysis.reduction_seed_rates(rows)
        self.assertEqual(result[0]["total_client_rounds"],18)
        self.assertEqual(result[0]["defined_client_rounds"],12)
        self.assertEqual(result[0]["reduction_fraction"],.5)
        self.assertIsNone(analysis.reduction_seed_rates([rows[2]])[0]["reduction_fraction"])
        json.dumps(result,allow_nan=False)
        episodes=[{"regime":"IID","episode":1,"status":"post_event_restored"},
                  {"regime":"IID","episode":1,"status":"restoration_not_observed_before_censoring"}]
        counts=analysis.episode_status_counts(episodes,["regime","episode"])
        self.assertEqual(counts[0]["episodes"],2)
        self.assertEqual(len(counts[0]["counts"]),2)
        json.dumps(counts,allow_nan=False)

    def test_paired_contrasts_match_seeds_before_subtracting(self):
        rows=[];index={}
        for seed in range(1,6):
            for method in ("FedReTrust","FedAvg"):
                rows.append({"phase":"final","regime":"IID","seed":seed,"method":method,"scenario":"clean",
                             "metric":"overall_mean_5_24_pp","value":seed*10+(2 if method=="FedReTrust" else 0)})
                index[("IID",seed,method,"clean")]=record("clean",method,seed)
        with patch.object(analysis,"cross_method_pair"):
            result=analysis.paired_contrasts(list(reversed(rows)),index)
        self.assertEqual(len(result),1)
        self.assertEqual(result[0]["paired_differences"],[2]*5)
        self.assertEqual(result[0]["sd"],0)


if __name__=="__main__": unittest.main()
