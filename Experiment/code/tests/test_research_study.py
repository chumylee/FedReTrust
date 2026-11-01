"""Scientific guardrails: test isolation, complete selection and exact manifests."""
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import research_runs as runs
import research_study as study
from experiment_core import TrainConfig
from scenario_metrics import longer_scenarios, validate_matched_runs


class ResearchStudyTests(unittest.TestCase):
    def test_official_dataset_is_never_constructed_without_freeze(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(runs.datasets,"MNIST") as constructor:
            with self.assertRaises(ValueError):
                runs.attach_official_test(None,TrainConfig(),Path(temp)/"missing.json",{"phase":"final"})
            constructor.assert_not_called()

    def test_changed_source_and_unregistered_jobs_block_test_access(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(runs.datasets,"MNIST") as constructor:
            path=Path(temp)/"freeze.json"
            path.write_text(json.dumps({"status":"frozen_before_official_test","training_source_sha256":{}}))
            with self.assertRaises(ValueError):
                runs.attach_official_test(None,TrainConfig(),path,{"phase":"final"})
            with patch.object(runs,"verify_freeze",return_value={"authorized_final_jobs":[]}):
                with self.assertRaises(ValueError):
                    runs.attach_official_test(None,TrainConfig(),path,{"phase":"final"})
            constructor.assert_not_called()

    def test_development_phase_and_unregistered_seed_cannot_request_test(self):
        request={"phase":"development"}
        frozen={"authorized_final_jobs":[request],"final_seeds":[20261014],
                "workload":{"train_images_per_client":512,"rounds":24}}
        with patch.object(runs,"verify_freeze",return_value=frozen), patch.object(runs.datasets,"MNIST") as constructor:
            for cfg in (TrainConfig(development_seed=99,rounds=24),TrainConfig(rounds=24)):
                with self.assertRaises(ValueError):
                    runs.attach_official_test(None,cfg,"unused",request)
            constructor.assert_not_called()

    def test_authorized_test_attachment_keeps_training_pool_separate(self):
        class SyntheticPool:
            data=torch.zeros(10000,28,28,dtype=torch.uint8)
            targets=torch.arange(10000)%10
            def __len__(self): return 10000
        request={"phase":"final"}
        frozen={"authorized_final_jobs":[request],"final_seeds":[20261101],
                "workload":{"train_images_per_client":512,"rounds":24}}
        original_pool=object()
        data=SimpleNamespace(pool=original_pool,selection={"clients":{"0":[1]},"development_validation":[2]},
                             manifest={"dataset_files":{name:{"raw_sha256":"synthetic"} for name in
                                         ("t10k-images-idx3-ubyte","t10k-labels-idx1-ubyte")},
                                       "indices":{"reserved_final_test":list(range(10000))},
                                       "normalization":{"mean":[.1307],"std":[.3081]}})
        with patch.object(runs,"verify_freeze",return_value=frozen),patch.object(runs,"file_sha256",return_value="synthetic"),patch.object(runs.datasets,"MNIST",return_value=SyntheticPool()) as constructor:
            result=runs.attach_official_test(data,TrainConfig(development_seed=20261101,rounds=24),"synthetic-freeze",request)
        self.assertIs(result.pool,original_pool)
        self.assertFalse(constructor.call_args.kwargs["train"])
        self.assertEqual(result.evaluation_metadata["examples_per_task"],10000)
        self.assertEqual(result.selection["clients"],{"0":[1]})
        self.assertEqual(set(result.validation_tasks),{"clean","shot_noise_s3","stripe"})
        for view in result.validation_tasks.values():
            self.assertEqual(len(view),10000)
            self.assertEqual(view[9][1],9)

    def test_matched_runs_reject_different_evaluation_splits(self):
        fields=dict(config={},method={},partition_sha256="a",selection_sha256="b",initial_state_sha256="c",
                    code_sha256={},versions={},validation_tasks=["clean"],evaluation_recorder={})
        obs={**fields,"scenario":longer_scenarios()["attack_stops"].to_dict(),"official_test_used":True,
             "evaluation_data":{"split":"official_test","freeze_sha256":"a"}}
        ref={**fields,"scenario":longer_scenarios()["clean"].to_dict(),"official_test_used":True,
             "evaluation_data":copy.deepcopy(obs["evaluation_data"])}
        self.assertTrue(validate_matched_runs(obs,ref))
        for key,value in (("official_test_used",False),("evaluation_data",{"split":"official_test","freeze_sha256":"b"})):
            altered={**ref,key:value}
            with self.assertRaises(ValueError): validate_matched_runs(obs,altered)

    def test_predeclaration_equal_budgets_controls_and_seeds(self):
        with tempfile.TemporaryDirectory() as temp:
            plan=study.initialize(temp)
            self.assertEqual(len(plan["development_jobs"]),720)
            self.assertEqual(len(plan["feasibility_jobs"]),30)
            self.assertFalse(set(plan["development_seeds"])&set(plan["final_seeds"]))
            self.assertEqual(sum(map(len,plan["targeted_controls"].values())),28)
            for scenarios in plan["targeted_controls"].values():
                self.assertIn("clean",scenarios)
                if "mixed_drift_and_attack" in scenarios: self.assertIn("mixed_drift_only_control",scenarios)
            for regime in plan["regimes"]:
                for method in plan["methods"]:
                    self.assertEqual(sum(j["regime"]==regime and j["method"]==method for j in plan["development_jobs"]),72)
            self.assertEqual(study.initialize(temp),plan)

    def test_incomplete_grid_cannot_select_or_freeze(self):
        with tempfile.TemporaryDirectory() as temp:
            study.initialize(temp)
            with self.assertRaises(ValueError): study.select_settings(temp)
            self.assertFalse((Path(temp)/"frozen_protocol.json").exists())

    def test_selection_counts_failure_and_tie_breaks_without_replacement(self):
        def result_for(directory,request):
            failed=request["method"]=="FedReTrust" and request["relative_rate"]==4.
            return {"job":request,"status":"numerical_failure" if failed else "complete", "job_id":study.canonical_hash(request)}
        def scores(result):
            rate=result["job"]["relative_rate"]
            return {r:(.9 if rate==4. else .8) for r in range(25)}
        with tempfile.TemporaryDirectory() as temp:
            study.initialize(temp)
            with patch.object(study,"result_for",side_effect=result_for),patch.object(study,"task_series",side_effect=scores):
                result=study.select_settings(temp)
            for regime in runs.PARTITIONS:
                self.assertEqual(result["selected"][regime]["FedReTrust"]["relative_rate"],1.)
                self.assertEqual(result["selected"][regime]["FedAvg"]["relative_rate"],4.)

    def test_final_manifest_contains_all_predeclared_runs_and_inherited_rates(self):
        with tempfile.TemporaryDirectory() as temp:
            plan=study.initialize(temp)
            study.save(Path(temp)/"feasibility_review.json",{"passed":True})
            selected={r:{m:{"relative_rate":2.} for m in runs.METHODS} for r in runs.PARTITIONS}
            selection={"selected":selected}
            study.save(Path(temp)/"development_selection.json",selection)
            with patch.object(study,"select_settings",return_value=selection):
                frozen=study.freeze(temp)
            jobs=frozen["authorized_final_jobs"]
            self.assertEqual(len(jobs),980)
            for phase,count in (("final",500),("ablation",280),("sensitivity",200)):
                self.assertEqual(sum(j["phase"]==phase for j in jobs),count)
            self.assertTrue(all(j["relative_rate"]==2. for j in jobs))
            self.assertEqual(runs.verify_freeze(Path(temp)/"frozen_protocol.json"),frozen)
            selection_hash=runs.file_sha256(Path(temp)/"development_selection.json")
            with patch.object(study,"result_for",side_effect=AssertionError("Do not reselect after freeze")):
                self.assertEqual(study.select_settings(temp),selection)
            self.assertEqual(runs.file_sha256(Path(temp)/"development_selection.json"),selection_hash)
            self.assertEqual(study.freeze(temp),frozen)
            (Path(temp)/"development_selection.json").write_text('{}')
            with self.assertRaises(ValueError):study.select_settings(temp)
            with self.assertRaises(ValueError):study.freeze(temp)
            with patch.object(runs,"source_hashes",return_value={}):
                with self.assertRaises(ValueError): runs.verify_freeze(Path(temp)/"frozen_protocol.json")


if __name__=="__main__":
    unittest.main()
