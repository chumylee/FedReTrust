"""Stage-order checks with fake workers; never construct an official dataset."""
from contextlib import ExitStack
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import contextlib
import io

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import research_pipeline as pipeline


class PipelineTests(unittest.TestCase):
    def run_fake(self,directory,passed):
        root=Path(directory)/"Experiment"; (root/"code"/"tests").mkdir(parents=True)
        study=Path(directory)/"study";study.mkdir()
        stages=[]
        with ExitStack() as stack:
            for name,value in (("ROOT",root),("read_plan",lambda _:{}),("review_feasibility",lambda _:{"passed":passed}),
                               ("run_stage",lambda stage,path:stages.append(stage)),
                               ("select_settings",lambda _:stages.append("select")),("freeze",lambda _:stages.append("freeze"))):
                stack.enter_context(patch.object(pipeline,name,value))
            stack.enter_context(patch.object(pipeline.shutil,"which",return_value=None))
            stack.enter_context(patch.object(pipeline.subprocess,"run"))
            stack.enter_context(patch("research_notebooks.build"))
            stack.enter_context(patch("research_analysis.analyse",return_value={"completed_runs":980,"planned_runs":980,
                "numerical_failures":0,"overall_predeclared_criterion_met":False}))
            if passed:
                with contextlib.redirect_stdout(io.StringIO()): pipeline.execute(study)
            else:
                with self.assertRaises(ValueError): pipeline.execute(study)
        return stages,json.loads((study/"pipeline_status.json").read_text())

    def test_failed_feasibility_stops_before_development_or_test(self):
        with tempfile.TemporaryDirectory() as directory:
            stages,status=self.run_fake(directory,False)
        self.assertEqual(stages,["feasibility"])
        self.assertEqual(status["status"],"stopped_on_error")
        self.assertEqual(status["stage"],"feasibility")

    def test_completion_requires_final_controls_sensitivity_and_analysis(self):
        with tempfile.TemporaryDirectory() as directory:
            stages,status=self.run_fake(directory,True)
        self.assertEqual(stages,["feasibility","development","select","freeze","final","ablation","sensitivity"])
        self.assertEqual(status["status"],"complete")
        self.assertFalse(status["overall_predeclared_criterion_met"])


if __name__=="__main__":unittest.main()
