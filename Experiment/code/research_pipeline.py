"""Execute the already-authorized study through Step 10, with stage checkpoints.

No scheduler or external service is used. This is one local computation process.
Restarting the same command validates and reuses completed stage artifacts.
"""
from pathlib import Path
import argparse
import fcntl
import os
import shutil
import subprocess
import sys
import traceback

from experiment_core import file_sha256
from research_runs import ROOT
from research_study import (DEFAULT_STUDY, read_plan, save, now, run_stage,
                            review_feasibility, select_settings, freeze)


def execute(study=DEFAULT_STUDY):
    study=Path(study)
    read_plan(study)
    current="verification"
    with (study/"pipeline.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        wake=None
        try:
            # Prevent idle sleep for this process only; display sleep is allowed.
            if shutil.which("caffeinate"):
                wake=subprocess.Popen(["caffeinate","-i","-w",str(os.getpid())],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            def status(stage,**extra):
                save(study/"pipeline_status.json",{"stage":stage,"status":"running","pid":os.getpid(),"updated_at":now(),**extra})
            status(current)
            with (study/"preflight_tests.txt").open("w") as log:
                subprocess.run([sys.executable,"-m","unittest","discover","-s",str(ROOT/"code"/"tests"),"-q"],stdout=log,stderr=subprocess.STDOUT,check=True)
            snapshot=study/"source_snapshot"; snapshot.mkdir(exist_ok=True)
            paths=list((ROOT/"code").glob("*.py"))+list((ROOT/"code"/"tests").glob("*.py"))
            hashes={}
            for source in paths:
                relative=source.relative_to(ROOT/"code")
                target=snapshot/relative;target.parent.mkdir(parents=True,exist_ok=True)
                digest=file_sha256(source)
                if target.exists() and file_sha256(target)!=digest:
                    raise ValueError(f"Source differs from saved pipeline snapshot: {relative}; investigate before resuming.")
                if not target.exists(): shutil.copyfile(source,target)
                hashes[str(relative)]=digest
            save(study/"execution_sources.json",{"at":now(),"sha256":hashes,"preflight_log_sha256":file_sha256(study/"preflight_tests.txt")})
            current="feasibility";status(current,note="Waiting for any existing stage coordinator, then validating/reusing completed feasibility runs.")
            with (study/"coordinator.lock").open("a") as coordinator:
                fcntl.flock(coordinator,fcntl.LOCK_EX)
            run_stage(current,study)
            review=review_feasibility(study)
            if not review["passed"]: raise ValueError("Common feasibility gate failed; study stops before development selection or official test.")
            from research_notebooks import build
            build(13,study=study)
            current="development";status(current);run_stage(current,study)
            current="selection_and_freeze";status(current)
            select_settings(study);freeze(study);build(13,study=study)
            for stage in ("final","ablation","sensitivity"):
                current=stage;status(current);run_stage(stage,study)
                if stage=="final":build(14,study=study)
            current="analysis";status(current)
            from research_analysis import analyse
            manifest=analyse(study)
            for number in (14,15,16): build(number,study=study)
            save(study/"pipeline_status.json",{"stage":"step_10_complete","status":"complete","updated_at":now(),
                                              "completed_runs":manifest["completed_runs"],"planned_final_runs":manifest["planned_runs"],
                                              "numerical_failures":manifest["numerical_failures"],
                                              "overall_predeclared_criterion_met":manifest["overall_predeclared_criterion_met"],
                                              "analysis_manifest":"analysis/analysis_manifest.json"})
            print("Study complete through Step 10. See analysis/analysis_manifest.json and notebooks 13–16.",flush=True)
        except BaseException as exc:
            save(study/"pipeline_status.json",{"stage":current,"status":"stopped_on_error","updated_at":now(),
                                              "error":repr(exc),"traceback":traceback.format_exc(),
                                              "instruction":"Preserve all artifacts and investigate; do not silently change frozen settings."})
            raise
        finally:
            if wake is not None and wake.poll() is None: wake.terminate()


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--study",type=Path,default=DEFAULT_STUDY)
    args=parser.parse_args();execute(args.study)
