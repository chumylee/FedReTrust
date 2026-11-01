"""Predeclared development and final manifests with resumable isolated workers.

Run stages explicitly. Final-test access is impossible until development has
completed and freeze() has saved the exact settings and all final jobs.
"""
from dataclasses import asdict
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path
import argparse
import fcntl
import json
import math
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd

from experiment_core import file_sha256
from fedretrust import FedReTrustConfig
from research_runs import ROOT, METHODS, PARTITIONS, source_hashes, canonical_hash, run_identity
from scenario_metrics import (PERFORMANCE_SCENARIOS, PRIMARY_SCENARIOS, longer_scenarios,
                              legitimate_task_weights, weighted_task_score, primary_development_score)


DEFAULT_STUDY = ROOT/"outputs"/"full_study"/"20261011_v1"


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    temporary.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


def job(phase, method, regime, seed, scenario, rate, **extra):
    return dict(phase=phase, method=method, regime=regime, seed=seed, scenario=scenario,
                relative_rate=float(rate), rounds=24, train_images=512, num_threads=4,
                root_steps=8, **extra)


def initialize(study=DEFAULT_STUDY):
    study = Path(study)
    path = study/"development_plan.json"
    if path.exists():
        return read_plan(study)
    policy = json.loads((ROOT/"fedretrust_policy_draft.json").read_text())
    controls = {}
    for variant, scenarios in policy["targeted_control_scenarios_draft"].items():
        names = set(scenarios)|{"clean"}
        if "mixed_drift_and_attack" in names:
            names.add("mixed_drift_only_control")
        controls[variant] = sorted(names)
    sensitivity = [
        {"parameter":"relative_loss_tolerance", "values":[.0025,.01],
         "scenarios":["clean","abrupt_shot_noise","mixed_drift_and_attack","mixed_drift_only_control"]},
        {"parameter":"restoration_increment", "values":[.125,.5],
         "scenarios":["clean","honest_change_stops","recurring_attack"]},
        {"parameter":"non_supportive_allowance", "values":[2.,8.],
         "scenarios":["clean","abrupt_shot_noise","recurring_attack"]},
    ]
    feasibility = [job("feasibility",m,r,20261018,s,1.) for r in PARTITIONS for m in METHODS
                   for s in ("clean","honest_change_stops","recurring_attack")]
    development = [job("development",m,r,seed,s,rate) for r in PARTITIONS for m in METHODS
                   for rate in (1.,2.,4.) for seed in (20261014,20261015,20261016) for s in PRIMARY_SCENARIOS]
    plan = {
        "schema_version":1, "created_at":now(), "status":"predeclared_before_this_development_grid",
        "authorization":"User requested continuation through Step 10 and full five-seed, two-regime study with targeted controls.",
        "training_source_sha256":source_hashes(),
        "partition_sha256":{r:file_sha256(ROOT/"data"/"partitions"/p/"indices.json") for r,p in PARTITIONS.items()},
        "draft_policy_sha256":file_sha256(ROOT/"fedretrust_policy_draft.json"),
        "methods":list(METHODS), "regimes":list(PARTITIONS),
        "development_seeds":[20261014,20261015,20261016], "feasibility_seed":20261018,
        "final_seeds":[20261101,20261102,20261103,20261104,20261105],
        "workload":{"rounds":24,"clients":10,"train_images_per_client":512,"local_epochs":1,
                    "batch_size":64,"local_learning_rate":.01,"momentum":.9,
                    "root_images":300,"root_optimizer_steps":8,"root_batch_size":64,
                    "development_evaluation_images_per_task":1000,"final_evaluation_images_per_task":10000,
                    "worker_threads":4,"concurrent_workers":2},
        "workload_scope":"Same fixed 512-image subset per client throughout a run, reselected by seed; 5,120 client images per run, not all 53,700 assigned images. Full 10,000-image official test after freeze.",
        "reference_access":"FLTrust and FedReTrust share the same disjoint 300 clean root images and eight SGD steps; FedReTrust additionally probes reference loss, whose cost is recorded.",
        "candidate_relative_rates":[1.,2.,4.],
        "candidate_rationale":"Earlier six-round clean pilot selected rate 2 at its upper grid boundary for all four baselines. This independent broader-scenario grid fixes 1,2,4 before outcomes. Three configurations per method/regime; published defence defaults and the proposed draft constants remain fixed. This is limited server-rate calibration, not exhaustive defence optimisation. Unused slots from the earlier maximum-12 draft are not transferred.",
        "fixed_fedretrust_config":asdict(FedReTrustConfig()),
        "server_rate_anchors":{"FedAvg":1.,"FLTrust":1.,"RFA":1.,"Viceroy":"1 / 10", "FedReTrust":1.},
        "selection_rule":"Maximize equal-primary-scenario, equal-round accuracy over rounds 5..24, averaged across all three development seeds. One setting per method/regime. Tie tolerance 1e-12, then nearest rate to 1 in log space, then smaller rate.",
        "failure_rule":"Any numerical failure makes that candidate ineligible; preserve and count failed jobs, no replacement candidate. Infrastructure/assertion/protocol failures halt for investigation. Final numerical failures remain failures; no retuning and no complete-evidence superiority claim.",
        "grid_revision_rule":"No adaptive candidate additions; boundary winners are disclosed. Any future grid revision requires a new recorded development protocol before test access.",
        "feasibility_gate":"All 30 common panel runs complete with finite outputs and checkpoint parity; clean FedAvg final development accuracy >=75% in each regime. No requirement for FedReTrust to win. Inspect costs and late learning; 24 rounds remains a bounded-horizon study.",
        "horizon_limits":"Four warm-up rounds; 14 rounds after a change ending at 10; six rounds after recurrence ending at 18. Recovery can be right-censored, and a first three-round restoration is not permanent recovery or convergence.",
        "fixed_partition_limits":"One fixed partition per regime, one fixed cohort assignment. Seed uncertainty covers model initialisation, selected training images, shuffles and corruptions; it does not establish generalisation across partitions, datasets or cohort assignments.",
        "primary_scenarios":list(PRIMARY_SCENARIOS),"performance_scenarios":list(PERFORMANCE_SCENARIOS),
        "targeted_controls":controls,"parameter_sensitivity":sensitivity,
        "saved_trace_sensitivity":{"contribution_reduction_thresholds":[.4,.5,.6],"contribution_restoration_thresholds":[.7,.8,.9],"consecutive_rounds":3},
        "success_criteria":{
            "primary_contrast":"FedReTrust minus Viceroy; all other baselines also reported",
            "honest_composite":"Equal mean of affected-honest task accuracy: abrupt 5..24 clients0..2; gradual 5..24 clients0..2; honest_change_stops 5..10 clients0..2; mixed 5..24 clients3..5.",
            "mean_honest_gain_pp":1.,"honest_gain_ci_lower_must_exceed_pp":0.,
            "clean_and_each_honest_component_and_return_ci_lower_at_least_pp":-1.,
            "clean_window":[5,24],"honest_return_window":[11,24],
            "attack_excess_damage_ci_upper_at_most_pp":1.,
            "attack_damage_scope":"Each attack scenario in each regime separately; own matched no-attack reference; signed active-window mean damage. Also report positive area, tail damage and all curves.",
            "uncertainty":"Five paired seeds: mean, sample SD, descriptive two-sided Student-t 95% CI (df4). Conditional on fixed partitions; no multiplicity-adjusted confirmatory inference.",
            "decision":"All coverage and primary gain/guard criteria must hold in BOTH regimes for the predeclared overall criterion. Otherwise distinguish observed criterion failures from inconclusive intervals; never force a winner.",
            "mechanisms":"Component controls share selected full-method settings and run their own trajectories. H3 compares full vs no_repeat_memory over the SAME second attack window 15..18, not a causal first/second episode comparison.",
        },
        "feasibility_jobs":feasibility,"development_jobs":development,
        "job_counts":{"feasibility":len(feasibility),"development":len(development),"final":500,"ablation":280,"sensitivity":200},
        "execution_cost_limits":"Two simultaneous fresh worker processes, four PyTorch CPU threads each. Per-run timings and peak RSS include concurrent contention/import/data overhead and are not isolated kernel benchmarks.",
    }
    save(path, plan)
    return plan


def read_plan(study):
    plan = json.loads((Path(study)/"development_plan.json").read_text())
    if plan["training_source_sha256"] != source_hashes():
        raise ValueError("Training sources changed since predeclaration; investigate and version the development study.")
    return plan


def result_for(study, request):
    directory = Path(study)/"jobs"/run_identity(request)
    if (directory/"result.json").exists():
        result = json.loads((directory/"result.json").read_text())
        if result["job"] != request or result["source_sha256"] != source_hashes():
            raise ValueError("Saved run provenance mismatch.")
        return result
    if (directory/"failure.json").exists():
        failure = json.loads((directory/"failure.json").read_text())
        if failure["exception_type"] != "FloatingPointError":
            raise RuntimeError(f"Investigate job {directory.name}: {failure['error']}")
        return {**failure,"job_id":directory.name,"status":"numerical_failure"}
    return None


def run_one(study, request):
    directory = Path(study)/"jobs"/run_identity(request)
    directory.mkdir(parents=True, exist_ok=True)
    found = result_for(study, request)
    if found and found["status"] == "numerical_failure":
        return found
    save(directory/"job.json", request)
    with (directory/"worker.log").open("a") as log:
        code = subprocess.call([sys.executable, str(ROOT/"code"/"research_runs.py"),
                                "--job",str(directory/"job.json"),"--study",str(study)],
                               stdout=log,stderr=subprocess.STDOUT, env={**os.environ,"PYTHONUNBUFFERED":"1"})
    result = result_for(study, request)
    if result is None or (code != 0 and result["status"] != "numerical_failure"):
        raise RuntimeError(f"Worker failed without an eligible numerical failure: {directory}")
    return result


def run_stage(stage, study=DEFAULT_STUDY):
    study = Path(study)
    plan = read_plan(study)
    if stage == "feasibility":
        jobs = plan["feasibility_jobs"]
    elif stage == "development":
        review = json.loads((study/"feasibility_review.json").read_text())
        if not review["passed"]:
            raise ValueError("Feasibility review has not passed.")
        jobs = plan["development_jobs"]
    else:
        from research_runs import verify_freeze
        freeze_record = verify_freeze(study/"frozen_protocol.json")
        jobs = [j for j in freeze_record["authorized_final_jobs"] if j["phase"] == stage]
        if not jobs:
            raise ValueError("Unknown or empty stage.")
    started = time.perf_counter()
    completed, failures, pending_jobs = [], [], []
    # Previously completed workers revalidate every recorded artifact on reuse.
    for request in jobs:
        found = result_for(study,request)
        if found is not None:
            found = run_one(study,request)
            (completed if found["status"] == "complete" else failures).append(found["job_id"])
        else:
            pending_jobs.append(request)
    def progress(active, status="running"):
        value = {"stage":stage,"status":status,"updated_at":now(),"total":len(jobs),
                 "completed":len(completed),"numerical_failures":len(failures),
                 "active_jobs":active,"completed_job_ids":completed,"failed_job_ids":failures,
                 "invocation_seconds":time.perf_counter()-started}
        save(study/f"progress_{stage}.json",value)
    with (study/"coordinator.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with ThreadPoolExecutor(max_workers=plan["workload"]["concurrent_workers"]) as executor:
            iterator = iter(pending_jobs)
            active = {}
            def submit_next():
                request = next(iterator,None)
                if request is not None:
                    active[executor.submit(run_one,study,request)] = request
            for _ in range(plan["workload"]["concurrent_workers"]):
                submit_next()
            while active:
                progress(list(active.values()))
                done,_ = wait(active,timeout=30,return_when=FIRST_COMPLETED)
                for future in done:
                    request = active.pop(future)
                    try:
                        result = future.result()
                    except Exception as exc:
                        progress(list(active.values()),"blocked_by_worker_error")
                        save(study/f"error_{stage}.json",{"at":now(),"job":request,"error":repr(exc)})
                        raise
                    (completed if result["status"] == "complete" else failures).append(result["job_id"])
                    print(json.dumps({"stage":stage,"completed":len(completed),"failed":len(failures),"total":len(jobs),"last":request}),flush=True)
                    submit_next()
        progress([],"complete" if not failures else "complete_with_numerical_failures")


def task_series(result, scenario_name=None, client_ids=None):
    if result["status"] != "complete":
        raise ValueError("Cannot score a failed run.")
    path = ROOT/result["run_path"]
    rows = pd.read_csv(path/"task_metrics.csv")
    counts = json.loads((path/"population_class_counts.json").read_text())
    counts = [counts[str(i)] for i in range(10)]
    scenario = longer_scenarios()[scenario_name or result["job"]["scenario"]]
    scores = {}
    for round_number,group in rows.groupby("round"):
        tasks = {row["task"]:row for row in group.to_dict("records")}
        weights = legitimate_task_weights(scenario,int(round_number),counts,client_ids)
        scores[int(round_number)] = weighted_task_score(tasks,weights)["accuracy"]
    if set(scores) != set(range(result["job"]["rounds"]+1)):
        raise ValueError("Incomplete round coverage.")
    return scores


def review_feasibility(study=DEFAULT_STUDY):
    study = Path(study)
    plan = read_plan(study)
    results = [result_for(study,j) for j in plan["feasibility_jobs"]]
    if any(r is None for r in results):
        raise ValueError("Complete the whole common feasibility panel first.")
    rows = []
    passed = all(r["status"] == "complete" for r in results)
    for result in results:
        if result["status"] != "complete":
            rows.append({**result["job"],"status":result["status"]})
            continue
        series = task_series(result)
        history = pd.read_csv(ROOT/result["run_path"]/"history.csv")
        summary = json.loads((ROOT/result["run_path"]/"summary.json").read_text())
        if result["job"]["method"] == "FedAvg" and result["job"]["scenario"] == "clean":
            passed &= series[24] >= .75
        passed &= summary["checkpoint_logits_identical"] and not summary["official_test_used"]
        rows.append({**result["job"],"status":"complete","accuracy_round24":series[24],
                     "late_gain_pp":100*(np.mean([series[r] for r in range(21,25)])-np.mean([series[r] for r in range(17,21)])),
                     "median_round_seconds":float(history.round_seconds.median()),
                     "peak_rss_bytes":result["process_lifetime_peak_rss_bytes"]})
    review = {"at":now(),"passed":bool(passed),"results":rows,"horizon_assessment":plan["horizon_limits"],
              "workload_decision":plan["workload_scope"],"interpretation":"Operability gate only; no superiority or convergence claim. Keep all declared candidate rates unchanged."}
    save(study/"feasibility_review.json",review)
    return review


def select_settings(study=DEFAULT_STUDY):
    study = Path(study)
    plan = read_plan(study)
    frozen_path = study/"frozen_protocol.json"
    if frozen_path.exists():
        frozen = json.loads(frozen_path.read_text())
        path = study/"development_selection.json"
        if file_sha256(path) != frozen["development_selection_sha256"]:
            raise ValueError("Frozen development selection changed; preserve it and investigate.")
        # Re-running a frozen study must not update even the selection timestamp.
        return json.loads(path.read_text())
    all_results = [result_for(study,j) for j in plan["development_jobs"]]
    if any(r is None for r in all_results):
        raise ValueError("Development grid is incomplete.")
    selected, table = {}, []
    for regime in PARTITIONS:
        selected[regime] = {}
        for method in METHODS:
            candidates = []
            for rate in plan["candidate_relative_rates"]:
                group = [r for r in all_results if r["job"]["regime"] == regime and r["job"]["method"] == method and r["job"]["relative_rate"] == rate]
                eligible = len(group) == 24 and all(r["status"] == "complete" for r in group)
                seed_scores = []
                if eligible:
                    for seed in plan["development_seeds"]:
                        observed = {r["job"]["scenario"]:{t:v for t,v in task_series(r).items() if t>=5}
                                    for r in group if r["job"]["seed"] == seed}
                        seed_scores.append(primary_development_score(observed)["accuracy"])
                row = {"regime":regime,"method":method,"relative_rate":rate,"eligible":eligible,
                       "mean_accuracy":float(np.mean(seed_scores)) if eligible else None,"seed_scores":seed_scores,
                       "failures":[r["job_id"] for r in group if r["status"] != "complete"]}
                table.append(row)
                if eligible:
                    candidates.append(row)
            if not candidates:
                raise ValueError(f"No eligible configuration for {method}/{regime}; stop before official test.")
            best = max(r["mean_accuracy"] for r in candidates)
            tied = [r for r in candidates if best-r["mean_accuracy"] <= 1e-12]
            winner = min(tied,key=lambda r:(abs(math.log(r["relative_rate"])),r["relative_rate"]))
            selected[regime][method] = {**winner,"boundary_winner":winner["relative_rate"] in [min(plan["candidate_relative_rates"]),max(plan["candidate_relative_rates"])]}
    result = {"selected":selected,"candidates":table,"at":now(),"plan_sha256":file_sha256(study/"development_plan.json")}
    save(study/"development_selection.json",result)
    return result


def freeze(study=DEFAULT_STUDY):
    study = Path(study)
    path = study/"frozen_protocol.json"
    if path.exists():
        from research_runs import verify_freeze
        frozen = verify_freeze(path)
        if (file_sha256(study/"development_selection.json") != frozen["development_selection_sha256"]
            or file_sha256(study/"development_plan.json") != frozen["development_plan_sha256"]):
            raise ValueError("A pretest record changed after freeze.")
        return frozen
    plan = read_plan(study)
    if not json.loads((study/"feasibility_review.json").read_text())["passed"]:
        raise ValueError("Feasibility gate failed.")
    selection = select_settings(study)
    jobs = []
    for regime in PARTITIONS:
        for seed in plan["final_seeds"]:
            for method in METHODS:
                rate = selection["selected"][regime][method]["relative_rate"]
                jobs += [job("final",method,regime,seed,s,rate) for s in PERFORMANCE_SCENARIOS]
            rate = selection["selected"][regime]["FedReTrust"]["relative_rate"]
            for variant, scenarios in plan["targeted_controls"].items():
                jobs += [job("ablation","FedReTrust",regime,seed,s,rate,variant=variant) for s in scenarios]
            for cell in plan["parameter_sensitivity"]:
                for value in cell["values"]:
                    jobs += [job("sensitivity","FedReTrust",regime,seed,s,rate,overrides={cell["parameter"]:value}) for s in cell["scenarios"]]
    if len({canonical_hash(j) for j in jobs}) != len(jobs):
        raise ValueError("Duplicate final manifest entries.")
    for phase in ("final","ablation","sensitivity"):
        if sum(j["phase"] == phase for j in jobs) != plan["job_counts"][phase]:
            raise ValueError("Final counts differ from predeclared budget.")
    value = {**{k:v for k,v in plan.items() if k not in {"feasibility_jobs","development_jobs","status","created_at"}},
             "status":"frozen_before_official_test","frozen_at":now(),
             "development_plan_sha256":file_sha256(study/"development_plan.json"),
             "development_selection_sha256":file_sha256(study/"development_selection.json"),
             "selected_settings":selection["selected"],"authorized_final_jobs":jobs,
             "pretest_review":{"data_separation":"official test not loaded in development jobs; worker checks this freeze before constructing train=False dataset",
                               "fairness":"equal three-candidate grid and 24 scenario/seed runs per candidate, same training/reference/root work, one selected rate per method/regime",
                               "limits":["limited server-rate tuning","single dataset and fixed partitions/cohorts","subsampled client data","bounded 24-round horizon","trusted reference assumption","descriptive small-seed intervals"]}}
    save(path,value)
    return value


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action",choices=["initialize","feasibility","review-feasibility","development","select","freeze","final","ablation","sensitivity"])
    parser.add_argument("--study",type=Path,default=DEFAULT_STUDY)
    args = parser.parse_args()
    functions = {"initialize":initialize,"review-feasibility":review_feasibility,"select":select_settings,"freeze":freeze}
    if args.action in functions:
        value = functions[args.action](args.study)
        print(json.dumps({"action":args.action,"study":str(args.study),"passed":value.get("passed"),"status":value.get("status")}),flush=True)
    else:
        run_stage(args.action,args.study)
