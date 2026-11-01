"""Declared exploratory measurement pilot; preserve all earlier training sources.

32 observed trajectories plus 640 local counterfactual trainings. No official
test loader and no learned diagnostic rule in the server aggregation path.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
import argparse
import fcntl
import json
import platform
import resource
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch

from experiment_core import TrainConfig, load_data, run_experiment, file_sha256, write_json
from research_runs import ROOT, PARTITIONS, canonical_hash, source_hashes, make_aggregator
from research_analysis import extract_run, cross_method_pair, cost_record
from scenario_metrics import ClasswiseEvaluator, ContributionRecorder
from scenario_metrics import longer_scenarios, matched_reference_name
from directional_study import verify as verify_v03, checked_result
from nnm_rfa import NNMRFAAggregator, NNMRFAConfig
from update_observer import UpdateObserver
from mechanism_diagnostics import DIAGNOSTIC_PROTOCOL, analyse_diagnostics, analyse_probe
from common_broadcast_probe import run_probe


STUDY = ROOT / "outputs/mechanism_pilot/20261030_v1"
OLD = ROOT / "outputs/full_study/20261011_v1"
V03 = ROOT / "outputs/reference_free/20261030_v03"
SEEDS = (20261014, 20261015)
SCENARIOS = ("clean", "abrupt_shot_noise", "gradual_stripe", "label_flip")
METHODS = ("FedAvg", "NNM-RFA")
BASELINES = ("FedAvg", "RFA", "FLTrust", "Viceroy", "FedReTrust")
PREVIOUS = ("FedReTrust-RF-recovery", "FedReTrust-Direction-loose")
NEW_SOURCES = ("nnm_rfa.py", "update_observer.py", "mechanism_diagnostics.py",
               "common_broadcast_probe.py", "mechanism_pilot.py")
TEST_SOURCES = ("test_nnm_rfa.py", "test_update_observer.py",
                "test_mechanism_diagnostics.py", "test_common_broadcast_probe.py")


def stamp():
    return datetime.now(timezone.utc).isoformat()


def hashes():
    from directional_study import hashes as old_hashes
    return {**old_hashes(), **{n: file_sha256(ROOT / "code" / n) for n in NEW_SOURCES}}


def key(j):
    return j["method"], j["regime"], j["scenario"], j["seed"]


def jobs():
    return [{"method": m, "regime": r, "scenario": s, "seed": seed,
             "phase": "development", "relative_rate": 1., "root_steps": 8,
             "rounds": 24, "train_images": 512, "num_threads": 4}
            for m in METHODS for seed in SEEDS for r in PARTITIONS for s in SCENARIOS]


def existing_records():
    """Read verified comparison artifacts, never re-evaluate their test data."""
    verify_v03(V03)
    records = {}
    paths = list((OLD / "jobs").glob("*/result.json"))
    candidates = [json.loads(p.read_text()) for p in paths]
    candidates += json.loads((V03 / "result_manifest.json").read_text())["records"]
    for result in candidates:
        j = result["job"]
        if (j["phase"] != "development" or j["relative_rate"] != 1.
                or j["method"] not in (*BASELINES, *PREVIOUS)
                or j["scenario"] not in SCENARIOS or j["seed"] not in SEEDS):
            continue
        if j.get("variant", "full") != "full" or j.get("overrides"):
            continue
        if j["method"] in BASELINES and result["source_sha256"] != source_hashes():
            raise ValueError("Reused baseline source mismatch.")
        records[key(j)] = checked_result(result)
    expected = {(m, r, s, seed) for m in BASELINES for r in PARTITIONS
                for s in SCENARIOS for seed in SEEDS}
    if not expected.issubset(records):
        raise ValueError(f"Missing original paired baseline records: {expected-set(records)}")
    return records


def prepare(study):
    study.mkdir(parents=True, exist_ok=True)
    if (study / "protocol.json").exists():
        verify(study)
        return
    old = json.loads((OLD / "frozen_protocol.json").read_text())["training_source_sha256"]
    if source_hashes() != old:
        raise ValueError("Original training sources changed.")
    existing = existing_records()
    protocol = {
        "created_at": stamp(), "status": "declared_before_pilot_training",
        "authorization": "User: go ahead; diagnostic pilot and independent NNM-RFA comparison.",
        "scope": "Exploratory development: these seeds, partitions and validation images have been used before. Not confirmatory or a novelty claim.",
        "training_source_sha256": hashes(),
        "partition_sha256": {r: file_sha256(ROOT / "data/partitions" / p / "indices.json") for r, p in PARTITIONS.items()},
        "jobs": jobs(), "new_trajectory_runs": 32,
        "workload": "24 rounds, 10 clients, 512 images per client, one local epoch, original SGD, 1000 development images per task, four threads per worker and at most two concurrent workers.",
        "methods": {"FedAvg": "Exact existing aggregation with observational diagnostics only.",
                    "NNM-RFA": "Equal arithmetic mean of n-f nearest deltas including self, fixed f=3 in every scenario, then unchanged RFAConfig solver. Original sample shares weight anchor mixtures; no claimed theorem for unequal counts or this local-SGD implementation."},
        "no_server_reference_images": True,
        "fairness": "Original sample selections and initializations retained; the 300 reserved images remain excluded from client training and unused by NNM or the observer. Historical FLTrust and original FedReTrust retain their original reference access.",
        "diagnostics": DIAGNOSTIC_PROTOCOL,
        "probe": {"source": "Clean observed FedAvg pre-update broadcasts", "rounds": [5, 8, 12, 20],
                  "scenarios": SCENARIOS, "regimes": list(PARTITIONS), "seeds": SEEDS,
                  "client_trainings": 640,
                  "interpretation": "Counterfactual current updates share the exact model, client selections and minibatch seeds. No evolution or evaluation feedback. Temporal features undefined. Descriptive cross-check of selected fixed score and whole-cosine baseline; cannot rescue a failed trajectory gate."},
        "gate_use": "A passing diagnostic gate only permits a separately declared new candidate experiment. No post-hoc choice of another score or threshold. A failing gate stops the planned combined prototype; NNM performance is reported regardless. A contradictory common-broadcast probe requires further investigation before any claim of reliable separation.",
        "evaluation": "All original legitimate-task accuracy metrics, own-method paired attack damage, clean and absolute attacked accuracy. Four persistent scenarios in this bounded pilot; no new recovery claim. Six clean/stripe/label targets retain best original-baseline targets; also preserve previous v0.3 IID stripe accuracy as a guard. Per-seed values and two-seed means, no significance claims.",
        "reuse_count": len(existing),
        "observer_audit": "Require every observed FedAvg global model hash in all24 rounds and every task evaluation to equal its original matched run exactly.",
        "stopping": "Exactly the32 declared trajectories and640 probe clienttrainings; no adaptive hyperparameter search. New rule requires diagnostic support and a separate frozen development protocol.",
    }
    write_json(study / "protocol.json", protocol)
    snapshot = study / "source_snapshot"; snapshot.mkdir()
    for name in hashes(): shutil.copy2(ROOT / "code" / name, snapshot / name)
    for name in TEST_SOURCES: shutil.copy2(ROOT / "code/tests" / name, snapshot / name)
    write_json(study / "reused_manifest.json", {"records": list(existing.values())})
    print(json.dumps({"protocol": str(study / "protocol.json"), "new_runs": 32,
                      "reused_records": len(existing), "probe_client_trainings": 640}), flush=True)


def verify(study):
    p = json.loads((study / "protocol.json").read_text())
    if p["training_source_sha256"] != hashes():
        raise ValueError("Sources changed after pilot declaration.")
    for regime, part in PARTITIONS.items():
        if file_sha256(ROOT / "data/partitions" / part / "indices.json") != p["partition_sha256"][regime]:
            raise ValueError("Partition changed after declaration.")
    verify_v03(V03)
    return p


def new_records(study):
    records = {}
    for path in (study / "jobs").glob("*/result.json"):
        result = checked_result(json.loads(path.read_text()))
        if result["source_sha256"] != hashes() or result["job"] not in jobs():
            raise ValueError("Unregistered observation run.")
        records[key(result["job"])] = result
    return records


def worker(j, study):
    verify(study)
    if j not in jobs(): raise ValueError("Undeclared job.")
    jid = canonical_hash({"job": j, "sources": hashes()})[:20]
    directory = study / "jobs" / jid; directory.mkdir(parents=True, exist_ok=True)
    with (directory / "worker.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (directory / "result.json").exists():
            return checked_result(json.loads((directory / "result.json").read_text()))
        if (directory / "failure.json").exists():
            raise ValueError("Investigate the preserved failed attempt before resuming.")
        write_json(directory / "request.json", {"job": j, "sources": hashes()})
        started = time.perf_counter()
        try:
            cfg = TrainConfig(partition_id=PARTITIONS[j["regime"]], development_seed=j["seed"],
                              rounds=24, max_train_per_client=512, max_validation=1000, num_threads=4)
            data = load_data(ROOT, cfg)
            base = make_aggregator("FedAvg", data, cfg) if j["method"] == "FedAvg" else ContributionRecorder(NNMRFAAggregator(NNMRFAConfig(f=3)))
            observed = UpdateObserver(base)
            run = run_experiment(data, cfg, longer_scenarios()[j["scenario"]], observed,
                                 notebook_code_sha256=jid, verbose=False, evaluator=ClasswiseEvaluator())
            torch.save(observed.checkpoints, run / "observation_checkpoints.pt")
            counts = {i: np.bincount(data.pool.targets[ix].numpy(), minlength=10).tolist()
                      for i, ix in data.manifest["indices"]["clients"].items()}
            write_json(run / "population_class_counts.json", counts)
            names = ("run.json", "selection.json", "summary.json", "history.csv", "task_metrics.csv",
                     "client_metrics.csv", "aggregation_diagnostics.json", "population_class_counts.json",
                     "initial_global_model_state.pt", "final_global_model_state.pt", "status.json",
                     "observation_checkpoints.pt")
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            result = {"job_id": jid, "job": j, "status": "complete", "run_path": str(run.relative_to(ROOT)),
                      "artifact_sha256": {n: file_sha256(run / n) for n in names}, "source_sha256": hashes(),
                      "worker_wall_seconds": time.perf_counter() - started,
                      "process_lifetime_peak_rss_bytes": int(rss if platform.system() == "Darwin" else rss * 1024)}
            write_json(directory / "result.json", result)
            return result
        except Exception as exc:
            write_json(directory / "failure.json", {"job": j, "error": repr(exc), "at": stamp()})
            raise


def run(study):
    verify(study)
    with (study / "coordinator.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            existing = new_records(study)
            pending = [j for j in jobs() if key(j) not in existing]
            requests = study / "requests"; requests.mkdir(exist_ok=True)
            write_json(study / "status.json", {"status": "running", "at": stamp()})
            write_json(study / "progress.json", {"stage": "observed_training", "completed": len(existing), "total": 32, "at": stamp()})
            def launch(j):
                ident = canonical_hash(j)[:20]
                request = requests / f"{ident}.json"; write_json(request, j)
                with (requests / f"{ident}.log").open("w") as log:
                    subprocess.run([sys.executable, str(Path(__file__).resolve()), "worker", "--study", str(study),
                                    "--job", str(request)], stdout=log, stderr=subprocess.STDOUT, check=True)
                return j
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = {pool.submit(launch, j): j for j in pending}
                for n, future in enumerate(as_completed(futures), 1):
                    done = future.result()
                    progress = {"stage": "observed_training", "completed": len(existing)+n,
                                "total": 32, "job": done, "at": stamp()}
                    write_json(study / "progress.json", progress)
                    print(json.dumps(progress), flush=True)
            records = list(new_records(study).values())
            analyse(study)
            write_json(study / "progress.json", {"stage": "common_broadcast_probes", "client_trainings": 640, "at": stamp()})
            run_probe(records, study / "probes")
            diagnostic = json.loads((study / "analysis/diagnostics/diagnostic_summary.json").read_text())
            analyse_probe(study / "probes", diagnostic, study / "analysis/diagnostics")
            write_json(study / "status.json", {"status": "complete", "new_runs": len(records), "at": stamp()})
        except BaseException as exc:
            write_json(study / "status.json", {"status": "stopped_on_error", "error": repr(exc), "at": stamp()})
            raise


def analyse(study):
    verify(study)
    new = new_records(study)
    if len(new) != 32: raise ValueError("Need all32 declared observed runs.")
    reused = {key(r["job"]): checked_result(r) for r in json.loads((study / "reused_manifest.json").read_text())["records"]}
    records = {**reused, **new}
    audits = []
    for k, result in new.items():
        method, regime, scenario, seed = k
        reference = reused[("FedAvg", regime, scenario, seed)]
        cross_method_pair(result, reference)
        path = ROOT / result["run_path"]
        diagnostics = json.loads((path / "aggregation_diagnostics.json").read_text())
        assert len(diagnostics) == 24
        assert all("update_observation" in row["values"] for row in diagnostics)
        assert all(row["values"].get("reference_forward_examples", 0) == 0 for row in diagnostics)
        audit = {"job_id": result["job_id"], "method": method, "rounds": 24,
                 "cross_method_pair": "passed", "reference_forward_examples": 0}
        if method == "FedAvg":
            oldpath = ROOT / reference["run_path"]
            a, b = pd.read_csv(path / "history.csv"), pd.read_csv(oldpath / "history.csv")
            assert a.global_state_sha256.tolist() == b.global_state_sha256.tolist()
            pd.testing.assert_frame_equal(pd.read_csv(path / "task_metrics.csv"), pd.read_csv(oldpath / "task_metrics.csv"), check_exact=True)
            audit["all24_global_states_and_all_task_metrics_identical"] = True
        audits.append(audit)
    write_json(study / "observation_access_audit.json", {"new_runs": 32, "new_rounds": 768,
                "trusted_examples": 0, "runs": audits})
    metrics, curves, episodes, costs = [], [], [], []
    for k, result in records.items():
        m, regime, scenario, seed = k
        reference = records.get((m, regime, matched_reference_name(scenario), seed))
        a, b, c = extract_run(result, reference if scenario != "clean" else None)
        metrics += a; curves += b; episodes += c; costs.append(cost_record(result))
    output = study / "analysis"; output.mkdir(exist_ok=True)
    for name, rows in (("seed_metrics", metrics), ("accuracy_curves", curves), ("model_episodes", episodes), ("costs", costs)):
        write_json(output / f"{name}.json", rows)
        pd.DataFrame(rows).to_csv(output / f"{name}.csv", index=False)
    table = pd.DataFrame(metrics)
    means = table.groupby(["regime", "scenario", "metric", "method"]).value.agg(["mean", "count"]).reset_index()
    means.to_csv(output / "means_with_coverage.csv", index=False)
    complete = means[means["count"] == 2]
    complete.pivot(index=["regime", "scenario", "metric"], columns="method", values="mean").to_csv(output / "two_seed_means.csv")
    targets = []
    for regime in PARTITIONS:
        for scenario, metric, direction in (("clean", "overall_mean_5_24_pp", 1), ("gradual_stripe", "affected_honest_event_pp", 1), ("label_flip", "attack_active_signed_damage_pp", -1)):
            x = complete[(complete.regime == regime) & (complete.scenario == scenario) & (complete.metric == metric)]
            baselines = x[x.method.isin(BASELINES)]
            best = baselines.loc[(baselines["mean"] * direction).idxmax()]
            value = float(x[x.method == "NNM-RFA"]["mean"].item())
            targets.append({"regime": regime, "scenario": scenario, "metric": metric,
                            "nnm_rfa_pp": value, "best_baseline": best.method, "target_pp": float(best["mean"]),
                            "advantage_pp": direction*(value-float(best["mean"])),
                            "target_met_descriptively": direction*(value-float(best["mean"])) >= 0})
    stripe = complete[(complete.regime == "IID") & (complete.scenario == "gradual_stripe") & (complete.metric == "affected_honest_event_pp")]
    guard = float(stripe[stripe.method == "NNM-RFA"]["mean"].item()) >= float(stripe[stripe.method == PREVIOUS[1]]["mean"].item())
    write_json(output / "target_check.json", {"all_six_baseline_targets_met": all(t["target_met_descriptively"] for t in targets),
               "preserves_v03_IID_stripe": guard, "comparisons": targets,
               "interpretation": "Two reused development seeds on fixed partitions; descriptive means, no superiority inference."})
    write_json(study / "result_manifest.json", {"records": list(records.values()), "observed_records": list(new.values()),
               "reused_original_records": len(reused), "at": stamp(), "official_test_used": False})
    analyse_diagnostics(list(new.values()), output / "diagnostics")
    print(json.dumps({"analysis": str(output), "comparison_records": len(records), "targets": targets}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "run", "worker", "analyse"))
    parser.add_argument("--study", type=Path, default=STUDY)
    parser.add_argument("--job", type=Path)
    args = parser.parse_args()
    if args.action == "worker": worker(json.loads(args.job.read_text()), args.study)
    else: globals()[args.action](args.study)
