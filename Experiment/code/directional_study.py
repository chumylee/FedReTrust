"""Small, separate, development-only experiment for FedReTrust v0.3.

The original frozen study and its training sources are never edited. Existing
paired development baselines are checked and reused. No official test loader
is called. Run `prepare`, inspect protocol.json, then `run` and `analyse`.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
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

from experiment_core import TrainConfig, load_data, run_experiment, file_sha256, write_json
from fedretrust_directional import CANDIDATE_CONFIGS, DirectionalFedReTrust
from reference_free_study import STUDY as PREVIOUS, verify as verify_previous
from research_runs import ROOT, PARTITIONS, canonical_hash, source_hashes, execute_job
from research_analysis import extract_run, cross_method_pair, cost_record
from scenario_metrics import ClasswiseEvaluator, ContributionRecorder, longer_scenarios, matched_reference_name


STUDY = ROOT / "outputs/reference_free/20261030_v03"
OLD = ROOT / "outputs/full_study/20261011_v1"
CORE = ("clean", "gradual_stripe", "label_flip")
BASELINES = ("FedAvg", "RFA", "FLTrust", "Viceroy", "FedReTrust")
CANDIDATES = tuple(CANDIDATE_CONFIGS)
PREVIOUS_METHOD = "FedReTrust-RF-recovery"
SCREEN_SEED, CHECK_SEED = 20261014, 20261015


def stamp():
    return datetime.now(timezone.utc).isoformat()


def hashes():
    return {**source_hashes(), **{name: file_sha256(ROOT / "code" / name)
            for name in ("fedretrust_reference_free.py", "reference_free_study.py", "fedretrust_directional.py", "directional_study.py")}}


def job(method, regime, scenario, seed):
    return {"method": method, "regime": regime, "scenario": scenario, "seed": seed,
            "phase": "development", "relative_rate": 1., "root_steps": 8,
            "rounds": 24, "train_images": 512, "num_threads": 4}


def key(j):
    return j["method"], j["regime"], j["scenario"], j["seed"]


def required_baselines():
    return [job(m, r, s, seed) for m in BASELINES for r in PARTITIONS
            for seed, scenarios in ((SCREEN_SEED, tuple(longer_scenarios())), (CHECK_SEED, CORE))
            for s in scenarios if s != "ambiguous_malicious_input"]


def prepare(study):
    study.mkdir(parents=True, exist_ok=True)
    path = study / "protocol.json"
    if path.exists():
        verify(study)
        return
    original = json.loads((OLD / "frozen_protocol.json").read_text())["training_source_sha256"]
    if source_hashes() != original:
        raise ValueError("Original frozen training sources have changed.")
    verify_previous(PREVIOUS)
    protocol = {
        "created_at": stamp(), "status": "declared_before_new_candidate_training",
        "scope": "Exploratory development after v0.2 and earlier final results; not confirmatory.",
        "authorization": "User requested another attempt to meet protection and honest-drift targets; include Viceroy.",
        "training_source_sha256": hashes(),
        "partition_sha256": {r: file_sha256(ROOT / "data/partitions" / p / "indices.json") for r, p in PARTITIONS.items()},
        "candidate_configs": {m: asdict(cfg) for m, cfg in CANDIDATE_CONFIGS.items()},
        "trusted_examples_for_candidates": 0,
        "fairness": "Original client/evaluation selections; 300 reserved root images remain excluded from clients and unused by candidates. FLTrust/v0.1 retain original root access. Viceroy retains calibrated 1/n server-rate adapter.",
        "workload": "24 rounds; 10 clients; 512 images/client; one local epoch; original optimizer; 1000 development images/task; four CPU threads; two workers.",
        "screen": {"seed": SCREEN_SEED, "scenarios": CORE, "regimes": list(PARTITIONS), "candidates": CANDIDATES, "new_runs": 18},
        "screen_selection": "Select one configuration across both regimes by minimum largest shortfall versus the BEST of all five baseline clean accuracy, stripe affected accuracy, and label-flip damage; tie by sum then declared candidate order. Zero tolerance (unlike v0.2). Selection does not establish success or statistical superiority.",
        "followup": {"selected_only": True, "screen_seed_all_10_scenarios": SCREEN_SEED, "second_seed_core": CHECK_SEED, "new_candidate_runs": 20},
        "baseline_reuse": "Verify original development and v0.2 artifacts. Five baselines, 130 records; four missing Viceroy controls expected. Also retain the 26 selected v0.2 records as historical paired comparisons.",
        "evaluation": "Original legitimate-task accuracy, own-method matched damage and recovery definitions. Score best-baseline shortfalls for two-seed core means plus per-seed results and all-scenario screen-seed panel. Core success requires all six mean shortfalls <= 0; full-panel success additionally requires all honest-event accuracies and attack damages to beat the best included baseline. Recovery is reported separately, preserving censoring. No official test access or five-seed confidence claims.",
        "stopping": "Exactly three declared variants; no adaptive thresholds or extra search after seeing new results. Follow up the least-bad candidate even if all fail and report failure.",
        "hypothesis": "Remove distance veto that admitted nearby opposing attacks; low saturation threshold retains aligned honest drift. Unit-vector RFA anchor tests whether separating magnitude from directional consensus improves protection.",
        "development_evidence_used": "v0.2 diagnostics on seed 20261014: most non-IID label-flip updates opposed their anchor but had weak distance penalties. Honest non-IID clients also sometimes opposed it, so clean/drift harm is an explicit competing risk.",
        "attribution": "Uses existing RFA solver and v0.2 bounded influence/restoration; replaces joint distance/direction weighting by positive directional evidence. Unit-anchor variant applies the solver to normalized deltas. No inherited theorem, novelty claim, or adaptive-attacker guarantee.",
    }
    write_json(path, protocol)
    snapshot = study / "source_snapshot"
    snapshot.mkdir()
    for name in hashes(): shutil.copy2(ROOT / "code" / name, snapshot / name)
    shutil.copy2(ROOT / "code/tests/test_fedretrust_directional.py", snapshot / "test_fedretrust_directional.py")
    print(json.dumps({"protocol": str(path), "new_runs_expected": 42}), flush=True)


def verify(study):
    p = json.loads((study / "protocol.json").read_text())
    if p["training_source_sha256"] != hashes():
        raise ValueError("Source changed since declaration; preserve this experiment.")
    for r, part in PARTITIONS.items():
        if file_sha256(ROOT / "data/partitions" / part / "indices.json") != p["partition_sha256"][r]:
            raise ValueError("Partition changed since declaration.")
    return p


def checked_result(result):
    if result["status"] != "complete": raise ValueError("Incomplete result.")
    for name, digest in result["artifact_sha256"].items():
        if file_sha256(ROOT / result["run_path"] / name) != digest:
            raise ValueError("Saved artifact changed.")
    meta = json.loads((ROOT / result["run_path"] / "run.json").read_text())
    if meta["official_test_used"] or meta.get("evaluation_data", {}).get("split") == "official_test":
        raise ValueError("This study only accepts development evaluation.")
    return result


def collect(study):
    records = {}
    desired = {key(j) for j in required_baselines()}
    for f in (OLD / "jobs").glob("*/result.json"):
        result = json.loads(f.read_text()); j = result["job"]
        if (j["phase"] == "development" and j["relative_rate"] == 1. and key(j) in desired):
            if result["source_sha256"] != source_hashes(): raise ValueError("Baseline source mismatch.")
            records[key(j)] = checked_result(result)
    verify_previous(PREVIOUS)
    previous = json.loads((PREVIOUS / "result_manifest.json").read_text())["records"]
    for result in previous:
        j = result["job"]
        if key(j) in desired or j["method"] == PREVIOUS_METHOD:
            if j["method"] == PREVIOUS_METHOD and result["source_sha256"] != json.loads((PREVIOUS / "protocol.json").read_text())["training_source_sha256"]:
                raise ValueError("Previous prototype source mismatch.")
            if j["method"] in BASELINES and result["source_sha256"] != source_hashes():
                raise ValueError("Previous baseline source mismatch.")
            records[key(j)] = checked_result(result)
    for f in (study / "jobs").glob("*/result.json"):
        result = checked_result(json.loads(f.read_text()))
        if result["job"]["method"] not in CANDIDATES and result["source_sha256"] != source_hashes():
            raise ValueError("New baseline source mismatch.")
        records[key(result["job"])] = result
    return records


def worker(j, study):
    verify(study)
    if j["method"] in BASELINES:
        if j not in required_baselines(): raise ValueError("Unregistered baseline job.")
        return execute_job(j, study)
    allowed = [job(m, r, s, SCREEN_SEED) for m in CANDIDATES for r in PARTITIONS for s in CORE]
    if (study / "selection.json").exists():
        selected = json.loads((study / "selection.json").read_text())["selected"]
        allowed += [job(selected, r, s, seed) for r in PARTITIONS
                    for seed, scenarios in ((SCREEN_SEED, tuple(longer_scenarios())), (CHECK_SEED, CORE))
                    for s in scenarios if s != "ambiguous_malicious_input"]
    if j not in allowed: raise ValueError("Unregistered candidate job.")
    jid = canonical_hash({"job": j, "sources": hashes()})[:20]
    directory = study / "jobs" / jid
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "worker.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (directory / "result.json").exists():
            return checked_result(json.loads((directory / "result.json").read_text()))
        if (directory / "failure.json").exists(): raise ValueError("Explicit failure investigation required.")
        write_json(directory / "request.json", {"job": j, "source_sha256": hashes()})
        started = time.perf_counter()
        try:
            cfg = TrainConfig(partition_id=PARTITIONS[j["regime"]], development_seed=j["seed"],
                              rounds=24, max_train_per_client=512, max_validation=1000, num_threads=4)
            data = load_data(ROOT, cfg)
            options = CANDIDATE_CONFIGS[j["method"]]
            # The aggregator is constructed with scalar settings only.
            agg = ContributionRecorder(DirectionalFedReTrust(options))
            run = run_experiment(data, cfg, longer_scenarios()[j["scenario"]], agg,
                                 notebook_code_sha256=jid, verbose=False, evaluator=ClasswiseEvaluator())
            counts = {i: np.bincount(data.pool.targets[ix].numpy(), minlength=10).tolist()
                      for i, ix in data.manifest["indices"]["clients"].items()}
            write_json(run / "population_class_counts.json", counts)
            names = ("run.json", "selection.json", "summary.json", "history.csv", "task_metrics.csv",
                     "client_metrics.csv", "aggregation_diagnostics.json", "population_class_counts.json",
                     "initial_global_model_state.pt", "final_global_model_state.pt", "status.json")
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            result = {"job_id": jid, "job": j, "status": "complete", "run_path": str(run.relative_to(ROOT)),
                      "artifact_sha256": {n: file_sha256(run / n) for n in names},
                      "source_sha256": hashes(), "worker_wall_seconds": time.perf_counter() - started,
                      "process_lifetime_peak_rss_bytes": int(rss if platform.system() == "Darwin" else rss * 1024)}
            write_json(directory / "result.json", result)
            return result
        except Exception as exc:
            write_json(directory / "failure.json", {"job": j, "error": repr(exc), "at": stamp()})
            raise


def stage(jobs, study, label):
    records = collect(study)
    pending = [j for j in jobs if key(j) not in records]
    requests = study / "requests"; requests.mkdir(exist_ok=True)
    def launch(j):
        ident = canonical_hash(j)[:20]
        request = requests / f"{ident}.json"
        write_json(request, j)
        with (requests / f"{ident}.log").open("w") as log:
            subprocess.run([sys.executable, str(Path(__file__).resolve()), "worker", "--study", str(study),
                            "--job", str(request)], stdout=log, stderr=subprocess.STDOUT, check=True)
        return j
    write_json(study / "progress.json", {"stage": label, "completed": 0, "pending": len(pending), "at": stamp()})
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(launch, j): j for j in pending}
        try:
            for n, future in enumerate(as_completed(futures), 1):
                done = future.result()
                progress = {"stage": label, "completed": n, "pending": len(pending) - n, "job": done, "at": stamp()}
                write_json(study / "progress.json", progress)
                print(json.dumps(progress), flush=True)
        except BaseException:
            for future in futures: future.cancel()
            raise


def metrics_for(records, k):
    r = records[k]; j = r["job"]
    ref = records.get((j["method"], j["regime"], matched_reference_name(j["scenario"]), j["seed"]))
    return {x["metric"]: x["value"] for x in extract_run(r, ref if j["scenario"] != "clean" else None)[0]}


def select(study):
    if (study / "selection.json").exists(): return json.loads((study / "selection.json").read_text())["selected"]
    records = collect(study); rows = []
    for m in CANDIDATES:
        violations = []; detail = []
        for r in PARTITIONS:
            clean = metrics_for(records, (m, r, "clean", SCREEN_SEED))["overall_mean_5_24_pp"]
            best_clean = max(metrics_for(records, (b, r, "clean", SCREEN_SEED))["overall_mean_5_24_pp"] for b in BASELINES)
            drift = metrics_for(records, (m, r, "gradual_stripe", SCREEN_SEED))["affected_honest_event_pp"]
            best_drift = max(metrics_for(records, (b, r, "gradual_stripe", SCREEN_SEED))["affected_honest_event_pp"] for b in BASELINES)
            damage = metrics_for(records, (m, r, "label_flip", SCREEN_SEED))["attack_active_signed_damage_pp"]
            best_damage = min(metrics_for(records, (b, r, "label_flip", SCREEN_SEED))["attack_active_signed_damage_pp"] for b in BASELINES)
            v = [max(0., best_clean - clean), max(0., best_drift - drift), max(0., damage - best_damage)]
            violations += v
            detail.append({"regime": r, "clean_pp": clean, "stripe_pp": drift, "damage_pp": damage,
                           "best_baseline_clean_pp": best_clean, "best_baseline_stripe_pp": best_drift,
                           "best_baseline_damage_pp": best_damage, "violations_pp": v})
        rows.append({"candidate": m, "largest_violation_pp": max(violations), "sum_violation_pp": sum(violations),
                     "all_screen_guards_pass": max(violations) == 0, "details": detail})
    chosen = min(rows, key=lambda x: (x["largest_violation_pp"], x["sum_violation_pp"], CANDIDATES.index(x["candidate"])))
    write_json(study / "selection.json", {"selected": chosen["candidate"], "at": stamp(), "candidates": rows,
               "interpretation": "Follow-up candidate; selection is not a finding of success. Second seed is an exploratory replication using previously used development data."})
    print(json.dumps({"selection": chosen}), flush=True)
    return chosen["candidate"]


def run(study):
    verify(study)
    with (study / "coordinator.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            screen = [job(m, r, s, SCREEN_SEED) for m in CANDIDATES for r in PARTITIONS for s in CORE]
            stage([j for j in required_baselines() if j["scenario"] in CORE and j["seed"] == SCREEN_SEED], study, "screen_controls")
            stage(screen, study, "screen")
            selected = select(study)
            followup = [job(selected, r, s, seed) for r in PARTITIONS
                        for seed, scenarios in ((SCREEN_SEED, tuple(longer_scenarios())), (CHECK_SEED, CORE))
                        for s in scenarios if s != "ambiguous_malicious_input"]
            stage(followup + required_baselines(), study, "followup_and_missing_baseline_controls")
            analyse(study)
            write_json(study / "status.json", {"status": "complete", "at": stamp(), "selected": selected})
        except BaseException as exc:
            write_json(study / "status.json", {"status": "stopped_on_error", "at": stamp(), "error": repr(exc)})
            raise


def analyse(study):
    verify(study)
    records = collect(study)
    metrics, curves, episodes, costs = [], [], [], []
    for k, result in records.items():
        m, regime, scenario, seed = k
        control = records.get((m, regime, matched_reference_name(scenario), seed))
        a, b, c = extract_run(result, control if scenario != "clean" else None)
        metrics += a; curves += b; episodes += c; costs.append(cost_record(result))
        if m in CANDIDATES:
            cross_method_pair(result, records[("FedAvg", regime, scenario, seed)])
    output = study / "analysis"; output.mkdir(exist_ok=True)
    for name, data in (("seed_metrics", metrics), ("accuracy_curves", curves), ("model_episodes", episodes), ("costs", costs)):
        write_json(output / f"{name}.json", data)
        pd.DataFrame(data).to_csv(output / f"{name}.csv", index=False)
    write_json(study / "result_manifest.json", {"at": stamp(), "records": list(records.values()),
               "official_test_used": False, "original_training_sources_unchanged": source_hashes() == json.loads((OLD / "frozen_protocol.json").read_text())["training_source_sha256"]})
    table = pd.DataFrame(metrics)
    selected = json.loads((study / "selection.json").read_text())["selected"]
    chosen = table[table.method.isin([*BASELINES, PREVIOUS_METHOD, selected])]
    for seed in (SCREEN_SEED, CHECK_SEED):
        compact = chosen[(chosen.seed == seed) & chosen.metric.isin(["overall_mean_5_24_pp", "affected_honest_event_pp", "attack_active_signed_damage_pp"])]
        compact.pivot(index=["regime", "scenario", "metric"], columns="method", values="value").round(3).to_csv(output / f"comparison_seed_{seed}.csv")
    core = chosen[chosen.scenario.isin(CORE) & chosen.metric.isin(["overall_mean_5_24_pp", "affected_honest_event_pp", "attack_active_signed_damage_pp"])]
    core_means = core.groupby(["regime", "scenario", "metric", "method"]).value.agg(["mean", "count"]).reset_index()
    if not (core_means["count"] == 2).all(): raise ValueError("Incomplete two-seed core means.")
    core_means.pivot(index=["regime", "scenario", "metric"], columns="method", values="mean").to_csv(output / "two_seed_core_means.csv")
    targets = []
    for regime in PARTITIONS:
        for scenario, metric, direction in (("clean", "overall_mean_5_24_pp", 1), ("gradual_stripe", "affected_honest_event_pp", 1), ("label_flip", "attack_active_signed_damage_pp", -1)):
            x = core_means[(core_means.regime == regime) & (core_means.scenario == scenario) & (core_means.metric == metric)]
            baselines = x[x.method.isin(BASELINES)]
            best = baselines.loc[(baselines["mean"] * direction).idxmax()]
            value = float(x[x.method == selected]["mean"].item())
            targets.append({"regime":regime, "scenario":scenario, "metric":metric, "candidate_pp":value,
                            "best_baseline":best.method, "target_pp":float(best["mean"]),
                            "advantage_pp":direction * (value-float(best["mean"])),
                            "target_met_descriptively":direction * (value-float(best["mean"])) >= 0})
    write_json(output / "target_check.json", {"core_mean_targets_all_met":all(t["target_met_descriptively"] for t in targets), "comparisons":targets, "warning":"Two reused development seeds, fixed partitions; no confirmatory superiority claim."})
    audit = []
    for result in records.values():
        if result["job"]["method"] not in CANDIDATES: continue
        ds = json.loads((ROOT/result["run_path"]/"aggregation_diagnostics.json").read_text())
        meta = json.loads((ROOT/result["run_path"]/"run.json").read_text())
        assert not meta["official_test_used"]
        assert all(d["values"]["reference_forward_examples"] == 0 for d in ds)
        audit.append({"job_id":result["job_id"], "rounds":len(ds), "reference_forward_examples":0, "paired_FedAvg_provenance":"passed"})
    write_json(study / "candidate_access_audit.json", {"runs":audit, "candidate_runs":len(audit), "candidate_rounds":sum(a["rounds"] for a in audit), "trusted_examples":0})
    print(json.dumps({"analysis": str(output), "records": len(records), "targets": targets}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("prepare", "run", "worker", "analyse"))
    p.add_argument("--study", type=Path, default=STUDY)
    p.add_argument("--job", type=Path)
    args = p.parse_args()
    if args.action == "worker": worker(json.loads(args.job.read_text()), args.study)
    else: globals()[args.action](args.study)
