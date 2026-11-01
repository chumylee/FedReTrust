"""Frozen feasibility pilot for untrusted client-local validation reports.

No global model is updated. Each candidate is one 10%-weighted local update at
an existing common clean broadcast. Local reporting and independent audit data
are isolated from the pure score rules. This is a simulation, not a distributed
privacy implementation or a reproduction of FLShield.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
import argparse
import fcntl
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
import torch

from experiment_core import (MnistCNN, clone_state, file_sha256, load_data,
                             state_sha256, train_client, write_json)
from research_runs import ROOT, PARTITIONS, canonical_hash
from mechanism_pilot import STUDY as PREVIOUS, verify as verify_previous, hashes as previous_hashes
from common_broadcast_probe import _validate_baseline
from scenario_metrics import longer_scenarios
from scenario_utils import Event, Scenario, client_view
from client_validation_local import build_local_validation_plan, evaluate_local_candidates
from client_validation_analysis import DIAGNOSTIC_PROTOCOL, analyse_study


STUDY = ROOT / "outputs/client_validation/20261030_v1"
SEEDS = (20261014, 20261015)
CHECKPOINTS = (5, 8, 12, 20)
SCENARIOS = ("clean", "abrupt_shot_noise", "gradual_stripe", "label_flip",
             "attack_stops", "mixed_stripe_attack")
NEW_SOURCES = ("client_validation_local.py", "client_validation_analysis.py", "client_validation_study.py")
NEW_TESTS = ("test_client_validation_local.py", "test_client_validation_analysis.py")
MARGINAL_RATE = .1


def stamp():
    return datetime.now(timezone.utc).isoformat()


def scenarios():
    result = {s: longer_scenarios()[s] for s in SCENARIOS if s != "mixed_stripe_attack"}
    result["mixed_stripe_attack"] = Scenario("mixed_stripe_attack", (
        Event("label_flip", (0, 1, 2), ((5, 24),), intent="malicious"),
        Event("input_change", (3, 4, 5), ((5, 24),), corruption="stripe", ramp_rounds=8),
    ))
    return result


def hashes():
    return {**previous_hashes(), **{n: file_sha256(ROOT / "code" / n) for n in NEW_SOURCES}}


def jobs():
    return [{"regime": r, "seed": seed, "round": round_number, "scenario": scenario}
            for r in PARTITIONS for seed in SEEDS for round_number in CHECKPOINTS for scenario in SCENARIOS]


def blocks():
    return [{"regime": r, "seed": seed, "round": round_number}
            for r in PARTITIONS for seed in SEEDS for round_number in CHECKPOINTS]


def clean_records():
    verify_previous(PREVIOUS)
    records = json.loads((PREVIOUS / "result_manifest.json").read_text())["observed_records"]
    return [r for r in records if r["job"]["method"] == "FedAvg" and r["job"]["scenario"] == "clean"]


def prepare(study):
    study.mkdir(parents=True, exist_ok=True)
    if (study / "protocol.json").exists():
        verify(study)
        return
    records = clean_records()
    plan = build_local_validation_plan(ROOT, records, report_size=128, audit_size=128)
    private = study / "client_side"; private.mkdir(exist_ok=True)
    write_json(private / "validation_plan.json", plan)
    write_json(study / "source_records.json", {"records": records})
    protocol = {
        "created_at": stamp(), "status": "declared_before_new_local_training_or_validation",
        "authorization": "User requested client-local validation pilot: ok do it.",
        "scope": "Exploratory utility-observation pilot on previously used development seeds and fixed partitions; not a new global training benchmark, FedReTrust version, FLShield reproduction, novelty claim or privacy guarantee.",
        "source_sha256": hashes(), "source_records_sha256": file_sha256(study / "source_records.json"),
        "private_validation_plan_sha256": file_sha256(private / "validation_plan.json"),
        "jobs": jobs(), "blocks": blocks(), "cases": 96, "local_trainings": 960,
        "parent_checkpoint_manifest_sha256": file_sha256(PREVIOUS / "result_manifest.json"),
        "scenarios": {k: v.to_dict() for k, v in scenarios().items()},
        "candidate_model": "broadcast + 0.1 * (submitted_local_model - broadcast); float64 arithmetic then cast to original dtype",
        "marginal_rate": MARGINAL_RATE,
        "training": "Original 512 training images per client, one local epoch and optimizer; ten clients; saved clean FedAvg pre-update checkpoints5,8,12,20; no feedback, selected-update aggregation or evolved global model.",
        "validation": "128 report and128 audit examples per client, sampled without label balancing from their original assigned pool. Globally exclude all images used by any of the four source trajectories, all server-reference and development-validation indices. Globally disjoint report/audit banks prevent role overlap across seeds/regimes. All images and labels stay in simulated local evaluators; the server rule receives a matrix of scalar scores only.",
        "utility": "Half multiclass Brier loss:0.5*sum((softmax-onehot)^2), bounded0..1. Report gain=mean loss(broadcast)-mean loss(candidate), bounded-1..1. Classwise Brier/CE and accuracy are secondary diagnostics; missing classes remain null. Honest input drift also changes the validator's local task. Audit labels always represent the legitimate task.",
        "controls": DIAGNOSTIC_PROTOCOL,
        "reference_access": {"trusted_server_images": 0, "global_development_forward_examples": 0, "official_test_used": False},
        "recovery_scope": "attack_stops label poisoning and dishonest reporting end afterround10. Readmission at12/20 is a stateless decision-level check at clean common broadcasts; these updates should equal the clean counterfactual. It is not learned rehabilitation, global accuracy recovery, or the original1pp/3consecutive-round recovery criterion.",
        "compute": "16 checkpoint blocks, two workers maximum, four CPU threads each. Each case trains10 local models and validates11 models (broadcast plus10 candidates) on128 report and128 audit images at each of10clients. Total960 localtrainings and2703360 validation forward examples; audit half is research-only overhead.",
        "communication": "A naive full-model deployment requires nine other candidates pervalidator,90 model transmissions per case, plus90 peer scalar reports after excluding self. Count float32 parameter payload separately from serialized diagnostics; no compressed/deployed communication implementation.",
        "stopping": "Exactly96 cases. Three fixed rules and four predeclared reporting profiles; no tuned thresholds or extra search. A failed gate stops this proposed combination. Any pass only justifies a separately declared aggregation experiment and does not establish all-update aggregate utility.",
        "sources": ["https://arxiv.org/html/2308.05832", "https://github.com/ehsanul9511/FLShield"],
        "source_distinction": "FLShield motivates local validation but validates mixed representatives and uses a different report filter. Our individual10%-scaled candidates, boundedBrier, self-exclusion and three score controls are an explicit diagnostic adaptation. Current author-code filtering also differs from the paper; no faithful reproduction is claimed.",
    }
    write_json(study / "protocol.json", protocol)
    snapshot = study / "source_snapshot"; snapshot.mkdir()
    for name in hashes(): shutil.copy2(ROOT / "code" / name, snapshot / name)
    for name in NEW_TESTS: shutil.copy2(ROOT / "code/tests" / name, snapshot / name)
    print(json.dumps({"protocol": str(study / "protocol.json"), "cases": 96, "local_trainings": 960}), flush=True)


def verify(study):
    protocol = json.loads((study / "protocol.json").read_text())
    if protocol["source_sha256"] != hashes():
        raise ValueError("A source changed after the local-validation pilot was declared.")
    for name, field in (("source_records.json", "source_records_sha256"),
                        ("client_side/validation_plan.json", "private_validation_plan_sha256")):
        if file_sha256(study / name) != protocol[field]:
            raise ValueError("Pilot input fingerprint changed.")
    if file_sha256(PREVIOUS / "result_manifest.json") != protocol["parent_checkpoint_manifest_sha256"]:
        raise ValueError("Parent checkpoint manifest changed.")
    verify_previous(PREVIOUS)
    return protocol


def checked_result(result):
    if result["status"] != "complete" or result["job"] not in jobs() or result["source_sha256"] != hashes():
        raise ValueError("Invalid local-validation result.")
    path = ROOT / result["run_path"]
    for name, digest in result["artifact_sha256"].items():
        if file_sha256(path / name) != digest:
            raise ValueError("Local-validation artifact changed.")
    return result


def results(study):
    return [checked_result(json.loads(path.read_text())) for path in sorted((study / "cases").glob("*/result.json"))]


def access_audit(study, records):
    rows = []
    for result in records:
        path = ROOT / result["run_path"]
        meta = json.loads((path / "metadata.json").read_text())
        replay = json.loads((path / "replay_audit.json").read_text())
        reports = json.loads((path / "reports.json").read_text())
        audit = json.loads((path / "audit.json").read_text())
        assert not meta["official_test_used"] and not meta["global_model_advanced"]
        assert meta["server_reference_forward_examples"] == meta["global_development_forward_examples"] == 0
        assert len(reports) == len(audit) == 100
        assert all(r["examples"] == 128 for r in reports+audit)
        assert all(r["exact_match"] for r in replay)
        report_count = sum(r["report_forward_examples"] for r in meta["local_validation_costs"])
        audit_count = sum(r["audit_forward_examples"] for r in meta["local_validation_costs"])
        assert report_count == audit_count == 14080
        rows.append({"job": result["job"], "exact_replayed_updates": len(replay),
                     "report_forward_examples": report_count, "audit_forward_examples": audit_count,
                     "naive_model_distribution_bytes": 90*meta["float32_model_payload_bytes"],
                     "peer_scalar_float64_payload_bytes": 90*8})
    assert len(rows) == 96 and sum(r["exact_replayed_updates"] for r in rows) == 912
    output = {"cases": 96, "local_trainings": 960, "exact_prior_update_replays": 912,
              "trusted_server_images": 0, "official_test_used": False,
              "report_forward_examples": sum(r["report_forward_examples"] for r in rows),
              "audit_forward_examples": sum(r["audit_forward_examples"] for r in rows),
              "global_development_forward_examples": 0, "global_models_advanced": 0, "cases_audit": rows,
              "communication_caveat": "Naive uncompressed float32 candidate-model distribution and float64 scalar-only payload estimates; metadata, transport, security and optional class-statistic overhead omitted. Audit inference is research-only, not required by a deployed score rule."}
    write_json(study / "access_audit.json", output)
    return output


def candidate_state(broadcast, trained):
    return {name: (value.double() + MARGINAL_RATE*(trained[name].double()-value.double())).to(value.dtype)
            for name, value in broadcast.items()}


def execute_case(job, study, data, cfg, broadcast, source_record, selections, old_training):
    ident = canonical_hash({"job": job, "sources": hashes()})[:20]
    directory = study / "cases" / ident
    if directory.exists():
        return checked_result(json.loads((directory / "result.json").read_text()))
    directory.parent.mkdir(exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{ident}.", dir=directory.parent))
    started = time.perf_counter()
    try:
        random.seed(cfg.development_seed); np.random.seed(cfg.development_seed)
        torch.manual_seed(cfg.development_seed)
        torch.set_num_threads(cfg.num_threads); torch.use_deterministic_algorithms(True)
        model = MnistCNN(); model.load_state_dict(broadcast)
        broadcast_hash = state_sha256(broadcast)
        scenario = scenarios()[job["scenario"]]
        candidates, training, replay_checks = {}, [], []
        for cid in range(10):
            view = client_view(data.pool, data.selection["clients"][str(cid)], data.manifest["normalization"],
                               cfg.development_seed, scenario, cid, job["round"])
            trained, metrics = train_client(model, view, cfg, cid, job["round"], broadcast)
            trained_hash = state_sha256(trained)
            candidates[cid] = candidate_state(broadcast, trained)
            training.append({"client_id": cid, "examples": len(view), **metrics,
                             "submitted_state_sha256": trained_hash,
                             "marginal_state_sha256": state_sha256(candidates[cid])})
            source_scenario = job["scenario"]
            if source_scenario == "attack_stops":
                source_scenario = "label_flip" if job["round"] <= 10 else "clean"
            if source_scenario == "mixed_stripe_attack":
                source_scenario = "label_flip" if cid in (0, 1, 2) else "clean" if cid in (6, 7, 8, 9) else None
            if source_scenario is not None:
                expected = old_training[(source_scenario, cid)]
                if trained_hash != expected["submitted_state_sha256"] or metrics["shuffle_seed"] != expected["shuffle_seed"]:
                    raise AssertionError("Repeated local training changed its saved counterfactual update.")
                replay_checks.append({"client_id": cid, "source_scenario": source_scenario, "exact_match": True})
        train_seconds = time.perf_counter()-started
        if state_sha256(clone_state(model)) != broadcast_hash:
            raise AssertionError("Local training changed the common model.")
        # Simulator creates each client's legitimate local task. Neither this
        # scenario nor any actor labels cross the pure scoring-rule interface.
        legitimate = Scenario(scenario.name, tuple(e for e in scenario.events
                                                   if e.kind == "input_change" and e.intent == "honest"))
        report_rows, audit_rows, costs = [], [], []
        for validator in range(10):
            local = selections[str(validator)]
            report_view = client_view(data.pool, local["report_indices"], data.manifest["normalization"],
                                      cfg.development_seed, legitimate, validator, job["round"])
            audit_view = client_view(data.pool, local["audit_indices"], data.manifest["normalization"],
                                     cfg.development_seed, legitimate, validator, job["round"])
            value = evaluate_local_candidates(broadcast, candidates, report_view, audit_view, cfg, validator)
            report_rows.extend(value["report_rows"]); audit_rows.extend(value["audit_rows"]); costs.append(value["cost"])
        if len(report_rows) != 100 or len(audit_rows) != 100:
            raise AssertionError("Every candidate needs all ten local evaluators.")
        if state_sha256(clone_state(model)) != broadcast_hash:
            raise AssertionError("Local validation changed the common model.")
        metadata = {"job": job, "scenario": scenario.to_dict(), "source_broadcast_sha256": broadcast_hash,
                    "source_record_job_id": source_record["job_id"], "source_run_path": source_record["run_path"],
                    "source_checkpoint_sha256": source_record["artifact_sha256"]["observation_checkpoints.pt"],
                    "config": vars(cfg), "marginal_rate": MARGINAL_RATE,
                    "private_plan_sha256": file_sha256(study / "client_side/validation_plan.json"),
                    "official_test_used": False, "server_reference_forward_examples": 0,
                    "global_development_forward_examples": 0, "global_model_advanced": False,
                    "training_seconds": train_seconds, "local_validation_costs": costs,
                    "cost": {"training_seconds": train_seconds,
                             "validation_seconds": sum(c["elapsed_seconds"] for c in costs),
                             "report_forward_examples": sum(c["report_forward_examples"] for c in costs),
                             "audit_forward_examples": sum(c["audit_forward_examples"] for c in costs),
                             "wall_seconds_before_artifact_save": time.perf_counter()-started},
                    "parameter_count": sum(v.numel() for v in broadcast.values()),
                    "float32_model_payload_bytes": sum(v.numel()*v.element_size() for v in broadcast.values()),
                    "evaluation_note": "reports.json contains truthful and corrupted-label counterfactual reports for offline attack construction. A scoring rule receives only one scalar report matrix at a time. audit.json is evaluation-only; no labels, indices, scenario or audit enters the rule."}
        for name, value in (("metadata.json", metadata), ("reports.json", report_rows), ("audit.json", audit_rows),
                            ("training.json", training), ("replay_audit.json", replay_checks)):
            write_json(temporary / name, value)
        names = ("metadata.json", "reports.json", "audit.json", "training.json", "replay_audit.json")
        result = {"status": "complete", "job_id": ident, "job": job,
                  "run_path": str(directory.relative_to(ROOT)), "source_sha256": hashes(),
                  "artifact_sha256": {n: file_sha256(temporary / n) for n in names},
                  "worker_wall_seconds": time.perf_counter()-started, "local_trainings": 10,
                  "at": stamp()}
        write_json(temporary / "result.json", result)
        os.replace(temporary, directory)
        return result
    except BaseException as exc:
        write_json(temporary / "failure.json", {"job": job, "error": repr(exc), "at": stamp()})
        raise


def worker(block, study):
    verify(study)
    if block not in blocks(): raise ValueError("Unregistered checkpoint block.")
    lockdir = study / "locks"; lockdir.mkdir(exist_ok=True)
    with (lockdir / f"{canonical_hash(block)[:20]}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        records = json.loads((study / "source_records.json").read_text())["records"]
        record = next(r for r in records if r["job"]["regime"] == block["regime"] and r["job"]["seed"] == block["seed"])
        _, cfg, _, source_selection, checkpoints, _ = _validate_baseline(record)
        data = load_data(ROOT, cfg, task_names=("clean",))
        if data.selection != source_selection:
            raise ValueError("Source training selections changed.")
        plan = json.loads((study / "client_side/validation_plan.json").read_text())
        selections = plan["clients"][block["regime"]][str(block["seed"])]
        previous = PREVIOUS / "probes" / f"{block['regime']}_seed_{block['seed']}_round_{block['round']:02d}"
        previous_manifest = json.loads((previous / "result.json").read_text())
        if file_sha256(previous / "training.json") != previous_manifest["artifact_sha256"]["training.json"]:
            raise ValueError("Parent counterfactual training artifact changed.")
        old_training = {(r["scenario"], r["client"]): r for r in json.loads((previous / "training.json").read_text())}
        for scenario in SCENARIOS:
            execute_case({**block, "scenario": scenario}, study, data, cfg, checkpoints[block["round"]],
                         record, selections, old_training)
        return block


def run(study):
    verify(study)
    with (study / "coordinator.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write_json(study / "status.json", {"status": "running", "at": stamp()})
        requests = study / "requests"; requests.mkdir(exist_ok=True)
        def launch(block):
            ident = canonical_hash(block)[:20]
            request = requests / f"{ident}.json"; write_json(request, block)
            with (requests / f"{ident}.log").open("w") as log:
                subprocess.run([sys.executable, str(Path(__file__).resolve()), "worker", "--study", str(study),
                                "--block", str(request)], stdout=log, stderr=subprocess.STDOUT, check=True)
            return block
        try:
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(launch, block) for block in blocks()]
                for n, future in enumerate(as_completed(futures), 1):
                    block = future.result()
                    progress = {"completed_blocks": n, "total_blocks": 16, "completed_cases": len(results(study)),
                                "total_cases": 96, "block": block, "at": stamp()}
                    write_json(study / "progress.json", progress); print(json.dumps(progress), flush=True)
            records = results(study)
            if len(records) != 96: raise ValueError("Incomplete pilot.")
            write_json(study / "result_manifest.json", {"records": records, "at": stamp(), "official_test_used": False,
                       "local_trainings": 960, "global_models_trained": 0, "trusted_server_images": 0})
            access_audit(study, records)
            analyse_study(study)
            write_json(study / "status.json", {"status": "complete", "cases": 96, "local_trainings": 960, "at": stamp()})
        except BaseException as exc:
            write_json(study / "status.json", {"status": "stopped_on_error", "error": repr(exc), "at": stamp()})
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "run", "worker", "analyse"))
    parser.add_argument("--study", type=Path, default=STUDY)
    parser.add_argument("--block", type=Path)
    args = parser.parse_args()
    if args.action == "worker": worker(json.loads(args.block.read_text()), args.study)
    elif args.action == "analyse":
        verify(args.study); analyse_study(args.study)
    else: globals()[args.action](args.study)
