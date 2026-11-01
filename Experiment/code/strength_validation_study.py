"""Controlled contribution-strength pilot with frozen matched inputs.

Four strengths per client, one client varied at a time; no global trajectory
is advanced. Endpoints are freshly evaluated and must reproduce the parent.
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
from aggregate_validation_study import (STUDY as PREVIOUS, verify as verify_previous,
                                     hashes as previous_hashes, checked_result as checked_previous_result)
from common_broadcast_probe import _validate_baseline
from scenario_metrics import longer_scenarios
from scenario_utils import Event, Scenario, client_view
from strength_validation_local import build_strength_models, evaluate_strength_models, verify_endpoint_rows, SEMANTICS
from strength_validation_analysis import DIAGNOSTIC_PROTOCOL, analyse_study


STUDY = ROOT / "outputs/strength_validation/20261030_v1"
SEEDS = (20261014, 20261015)
CHECKPOINTS = (5, 8, 12, 20)
SCENARIOS = ("clean", "abrupt_shot_noise", "gradual_stripe", "label_flip",
             "attack_stops", "mixed_stripe_attack")
NEW_SOURCES = ("strength_validation_local.py", "strength_validation_analysis.py", "strength_validation_study.py")
NEW_TESTS = ("test_strength_validation.py",)
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
    return json.loads((PREVIOUS / "source_records.json").read_text())["records"]


def prepare(study):
    study.mkdir(parents=True, exist_ok=True)
    if (study / "protocol.json").exists():
        verify(study)
        return
    clean_records()  # Verify the frozen parent before copying its exact inputs.
    private = study / "client_side"; private.mkdir(exist_ok=True)
    shutil.copy2(PREVIOUS / "client_side/validation_plan.json", private / "validation_plan.json")
    shutil.copy2(PREVIOUS / "source_records.json", study / "source_records.json")
    protocol = {
        "created_at": stamp(), "status": "declared_before_new_local_training_or_validation",
        "authorization": "User authorized fixed 0%,25%,50%,100% contribution-strength test: do it; build and train adaptive weighting only if a useful, robust balance exists.",
        "scope": "Exploratory contribution-strength utility pilot on the same reused development seeds, fixed partitions, local data and updates as the preceding aggregate-validation pilot. No global trajectory is advanced; no FedReTrust performance, FLShield reproduction, novelty or privacy guarantee is claimed.",
        "source_sha256": hashes(), "source_records_sha256": file_sha256(study / "source_records.json"),
        "private_validation_plan_sha256": file_sha256(private / "validation_plan.json"),
        "jobs": jobs(), "blocks": blocks(), "cases": 96, "local_trainings": 960,
        "parent_checkpoint_manifest_sha256": file_sha256(PREVIOUS / "result_manifest.json"),
        "parent_protocol_sha256": file_sha256(PREVIOUS / "protocol.json"),
        "parent_comparison_sha256": {name: file_sha256(PREVIOUS / "analysis" / name)
                                     for name in ("decision_rows.csv", "grouped_rates.csv", "audit_utility.csv")},
        "scenarios": {k: v.to_dict() for k, v in scenarios().items()},
        "candidate_model": SEMANTICS,
        "scales": [0., .25, .5, 1.],
        "marginal_rate": MARGINAL_RATE,
        "training": "Original 512 training images per client, one local epoch and optimizer; ten clients; saved clean FedAvg pre-update checkpoints 5, 8, 12, 20; no feedback, selected-update aggregation or evolved global model.",
        "validation": "Exactly the preceding pilot's 128 report and 128 audit examples per client, copied byte-identically rather than resampled; originally sampled without label balancing from each client's assigned pool. Globally exclude all images used by any of the four source trajectories, all server-reference and development-validation indices. Globally disjoint report/audit banks prevent role overlap across seeds/regimes. All images and labels stay in simulated local evaluators; the server rule receives a matrix of scalar scores only.",
        "utility": "Half multiclass Brier loss: 0.5*sum((softmax-onehot)^2), bounded 0..1. Contribution gain=mean loss(without_client)-mean loss(model_at_selected_strength), bounded -1..1; positive means that strength improves the aggregate compared with omission. Classwise Brier/CE and accuracy are secondary diagnostics; missing classes remain null. Honest input drift also changes the validator's local task. Audit labels always represent the legitimate task.",
        "controls": DIAGNOSTIC_PROTOCOL,
        "endpoint_replay": "Freshly evaluate all endpoint models: 100% reports/audits must exactly equal the parent; zero action gains must equal zero. Full and ten omitted-model hashes must match too.",
        "reference_access": {"trusted_server_images": 0, "global_development_forward_examples": 0, "official_test_used": False},
        "recovery_scope": "attack_stops label poisoning and dishonest reporting end after round 10. Readmission at 12/20 is a stateless decision-level check at clean common broadcasts; these updates should equal the clean counterfactual. It is not learned rehabilitation, global accuracy recovery, or the original 1 pp / 3 consecutive-round recovery criterion.",
        "compute": "16 blocks, two workers maximum, four CPU threads each. Each case replays ten local trainings and evaluates 31 unique models on 128 report and 128 audit images per validator: shared full model, ten omissions and twenty fractional models. Total 960 local trainings and 7,618,560 forward examples (3,809,280 per role). All endpoints are recomputed, not reused as fresh results.",
        "communication": "Naive server-to-client scoring sends full aggregate plus nine peer omissions and eighteen fractional models to each validator: 280 float32 model payloads/case, plus 270 float64 scalar reports across three nonzero scales. Own-candidate evaluations support offline audit. These are uncompressed payload estimates, not measured traffic.",
        "stopping": DIAGNOSTIC_PROTOCOL["continuation"],
        "sources": ["https://arxiv.org/html/2308.05832", "https://github.com/ehsanul9511/FLShield"],
        "source_distinction": "FLShield motivates local validation but validates mixed representatives and uses a different report filter. Our fixed-weight full/omitted/fractional models, bounded Brier, self-exclusion and three score controls are an explicit diagnostic adaptation. Current author-code filtering also differs from the paper; no faithful reproduction is claimed.",
    }
    write_json(study / "protocol.json", protocol)
    snapshot = study / "source_snapshot"; snapshot.mkdir()
    for name in hashes(): shutil.copy2(ROOT / "code" / name, snapshot / name)
    for name in NEW_TESTS: shutil.copy2(ROOT / "code/tests" / name, snapshot / name)
    print(json.dumps({"protocol": str(study / "protocol.json"), "cases": 96, "local_trainings": 960}), flush=True)


def verify(study):
    protocol = json.loads((study / "protocol.json").read_text())
    if protocol["source_sha256"] != hashes():
        raise ValueError("A source changed after the strength-validation pilot was declared.")
    for name, field in (("source_records.json", "source_records_sha256"),
                        ("client_side/validation_plan.json", "private_validation_plan_sha256")):
        if file_sha256(study / name) != protocol[field]:
            raise ValueError("Pilot input fingerprint changed.")
    if file_sha256(PREVIOUS / "result_manifest.json") != protocol["parent_checkpoint_manifest_sha256"]:
        raise ValueError("Parent checkpoint manifest changed.")
    if file_sha256(PREVIOUS / "protocol.json") != protocol["parent_protocol_sha256"]:
        raise ValueError("Parent protocol changed.")
    for name, digest in protocol["parent_comparison_sha256"].items():
        if file_sha256(PREVIOUS / "analysis" / name) != digest:
            raise ValueError("Parent comparison changed.")
    for name in ("source_records.json", "client_side/validation_plan.json"):
        if file_sha256(study / name) != file_sha256(PREVIOUS / name):
            raise ValueError("Paired pilot inputs differ from the preceding study.")
    verify_previous(PREVIOUS)
    return protocol


def checked_result(result):
    if result["status"] != "complete" or result["job"] not in jobs() or result["source_sha256"] != hashes():
        raise ValueError("Invalid strength-validation result.")
    path = ROOT / result["run_path"]
    for name, digest in result["artifact_sha256"].items():
        if file_sha256(path / name) != digest:
            raise ValueError("Strength-validation artifact changed.")
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
        assert len(reports) == len(audit) == 400
        assert all(r["examples"] == 128 for r in reports+audit)
        assert all(r["exact_match"] for r in replay)
        report_count = sum(r["report_forward_examples"] for r in meta["local_validation_costs"])
        audit_count = sum(r["audit_forward_examples"] for r in meta["local_validation_costs"])
        assert report_count == audit_count == 39680
        rows.append({"job": result["job"], "exact_replayed_updates": len(replay),
                     "report_forward_examples": report_count, "audit_forward_examples": audit_count,
                     "exact_endpoint_rows": meta["exact_endpoint_rows"],
                     "naive_model_distribution_bytes": 280*meta["float32_model_payload_bytes"],
                     "peer_scalar_float64_payload_bytes": 270*8})
    assert len(rows) == 96 and sum(r["exact_replayed_updates"] for r in rows) == 960
    assert sum(r["exact_endpoint_rows"] for r in rows) == 19200
    output = {"cases": 96, "local_trainings": 960, "exact_prior_update_replays": 960,
              "trusted_server_images": 0, "official_test_used": False,
              "exact_endpoint_rows": sum(r["exact_endpoint_rows"] for r in rows),
              "report_forward_examples": sum(r["report_forward_examples"] for r in rows),
              "audit_forward_examples": sum(r["audit_forward_examples"] for r in rows),
              "global_development_forward_examples": 0, "global_models_advanced": 0, "cases_audit": rows,
              "communication_caveat": "Naive uncompressed float32 full, omitted and fractional model distribution and float64 scalar-only payload estimates; metadata, transport, security and optional class-statistic overhead omitted. Audit inference is research-only, not required by a deployed score rule."}
    write_json(study / "access_audit.json", output)
    return output


def assert_exact_replay(trained_hash, metrics, examples, expected):
    """Abort on any changed local update, optimizer schedule, loss or data count."""
    if (trained_hash != expected["submitted_state_sha256"]
            or examples != expected["examples"]
            or any(metrics[name] != expected[name]
                   for name in ("shuffle_seed", "optimizer_steps", "online_train_loss"))):
        raise AssertionError("Repeated local training changed its saved paired update.")


def execute_case(job, study, data, cfg, broadcast, source_record, selections, old_training, parent_case):
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
        submitted, training, replay_checks = {}, [], []
        for cid in range(10):
            view = client_view(data.pool, data.selection["clients"][str(cid)], data.manifest["normalization"],
                               cfg.development_seed, scenario, cid, job["round"])
            trained, metrics = train_client(model, view, cfg, cid, job["round"], broadcast)
            trained_hash = state_sha256(trained)
            submitted[cid] = trained
            training.append({"client_id": cid, "examples": len(view), **metrics,
                             "submitted_state_sha256": trained_hash})
            expected = old_training[(job["scenario"], cid)]
            assert_exact_replay(trained_hash, metrics, len(view), expected)
            replay_checks.append({"client_id": cid, "source_scenario": job["scenario"],
                                  "source_submitted_state_sha256": expected["submitted_state_sha256"],
                                  "exact_match": True})
        models = build_strength_models(broadcast, submitted)
        parent_path = ROOT / parent_case["run_path"]
        parent_meta = json.loads((parent_path / "metadata.json").read_text())
        full = models[1.][0]
        if state_sha256(full) != parent_meta["full_aggregate_state_sha256"]:
            raise AssertionError("Full aggregate changed from the parent.")
        if parent_meta["private_plan_sha256"] != file_sha256(study / "client_side/validation_plan.json"):
            raise AssertionError("Paired local views changed.")
        for row in training:
            cid = row["client_id"]
            row["without_client_state_sha256"] = state_sha256(models[0.][cid])
            row["quarter_strength_state_sha256"] = state_sha256(models[.25][cid])
            row["half_strength_state_sha256"] = state_sha256(models[.5][cid])
            if row["without_client_state_sha256"] != old_training[(job["scenario"],cid)]["without_client_state_sha256"]:
                raise AssertionError("Omitted-client endpoint changed.")
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
            value = evaluate_strength_models(models, report_view, audit_view, cfg, validator)
            report_rows.extend(value["report_rows"]); audit_rows.extend(value["audit_rows"]); costs.append(value["cost"])
        if len(report_rows) != 400 or len(audit_rows) != 400:
            raise AssertionError("Every candidate needs all ten local evaluators.")
        if state_sha256(clone_state(model)) != broadcast_hash:
            raise AssertionError("Local validation changed the common model.")
        endpoint_rows = verify_endpoint_rows(report_rows, json.loads((parent_path/"reports.json").read_text()))
        endpoint_rows += verify_endpoint_rows(audit_rows, json.loads((parent_path/"audit.json").read_text()))
        metadata = {"job": job, "scenario": scenario.to_dict(), "source_broadcast_sha256": broadcast_hash,
                    "source_record_job_id": source_record["job_id"], "source_run_path": source_record["run_path"],
                    "source_checkpoint_sha256": source_record["artifact_sha256"]["observation_checkpoints.pt"],
                    "config": vars(cfg), "marginal_rate": MARGINAL_RATE,
                    "full_aggregate_state_sha256": state_sha256(full),
                    "evaluation_semantics": SEMANTICS, "exact_endpoint_rows": endpoint_rows,
                    "parent_case_run_path": parent_case["run_path"],
                    "parent_case_artifact_sha256": parent_case["artifact_sha256"],
                    "fixed_client_weights": {str(cid): MARGINAL_RATE for cid in range(10)},
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
        previous_records = json.loads((PREVIOUS / "result_manifest.json").read_text())["records"]
        old_training, parent_cases = {}, {}
        for previous_record in previous_records:
            if all(previous_record["job"][k] == v for k, v in block.items()):
                checked_previous_result(previous_record)
                path = ROOT / previous_record["run_path"]
                scenario_name = previous_record["job"]["scenario"]
                parent_cases[scenario_name] = previous_record
                for row in json.loads((path / "training.json").read_text()):
                    key = (scenario_name, row["client_id"])
                    if key in old_training: raise ValueError("Duplicate parent training update.")
                    old_training[key] = row
        if len(old_training) != 60:
            raise ValueError("All sixty paired parent updates are required per checkpoint block.")
        for scenario in SCENARIOS:
            execute_case({**block, "scenario": scenario}, study, data, cfg, checkpoints[block["round"]],
                         record, selections, old_training, parent_cases[scenario])
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
