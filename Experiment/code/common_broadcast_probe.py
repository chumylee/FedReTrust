"""Offline instantaneous update probes from identical clean FedAvg broadcasts.

Call run_probe only after the parent study verifies its frozen pilot protocol.
Each of four checkpoint broadcasts is replayed under four prescribed conditions;
no resulting aggregate is installed in a model or used for a future round.
The data loader opens only MNIST's training split. There is no evaluation call.
"""
from pathlib import Path
import csv
import fcntl
import hashlib
import json
import os
import random
import tempfile
import time

import numpy as np
import torch

from baseline_calibration import ServerLearningRate
from experiment_core import (ClientSubmission, FedAvgAggregator, MnistCNN, TrainConfig,
                             clone_state, file_sha256, load_data, state_sha256, train_client, write_json)
from research_runs import PARTITIONS, ROOT, canonical_hash
from scenario_metrics import longer_scenarios
from scenario_utils import client_view, truth_row
from update_observer import LAYER_NAMES, UpdateObserver


CHECKPOINT_ROUNDS = (5, 8, 12, 20)
CONDITIONS = ("clean", "abrupt_shot_noise", "gradual_stripe", "label_flip")
SEEDS = (20261014, 20261015)
SOURCE_NAMES = ("common_broadcast_probe.py", "update_observer.py", "experiment_core.py",
                "scenario_utils.py", "scenario_metrics.py", "rfa.py", "research_runs.py",
                "baseline_calibration.py")


def _source_hashes():
    return {name: file_sha256(Path(__file__).parent / name) for name in SOURCE_NAMES}


def _read_json(path):
    return json.loads(Path(path).read_text())


def _write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _flat_features(rows):
    output = []
    for row in rows:
        features = row["features"]
        flat = {key: value for key, value in row.items() if key != "features"}
        flat.update({key: value for key, value in features.items() if not isinstance(value, list)})
        flat.update({f"layer_cosine_{name}": value for name, value in zip(LAYER_NAMES, features["layer_cosines"])})
        flat.update({f"head_class_cosine_{digit}": value for digit, value in enumerate(features["head_class_cosines"])})
        output.append(flat)
    return output


def _atomic_json(path, value):
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def _select_records(records):
    selected = {}
    for record in records.values() if isinstance(records, dict) else records:
        job = record["job"]
        if job["method"] == "FedAvg" and job["scenario"] == "clean":
            pair = (job["regime"], job["seed"])
            if pair in selected:
                raise ValueError("Duplicate clean FedAvg record for a seed/regime pair.")
            selected[pair] = record
    expected = {(regime, seed) for regime in PARTITIONS for seed in SEEDS}
    if set(selected) != expected:
        raise ValueError("Probe requires exactly the four declared clean FedAvg seed/regime pairs.")
    return [selected[pair] for pair in sorted(expected)]


def _validate_baseline(record):
    """Validate saved source/artifact provenance before loading or training."""
    job = record["job"]
    if (record["status"] != "complete" or job["method"] != "FedAvg" or job["scenario"] != "clean"
            or job.get("phase", "development") != "development"):
        raise ValueError("A completed development clean FedAvg run is required.")
    run = (ROOT / record["run_path"]).resolve()
    if not run.is_relative_to(ROOT.resolve()):
        raise ValueError("Baseline run must be inside the experiment workspace.")
    required = {"run.json", "selection.json", "status.json", "client_metrics.csv", "observation_checkpoints.pt"}
    if not required.issubset(record["artifact_sha256"]):
        raise ValueError("Missing baseline or observation-checkpoint artifact fingerprints.")
    for name, digest in record["artifact_sha256"].items():
        path = (run / name).resolve()
        if not path.is_relative_to(run) or file_sha256(path) != digest:
            raise ValueError(f"Baseline artifact fingerprint changed: {name}")
    if not record.get("source_sha256"):
        raise ValueError("Missing baseline source fingerprints.")
    for name, digest in record["source_sha256"].items():
        path = (ROOT / "code" / name).resolve()
        if not path.is_relative_to((ROOT / "code").resolve()) or file_sha256(path) != digest:
            raise ValueError(f"Baseline source fingerprint changed: {name}")
    metadata, status = _read_json(run / "run.json"), _read_json(run / "status.json")
    if (metadata["official_test_used"] or metadata.get("evaluation_data", {}).get("split") == "official_test"
            or status["status"] != "complete"
            or metadata["scenario"]["name"] != "clean"):
        raise ValueError("Invalid clean development baseline metadata.")
    # The original study's FedAvg factory preserves its model exactly through
    # this rate-one adapter, whose recorded name is FedAvg_server_lr_1.
    expected_adapter = ServerLearningRate(FedAvgAggregator(), 1.0)
    method = metadata["method"]
    if (method["name"] != expected_adapter.name
            or any(method["settings"].get(name) != value for name, value in expected_adapter.settings.items())):
        raise ValueError("Probe requires the original verified rate-one FedAvg adapter.")
    cfg = TrainConfig(**metadata["config"])
    if (cfg.partition_id != PARTITIONS[job["regime"]] or cfg.development_seed != job["seed"]
            or cfg.rounds != 24 or cfg.max_train_per_client != 512 or cfg.num_threads != 4
            or cfg.local_epochs != 1 or cfg.batch_size != 64 or cfg.learning_rate != .01 or cfg.momentum != .9):
        raise ValueError("Baseline differs from the declared pilot training configuration.")
    for name, digest in metadata["code_sha256"].items():
        if file_sha256(ROOT / "code" / name) != digest:
            raise ValueError("Baseline training implementation changed.")
    observer_settings = metadata["method"]["settings"].get("update_observation", {})
    if observer_settings.get("implementation_sha256") != file_sha256(Path(__file__).parent / "update_observer.py"):
        raise ValueError("Baseline observer provenance is missing or differs.")
    selection = _read_json(run / "selection.json")
    if canonical_hash(selection) != metadata["selection_sha256"]:
        raise ValueError("Saved baseline selection fingerprint differs.")
    checkpoints = torch.load(run / "observation_checkpoints.pt", weights_only=True, map_location="cpu")
    if set(checkpoints) != set(CHECKPOINT_ROUNDS):
        raise ValueError("Checkpoint rounds differ from the declared probe.")
    with (run / "client_metrics.csv").open(newline="") as handle:
        source_rows = [row for row in csv.DictReader(handle) if int(row["round"]) in CHECKPOINT_ROUNDS]
    rows_by_round = {}
    for round_number in CHECKPOINT_ROUNDS:
        rows = [row for row in source_rows if int(row["round"]) == round_number]
        if len(rows) != 10 or {int(row["client"]) for row in rows} != set(range(10)):
            raise ValueError("Baseline must contain ten unique submissions for every checkpoint.")
        digest = state_sha256(checkpoints[round_number])
        if {row["broadcast_state_sha256"] for row in rows} != {digest}:
            raise ValueError("Saved observation checkpoint is not the baseline's pre-update broadcast.")
        rows_by_round[round_number] = {int(row["client"]): row for row in rows}
    return run, cfg, metadata, selection, checkpoints, rows_by_round


def probe_case(data, config, broadcast, scenario, round_number):
    """Train one condition at a fixed broadcast; return features and separate truth.

    The observer is newly constructed for EVERY call. Its temporal scores must
    remain undefined because these probes do not form a client trajectory.
    """
    random.seed(config.development_seed)
    np.random.seed(config.development_seed)
    torch.manual_seed(config.development_seed)
    torch.set_num_threads(config.num_threads)
    torch.use_deterministic_algorithms(True)
    model = MnistCNN()
    model.load_state_dict(broadcast, strict=True)
    broadcast_hash = state_sha256(broadcast)
    client_ids = sorted(int(key) for key in data.selection["clients"])
    if client_ids != list(range(10)):
        raise ValueError("Each probe condition requires all ten clients.")
    scenario.validate(client_ids, config.rounds)
    submissions, training_rows, offline_truth = [], [], []
    for cid in client_ids:
        view = client_view(data.pool, data.selection["clients"][str(cid)], data.manifest["normalization"],
                           config.development_seed, scenario, cid, round_number)
        local_state, metrics = train_client(model, view, config, cid, round_number, broadcast)
        submissions.append(ClientSubmission(cid, len(view), local_state))
        training_rows.append({"client": cid, "examples": len(view), **metrics,
                              "submitted_state_sha256": state_sha256(local_state)})
        truth = truth_row(scenario, cid, round_number, view)
        honest_change = any(event.kind == "input_change" and event.intent == "honest"
                            for event in scenario.active(cid, round_number))
        offline_truth.append({**truth, "is_attack": bool(truth["attack_active"]),
                              "is_honest_change": honest_change,
                              "group": "attack" if truth["attack_active"] else "honest_change" if honest_change else "clean"})
    if state_sha256(clone_state(model)) != broadcast_hash or state_sha256(broadcast) != broadcast_hash:
        raise AssertionError("Probe local training modified its common broadcast.")
    observer = UpdateObserver(FedAvgAggregator(), checkpoint_rounds=())
    result = observer.aggregate(submissions, broadcast, round_number)
    observation = result.diagnostics["update_observation"]
    for features in observation["feature_rows"].values():
        if features["temporal_cosine"] is not None or features["temporal_relative_change"] is not None:
            raise AssertionError("Independent common-broadcast probes must not acquire temporal history.")
    # The returned FedAvg state is deliberately discarded; no load_state_dict follows.
    return observation, training_rows, offline_truth


def _verify_block(directory, provenance):
    manifest = _read_json(directory / "result.json")
    if manifest["status"] != "complete" or manifest["provenance"] != provenance:
        raise ValueError("Saved common-broadcast block provenance differs; refusing reuse.")
    for name, digest in manifest["artifact_sha256"].items():
        if file_sha256(directory / name) != digest:
            raise ValueError("Saved common-broadcast block artifact changed.")
    return manifest


def run_probe(records, output_dir):
    """Run or verify 16 atomic blocks, bounded to 640 new local trainings.

    Input records are the parent study's verified job results. Completed blocks
    resume only when baseline artifacts, sources, selection and broadcast agree.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = _select_records(records)
    sources = _source_hashes()
    scenarios = longer_scenarios()
    blocks = []
    with (output_dir / "probe.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for record in selected:
            run, cfg, metadata, selection, checkpoints, baseline_rows = _validate_baseline(record)
            data = load_data(ROOT, cfg, task_names=("clean",))
            if (data.selection != selection or canonical_hash(data.selection) != metadata["selection_sha256"]
                    or data.manifest_sha256 != metadata["partition_sha256"]):
                raise ValueError("Reconstructed data selection or partition differs from the baseline.")
            if any(len(indices) != 512 for indices in data.selection["clients"].values()):
                raise ValueError("Declared probes require exactly 512 training examples per client.")
            job = record["job"]
            for round_number in CHECKPOINT_ROUNDS:
                broadcast = checkpoints[round_number]
                broadcast_hash = state_sha256(broadcast)
                identity = {"regime": job["regime"], "seed": cfg.development_seed, "round": round_number,
                            "broadcast_state_sha256": broadcast_hash}
                provenance = {**identity, "baseline_run_path": str(run.relative_to(ROOT)),
                              "baseline_artifact_sha256": record["artifact_sha256"],
                              "selection_sha256": metadata["selection_sha256"],
                              "partition_sha256": metadata["partition_sha256"],
                              "config": metadata["config"], "source_sha256": sources,
                              "conditions": list(CONDITIONS),
                              "scenario_definitions_sha256": canonical_hash({name: scenarios[name].to_dict() for name in CONDITIONS})}
                directory = output_dir / f"{job['regime']}_seed_{cfg.development_seed}_round_{round_number:02d}"
                if directory.exists():
                    manifest = _verify_block(directory, provenance)
                else:
                    temporary = Path(tempfile.mkdtemp(prefix=f".{directory.name}.", dir=output_dir))
                    started = time.perf_counter()
                    feature_rows, truth_rows, training_rows, observations = [], [], [], []
                    try:
                        for name in CONDITIONS:
                            observation, trained, truth = probe_case(data, cfg, broadcast, scenarios[name], round_number)
                            if name == "clean":
                                for row in trained:
                                    original = baseline_rows[round_number][row["client"]]
                                    if (row["submitted_state_sha256"] != original["submitted_state_sha256"]
                                            or row["shuffle_seed"] != int(original["shuffle_seed"])):
                                        raise AssertionError("Clean probe did not exactly replay the original baseline local update.")
                            case = {**identity, "scenario": name}
                            feature_rows.extend({**case, "client": int(cid), "features": values}
                                                for cid, values in observation["feature_rows"].items())
                            truth_rows.extend({**case, **row} for row in truth)
                            training_rows.extend({**case, **row} for row in trained)
                            observations.append({**case, **observation})
                        write_json(temporary / "features.json", feature_rows)
                        _write_csv(temporary / "features.csv", _flat_features(feature_rows))
                        write_json(temporary / "offline_truth.json", truth_rows)
                        _write_csv(temporary / "offline_truth.csv", truth_rows)
                        write_json(temporary / "training.json", training_rows)
                        write_json(temporary / "diagnostics.json", observations)
                        names = ("features.json", "features.csv", "offline_truth.json", "offline_truth.csv", "training.json", "diagnostics.json")
                        manifest = {"status": "complete", "provenance": provenance,
                                    "artifact_sha256": {name: file_sha256(temporary / name) for name in names},
                                    "local_trainings": 40, "local_training_examples": 40 * 512,
                                    "reference_forward_examples": 0, "evaluation_forward_examples": 0,
                                    "official_test_used": False, "temporal_features_usable": False,
                                    "clean_replay_identical": True,
                                    "elapsed_seconds": time.perf_counter() - started}
                        write_json(temporary / "result.json", manifest)
                        os.replace(temporary, directory)
                    except Exception as exc:
                        write_json(temporary / "failure.json", {"error": repr(exc), "provenance": provenance})
                        raise
                blocks.append({"directory": directory.name, "result_sha256": file_sha256(directory / "result.json"),
                               "provenance": provenance, "elapsed_seconds": manifest["elapsed_seconds"]})
                _atomic_json(output_dir / "progress.json", {"status": "running", "completed_blocks": len(blocks),
                                                            "total_blocks": 16, "last_block": directory.name})
                print(json.dumps({"common_broadcast_probe": directory.name, "completed_blocks": len(blocks), "total_blocks": 16}), flush=True)
        features, truth = [], []
        for block in blocks:
            features.extend(_read_json(output_dir / block["directory"] / "features.json"))
            truth.extend(_read_json(output_dir / block["directory"] / "offline_truth.json"))
        _atomic_json(output_dir / "features.json", features)
        _atomic_json(output_dir / "offline_truth.json", truth)
        _write_csv(output_dir / "features.csv", _flat_features(features))
        _write_csv(output_dir / "offline_truth.csv", truth)
        summary = {"status": "complete", "blocks": blocks, "local_trainings": 640,
                   "local_training_examples": 640 * 512, "feature_rows": len(features),
                   "official_test_used": False, "reference_forward_examples": 0,
                   "evaluation_forward_examples": 0, "temporal_features_usable": False,
                   "interpretation": "Instantaneous observational features at common clean broadcasts; no model trajectory or accuracy result.",
                   "artifact_sha256": {name: file_sha256(output_dir / name)
                                       for name in ("features.json", "features.csv", "offline_truth.json", "offline_truth.csv")}}
        _atomic_json(output_dir / "summary.json", summary)
        _atomic_json(output_dir / "progress.json", {"status": "complete", "completed_blocks": 16, "total_blocks": 16})
    return summary
