"""Client-owned holdouts and local paired candidate evaluation for the pilot.

The plan contains private example indices and belongs under client_side/ only.
The evaluator returns aggregate gains/counts; images, labels, indices and logits
remain local. This is a simulated interface boundary, not cryptographic privacy
or a deployment of distributed clients. Audit rows must never enter scoring.
"""
from pathlib import Path
import copy
import hashlib
import json
import math
import struct
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from common_broadcast_probe import SEEDS, _validate_baseline
from experiment_core import MnistCNN, file_sha256
from research_runs import PARTITIONS, canonical_hash
from scenario_utils import seed_for


ROLE_SEED = 20261030
ROLE_STREAM = "client_local_validation_roles_v1"
SAMPLE_STREAM = "client_local_validation_sample_v1"


def _index_set(values, size, description):
    if (any(type(value) is not int or not 0 <= value < size for value in values)
            or len(set(values)) != len(values)):
        raise ValueError(f"Invalid or duplicate official-training indices: {description}")
    return set(values)


def build_local_validation_plan(root, clean_records, report_size=128, audit_size=128):
    """Return a deterministic JSON-compatible private plan; do not save it here.

    plan['clients'][regime][str(seed)][str(client_id)] contains report_indices,
    audit_indices, their class_counts, eligible_count and role-bank counts.
    The globally disjoint role banks prevent report/audit overlap even across
    seeds and partition regimes. Labels count coverage AFTER index-only draws;
    they never stratify, balance, retry or otherwise influence selection.
    """
    if any(type(size) is not int or size <= 0 for size in (report_size, audit_size)):
        raise ValueError("Report and audit sizes must be positive integers.")
    root = Path(root).resolve()
    records = list(clean_records.values()) if isinstance(clean_records, dict) else list(clean_records)
    keyed = {}
    for record in records:
        job = record["job"]
        key = (job["regime"], job["seed"])
        if job["method"] != "FedAvg" or job["scenario"] != "clean" or key in keyed:
            raise ValueError("Provide only four distinct observed clean FedAvg records.")
        keyed[key] = record
    if set(keyed) != {(regime, seed) for regime in PARTITIONS for seed in SEEDS}:
        raise ValueError("Require both declared seeds and both partition regimes.")
    partitions, selections, provenance = {}, {}, []
    reference, development, training = set(), set(), set()
    label_digest = None
    for key, record in sorted(keyed.items()):
        regime, seed = key
        run, config, metadata, selection, _, _ = _validate_baseline(record)
        run = run.resolve()
        if not run.is_relative_to(root):
            raise ValueError("Validated baseline belongs to another experiment root.")
        if config.development_seed != seed or config.partition_id != PARTITIONS[regime]:
            raise ValueError("Baseline seed or partition differs from its record.")
        path = root / "data/partitions" / config.partition_id / "indices.json"
        if file_sha256(path) != metadata["partition_sha256"]:
            raise ValueError("Baseline partition fingerprint changed.")
        partition = json.loads(path.read_text())
        if (partition["official_train_size"] != 60000
                or any(partition["index_spaces"][name] != "official_train"
                       for name in ("clients", "server_reference", "development_validation"))
                or partition["index_spaces"]["reserved_final_test"] != "official_test"):
            raise ValueError("Holdouts must use the saved official-training index space only.")
        indices = partition["indices"]
        if set(indices["clients"]) != {str(cid) for cid in range(10)}:
            raise ValueError("Require the ten saved client pools.")
        pools = {cid: _index_set(ix, 60000, f"client {cid}") for cid, ix in indices["clients"].items()}
        reserved_reference = _index_set(indices["server_reference"], 60000, "reference")
        reserved_development = _index_set(indices["development_validation"], 60000, "development")
        parts = [reserved_reference, reserved_development, *pools.values()]
        if sum(map(len, parts)) != 60000 or set().union(*parts) != set(range(60000)):
            raise ValueError("Saved client/reference/development pools must be disjoint and cover the training split.")
        if canonical_hash(selection) != metadata["selection_sha256"]:
            raise ValueError("Selection differs from the checkpoint's recorded selection.")
        if set(selection["clients"]) != set(pools):
            raise ValueError("Baseline training selection lacks a client.")
        for cid, values in selection["clients"].items():
            chosen = _index_set(values, 60000, f"training selection {cid}")
            if len(chosen) != 512 or not chosen.issubset(pools[cid]):
                raise ValueError("Baseline training indices must be 512 examples in their own client pool.")
            training.update(chosen)
        reference.update(reserved_reference)
        development.update(reserved_development)
        digest = partition["dataset_files"]["train-labels-idx1-ubyte"]["raw_sha256"]
        if label_digest is not None and label_digest != digest:
            raise ValueError("Partitions refer to different training-label files.")
        label_digest = digest
        partitions[regime] = partition
        selections[key] = selection
        provenance.append({"regime": regime, "seed": seed, "run_path": str(run.relative_to(root)),
                           "partition_sha256": metadata["partition_sha256"],
                           "selection_sha256": metadata["selection_sha256"],
                           "artifact_sha256": copy.deepcopy(record["artifact_sha256"])})
    if training & (reference | development) or reference & development:
        raise ValueError("Training and reserved pools overlap across the partition regimes.")
    label_path = root / "data/MNIST/raw/train-labels-idx1-ubyte"
    if file_sha256(label_path) != label_digest:
        raise ValueError("Official-training label fingerprint changed.")
    raw = label_path.read_bytes()
    if len(raw) < 8 or struct.unpack(">II", raw[:8]) != (2049, 60000):
        raise ValueError("Unexpected official-training label file.")
    labels = np.frombuffer(raw[8:], dtype=np.uint8)
    if len(labels) != 60000 or int(labels.max()) > 9:
        raise ValueError("Expected 60000 ten-class training labels.")
    excluded = training | reference | development
    eligible = sorted(set(range(60000)) - excluded)
    shuffled = np.random.default_rng(seed_for(ROLE_SEED, ROLE_STREAM)).permutation(
        np.asarray(eligible, dtype=np.int64))
    midpoint = len(shuffled) // 2
    banks = {"report": set(map(int, shuffled[:midpoint])), "audit": set(map(int, shuffled[midpoint:]))}
    clients = {}
    for regime in sorted(PARTITIONS):
        clients[regime] = {}
        for seed in SEEDS:
            clients[regime][str(seed)] = {}
            for cid in range(10):
                pool = set(partitions[regime]["indices"]["clients"][str(cid)])
                residual = sorted(pool - excluded)
                split = {"eligible_count": len(residual),
                         "eligible_class_counts": np.bincount(labels[residual], minlength=10).tolist()}
                for role, count in (("report", report_size), ("audit", audit_size)):
                    available = sorted(pool & banks[role])
                    if len(available) < count:
                        raise ValueError(f"Insufficient private {role} examples for {regime} client {cid}.")
                    rng = np.random.default_rng(seed_for(seed, SAMPLE_STREAM, regime, cid, role))
                    selected = sorted(map(int, rng.permutation(np.asarray(available, dtype=np.int64))[:count]))
                    split[f"{role}_indices"] = selected
                    split[f"{role}_class_counts"] = np.bincount(labels[selected], minlength=10).tolist()
                    split[f"{role}_bank_count"] = len(available)
                clients[regime][str(seed)][str(cid)] = split
    return {
        "schema_version": 1, "index_space": "official_train", "storage_boundary": "client_side_only",
        "privacy_scope": "Simulated client-owned data boundary; not secure distributed deployment.",
        "parameters": {"report_size": report_size, "audit_size": audit_size, "role_seed": ROLE_SEED,
                       "role_namespace": ROLE_STREAM, "sample_namespace": SAMPLE_STREAM,
                       "sampling": "unstratified_without_replacement; return_sorted_indices",
                       "shared_across": "methods, scenarios and rounds within each seed/regime/client"},
        "exclusions": {"training_indices": sorted(training), "reference_indices": sorted(reference),
                       "development_indices": sorted(development), "training_union_size": len(training),
                       "reference_union_size": len(reference), "development_union_size": len(development)},
        "role_banks": {"report_indices": sorted(banks["report"]), "audit_indices": sorted(banks["audit"]),
                       "report_count": len(banks["report"]), "audit_count": len(banks["audit"])},
        "eligible_total": len(eligible), "clients": clients, "provenance": provenance,
        "training_labels_sha256": label_digest, "implementation_sha256": file_sha256(__file__),
        "official_test_loaded": False,
    }


def _validate_views(report_view, audit_view):
    selected = []
    for view in (report_view, audit_view):
        if not hasattr(view, "indices") or len(view) == 0:
            raise ValueError("Local validation views must expose nonempty private training indices.")
        indices = _index_set(list(view.indices), 60000, "local validation view")
        if len(indices) != len(view) or bool(getattr(view, "flip_labels", False)):
            raise ValueError("Local views must have distinct indices and correct, unflipped labels.")
        if getattr(getattr(view, "pool", None), "train", True) is not True:
            raise ValueError("Official-test views are forbidden.")
        selected.append(indices)
    if selected[0] & selected[1]:
        raise ValueError("Reporting and independent audit views must be disjoint.")


def _validate_models(broadcast, candidate_states, validator_id):
    if type(validator_id) is not int or validator_id not in range(10):
        raise ValueError("Validator ID must be one of the ten clients.")
    if (set(candidate_states) != set(range(10))
            or any(type(cid) is not int for cid in candidate_states) or not broadcast):
        raise ValueError("Supply all ten candidate models and a nonempty broadcast.")
    for state in (broadcast, *candidate_states.values()):
        if set(state) != set(broadcast):
            raise ValueError("Candidate model keys differ from the broadcast.")
        for name, value in state.items():
            reference = broadcast[name]
            if (not isinstance(value, torch.Tensor) or value.device.type != "cpu"
                    or not value.is_floating_point() or value.numel() == 0
                    or value.dtype != reference.dtype or value.shape != reference.shape):
                raise ValueError("Expected compatible nonempty floating-point CPU model states.")
            if not torch.isfinite(value).all():
                raise FloatingPointError("Non-finite candidate or broadcast model parameter.")


def _loss_vectors(logits, targets):
    if logits.ndim != 2 or logits.shape != (len(targets), 10) or not torch.isfinite(logits).all():
        raise ValueError("Expected finite ten-class logits for every local example.")
    if targets.ndim != 1 or targets.dtype != torch.int64 or not ((targets >= 0) & (targets < 10)).all():
        raise ValueError("Expected correct integer targets in the ten-class label space.")
    log_probabilities = logits.double().log_softmax(dim=1)
    probabilities = log_probabilities.exp()
    one_hot = torch.nn.functional.one_hot(targets, num_classes=10).double()
    return {"brier": .5 * (probabilities - one_hot).square().sum(dim=1),
            "cross_entropy": -log_probabilities.gather(1, targets[:, None]).squeeze(1),
            "correct": (logits.argmax(dim=1) == targets).double()}


def _gain_summary(baseline, candidate, targets):
    counts = torch.bincount(targets, minlength=10)
    brier_gain = baseline["brier"] - candidate["brier"]
    ce_gain = baseline["cross_entropy"] - candidate["cross_entropy"]
    return {
        "examples": len(targets), "class_counts": counts.tolist(),
        "mean_brier_gain": float(brier_gain.mean()), "mean_cross_entropy_gain": float(ce_gain.mean()),
        "accuracy_gain": float((candidate["correct"] - baseline["correct"]).mean()),
        "baseline_mean_brier": float(baseline["brier"].mean()),
        "candidate_mean_brier": float(candidate["brier"].mean()),
        "baseline_mean_cross_entropy": float(baseline["cross_entropy"].mean()),
        "candidate_mean_cross_entropy": float(candidate["cross_entropy"].mean()),
        "baseline_accuracy": float(baseline["correct"].mean()),
        "candidate_accuracy": float(candidate["correct"].mean()),
        "class_brier_gains": [float(brier_gain[targets == digit].mean()) if counts[digit] else None for digit in range(10)],
        "class_cross_entropy_gains": [float(ce_gain[targets == digit].mean()) if counts[digit] else None for digit in range(10)],
    }


def evaluate_local_candidates(broadcast, candidate_states, report_view, audit_view, config, validator_id):
    """Evaluate broadcast and ten supplied marginal models on two private views.

    Correct labels stay in both views. Flipped report summaries use 9-y locally
    from the SAME logits, with no extra model forward pass. Models and dataset
    batches are never returned. Losses use 0.5*sum((p-onehot(y))**2), in [0,1].
    Positive paired gains mean improvement relative to the broadcast. Audit
    summaries are returned separately for offline evaluation, never scoring.
    """
    _validate_models(broadcast, candidate_states, validator_id)
    _validate_views(report_view, audit_view)
    if type(config.evaluation_batch_size) is not int or config.evaluation_batch_size < 1:
        raise ValueError("Evaluation batch size must be a positive integer.")
    batch_size = min(128, config.evaluation_batch_size)
    started = time.perf_counter()
    outputs, batches_used = {}, 0
    # Model construction initializes parameters. Restore the caller's CPU RNG
    # afterward, including on error; loaders use their own private generators.
    with torch.random.fork_rng(devices=[]), torch.inference_mode():
        model = MnistCNN()
        model.eval()
        for role, view in (("report", report_view), ("audit", audit_view)):
            loader = DataLoader(view, batch_size=batch_size, shuffle=False, num_workers=0,
                                generator=torch.Generator().manual_seed(0))
            # Cache locally so every candidate sees precisely the same examples.
            batches = [(images.detach().clone(), targets.detach().clone()) for images, targets in loader]
            if any(images.device.type != "cpu" or targets.device.type != "cpu" for images, targets in batches):
                raise ValueError("Local validation batches must remain on the CPU.")
            targets = torch.cat([labels for _, labels in batches])
            if len(targets) != len(view):
                raise ValueError("Local validation loader did not cover its private view exactly.")

            def forward(state):
                nonlocal batches_used
                model.load_state_dict(state, strict=True)
                logits = torch.cat([model(images) for images, _ in batches])
                batches_used += len(batches)
                return logits

            baseline_logits = forward(broadcast)
            baseline = _loss_vectors(baseline_logits, targets)
            flipped_targets = 9 - targets
            flipped_baseline = _loss_vectors(baseline_logits, flipped_targets) if role == "report" else None
            rows = []
            for cid in sorted(candidate_states):
                logits = forward(candidate_states[cid])
                candidate = _loss_vectors(logits, targets)
                row = {"candidate_id": cid, "validator_id": validator_id,
                       **_gain_summary(baseline, candidate, targets)}
                if role == "report":
                    flipped = _gain_summary(flipped_baseline, _loss_vectors(logits, flipped_targets), flipped_targets)
                    row.update({f"flipped_{key}": value for key, value in flipped.items() if key != "examples"})
                rows.append(row)
            outputs[f"{role}_rows"] = rows
    examples = len(report_view) + len(audit_view)
    outputs["cost"] = {"validator_id": validator_id, "report_examples": len(report_view),
                       "audit_examples": len(audit_view), "models_per_role": 11,
                       "report_models_evaluated": 11, "audit_models_evaluated": 11,
                       "report_forward_examples": 11 * len(report_view),
                       "audit_forward_examples": 11 * len(audit_view),
                       "broadcast_forward_examples": examples, "candidate_forward_examples": 10 * examples,
                       "total_forward_examples": 11 * examples, "forward_batches": batches_used,
                       "evaluation_batch_size": batch_size, "optimizer_steps": 0,
                       "additional_training_examples": 0, "elapsed_seconds": time.perf_counter() - started}
    return outputs
