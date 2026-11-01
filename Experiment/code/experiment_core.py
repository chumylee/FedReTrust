"""Shared CPU training and recording for the executable experiment notebooks.

Scenario truth belongs to the simulator and saved evaluation files. Aggregators
receive only client IDs, sample counts, submitted model states and the broadcast.
"""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
import copy
import hashlib
import json
import math
import platform
import random
import time
import traceback

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets

from scenario_utils import ScenarioDataset, client_view, gaussian_update_corruption, seed_for, truth_row


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


@dataclass(frozen=True)
class TrainConfig:
    partition_id: str = "mnist_iid_ee88d9959326"
    development_seed: int = 20261014
    rounds: int = 6
    local_epochs: int = 1
    batch_size: int = 64
    evaluation_batch_size: int = 256
    learning_rate: float = 0.01
    momentum: float = 0.9
    num_threads: int = 4
    max_train_per_client: int | None = 512
    max_validation: int | None = 1000

    def __post_init__(self):
        for name in ("rounds", "local_epochs", "batch_size", "evaluation_batch_size", "num_threads"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer.")
        for name in ("max_train_per_client", "max_validation"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 1):
                raise ValueError(f"{name} must be a positive integer or None.")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0 or not 0 <= self.momentum < 1:
            raise ValueError("Invalid SGD settings.")
        if self.max_validation is not None and self.max_validation % 10:
            raise ValueError("Validation limit must allow equal counts for all ten digits.")


class MnistCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(64 * 7 * 7, 128), nn.ReLU(), nn.Linear(128, num_classes),
        )

    def forward(self, images):
        return self.classifier(self.features(images))


def clone_state(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def state_sha256(state):
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(f"{name}:{value.dtype}:{tuple(value.shape)}".encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def state_distance(left, right):
    return math.sqrt(sum((left[name].double() - right[name].double()).square().sum().item() for name in left))


@dataclass
class ClientSubmission:
    client_id: int
    num_examples: int
    state: dict[str, torch.Tensor]


@dataclass
class AggregationResult:
    state: dict[str, torch.Tensor]
    coefficients: dict[int, float]
    diagnostics: dict


def fedavg(client_states, sample_counts):
    if not client_states or len(client_states) != len(sample_counts):
        raise ValueError("Provide one positive sample count for each client state.")
    if any(type(count) is not int or count <= 0 for count in sample_counts):
        raise ValueError("Sample counts must be positive integers.")
    reference = client_states[0]
    for state in client_states:
        if set(state) != set(reference):
            raise ValueError("Client model parameters differ.")
        for name, tensor in state.items():
            if tensor.shape != reference[name].shape or tensor.dtype != reference[name].dtype:
                raise ValueError(f"Incompatible tensor: {name}")
            if tensor.device.type != "cpu" or not tensor.is_floating_point():
                raise ValueError("This CNN aggregator expects floating-point CPU states.")
            if not torch.isfinite(tensor).all():
                raise FloatingPointError(f"Non-finite client parameter: {name}")
    total = sum(sample_counts)
    averaged = {}
    for name, tensor in reference.items():
        accumulator = torch.zeros_like(tensor, dtype=torch.float64)
        for state, count in zip(client_states, sample_counts):
            accumulator.add_(state[name].double(), alpha=count / total)
        averaged[name] = accumulator.to(dtype=tensor.dtype)
    return averaged


class FedAvgAggregator:
    name = "FedAvg"
    settings = {"weighting": "sample_count", "accumulation_dtype": "float64"}

    def aggregate(self, submissions, global_state, round_number):
        ids = [item.client_id for item in submissions]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate client submission.")
        counts = [item.num_examples for item in submissions]
        state = fedavg([item.state for item in submissions], counts)
        return AggregationResult(state, {i: n / sum(counts) for i, n in zip(ids, counts)}, {})


@dataclass
class ExperimentData:
    root: Path
    pool: object
    manifest: dict
    manifest_sha256: str
    selection: dict
    validation_tasks: dict
    evaluation_metadata: dict | None = None


def load_data(root, config, task_names=("clean", "shot_noise_s3", "stripe")):
    root = Path(root).resolve()
    path = root / "data" / "partitions" / config.partition_id / "indices.json"
    manifest = json.loads(path.read_text())
    if manifest["schema_version"] != 1 or manifest["partition_id"] != config.partition_id:
        raise ValueError("Unexpected saved partition.")
    for group in ("clients", "development_validation", "server_reference"):
        if manifest["index_spaces"][group] != "official_train":
            raise ValueError("Development data must come from the official training split.")
    for filename in ("train-images-idx3-ubyte", "train-labels-idx1-ubyte"):
        if file_sha256(root / "data" / "MNIST" / "raw" / filename) != manifest["dataset_files"][filename]["raw_sha256"]:
            raise ValueError(f"Training-file fingerprint changed: {filename}")
    pool = datasets.MNIST(root=root / "data", train=True, download=False, transform=None)
    indices = manifest["indices"]
    if sorted(int(k) for k in indices["clients"]) != list(range(10)):
        raise ValueError("This development experiment needs the ten saved clients.")
    flat = np.concatenate([indices["server_reference"], indices["development_validation"], *indices["clients"].values()])
    if not np.array_equal(np.sort(flat), np.arange(len(pool))):
        raise ValueError("Saved assignments must be disjoint and cover the training pool.")
    clients = {}
    for key, original in indices["clients"].items():
        count = config.max_train_per_client
        if count is not None and count > len(original):
            raise ValueError("Client subset exceeds its saved partition.")
        positions = np.arange(len(original)) if count is None else np.sort(
            np.random.default_rng(seed_for(config.development_seed, "subset", key)).choice(len(original), count, replace=False))
        clients[key] = [original[int(i)] for i in positions]
    validation = list(indices["development_validation"])
    if config.max_validation is not None:
        selected = []
        for digit in range(10):
            candidates = [i for i in validation if int(pool.targets[i]) == digit]
            per_class = config.max_validation // 10
            if per_class > len(candidates):
                raise ValueError("Validation subset exceeds a saved digit group.")
            selected.extend(int(i) for i in np.random.default_rng(
                seed_for(config.development_seed, "validation_subset", digit)).choice(candidates, per_class, replace=False))
        selected_set = set(selected)
        validation = [i for i in validation if i in selected_set]
    selection = {"clients": clients, "development_validation": validation,
                 "index_space": "official_train", "selection_seed": config.development_seed}
    normalization = manifest["normalization"]
    tasks = {}
    for name in task_names:
        if name not in {"clean", "shot_noise_s3", "stripe"}:
            raise ValueError(f"Unknown validation task: {name}")
        corruption = None if name == "clean" else "shot_noise" if name == "shot_noise_s3" else "stripe"
        tasks[name] = ScenarioDataset(pool, validation, normalization["mean"][0], normalization["std"][0],
                                      config.development_seed, corruption=corruption, fraction=float(corruption is not None))
    if "clean" not in tasks:
        raise ValueError("Include the clean development task.")
    return ExperimentData(root, pool, manifest, file_sha256(path), selection, tasks)


def train_client(global_model, view, config, client_id, round_number, broadcast_state):
    local_model = copy.deepcopy(global_model)
    assert all(torch.equal(value, broadcast_state[name]) for name, value in local_model.state_dict().items())
    optimizer = torch.optim.SGD(local_model.parameters(), lr=config.learning_rate, momentum=config.momentum)
    assert len(optimizer.state) == 0
    shuffle_seed = (config.development_seed * 1_000_003 + round_number * 1009 + client_id) % (2**63 - 1)
    loader = DataLoader(view, batch_size=config.batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(shuffle_seed), num_workers=0)
    criterion = nn.CrossEntropyLoss()
    local_model.train()
    losses = []
    for _ in range(config.local_epochs):
        total, count = 0.0, 0
        for images, targets in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(local_model(images), targets)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss.")
            loss.backward()
            optimizer.step()
            total += loss.item() * targets.numel()
            count += targets.numel()
        losses.append(total / count)
    state = clone_state(local_model)
    if not all(torch.isfinite(value).all() for value in state.values()):
        raise FloatingPointError("Non-finite local state.")
    return state, {"online_train_loss": float(np.mean(losses)), "shuffle_seed": shuffle_seed,
                   "optimizer_steps": len(loader) * config.local_epochs}


@torch.inference_mode()
def evaluate(model, view, config):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    loader = DataLoader(view, batch_size=config.evaluation_batch_size, shuffle=False, num_workers=0)
    loss_sum, correct, count = 0.0, 0, 0
    for images, targets in loader:
        scores = model(images)
        loss = criterion(scores, targets)
        if not torch.isfinite(scores).all() or not torch.isfinite(loss):
            raise FloatingPointError("Non-finite evaluation output.")
        loss_sum += loss.item() * targets.numel()
        correct += int((scores.argmax(1) == targets).sum())
        count += targets.numel()
    return {"loss": loss_sum / count, "accuracy": correct / count, "examples": count}


@torch.inference_mode()
def checkpoint_matches(model, checkpoint_path, view, config):
    restored = MnistCNN()
    restored.load_state_dict(torch.load(checkpoint_path, weights_only=True, map_location="cpu"))
    model.eval()
    restored.eval()
    for images, _ in DataLoader(view, batch_size=config.evaluation_batch_size, shuffle=False, num_workers=0):
        if not torch.equal(model(images), restored(images)):
            return False
    return True


def run_experiment(data, config, scenario, aggregator=None, notebook_code_sha256=None, verbose=True, evaluator=None):
    """One fresh deterministic run, preserving every completed or failed attempt."""
    aggregator = FedAvgAggregator() if aggregator is None else aggregator
    # A stateful method must reset here, before seeing this run's observations.
    if hasattr(aggregator, "reset"):
        aggregator.reset()
    client_ids = sorted(int(k) for k in data.selection["clients"])
    scenario.validate(client_ids, config.rounds)
    if data.selection["selection_seed"] != config.development_seed:
        raise ValueError("Load selections with the run's development seed.")
    random.seed(config.development_seed)
    np.random.seed(config.development_seed)
    torch.manual_seed(config.development_seed)
    torch.set_num_threads(config.num_threads)
    torch.use_deterministic_algorithms(True)
    model = MnistCNN()
    initial_state = clone_state(model)
    evaluation_metadata = getattr(data, "evaluation_metadata", None)
    metadata = {
        "config": asdict(config), "scenario": scenario.to_dict(),
        "method": {"name": aggregator.name, "settings": aggregator.settings},
        "model": "MNIST_CNN_conv32_conv64_dense128", "parameters": 421642, "device": "cpu",
        "partition_sha256": data.manifest_sha256,
        "selection_sha256": hashlib.sha256(json.dumps(data.selection, sort_keys=True).encode()).hexdigest(),
        "initial_state_sha256": state_sha256(initial_state),
        "code_sha256": {name: file_sha256(Path(__file__).parent / name) for name in ("experiment_core.py", "scenario_utils.py")},
        "notebook_code_sha256": notebook_code_sha256,
        "mnist_c_source_manifest_sha256": file_sha256(data.root / "resources/mnist_c_reference/source_manifest.json"),
        "versions": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__, "pandas": pd.__version__},
        "validation_tasks": list(data.validation_tasks),
        "official_test_used": bool(evaluation_metadata and evaluation_metadata.get("split") == "official_test"),
    }
    if evaluation_metadata is not None:
        metadata["evaluation_data"] = copy.deepcopy(evaluation_metadata)
    if evaluator is not None:
        metadata["evaluation_recorder"] = {"name": evaluator.name, "settings": evaluator.settings}
    fingerprint = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()[:12]
    base = data.root / "outputs/scenarios" / f"{scenario.name}_seed_{config.development_seed}_{fingerprint}"
    base.mkdir(parents=True, exist_ok=True)
    attempt = 1
    while True:
        out = base / f"attempt_{attempt:04d}"
        try:
            out.mkdir()
            break
        except FileExistsError:
            attempt += 1
    (out / "truth").mkdir()
    write_json(out / "run.json", {**metadata, "created_at": datetime.now(timezone.utc).isoformat(), "attempt": attempt})
    write_json(out / "selection.json", data.selection)
    torch.save(initial_state, out / "initial_global_model_state.pt")
    write_json(out / "status.json", {"status": "running", "round": 0})
    history, task_rows, client_rows, truth_rows, diagnostics = [], [], [], [], []
    start_time = time.perf_counter()

    def measure(round_number):
        for name, view in data.validation_tasks.items():
            metrics = evaluate(model, view, config) if evaluator is None else evaluator.evaluate(model, view, config)
            task_rows.append({"round": round_number, "task": name, **metrics})

    def save_progress():
        pd.DataFrame(history).to_csv(out / "history.csv", index=False)
        pd.DataFrame(task_rows).to_csv(out / "task_metrics.csv", index=False)
        pd.DataFrame(client_rows).to_csv(out / "client_metrics.csv", index=False)
        pd.DataFrame(truth_rows).to_csv(out / "truth/scenario_truth.csv", index=False)
        write_json(out / "aggregation_diagnostics.json", diagnostics)

    try:
        measure(0)
        for round_number in range(1, config.rounds + 1):
            round_start = time.perf_counter()
            broadcast = clone_state(model)
            broadcast_hash = state_sha256(broadcast)
            submissions, round_clients = [], []
            for client_id in client_ids:
                view = client_view(data.pool, data.selection["clients"][str(client_id)], data.manifest["normalization"],
                                   config.development_seed, scenario, client_id, round_number)
                local_start = time.perf_counter()
                local_state, metrics = train_client(model, view, config, client_id, round_number, broadcast)
                training_seconds = time.perf_counter() - local_start
                gaussian_sigma = 0.0
                if any(event.kind == "gaussian_update" for event in scenario.active(client_id, round_number)):
                    local_state, gaussian_sigma = gaussian_update_corruption(local_state, broadcast, config.development_seed, client_id, round_number)
                submissions.append(ClientSubmission(client_id, len(view), local_state))
                round_clients.append({"round": round_number, "client": client_id, "examples": len(view), **metrics,
                                      "submitted_update_l2_norm": state_distance(local_state, broadcast),
                                      "submitted_state_sha256": state_sha256(local_state),
                                      "broadcast_state_sha256": broadcast_hash, "training_seconds": training_seconds})
                truth_rows.append({**truth_row(scenario, client_id, round_number, view), "gaussian_noise_std": gaussian_sigma})
            assert state_sha256(clone_state(model)) == broadcast_hash, "Local training modified the server."
            aggregation_start = time.perf_counter()
            # No scenario, labels, validation results or oracle truth cross this interface.
            result = aggregator.aggregate(submissions, broadcast, round_number)
            aggregation_seconds = time.perf_counter() - aggregation_start
            if not all(torch.isfinite(t).all() for t in result.state.values()):
                raise FloatingPointError("Non-finite aggregated state.")
            model.load_state_dict(result.state, strict=True)
            for row in round_clients:
                row["aggregation_coefficient"] = result.coefficients.get(row["client"], float("nan"))
            client_rows.extend(round_clients)
            diagnostics.append({"round": round_number, "values": result.diagnostics})
            evaluation_start = time.perf_counter()
            measure(round_number)
            evaluation_seconds = time.perf_counter() - evaluation_start
            clean = next(row for row in reversed(task_rows) if row["task"] == "clean")
            history.append({"round": round_number, "clean_accuracy": clean["accuracy"], "clean_loss": clean["loss"],
                            "training_seconds": sum(row["training_seconds"] for row in round_clients),
                            "aggregation_seconds": aggregation_seconds, "evaluation_seconds": evaluation_seconds,
                            "round_seconds": time.perf_counter() - round_start,
                            "global_update_l2_norm": state_distance(result.state, broadcast),
                            "global_state_sha256": state_sha256(result.state)})
            save_progress()
            write_json(out / "status.json", {"status": "running", "round": round_number})
            if verbose:
                print(f"{scenario.name}: round {round_number}/{config.rounds}, clean validation {clean['accuracy']:.1%}", flush=True)
        torch.save(clone_state(model), out / "final_global_model_state.pt")
        checkpoint_ok = all(checkpoint_matches(model, out / "final_global_model_state.pt", view, config)
                            for view in data.validation_tasks.values())
        if not checkpoint_ok:
            raise AssertionError("Restored checkpoint changed validation logits.")
        summary = {"scenario": scenario.name, "method": aggregator.name, "rounds": config.rounds,
                   "client_updates": len(client_rows), "elapsed_seconds": time.perf_counter() - start_time,
                   "checkpoint_logits_identical": checkpoint_ok,
                   "final_tasks": {row["task"]: {k: row[k] for k in ("accuracy", "loss", "examples")}
                                   for row in task_rows if row["round"] == config.rounds},
                   "official_test_used": metadata["official_test_used"]}
        write_json(out / "summary.json", summary)
        write_json(out / "status.json", {"status": "complete", "round": config.rounds})
        return out
    except Exception as exc:
        save_progress()
        (out / "failure.txt").write_text(traceback.format_exc())
        write_json(out / "status.json", {"status": "failed", "last_completed_round": len(history)})
        exc.run_output_path = str(out)
        raise
