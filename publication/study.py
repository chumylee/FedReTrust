"""Run the paired comparison and save round measurements and summaries."""
import argparse
import csv
import json
import math
import os
import platform
import sys
from dataclasses import asdict
from pathlib import Path
from statistics import mean, stdev
from time import monotonic
from zipfile import ZIP_DEFLATED, ZipFile

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
CODE = HERE.parent / "Experiment" / "code"
sys.path.insert(0, str(CODE))

from data import SEEDS, har_plan, image_plan, load_har, load_images
from experiment_core import ClientSubmission, FedAvgAggregator, clone_state, file_sha256, state_sha256
from fedretrust_v06 import FedReTrustV06
from fltrust import FLTrustConfig, flatten_delta, fltrust_rule
from rfa import RFAAggregator
from scenario_utils import seed_for
from viceroy import ViceroyAggregator

METHODS = {"FedAvg": FedAvgAggregator, "RFA": RFAAggregator, "FLTrust": None,
           "Viceroy": ViceroyAggregator, "FedReTrust-v06": FedReTrustV06}
CONDITIONS = ("clean", "honest_drift", "label_flip")
ONSET = 5
RAMP = 8


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def write_csv(path, rows):
    temp = path.with_suffix(".tmp")
    with temp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def train(model, broadcast, x, y, lr, batches):
    model.load_state_dict(broadcast)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=.9)
    for batch in batches:
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x[batch]), y[batch])
        loss.backward()
        optimizer.step()
    return clone_state(model)


@torch.inference_mode()
def evaluate(model, x, y, classes):
    model.eval()
    predictions = torch.cat([model(batch).argmax(dim=1) for batch in x.split(256)])
    # Keep counting and weighted evaluation on CPU across backends.
    labels = y.cpu()
    good = predictions.cpu() == labels
    count = torch.bincount(labels, minlength=classes)
    correct = torch.bincount(labels[good], minlength=classes)
    return int(good.sum()) / len(y), (correct.double() / count).numpy()


def to_device(data, device):
    data["clients"] = {i: tuple(t.to(device) for t in values)
                       for i, values in data["clients"].items()}
    for key in ("test", "root"):
        data[key] = tuple(t.to(device) for t in data[key])
    data["permutation"] = data["permutation"].to(device)


def run_job(data, method, condition, rounds, device):
    start = monotonic()
    seed = data["plan"]["seed"]
    torch.manual_seed(seed)
    model = data["model"]().to(device)
    local = data["model"]().to(device)
    initial_hash = state_sha256(clone_state(model))
    aggregator = METHODS[method]() if method != "FLTrust" else None
    root_options = FLTrustConfig(root_learning_rate=data["lr"])
    settings = (aggregator.settings if aggregator is not None else
                {**asdict(root_options), "clean_reference_examples": len(data["root"][1]),
                 "implementation_sha256": file_sha256(CODE / "fltrust.py")})
    rows = []
    test, shifted_test, test_y = data["test"]
    for round_id in range(rounds + 1):
        fraction = min(1.0, max(0.0, (round_id - ONSET + 1) / RAMP))
        if round_id:
            broadcast = clone_state(model)
            submissions = []
            for client, (clean, shifted, labels, drift_order) in data["clients"].items():
                x, y = clean, labels
                if round_id >= ONSET and client in data["affected"]:
                    if condition == "honest_drift":
                        positions = drift_order[:math.floor(fraction * len(labels))]
                        x = clean.clone()
                        x[positions] = shifted[positions]
                    elif condition == "label_flip":
                        y = data["permutation"][labels]
                shuffle = (seed * 1_000_003 + round_id * 1009 + client) % (2**63 - 1)
                generator = torch.Generator().manual_seed(shuffle)
                # DataLoader draws a base seed and RandomSampler then draws the permutation.
                torch.empty((), dtype=torch.int64).random_(generator=generator)
                batches = torch.randperm(len(y), generator=generator).to(device).split(64)
                state = train(local, broadcast, x, y, data["lr"], batches)
                submissions.append(ClientSubmission(client, len(y), state))
            if aggregator is None:
                root_x, root_y = data["root"]
                rng = torch.Generator().manual_seed(seed_for(seed, "fltrust_root_batches", round_id))
                root_batches = [torch.randperm(len(root_y), generator=rng)[:root_options.root_batch_size].to(device)
                                for _ in range(root_options.root_steps)]
                root_state = train(local, broadcast, root_x, root_y, root_options.root_learning_rate,
                                   root_batches)
                result = fltrust_rule(submissions, broadcast, flatten_delta(root_state, broadcast),
                                      root_options.global_learning_rate, root_options.epsilon)
            else:
                result = aggregator.aggregate(submissions, broadcast, round_id)
            if any(not torch.isfinite(value).all() for value in result.state.values()):
                raise FloatingPointError(f"Non-finite {method} model at round {round_id}")
            model.load_state_dict(result.state)
        accuracy, clean_classes = evaluate(model, test, test_y, data["classes"])
        drift = None
        if condition == "honest_drift":
            _, shifted_classes = evaluate(model, shifted_test, test_y, data["classes"])
            drift = float(data["weights"] @ ((1 - fraction) * clean_classes + fraction * shifted_classes))
        rows.append({"round": round_id, "accuracy_pct": 100 * accuracy,
                     "drift_pct": 100 * drift if drift is not None else None})
        if round_id and (round_id % 8 == 0 or round_id == rounds):
            print(f"    Round {round_id}/{rounds}: {100 * accuracy:.2f}% clean-input accuracy", flush=True)
    metric = "drift_pct" if condition == "honest_drift" else "accuracy_pct"
    return rows, {
        "initial_model_sha256": initial_hash,
        "settings": settings, "seconds": monotonic() - start,
        "mean_pct": mean(row[metric] for row in rows[ONSET:]),
        "round24_pct": rows[24][metric] if rounds >= 24 else None,
        "final_pct": rows[-1][metric],
    }


def summarize(output, records, seeds):
    by_seed = []
    for dataset, dataset_seeds in seeds.items():
        for seed in dataset_seeds:
            for method in METHODS:
                jobs = {name: records[dataset, seed, method, name] for name in CONDITIONS}
                row = {"dataset": dataset, "seed": seed, "method": method}
                for condition, label in zip(CONDITIONS, ("C", "H", "P")):
                    for key in ("mean_pct", "round24_pct", "final_pct"):
                        row[f"{label}_{key}"] = jobs[condition][key]
                row["poisoning_damage_pp"] = row["C_mean_pct"] - row["P_mean_pct"]
                by_seed.append(row)

    summary, pairs = [], []
    for dataset, dataset_seeds in seeds.items():
        for method in METHODS:
            group = [row for row in by_seed if (row["dataset"], row["method"]) == (dataset, method)]
            summary.append({
                "dataset": dataset, "method": method, "repeats": len(group),
                **{key: mean(row[key] for row in group) if group[0][key] is not None else None
                   for key in group[0] if key not in ("dataset", "seed", "method")},
            })
            if method == "FedReTrust-v06":
                continue
            for metric in ("C_mean_pct", "P_mean_pct", "H_mean_pct", "poisoning_damage_pp"):
                v06 = {row["seed"]: row[metric] for row in by_seed
                       if (row["dataset"], row["method"]) == (dataset, "FedReTrust-v06")}
                differences = [v06[row["seed"]] - row[metric] for row in group]
                centre = mean(differences)
                half = (2.7764451052 * stdev(differences) / math.sqrt(5)
                        if len(differences) == 5 else None)
                pairs.append({"dataset": dataset, "comparator": method, "metric": metric,
                              "repeats": len(dataset_seeds), "v06_minus_comparator_pp": centre,
                              "nominal_95_ci_low_pp": centre - half if half is not None else None,
                              "nominal_95_ci_high_pp": centre + half if half is not None else None})
    write_csv(output / "results_by_seed.csv", by_seed)
    write_csv(output / "results_table.csv", summary)
    write_csv(output / "paired_differences.csv", pairs)


def package_results(output, destination):
    temporary = destination.with_suffix(".tmp")
    with ZipFile(temporary, "w", ZIP_DEFLATED) as archive:
        for path in sorted(output.rglob("*")):
            if path.suffix in (".csv", ".json") and path.is_file():
                archive.write(path, path.relative_to(output))
    temporary.replace(destination)


def run_study(args, parser, output):
    backend = args.device
    if backend == "auto":
        backend = ("cuda" if torch.cuda.is_available() else
                   "mps" if torch.backends.mps.is_available() else "cpu")
    if (backend == "cuda" and not torch.cuda.is_available()
            or backend == "mps" and not torch.backends.mps.is_available()):
        parser.error(f"{backend} is unavailable. Check your PyTorch install or select --device cpu.")
    device = torch.device(backend)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    # CUDA uses deterministic cuBLAS/cuDNN paths. MPS may need CPU fallback for some ops.
    if backend != "mps":
        torch.use_deterministic_algorithms(True)
    rounds = 6 if args.smoke else 48
    seeds = {dataset: list(values[:1] if args.smoke else values) for dataset, values in SEEDS.items()}
    total = sum(map(len, seeds.values())) * len(METHODS) * len(CONDITIONS)
    manifest = {
        "rounds": rounds, "report_rounds": [ONSET, rounds], "local_epochs": 1,
        "batch_size": 64, "momentum": .9, "drift_ramp_rounds": RAMP,
        "methods": list(METHODS), "conditions": CONDITIONS, "seeds": seeds,
        "jobs": total, "device": backend, "machine": platform.machine(),
        "gpu": torch.cuda.get_device_name(0) if backend == "cuda" else None,
        "torch": str(torch.__version__), "numpy": np.__version__, "cuda": torch.version.cuda,
        "code_sha256": {path.name: file_sha256(path) for path in (
            *sorted(HERE.glob("*.py")), *(CODE / name for name in (
                "experiment_core.py", "rfa.py", "fltrust.py", "viceroy.py",
                "fedretrust_v06.py", "fedretrust_v05.py", "fedretrust_directional.py",
                "fedretrust_reference_free.py", "non_iid_partitions.py", "scenario_utils.py")))},
    }
    manifest = json.loads(json.dumps(manifest))
    manifest_path = output / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        parser.error(f"Settings differ from an existing run. Move {output} aside to start a fresh run.")
    write_json(manifest_path, manifest)

    print(f"Training on {backend}. {total} jobs, {rounds} rounds per job. Results: {output}", flush=True)
    records = {}
    for dataset, dataset_seeds in seeds.items():
        loaded = load_har() if dataset == "har" else load_images(dataset)
        for seed in dataset_seeds:
            data = har_plan(loaded, seed) if dataset == "har" else image_plan(dataset, loaded, seed)
            plan_path = output / "plans" / dataset / f"{seed}.json"
            if plan_path.exists() and json.loads(plan_path.read_text()) != data["plan"]:
                raise ValueError(f"Data plan changed: {plan_path}")
            write_json(plan_path, data["plan"])
            to_device(data, device)
            initial_hash = None
            for method in METHODS:
                for condition in CONDITIONS:
                    key = dataset, seed, method, condition
                    path = output / "runs" / dataset / str(seed) / method / condition
                    path.mkdir(parents=True, exist_ok=True)
                    done = path / "result.json"
                    measurements = path / "per_round.csv"
                    name = f"[{len(records) + 1}/{total}] {dataset} seed {seed} {method} {condition}"
                    if done.exists():
                        record = json.loads(done.read_text())
                        if file_sha256(measurements) != record["measurements_sha256"]:
                            raise ValueError(f"Saved round measurements changed: {measurements}")
                        print(f"{name} — already complete", flush=True)
                    else:
                        print(name, flush=True)
                        rows, record = run_job(data, method, condition, rounds, device)
                        write_csv(measurements, rows)
                        record["measurements_sha256"] = file_sha256(measurements)
                        write_json(done, record)
                    if initial_hash is not None and record["initial_model_sha256"] != initial_hash:
                        raise ValueError(f"Initial models differ in {dataset}, seed {seed}")
                    initial_hash = record["initial_model_sha256"]
                    records[key] = record
    summarize(output, records, seeds)
    write_json(output / "completed.json", {"jobs": len(records), "rounds": rounds})
    print(f"Complete. Open {output / 'results_table.csv'}", flush=True)
    if not args.smoke:
        destination = HERE / "FedReTrust-results.zip"
        package_results(output, destination)
        print(f"Send this file to Chuma: {destination}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument("--smoke", action="store_true", help="Six rounds and one seed per dataset.")
    args = parser.parse_args()
    output = HERE / "results" / ("smoke" if args.smoke else "full")
    output.mkdir(parents=True, exist_ok=True)
    # The OS releases the lock even if training is interrupted.
    with (output / ".run.lock").open("a+b") as lock:
        try:
            if sys.platform == "win32":
                import msvcrt
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            parser.error(f"Another process is already using {output}")
        run_study(args, parser, output)


if __name__ == "__main__":
    main()
