"""Replay the Table 2 jobs with confusion counts and macro metrics."""
import fcntl
import json
import os
from pathlib import Path
from statistics import mean

import numpy as np
import torch

import study

BASE = study.HERE / "results" / "full"
OUTPUT = study.HERE / "results" / "table2_metrics"
METHODS = ("RFA", "FLTrust", "Viceroy", "FedReTrust-v06")
CONDITIONS = ("label_flip", "honest_drift")
METRICS = ("accuracy", "macro_precision", "macro_recall")


@torch.inference_mode()
def confusion(model, x, y, classes):
    model.eval()
    predicted = torch.cat([model(batch).argmax(dim=1) for batch in x.split(256)]).cpu()
    return torch.bincount(y.cpu() * classes + predicted, minlength=classes**2).reshape(
        classes, classes).numpy()


def scores(matrix):
    correct = matrix.diagonal()
    predicted = matrix.sum(axis=0)
    actual = matrix.sum(axis=1)
    precision = np.divide(correct, predicted, out=np.zeros(len(correct)), where=predicted > 0)
    recall = np.divide(correct, actual, out=np.zeros(len(correct)), where=actual > 0)
    return {name + "_pct": 100 * float(value) for name, value in zip(
        METRICS, (correct.sum() / matrix.sum(), precision.mean(), recall.mean()))}


def evaluate_round(data, condition, counts, model, round_id, fraction):
    test, shifted, y = data["test"]
    clean = confusion(model, test, y, data["classes"])
    counts["unchanged"].append(clean)
    if condition == "honest_drift":
        drift = confusion(model, shifted, y, data["classes"])
        counts["shifted"].append(drift)
        # Reconstruct the existing affected-client mixture as a joint distribution.
        # Rows are true classes; each class supplies the original client-profile mass.
        mix = ((1 - fraction) * clean + fraction * drift) / clean.sum(axis=1)[:, None]
        matrix = data["weights"][:, None] * mix
    else:
        matrix = clean
    metrics = scores(matrix)
    clean_acc = 100 * float(clean.trace() / clean.sum())
    return {"round": round_id, "accuracy_pct": clean_acc,
            "drift_pct": metrics["accuracy_pct"] if condition == "honest_drift" else None,
            "macro_precision_pct": metrics["macro_precision_pct"],
            "macro_recall_pct": metrics["macro_recall_pct"]}


def run_one(data, method, condition, rounds, device, path, old_record):
    result_path = path / "result.json"
    if result_path.exists():
        record = json.loads(result_path.read_text())
        for filename, digest in record["files"].items():
            if study.file_sha256(path / filename) != digest:
                raise ValueError(f"Saved output changed: {path / filename}")
        return record
    counts = {"unchanged": [], "shifted": []}
    rows, record = study.run_job(
        data, method, condition, rounds, device,
        evaluate_round=lambda model, round_id, fraction: evaluate_round(
            data, condition, counts, model, round_id, fraction))
    if record["initial_model_sha256"] != old_record["initial_model_sha256"]:
        raise ValueError(f"Initial model differs from the original run: {path}")
    if any(len(v) not in (0, len(rows)) for v in counts.values()):
        raise ValueError(f"Round counts and scores differ: {path}")
    metric_names = {"accuracy": "drift_pct" if condition == "honest_drift" else "accuracy_pct",
                    "macro_precision": "macro_precision_pct", "macro_recall": "macro_recall_pct"}
    record["metrics"] = {name: mean(row[key] for row in rows[study.ONSET:]) if
                         record["status"] == "completed" else None
                         for name, key in metric_names.items()}
    path.mkdir(parents=True, exist_ok=True)
    study.write_csv(path / "per_round.csv", rows)
    np.savez_compressed(path / "confusions.npz", **{key: np.stack(value) for key, value
                                                  in counts.items() if value})
    record["files"] = {name: study.file_sha256(path / name)
                       for name in ("per_round.csv", "confusions.npz")}
    study.write_json(result_path, record)
    return record


def summarize(records):
    summary = []
    for dataset in study.SEEDS:
        for method in METHODS:
            for condition in CONDITIONS:
                jobs = [row for row in records if (row["dataset"], row["method"], row["condition"])
                        == (dataset, method, condition)]
                completed = [row for row in jobs if row["status"] == "completed"]
                row = {"dataset": dataset, "method": method, "condition": condition,
                       "completed": len(completed), "planned": len(jobs)}
                for metric in METRICS:
                    row[metric + "_pct"] = mean(job[metric + "_pct"] for job in completed) if completed else None
                summary.append(row)
    study.write_csv(OUTPUT / "results_table.csv", summary)
    study.write_csv(OUTPUT / "results_by_seed.csv", records)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / ".run.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        replay()


def replay():
    original_path = BASE / "manifest.json"
    original = json.loads(original_path.read_text())
    device = torch.device(original["device"])
    if device.type != "mps" or not torch.backends.mps.is_available():
        raise RuntimeError("This replay requires the Apple GPU used for the original study.")
    if str(torch.__version__) != original["torch"] or np.__version__ != original["numpy"]:
        raise ValueError("Use the same PyTorch and NumPy versions as the original study.")
    for filename, digest in original["code_sha256"].items():
        if filename in ("run.py", "study.py"):
            continue
        path = study.HERE / filename if filename == "data.py" else study.CODE / filename
        if study.file_sha256(path) != digest:
            raise ValueError(f"Experiment code changed since the original run: {path}")
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    rounds = original["rounds"]
    total = sum(map(len, original["seeds"].values())) * len(METHODS) * len(CONDITIONS)
    manifest = {"jobs": total, "rounds": rounds, "device": str(device),
                "source_manifest_sha256": study.file_sha256(original_path),
                "runner_sha256": study.file_sha256(Path(__file__)),
                "study_sha256": study.file_sha256(Path(study.__file__)),
                "metrics": list(METRICS), "report_rounds": original["report_rounds"],
                "macro_zero_division": 0,
                "drift": "Mix class-conditional confusions by round and weight rows by the saved affected-client class profile."}
    manifest_path = OUTPUT / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Replay settings differ from the existing Table 2 run.")
    study.write_json(manifest_path, manifest)
    print(f"Training on {device}. {total} Table 2 jobs, {rounds} rounds each. Results: {OUTPUT}", flush=True)
    records = []
    for dataset, seeds in original["seeds"].items():
        loaded = study.load_har() if dataset == "har" else study.load_images(dataset)
        for seed in seeds:
            data = study.har_plan(loaded, seed) if dataset == "har" else study.image_plan(dataset, loaded, seed)
            saved_plan = json.loads((BASE / "plans" / dataset / f"{seed}.json").read_text())
            if data["plan"] != saved_plan:
                raise ValueError(f"Data plan differs from the original run: {dataset}, {seed}")
            study.to_device(data, device)
            for method in METHODS:
                for condition in CONDITIONS:
                    relative = Path("runs") / dataset / str(seed) / method / condition
                    old = json.loads((BASE / relative / "result.json").read_text())
                    print(f"[{len(records)+1}/{total}] {dataset} seed {seed} {method} {condition}", flush=True)
                    record = run_one(data, method, condition, rounds, device, OUTPUT / relative, old)
                    row = {"dataset": dataset, "seed": seed, "method": method, "condition": condition,
                           "status": record["status"], "failure_round": (record["failure"] or {}).get("round")}
                    row.update({name + "_pct": value for name, value in record["metrics"].items()})
                    records.append(row)
                    if record["status"] == "diverged":
                        print(f"    Diverged at round {row['failure_round']}. Continuing.", flush=True)
                    else:
                        print("    Saved " + ", ".join(f"{name}={value:.2f}%" for name, value
                                                       in record["metrics"].items()), flush=True)
    summarize(records)
    study.write_json(OUTPUT / "completed.json", {"jobs": len(records),
                                               "diverged": sum(r["status"] == "diverged" for r in records)})
    print(f"Complete. Open {OUTPUT / 'results_table.csv'}", flush=True)


if __name__ == "__main__":
    main()
