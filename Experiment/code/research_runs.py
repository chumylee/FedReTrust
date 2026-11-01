"""One isolated, resumable research run; official test access requires a freeze.

Algorithms never receive this module's scenario or evaluation data. The factory
passes only counts, seed, settings and the copied reserved root reference.
"""
from dataclasses import asdict
from pathlib import Path
import argparse
import fcntl
import hashlib
import json
import platform
import resource
import sys
import time

import numpy as np
import torch
from torchvision import datasets

from experiment_core import (TrainConfig, FedAvgAggregator, load_data,
                             run_experiment, file_sha256, write_json)
from baseline_calibration import ServerLearningRate
from fltrust import FLTrustAggregator, FLTrustConfig, build_root_reference
from rfa import RFAAggregator
from viceroy import ViceroyAggregator
from fedretrust import FedReTrustAggregator, FedReTrustConfig
from scenario_metrics import ContributionRecorder, ClasswiseEvaluator, longer_scenarios
from scenario_utils import ScenarioDataset


ROOT = Path(__file__).resolve().parents[1]
TRAINING_SOURCES = (
    "experiment_core.py", "scenario_utils.py", "scenario_metrics.py", "baseline_calibration.py",
    "fltrust.py", "rfa.py", "viceroy.py", "fedretrust.py", "research_runs.py",
)
PARTITIONS = {"IID": "mnist_iid_ee88d9959326", "non-IID": "mnist_noniid_78ca1bdc3481"}
METHODS = ("FedAvg", "FLTrust", "RFA", "Viceroy", "FedReTrust")


def source_hashes():
    return {name: file_sha256(ROOT/"code"/name) for name in TRAINING_SOURCES}


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def verify_freeze(path):
    """This check must precede any official-test dataset construction."""
    path = Path(path)
    if not path.is_file():
        raise ValueError("A saved frozen protocol is required before official-test access.")
    freeze = json.loads(path.read_text())
    if freeze.get("status") != "frozen_before_official_test" or freeze["training_source_sha256"] != source_hashes():
        raise ValueError("Missing freeze or training implementation changed after freeze.")
    if set(freeze["partition_sha256"]) != set(PARTITIONS):
        raise ValueError("Both saved partitions must be frozen.")
    for regime, partition in freeze["partition_sha256"].items():
        if file_sha256(ROOT/"data"/"partitions"/PARTITIONS[regime]/"indices.json") != partition:
            raise ValueError("Partition changed after freeze.")
    return freeze


def attach_official_test(data, config, freeze_path, job):
    freeze = verify_freeze(freeze_path)
    if job not in freeze["authorized_final_jobs"]:
        raise ValueError("Job is absent from the predeclared final manifest.")
    if config.development_seed not in freeze["final_seeds"]:
        raise ValueError("Unregistered final seed.")
    if (config.max_train_per_client != freeze["workload"]["train_images_per_client"]
        or config.rounds != freeze["workload"]["rounds"]):
        raise ValueError("Final workload differs from frozen protocol.")
    if job["phase"] not in {"final", "ablation", "sensitivity"}:
        raise ValueError("Development jobs cannot request official test data.")
    for filename in ("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte"):
        if file_sha256(ROOT/"data"/"MNIST"/"raw"/filename) != data.manifest["dataset_files"][filename]["raw_sha256"]:
            raise ValueError("Official test file fingerprint changed.")
    pool = datasets.MNIST(root=ROOT/"data", train=False, download=False, transform=None)
    indices = list(data.manifest["indices"]["reserved_final_test"])
    if sorted(indices) != list(range(10000)) or len(pool) != 10000:
        raise ValueError("Expected all 10,000 reserved official test images.")
    normalization = data.manifest["normalization"]
    tasks = {}
    for name, corruption in (("clean", None), ("shot_noise_s3", "shot_noise"), ("stripe", "stripe")):
        tasks[name] = ScenarioDataset(pool, indices, normalization["mean"][0],
                                      normalization["std"][0], config.development_seed,
                                      corruption=corruption, fraction=float(corruption is not None))
    data.validation_tasks = tasks
    data.selection = {**data.selection, "official_test": indices,
                      "evaluation_index_space": "official_test"}
    data.evaluation_metadata = {"split": "official_test", "examples_per_task": len(indices),
                                "index_sha256": canonical_hash(indices),
                                "freeze_sha256": file_sha256(freeze_path)}
    return data


def make_aggregator(method, data, config, relative_rate=1., variant="full", overrides=None, root_steps=8):
    root_options = FLTrustConfig(root_steps=root_steps)
    if method == "FedReTrust":
        changes = dict(overrides or {})
        if "server_learning_rate" in changes or "variant" in changes:
            raise ValueError("Specify rate and variant in their dedicated fields.")
        options = FedReTrustConfig(server_learning_rate=relative_rate, variant=variant, **changes)
        reference = None if variant == "clipped_FedAvg" else build_root_reference(data)
        counts = {int(i):len(v) for i,v in data.manifest["indices"]["clients"].items()}
        selected = {int(i):len(v) for i,v in data.selection["clients"].items()}
        base = FedReTrustAggregator(reference, config.development_seed, counts, options, root_options, selected)
    else:
        if variant != "full" or overrides:
            raise ValueError("Do not alter a published baseline through proposed-policy overrides.")
        factories = {
            "FedAvg": lambda: FedAvgAggregator(),
            "FLTrust": lambda: FLTrustAggregator(build_root_reference(data), config.development_seed, root_options),
            "RFA": lambda: RFAAggregator(),
            "Viceroy": lambda: ViceroyAggregator(),
        }
        if method not in factories:
            raise ValueError("Unknown method.")
        rate = relative_rate/len(data.selection["clients"]) if method == "Viceroy" else relative_rate
        base = ServerLearningRate(factories[method](), rate)
    return ContributionRecorder(base)


def run_identity(job):
    return canonical_hash({"job":job, "training_source_sha256":source_hashes()})[:20]


def execute_job(job, study):
    directory = Path(study)/"jobs"/run_identity(job)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory/"worker.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _execute_job(job, study)


def _execute_job(job, study):
    start = time.perf_counter()
    study = Path(study)
    job_id = run_identity(job)
    directory = study/"jobs"/job_id
    directory.mkdir(parents=True, exist_ok=True)
    result_path = directory/"result.json"
    if result_path.exists():
        result = json.loads(result_path.read_text())
        if result["job"] != job:
            raise ValueError("Run identity mismatch.")
        run_path = ROOT/result["run_path"]
        if all(file_sha256(run_path/name) == digest for name, digest in result["artifact_sha256"].items()):
            print(json.dumps({"job_id":job_id, "status":"reused"}), flush=True)
            return result
        raise ValueError("A saved completed run has changed; preserve it and investigate.")
    if (directory/"failure.json").exists():
        raise ValueError("Previous failed job requires an explicit recorded retry decision.")
    write_json(directory/"request.json", {"job":job, "source_sha256":source_hashes()})
    if job["phase"] in {"final", "ablation", "sensitivity"}:
        freeze = verify_freeze(study/"frozen_protocol.json")
        if job not in freeze["authorized_final_jobs"]:
            raise ValueError("Job is absent from the predeclared final manifest.")
    cfg = TrainConfig(partition_id=PARTITIONS[job["regime"]], development_seed=job["seed"],
                      rounds=job.get("rounds",24), max_train_per_client=job.get("train_images",512),
                      max_validation=1000, num_threads=job.get("num_threads",4))
    try:
        data = load_data(ROOT, cfg)
        if job["phase"] in {"final", "ablation", "sensitivity"}:
            data = attach_official_test(data, cfg, study/"frozen_protocol.json", job)
        aggregator = make_aggregator(job["method"], data, cfg, job["relative_rate"],
                                    job.get("variant","full"), job.get("overrides"), job.get("root_steps",8))
        scenario = longer_scenarios()[job["scenario"]]
        if cfg.rounds < 24 and scenario.events:
            raise ValueError("Only clean runs may use a shorter integration horizon.")
        run_path = run_experiment(data, cfg, scenario, aggregator,
                                  notebook_code_sha256=job_id, verbose=False, evaluator=ClasswiseEvaluator())
        counts = {i:np.bincount(data.pool.targets[indices].numpy(), minlength=10).tolist()
                  for i,indices in data.manifest["indices"]["clients"].items()}
        write_json(run_path/"population_class_counts.json", counts)
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        rss_bytes = int(rss if platform.system() == "Darwin" else rss*1024)
        files = ("run.json","selection.json","summary.json","history.csv","task_metrics.csv",
                 "client_metrics.csv","aggregation_diagnostics.json","population_class_counts.json",
                 "initial_global_model_state.pt","final_global_model_state.pt","status.json")
        result = {"job_id":job_id, "job":job, "status":"complete",
                  "run_path":str(run_path.relative_to(ROOT)),
                  "artifact_sha256":{name:file_sha256(run_path/name) for name in files},
                  "worker_wall_seconds":time.perf_counter()-start,
                  "process_lifetime_peak_rss_bytes":rss_bytes,
                  "memory_interpretation":"fresh worker process high-water RSS; includes imports/data; not allocated method memory",
                  "source_sha256":source_hashes()}
        write_json(result_path, result)
        print(json.dumps({"job_id":job_id, "status":"complete", "seconds":result["worker_wall_seconds"],
                          "method":job["method"], "scenario":job["scenario"], "regime":job["regime"]}), flush=True)
        return result
    except Exception as exc:
        write_json(directory/"failure.json", {"job":job, "error":repr(exc),
                                            "exception_type":type(exc).__name__,
                                            "run_path":getattr(exc,"run_output_path",None),
                                            "elapsed_seconds":time.perf_counter()-start})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True)
    parser.add_argument("--study", required=True)
    args = parser.parse_args()
    execute_job(json.loads(Path(args.job).read_text()), args.study)
