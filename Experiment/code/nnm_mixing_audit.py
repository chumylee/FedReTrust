"""Post-pilot, offline explanation of NNM mixing and effective client weights.

This analysis never imports an aggregator or a training routine. It cannot alter
the frozen diagnostic gate, thresholds, models or original experiment files.
Simulator groups are derived from active saved events only after training.
"""
from collections import defaultdict
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import statistics


ROOT = Path(__file__).resolve().parents[1]
METHODS = ("FedAvg", "RFA", "FLTrust", "FedReTrust-RF-recovery",
           "FedReTrust-Direction-loose", "NNM-RFA")
SCENARIOS = ("clean", "abrupt_shot_noise", "gradual_stripe", "label_flip")
REGIMES = ("IID", "non-IID")
SEEDS = (20261014, 20261015)
ROUNDS = tuple(range(5, 25))
CONVENTION = "effective_coefficients_on_raw_client_deltas"


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path):
    return json.loads(Path(path).read_text())


def _write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _finite_nonnegative(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def _groups(scenario, round_number, clients):
    labels, strengths = {}, {}
    for cid in clients:
        active = []
        fractions = []
        for event in scenario["events"]:
            if cid not in event["clients"]:
                continue
            for start, end in event["windows"]:
                if start <= round_number <= end:
                    active.append(event)
                    if event["kind"] == "input_change" and event["intent"] == "honest":
                        ramp = event["ramp_rounds"]
                        fractions.append(min(1., (round_number - start + 1) / ramp) if ramp else 1.)
                    break
        labels[cid] = ("attack" if any(e["intent"] == "malicious" for e in active) else
                       "honest_changed" if fractions else "unchanged_honest")
        strengths[cid] = max(fractions, default=0.)
    return labels, strengths


def _audit_nnm(values, coefficients):
    """Validate the recorded mixing operator and independently expand A^T w."""
    order = values["client_order"]
    if (len(order) != 10 or any(type(cid) is not int for cid in order)
            or set(order) != set(coefficients) or len(set(order)) != len(order)):
        raise ValueError("NNM client order must cover all ten distinct submissions.")
    if values["declared_f"] != 3 or values["neighborhood_size"] != 7:
        raise ValueError("NNM does not match the fixed declared n=10, f=3 control.")
    matrix, weights = values["mixing_matrix"], values["mixture_weights"]
    if (len(matrix) != 10 or any(len(row) != 10 for row in matrix) or len(weights) != 10
            or any(not _finite_nonnegative(x) for row in matrix for x in row)
            or any(not _finite_nonnegative(x) for x in weights)
            or not math.isclose(sum(weights), 1., abs_tol=1e-12, rel_tol=0)):
        raise ValueError("Invalid NNM matrix or RFA mixture weights.")
    members = [0] * 10
    for i, cid in enumerate(order):
        neighbors = values["neighbor_ids"][str(cid)]
        if (len(neighbors) != 7 or len(set(neighbors)) != 7 or cid not in neighbors
                or not set(neighbors).issubset(order)):
            raise ValueError("NNM neighborhood must contain self and six distinct other clients.")
        for j, original in enumerate(order):
            expected = 1/7 if original in neighbors else 0.
            if not math.isclose(matrix[i][j], expected, abs_tol=1e-12, rel_tol=0):
                raise ValueError("NNM matrix differs from equal arithmetic neighbor mixing.")
            members[j] += int(original in neighbors)
    expanded = [sum(weights[i] * matrix[i][j] for i in range(10)) for j in range(10)]
    errors = [abs(expanded[j] - coefficients[cid]) for j, cid in enumerate(order)]
    if max(errors) > 1e-12:
        raise ValueError("Recorded applied coefficients do not equal A^T times mixture weights.")
    originals = values["original_client_coefficients"]
    if len(originals) != 10 or any(abs(originals[j] - expanded[j]) > 1e-12 for j in range(10)):
        raise ValueError("Original-client coefficient diagnostics differ from A^T w.")
    for j, cid in enumerate(order):
        if values["client_metrics"][str(cid)]["mixture_membership_count"] != members[j]:
            raise ValueError("Recorded mixture membership count differs from the matrix.")
    return order, matrix, weights, dict(zip(order, members)), max(errors)


def round_group_rows(job, scenario, round_number, values):
    """Return group masses and NNM origin-group/anchor-group coefficient paths."""
    if values.get("coefficient_convention") != CONVENTION:
        raise ValueError("Effective raw-client-delta coefficients are required.")
    clients = list(range(10))
    metrics = values["client_metrics"]
    if set(metrics) != {str(cid) for cid in clients}:
        raise ValueError("Missing or extra clients in saved coefficient diagnostics.")
    coefficients = {cid: metrics[str(cid)]["applied_delta_coefficient"] for cid in clients}
    if any(not _finite_nonnegative(value) for value in coefficients.values()):
        raise ValueError("Effective coefficients must be finite and nonnegative.")
    labels, strengths = _groups(scenario, round_number, clients)
    groups = {label: [cid for cid in clients if labels[cid] == label] for label in sorted(set(labels.values()))}
    total = sum(coefficients.values())
    base = {"method": job["method"], "regime": job["regime"], "scenario": job["scenario"],
            "seed": job["seed"], "round": round_number}
    order = matrix = weights = memberships = None
    error = None
    flows = []
    if job["method"] == "NNM-RFA":
        order, matrix, weights, memberships, error = _audit_nnm(values, coefficients)
        for origin_group in groups:
            for anchor_group in groups:
                mass = sum(weights[i] * matrix[i][j] for i, anchor in enumerate(order)
                           for j, origin in enumerate(order)
                           if labels[anchor] == anchor_group and labels[origin] == origin_group)
                flows.append({**base, "origin_group": origin_group, "anchor_group": anchor_group,
                              "coefficient_mass": mass})
    rows = []
    for group, ids in groups.items():
        mass = sum(coefficients[cid] for cid in ids)
        row = {**base, "group": group, "group_clients": len(ids),
               "mean_input_change_fraction": statistics.mean(strengths[cid] for cid in ids),
               "total_effective_coefficient_mass": total, "group_effective_coefficient_mass": mass,
               "group_share_of_effective_coefficient_mass": mass / total if total else None,
               "mean_effective_coefficient_per_client": mass / len(ids),
               "uniform_sample_share": len(ids) / 10,
               "coefficient_mass_over_uniform_sample_share": mass / (len(ids) / 10),
               "same_group_neighbor_fraction_per_anchor": None,
               "mean_mixture_memberships_per_client": None,
               "minimum_mixture_memberships_per_client": None,
               "maximum_mixture_memberships_per_client": None,
               "anchor_mixture_weight_mass": None,
               "group_coefficient_mass_from_same_group_anchors": None,
               "group_coefficient_mass_from_other_group_anchors": None,
               "group_coefficient_fraction_from_other_group_anchors": None,
               "coefficient_expansion_max_abs_error": error}
        if matrix is not None:
            indices = [i for i, cid in enumerate(order) if cid in ids]
            same = sum(flow["coefficient_mass"] for flow in flows
                       if flow["origin_group"] == group and flow["anchor_group"] == group)
            other = sum(flow["coefficient_mass"] for flow in flows
                        if flow["origin_group"] == group and flow["anchor_group"] != group)
            if not math.isclose(same + other, mass, abs_tol=1e-12, rel_tol=0):
                raise ValueError("Group flow decomposition does not reproduce its coefficient mass.")
            row.update({
                "same_group_neighbor_fraction_per_anchor": statistics.mean(
                    sum(matrix[i][j] for j, cid in enumerate(order) if labels[cid] == group) for i in indices),
                "mean_mixture_memberships_per_client": statistics.mean(memberships[cid] for cid in ids),
                "minimum_mixture_memberships_per_client": min(memberships[cid] for cid in ids),
                "maximum_mixture_memberships_per_client": max(memberships[cid] for cid in ids),
                "anchor_mixture_weight_mass": sum(weights[i] for i in indices),
                "group_coefficient_mass_from_same_group_anchors": same,
                "group_coefficient_mass_from_other_group_anchors": other,
                "group_coefficient_fraction_from_other_group_anchors": other / mass if mass else None,
            })
        rows.append(row)
    return rows, flows


def _summarize(rows, keys, metrics):
    buckets = defaultdict(list)
    for row in rows:
        buckets[tuple(row[key] for key in keys)].append(row)
    output = []
    for identity, items in sorted(buckets.items()):
        summary = dict(zip(keys, identity))
        for metric in metrics:
            values = [item[metric] for item in items]
            summary[metric] = statistics.mean(values) if all(value is not None for value in values) else None
        summary["observations_averaged"] = len(items)
        output.append(summary)
    return output


def analyse_mixing(manifest_path, output_dir=None):
    manifest_path = Path(manifest_path).resolve()
    if _read(manifest_path.parent / "status.json")["status"] != "complete":
        raise ValueError("The pilot must be complete before this post-pilot descriptive audit.")
    manifest = _read(manifest_path)
    if manifest.get("official_test_used") is not False:
        raise ValueError("Expected a development-only completed pilot manifest.")
    output_dir = manifest_path.parent / "analysis" if output_dir is None else Path(output_dir)
    rows, flows, inputs, seen = [], [], [], set()
    for record in manifest["records"]:
        job = record["job"]
        if job["method"] not in METHODS:
            continue
        identity = (job["method"], job["regime"], job["scenario"], job["seed"])
        if (identity in seen or job["regime"] not in REGIMES or job["scenario"] not in SCENARIOS
                or job["seed"] not in SEEDS or record["status"] != "complete"):
            raise ValueError("Unexpected, duplicate or incomplete comparison record.")
        seen.add(identity)
        path = (ROOT / record["run_path"]).resolve()
        if not path.is_relative_to(ROOT):
            raise ValueError("Run path is outside the experiment workspace.")
        fingerprints = {}
        for name in ("run.json", "aggregation_diagnostics.json"):
            fingerprints[name] = _hash(path / name)
            if fingerprints[name] != record["artifact_sha256"][name]:
                raise ValueError(f"Saved audit input changed: {name}")
        metadata = _read(path / "run.json")
        if (metadata["official_test_used"] or metadata["scenario"]["name"] != job["scenario"]
                or metadata["config"]["development_seed"] != job["seed"]):
            raise ValueError("Saved run identity differs from its comparison job.")
        diagnostics = _read(path / "aggregation_diagnostics.json")
        by_round = {item["round"]: item["values"] for item in diagnostics}
        if len(by_round) != len(diagnostics) or set(by_round) != set(range(1, 25)):
            raise ValueError("Audit requires all24 rounds exactly once.")
        for round_number in ROUNDS:
            grouped, paths = round_group_rows(job, metadata["scenario"], round_number, by_round[round_number])
            rows.extend(grouped)
            flows.extend(paths)
        inputs.append({"job": job, "run_path": record["run_path"], "artifact_sha256": fingerprints})
    required = {(method, regime, scenario, seed) for method in ("NNM-RFA", "FedAvg", "RFA", "FLTrust")
                for regime in REGIMES for scenario in SCENARIOS for seed in SEEDS}
    if not required.issubset(seen):
        raise ValueError("Missing required NNM or original comparison runs.")
    keys = ("method", "regime", "scenario", "seed", "group")
    metrics = [name for name in rows[0] if name not in (*keys, "round")]
    per_seed = _summarize(rows, keys, metrics)
    seed_means = _summarize(per_seed, ("method", "regime", "scenario", "group"), metrics)
    for row in per_seed:
        row["aggregation_level"] = "per_seed_mean_over_rounds_5_24"
        row["seed_count"] = 1
        row["round_observations"] = row.pop("observations_averaged")
    for row in seed_means:
        row["aggregation_level"] = "available_seed_mean_of_round_means"
        row["seed"] = None
        row["seed_count"] = row.pop("observations_averaged")
        row["round_observations"] = row["seed_count"] * len(ROUNDS)
    flow_metrics = ["coefficient_mass"]
    flow_keys = ("method", "regime", "scenario", "seed", "origin_group", "anchor_group")
    flow_summary = _summarize(flows, flow_keys, flow_metrics)
    coverage = [{"method": method, "regime": regime, "scenario": scenario,
                 "seeds": [seed for seed in SEEDS if (method, regime, scenario, seed) in seen]}
                for method in METHODS for regime in REGIMES for scenario in SCENARIOS]
    result = {
        "analysis": "Post-pilot descriptive NNM coefficient and neighborhood audit",
        "scope": "Exploratory interpretation after the planned pilot; no training, model changes, threshold selection or gate changes.",
        "labels": "Saved active simulator events used offline only; clean runs contain no labelled attacker or drift group.",
        "units": "Means over rounds5-24 within each seed, then equal-weight means over available seeds; these are correlated observations, not independent replicates.",
        "coefficient_caveat": "Applied coefficients multiply raw client deltas. Coefficient mass is not contribution norm, attack detection, accuracy or attack damage. Clipped methods may have total coefficient mass below one; both raw mass and normalized share are retained.",
        "neighbor_caveat": "NNM's seven-neighbor sets include self. Same-group fractions describe membership, not reliability; weight from other-group anchors is an accounting identity, not evidence of successful poisoning.",
        "membership_extrema": "Summary minimum/maximum membership columns average each round's group minimum/maximum; they are not extrema across the whole run.",
        "source_sha256": _hash(__file__), "manifest_path": str(manifest_path),
        "manifest_sha256": _hash(manifest_path), "inputs": inputs, "coverage": coverage,
        "per_seed_group_means": per_seed, "available_seed_group_means": seed_means,
        "nnm_flow_means_by_seed": flow_summary, "group_round_rows": rows,
        "nnm_flow_round_rows": flows,
        "maximum_nnm_coefficient_expansion_error": max(row["coefficient_expansion_max_abs_error"]
            for row in rows if row["method"] == "NNM-RFA"),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write(output_dir / "nnm_mixing_audit.json", result)
    csv_rows = per_seed + seed_means
    with (output_dir / "nnm_mixing_audit.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_seed[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    arguments = parser.parse_args()
    result = analyse_mixing(arguments.manifest, arguments.output_dir)
    print(json.dumps({"audit_inputs": len(result["inputs"]),
                      "maximum_nnm_coefficient_expansion_error": result["maximum_nnm_coefficient_expansion_error"]}))
