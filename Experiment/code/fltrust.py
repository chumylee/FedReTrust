"""FLTrust direction scoring and norm normalization, Cao et al., NDSS 2021.

Core rule: section IV-B, equations (2)-(4), and Algorithm 2. We use descent
deltas (trained minus broadcast) and ADD the aggregate to the broadcast.
Common training settings and undefined-case policies are recorded explicitly.
"""
from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import json
import math
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset

from experiment_core import AggregationResult, MnistCNN, clone_state, file_sha256, state_sha256
from scenario_utils import ScenarioDataset, seed_for


@dataclass(frozen=True)
class FLTrustConfig:
    root_steps: int = 8
    root_batch_size: int = 64
    root_learning_rate: float = 0.01
    root_momentum: float = 0.9
    global_learning_rate: float = 1.0
    epsilon: float = 1e-12

    def __post_init__(self):
        if type(self.root_steps) is not int or self.root_steps < 1:
            raise ValueError("Root steps must be positive.")
        if type(self.root_batch_size) is not int or self.root_batch_size < 1:
            raise ValueError("Root batch size must be positive.")
        for name in ("root_learning_rate", "global_learning_rate", "epsilon"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        if not 0 <= self.root_momentum < 1:
            raise ValueError("Root momentum must be in [0,1).")


@dataclass(frozen=True)
class RootReference:
    dataset: TensorDataset
    metadata: dict


def build_root_reference(data):
    """Copy ONLY the saved clean server images; do not hand the full pool to FLTrust."""
    indices = data.manifest["indices"]["server_reference"]
    client_indices = {i for group in data.manifest["indices"]["clients"].values() for i in group}
    validation_indices = set(data.manifest["indices"]["development_validation"])
    if not indices or len(set(indices)) != len(indices) or set(indices) & (client_indices | validation_indices):
        raise ValueError("Trusted reference indices must be distinct and disjoint from all client/validation data.")
    normalization = data.manifest["normalization"]
    clean = ScenarioDataset(data.pool, indices, normalization["mean"][0], normalization["std"][0], seed=0)
    items = [clean[i] for i in range(len(clean))]
    images = torch.stack([image for image, _ in items])
    targets = torch.tensor([label for _, label in items], dtype=torch.int64)
    digest = hashlib.sha256(images.numpy().tobytes() + targets.numpy().tobytes()).hexdigest()
    metadata = {"examples": len(indices), "original_indices": list(indices), "index_space": "official_train",
                "tensor_sha256": digest, "class_counts": {str(i): int((targets == i).sum()) for i in range(10)},
                "partition_sha256": data.manifest_sha256, "disjoint_from_clients_and_validation": True}
    return RootReference(TensorDataset(images, targets), metadata)


def flatten_delta(state, broadcast):
    if set(state) != set(broadcast) or not broadcast:
        raise ValueError("State keys must match the nonempty broadcast.")
    pieces = []
    for name, reference in broadcast.items():
        tensor = state[name]
        if tensor.shape != reference.shape or tensor.dtype != reference.dtype:
            raise ValueError(f"Incompatible parameter: {name}")
        if tensor.device.type != "cpu" or reference.device.type != "cpu" or not tensor.is_floating_point():
            raise ValueError("Expected floating-point CPU parameters.")
        if not torch.isfinite(tensor).all() or not torch.isfinite(reference).all():
            raise FloatingPointError("Non-finite model state.")
        # Convert before subtraction to avoid low-precision accumulation errors.
        pieces.append((tensor.double() - reference.double()).flatten())
    return torch.cat(pieces)


def fltrust_rule(submissions, broadcast, root_delta, global_learning_rate=1.0, epsilon=1e-12):
    """Equations (2)-(4); coefficients are trust shares BEFORE norm rescaling.

    Zero client updates get zero trust. A zero reference or no positive trust
    keeps the broadcast unchanged. These are explicit numerical policies for
    cases in which the published divisions are undefined, not a new defence.
    """
    if not submissions or len({s.client_id for s in submissions}) != len(submissions):
        raise ValueError("Provide distinct client submissions.")
    if not math.isfinite(global_learning_rate) or global_learning_rate <= 0 or not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("Invalid global learning rate or numerical tolerance.")
    if root_delta.device.type != "cpu" or root_delta.ndim != 1 or not torch.isfinite(root_delta).all():
        raise ValueError("Root delta must be a finite CPU vector.")
    root_delta = root_delta.double()
    deltas = [flatten_delta(s.state, broadcast) for s in submissions]
    if root_delta.shape != deltas[0].shape:
        raise ValueError("Root/client update dimensions differ.")
    root_norm = float(torch.linalg.vector_norm(root_delta))
    norms = [float(torch.linalg.vector_norm(delta)) for delta in deltas]
    cosines, scores, scales = [], [], []
    for delta, norm in zip(deltas, norms):
        valid = root_norm > epsilon and norm > epsilon
        cosine = max(-1.0, min(1.0, float(torch.dot(delta / norm, root_delta / root_norm)))) if valid else 0.0
        cosines.append(cosine)
        scores.append(max(0.0, cosine))
        scales.append(root_norm / norm if valid else 0.0)
    trust_sum = math.fsum(scores)
    reason = "zero_reference_update" if root_norm <= epsilon else "no_positive_trust" if trust_sum <= epsilon else None
    weights = [score / trust_sum if reason is None else 0.0 for score in scores]
    aggregate = torch.zeros_like(root_delta)
    if reason is None:
        for delta, norm, weight in zip(deltas, norms, weights):
            if weight:
                # Unit direction first: stable even when the norm ratio is large.
                aggregate.add_(delta / norm, alpha=weight * root_norm)
    result, offset = {}, 0
    for name, reference in broadcast.items():
        piece = aggregate[offset:offset + reference.numel()].reshape(reference.shape)
        result[name] = (reference.double() + global_learning_rate * piece).to(reference.dtype)
        offset += reference.numel()
    client_metrics = {}
    for s, norm, cosine, score, scale, weight in zip(submissions, norms, cosines, scores, scales, weights):
        client_metrics[str(s.client_id)] = {
            "cosine_similarity": cosine, "trust_score": score, "raw_update_l2_norm": norm,
            "norm_scale": scale, "normalized_update_l2_norm": root_norm if norm > epsilon and root_norm > epsilon else 0.0,
            "trust_weight": weight, "effective_update_coefficient": global_learning_rate * weight * scale,
            "excluded": weight == 0.0,
        }
    diagnostics = {"root_update_l2_norm": root_norm, "trust_sum": trust_sum, "fallback_reason": reason,
                   "aggregate_update_l2_norm": float(torch.linalg.vector_norm(aggregate)), "client_metrics": client_metrics}
    return AggregationResult(result, {s.client_id: weight for s, weight in zip(submissions, weights)}, diagnostics)


def train_root(broadcast, reference, options, development_seed, round_number):
    if options.root_batch_size > len(reference.dataset):
        raise ValueError("Root batches sample without replacement within each batch; batch exceeds root size.")
    images, targets = reference.dataset.tensors
    batch_seed = seed_for(development_seed, "fltrust_root_batches", round_number)
    generator = torch.Generator().manual_seed(batch_seed)
    # Creating a reference model must not consume a client's global RNG stream.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed_for(development_seed, "fltrust_root_model", round_number))
        model = MnistCNN()
        model.load_state_dict(broadcast, strict=True)
        initial_hash = state_sha256(clone_state(model))
        optimizer = torch.optim.SGD(model.parameters(), lr=options.root_learning_rate, momentum=options.root_momentum)
        assert not optimizer.state
        criterion = nn.CrossEntropyLoss()
        model.train()
        losses = []
        for _ in range(options.root_steps):
            positions = torch.randperm(len(reference.dataset), generator=generator)[:options.root_batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images[positions]), targets[positions])
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite root training loss.")
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        delta = flatten_delta(clone_state(model), broadcast)
    return delta, {"root_online_train_loss": float(np.mean(losses)), "root_optimizer_steps": options.root_steps,
                   "root_example_visits": options.root_steps * options.root_batch_size, "root_batch_seed": batch_seed,
                   "root_broadcast_state_sha256": initial_hash}


class FLTrustAggregator:
    name = "FLTrust"

    def __init__(self, reference, development_seed, options=None):
        self.reference = reference
        self.development_seed = development_seed
        self.options = FLTrustConfig() if options is None else options
        self.settings = {
            **asdict(self.options), "development_seed": development_seed,
            "root_reference": json.loads(json.dumps(reference.metadata)),
            "root_sampling": "independent random minibatches, without replacement within each minibatch",
            "delta_convention": "trained_minus_broadcast; add_aggregate",
            "weighting": "ReLU_cosine_only; no_sample_count_factor; no_additive_root_update",
            "zero_policy": "zero_client_trust; zero_reference_or_no_positive_trust_keep_global",
            "implementation_sha256": file_sha256(Path(__file__)),
            "paper_sha256": file_sha256(Path(__file__).parents[1] / "resources/FLTrust_Cao_2021.pdf"),
        }

    def aggregate(self, submissions, global_state, round_number):
        root_start = time.perf_counter()
        root_delta, root_metrics = train_root(global_state, self.reference, self.options, self.development_seed, round_number)
        root_seconds = time.perf_counter() - root_start
        rule_start = time.perf_counter()
        result = fltrust_rule(submissions, global_state, root_delta,
                              self.options.global_learning_rate, self.options.epsilon)
        result.diagnostics.update(root_metrics)
        result.diagnostics["root_training_seconds"] = root_seconds
        result.diagnostics["rule_seconds"] = time.perf_counter() - rule_start
        return result
