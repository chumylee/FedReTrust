"""Passive update diagnostics for the ten-class MnistCNN experiment.

The observer has no data, scenario, simulator-label or evaluator interface. It
does not alter aggregation: all scores are saved for subsequent offline study.
Layer slices are compared with the corresponding slices of ONE current-round,
sample-weighted RFA reference, not independently fitted layer medians.
"""
from dataclasses import asdict
from pathlib import Path
import copy
import time

import torch

from experiment_core import file_sha256
from rfa import RFAConfig, _flatten_submissions, smoothed_weiszfeld


EPSILON = 1e-12
LAYER_NAMES = ("features.0", "features.3", "classifier.1", "classifier.3")
PARAMETER_NAMES = tuple(f"{layer}.{part}" for layer in LAYER_NAMES for part in ("weight", "bias"))
SCALAR_FEATURES = (
    "whole_cosine", "penultimate_cosine", "head_cosine",
    "head_class_cosine_mean", "head_class_cosine_min", "head_relative_norm",
    "temporal_cosine", "temporal_relative_change", "layer_disagreement",
)


def _schema(broadcast):
    if set(broadcast) != set(PARAMETER_NAMES):
        raise ValueError("UpdateObserver expects the four MnistCNN weight/bias modules.")
    for layer in LAYER_NAMES:
        weight, bias = broadcast[f"{layer}.weight"], broadcast[f"{layer}.bias"]
        rank = 4 if layer.startswith("features.") else 2
        if weight.ndim != rank or bias.ndim != 1 or weight.shape[0] != bias.shape[0]:
            raise ValueError(f"Incompatible MnistCNN module shapes: {layer}")
    if broadcast["classifier.3.weight"].shape[0] != 10:
        raise ValueError("UpdateObserver expects all ten output classes.")
    if broadcast["classifier.3.weight"].shape[1] != broadcast["classifier.1.weight"].shape[0]:
        raise ValueError("Output and penultimate layer widths must agree.")
    if broadcast["features.3.weight"].shape[1] != broadcast["features.0.weight"].shape[0]:
        raise ValueError("Convolutional feature widths must agree.")
    ordered = {name: broadcast[name] for name in PARAMETER_NAMES}
    slices, offset = {}, 0
    for name, tensor in ordered.items():
        slices[name] = slice(offset, offset + tensor.numel())
        offset += tensor.numel()
    return ordered, slices


def _cosine(left, right):
    # A zero vector has no direction. Keep that uncertainty in offline records.
    left_norm = float(torch.linalg.vector_norm(left))
    right_norm = float(torch.linalg.vector_norm(right))
    if left_norm == 0.0 or right_norm == 0.0:
        return None
    value = float(torch.dot(left / max(left_norm, EPSILON), right / max(right_norm, EPSILON)))
    return max(-1.0, min(1.0, value))


def _layer_vector(vector, slices, layer):
    return torch.cat((vector[slices[f"{layer}.weight"]], vector[slices[f"{layer}.bias"]]))


def _class_vectors(vector, slices, head_shape):
    weight = vector[slices["classifier.3.weight"]].reshape(head_shape)
    bias = vector[slices["classifier.3.bias"]]
    return torch.cat((weight, bias[:, None]), dim=1)


@torch.inference_mode()
def observe_updates(submissions, broadcast, previous_deltas=None):
    """Return JSON-friendly observations and detached current deltas by client ID.

    ``previous_deltas`` must contain only earlier observations of the SAME client
    and use PARAMETER_NAMES order. The wrapper enforces forward round order. A
    caller making independent checkpoint probes should omit previous_deltas.
    Inputs and any supplied history are read-only. No randomness is consumed.
    """
    started = time.perf_counter()
    ordered, slices = _schema(broadcast)
    if any(type(s.client_id) is not int for s in submissions):
        raise ValueError("Client IDs must be integers.")
    points = _flatten_submissions(submissions, ordered)
    reference, _, _ = smoothed_weiszfeld(points, [s.num_examples for s in submissions], RFAConfig())
    previous_deltas = {} if previous_deltas is None else previous_deltas
    reference_layers = [_layer_vector(reference, slices, layer) for layer in LAYER_NAMES]
    reference_classes = _class_vectors(reference, slices, ordered["classifier.3.weight"].shape)
    rows, current = {}, {}
    for submission, point in zip(submissions, points):
        layers = [_layer_vector(point, slices, layer) for layer in LAYER_NAMES]
        layer_cosines = [_cosine(value, ref) for value, ref in zip(layers, reference_layers)]
        classes = _class_vectors(point, slices, ordered["classifier.3.weight"].shape)
        class_cosines = [_cosine(value, ref) for value, ref in zip(classes, reference_classes)]
        previous = previous_deltas.get(submission.client_id)
        temporal_cosine = temporal_change = None
        if previous is not None:
            if (previous.shape != point.shape or previous.device.type != "cpu"
                    or not previous.is_floating_point() or not torch.isfinite(previous).all()):
                raise ValueError("Previous client delta must be a compatible finite CPU vector.")
            previous = previous.detach().double()
            temporal_cosine = _cosine(point, previous)
            previous_norm = float(torch.linalg.vector_norm(previous))
            if previous_norm != 0.0:
                temporal_change = float(torch.linalg.vector_norm(point - previous)) / (previous_norm + EPSILON)
        whole_norm = float(torch.linalg.vector_norm(point))
        classes_defined = all(value is not None for value in class_cosines)
        layers_defined = all(value is not None for value in layer_cosines)
        rows[str(submission.client_id)] = {
            "whole_cosine": _cosine(point, reference),
            "penultimate_cosine": layer_cosines[2],
            "head_cosine": layer_cosines[3],
            "head_class_cosine_mean": sum(class_cosines) / 10 if classes_defined else None,
            "head_class_cosine_min": min(class_cosines) if classes_defined else None,
            "head_relative_norm": (float(torch.linalg.vector_norm(layers[3])) / (whole_norm + EPSILON)
                                   if whole_norm != 0.0 else None),
            "temporal_cosine": temporal_cosine,
            "temporal_relative_change": temporal_change,
            "layer_disagreement": max(layer_cosines) - min(layer_cosines) if layers_defined else None,
            "layer_cosines": layer_cosines,
            "head_class_cosines": class_cosines,
        }
        current[submission.client_id] = point.detach().clone()
    return {
        "client_order": [s.client_id for s in submissions],
        "feature_rows": rows,
        "diagnostics_seconds": time.perf_counter() - started,
    }, current


class UpdateObserver:
    """Delegate aggregation exactly while retaining causal, passive observations."""

    def __init__(self, base, checkpoint_rounds=(5, 8, 12, 20)):
        checkpoint_rounds = tuple(checkpoint_rounds)
        if any(type(value) is not int or value < 1 for value in checkpoint_rounds):
            raise ValueError("Checkpoint rounds must be positive integers.")
        self.base = base
        self.name = base.name
        self.checkpoint_rounds = tuple(sorted(set(checkpoint_rounds)))
        self.settings = copy.deepcopy(base.settings)
        self.settings["update_observation"] = {
            "schema_version": 1,
            "mode": "passive_offline_diagnostics_only",
            "implementation_sha256": file_sha256(Path(__file__)),
            "layer_order": list(LAYER_NAMES),
            "scalar_features": list(SCALAR_FEATURES),
            "output_classes": list(range(10)),
            "reference": "current_whole_delta_sample_weighted_RFA",
            "reference_rfa": asdict(RFAConfig()),
            "epsilon": EPSILON,
            "zero_norm_cosine": None,
            "undefined_component_summary": None,
            "zero_denominator_relative_norm": None,
            "accumulation_dtype": "float64",
            "delta_convention": "trained_minus_broadcast",
            "temporal_reference": "previous_observed_delta_of_same_client",
            "first_observation_temporal_values": None,
            "checkpoint_rounds": list(self.checkpoint_rounds),
            "checkpoint_state": "broadcast_before_aggregation",
        }
        self.checkpoints = {}
        self._previous_deltas = {}
        self._last_round = None

    def reset(self):
        if hasattr(self.base, "reset"):
            self.base.reset()
        self.checkpoints.clear()
        self._previous_deltas.clear()
        self._last_round = None

    def aggregate(self, submissions, broadcast, round_number):
        if type(round_number) is not int or round_number < 1:
            raise ValueError("Observation round must be a positive integer.")
        if self._last_round is not None and round_number <= self._last_round:
            raise ValueError("Observation rounds must increase; reset before starting a new run.")
        observation, current = observe_updates(submissions, broadcast, self._previous_deltas)
        checkpoint = None
        if round_number in self.checkpoint_rounds:
            checkpoint = {name: tensor.detach().clone() for name, tensor in broadcast.items()}
        # Deliberately outside inference_mode: the wrapper must not change the
        # execution context or inputs seen by the underlying aggregator.
        result = self.base.aggregate(submissions, broadcast, round_number)
        if "update_observation" in result.diagnostics:
            raise ValueError("The wrapped aggregator already supplies update_observation diagnostics.")
        result.diagnostics["update_observation"] = observation
        self._previous_deltas.update(current)
        self._last_round = round_number
        if checkpoint is not None:
            self.checkpoints[round_number] = checkpoint
        return result
