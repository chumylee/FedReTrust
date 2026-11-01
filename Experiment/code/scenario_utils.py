"""Recorded client interventions for notebook 04 and later matched experiments.

MNIST-C shot-noise and stripe recipes are adapted from Google Research's
corruptions.py, copyright 2018 Google LLC, licensed under Apache-2.0.
The pinned source, licence and adaptation notes are in resources/mnist_c_reference.
"""

from dataclasses import asdict, dataclass
import hashlib
import math
import re

import numpy as np
import torch
from torch.utils.data import Dataset


def seed_for(seed, stream, *parts):
    """Stable private RNG streams, independent of method and scenario names."""
    text = ":".join(map(str, (seed, stream, *parts)))
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big") % (2**63 - 1)


def mnist_c_corruption(image, name, severity=3, rng=None):
    """Return float32 pixels in [0,255], without modifying the source image."""
    if not isinstance(severity, int) or not 1 <= severity <= 5:
        raise ValueError("Severity must be an integer from 1 to 5.")
    image = np.asarray(image)
    if image.shape != (28, 28) or not np.isfinite(image).all() or image.min() < 0 or image.max() > 255:
        raise ValueError("Expected a finite 28x28 MNIST image in [0,255].")
    if name == "shot_noise":
        if rng is None:
            raise ValueError("Supply an explicit RNG for shot noise.")
        rate = [60, 25, 12, 5, 3][severity - 1]
        pixels = np.asarray(image) / 255.0
        return (np.clip(rng.poisson(pixels * rate) / float(rate), 0, 1) * 255).astype(np.float32)
    if name == "stripe":
        pixels = np.array(image, dtype=np.float32, copy=True)
        pixels[:, :7] = 255.0 - pixels[:, :7]
        pixels[:, 21:] = 255.0 - pixels[:, 21:]
        return pixels
    raise ValueError(f"Unsupported MNIST-C corruption: {name}")


@dataclass(frozen=True)
class Event:
    kind: str
    clients: tuple[int, ...]
    windows: tuple[tuple[int, int], ...]
    corruption: str | None = None
    severity: int = 3
    ramp_rounds: int = 0
    intent: str = "honest"

    def __post_init__(self):
        if self.kind not in {"input_change", "label_flip", "gaussian_update"}:
            raise ValueError("Unknown event kind.")
        if not self.clients or len(set(self.clients)) != len(self.clients) or any(type(c) is not int or c < 0 for c in self.clients):
            raise ValueError("Events need distinct nonnegative client IDs.")
        if not self.windows:
            raise ValueError("Events need at least one inclusive active window.")
        previous_end = 0
        for start, end in self.windows:
            if not isinstance(start, int) or not isinstance(end, int) or start <= previous_end or end < start:
                raise ValueError("Windows must be positive, ordered and non-overlapping.")
            previous_end = end
        if self.intent not in {"honest", "malicious"}:
            raise ValueError("Intent is simulator truth: honest or malicious.")
        if self.kind == "input_change":
            if self.corruption not in {"shot_noise", "stripe"}:
                raise ValueError("Choose a supported MNIST-C input corruption.")
        elif self.corruption is not None or self.ramp_rounds != 0 or self.intent != "malicious":
            raise ValueError("Attack events use malicious intent, no input corruption and no ramp.")
        if not isinstance(self.ramp_rounds, int) or self.ramp_rounds < 0:
            raise ValueError("Ramp length must be a nonnegative integer.")
        if not isinstance(self.severity, int) or not 1 <= self.severity <= 5:
            raise ValueError("Severity must be from 1 to 5.")

    def strength(self, round_number):
        for start, end in self.windows:
            if start <= round_number <= end:
                return min(1.0, (round_number - start + 1) / self.ramp_rounds) if self.ramp_rounds else 1.0
        return 0.0


@dataclass(frozen=True)
class Scenario:
    name: str
    events: tuple[Event, ...] = ()

    def to_dict(self):
        return asdict(self)

    def validate(self, client_ids, rounds):
        if not re.fullmatch(r"[a-z][a-z0-9_]*", self.name) or any(c not in client_ids for event in self.events for c in event.clients):
            raise ValueError("Scenario name or affected clients are invalid.")
        if any(end > rounds for event in self.events for _, end in event.windows):
            raise ValueError("Scenario window extends beyond the run.")
        for client in client_ids:
            for round_number in range(1, rounds + 1):
                active = self.active(client, round_number)
                kinds = [event.kind for event in active]
                if len(kinds) != len(set(kinds)):
                    raise ValueError("Overlapping interventions of the same kind are ambiguous.")

    def active(self, client_id, round_number):
        return tuple(event for event in self.events if client_id in event.clients and event.strength(round_number) > 0)


def scenario_catalog(rounds=6):
    """Explicit development schedules; 24-round schedules are provisional."""
    if rounds == 6:
        onset, stop, ramp, recurring = 3, 4, 4, ((2, 3), (5, 6))
        gradual_onset = 2
    elif rounds == 24:
        onset, stop, ramp, recurring = 9, 16, 8, ((5, 8), (13, 16), (21, 24))
        gradual_onset = 9
    else:
        raise ValueError("Use the explicit 6- or 24-round catalogue, or construct a Scenario.")
    clients = (0, 1, 2)
    return {
        "clean": Scenario("clean"),
        "abrupt_shot_noise": Scenario("abrupt_shot_noise", (
            Event("input_change", clients, ((onset, rounds),), corruption="shot_noise", severity=3),)),
        "gradual_stripe": Scenario("gradual_stripe", (
            Event("input_change", clients, ((gradual_onset, rounds),), corruption="stripe", ramp_rounds=ramp),)),
        "label_flip": Scenario("label_flip", (
            Event("label_flip", clients, ((onset, rounds),), intent="malicious"),)),
        "gaussian_update": Scenario("gaussian_update", (
            Event("gaussian_update", clients, ((onset, rounds),), intent="malicious"),)),
        "attack_stops": Scenario("attack_stops", (
            Event("label_flip", clients, ((onset, stop),), intent="malicious"),)),
        "recurring_attack": Scenario("recurring_attack", (
            Event("label_flip", clients, recurring, intent="malicious"),)),
    }


class ScenarioDataset(Dataset):
    """Client-side or evaluation-only view of original official-training indices."""

    def __init__(self, pool, indices, mean, std, seed, corruption=None, severity=3,
                 fraction=0.0, flip_labels=False):
        if not indices or not 0 <= fraction <= 1 or std <= 0:
            raise ValueError("Invalid dataset view.")
        self.pool, self.indices = pool, tuple(int(i) for i in indices)
        if len(set(self.indices)) != len(self.indices) or any(i < 0 or i >= len(pool.data) for i in self.indices):
            raise ValueError("View indices must be distinct valid original image indices.")
        self.mean, self.std, self.seed = mean, std, seed
        self.corruption, self.severity, self.flip_labels = corruption, severity, flip_labels
        if fraction > 0 and corruption is None:
            raise ValueError("A changed image fraction requires a corruption.")
        indices_hash = hashlib.sha256(np.asarray(self.indices, dtype=np.int64).tobytes()).hexdigest()
        order = np.random.default_rng(seed_for(seed, "image_selection", indices_hash)).permutation(len(self.indices))
        self.changed = np.zeros(len(self.indices), dtype=bool)
        self.changed[order[:math.floor(fraction * len(self.indices))]] = True

    def __len__(self):
        return len(self.indices)

    def pixels_and_label(self, position):
        index = self.indices[position]
        pixels = self.pool.data[index].numpy()
        if self.changed[position]:
            rng = np.random.default_rng(seed_for(self.seed, self.corruption, index))
            pixels = mnist_c_corruption(pixels, self.corruption, self.severity, rng)
        else:
            pixels = np.array(pixels, dtype=np.float32, copy=True)
        label = int(self.pool.targets[index])
        return pixels, 9 - label if self.flip_labels else label

    def __getitem__(self, position):
        pixels, label = self.pixels_and_label(position)
        image = torch.from_numpy(pixels).unsqueeze(0).div(255.0)
        image.sub_(self.mean).div_(self.std)
        return image, label


def client_view(pool, indices, normalization, seed, scenario, client_id, round_number):
    active = scenario.active(client_id, round_number)
    changes = [event for event in active if event.kind == "input_change"]
    event = changes[0] if changes else None
    return ScenarioDataset(pool, indices, normalization["mean"][0], normalization["std"][0], seed,
                           corruption=event.corruption if event else None,
                           severity=event.severity if event else 3,
                           fraction=event.strength(round_number) if event else 0.0,
                           flip_labels=any(item.kind == "label_flip" for item in active))


def gaussian_update_corruption(local_state, global_state, seed, client_id, round_number):
    """RFA Gaussian corruption: local model + N(0, Var(local-global) I)."""
    names = list(local_state)
    delta = torch.cat([(local_state[name] - global_state[name]).flatten().double() for name in names])
    sigma = float(delta.var(correction=0).sqrt())
    rng = np.random.default_rng(seed_for(seed, "gaussian_update", round_number, client_id))
    noise = rng.normal(0.0, sigma, size=delta.numel())
    result, offset = {}, 0
    for name in names:
        tensor = local_state[name]
        piece = torch.from_numpy(noise[offset:offset + tensor.numel()]).reshape(tensor.shape)
        result[name] = (tensor.double() + piece).to(tensor.dtype)
        offset += tensor.numel()
    return result, sigma


def truth_row(scenario, client_id, round_number, view):
    """Evaluation-only information. Never part of a server submission."""
    events = [event for event in scenario.events if client_id in event.clients]
    active = scenario.active(client_id, round_number)
    return {
        "round": round_number, "client": client_id,
        "honest_drift_client": any(e.kind == "input_change" and e.intent == "honest" for e in events),
        "malicious_client": any(e.intent == "malicious" for e in events),
        "attack_active": any(e.intent == "malicious" for e in active),
        "input_changed_examples": int(view.changed.sum()),
        "labels_changed_examples": len(view) if view.flip_labels else 0,
        "update_manipulated": any(e.kind == "gaussian_update" for e in active),
    }
