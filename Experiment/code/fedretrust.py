"""FedReTrust draft v0.1: explicitly proposed, not a published baseline.

The server receives observations only. Reference probes use maximum clipped
sample shares, independently of retained trust. State commits are atomic.
"""
from dataclasses import asdict, dataclass, replace
from pathlib import Path
import copy
import math
import statistics
import time

import torch
from torch import nn

from experiment_core import AggregationResult, MnistCNN, file_sha256
from fltrust import FLTrustConfig, flatten_delta, train_root


VARIANTS = {"full", "no_history", "no_recovery", "no_repeat_memory",
            "ambiguous_as_adverse", "no_finite_allowance", "clipped_FedAvg", "mass_only"}


@dataclass(frozen=True)
class FedReTrustConfig:
    clip_multiplier: float = 2.0
    relative_loss_tolerance: float = .005
    severe_loss_multiplier: float = 4.0
    direction_history_decay: float = .5
    confirmation_rounds: int = 2
    episode_window_rounds: int = 8
    uncertain_cap: float = .5
    penalty_cap: float = .1
    restoration_increment: float = .25
    non_supportive_allowance: float = 4.0
    server_learning_rate: float = 1.0
    norm_epsilon: float = 1e-12
    loss_epsilon: float = 1e-8
    variant: str = "full"

    def __post_init__(self):
        for key, value in asdict(self).items():
            if key in {"variant", "confirmation_rounds", "episode_window_rounds"}:
                continue
            if isinstance(value, bool) or not math.isfinite(value):
                raise ValueError(f"Invalid finite scalar {key}")
        for key in ("confirmation_rounds", "episode_window_rounds"):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(f"Invalid positive integer {key}")
        for key in ("clip_multiplier", "relative_loss_tolerance", "non_supportive_allowance",
                    "server_learning_rate", "norm_epsilon", "loss_epsilon"):
            if getattr(self, key) <= 0:
                raise ValueError(f"Expected positive {key}")
        if (self.severe_loss_multiplier <= 1 or not 0 <= self.direction_history_decay < 1
                or not 0 < self.penalty_cap < self.uncertain_cap <= 1
                or not 0 < self.restoration_increment <= 1 or self.variant not in VARIANTS):
            raise ValueError("Invalid policy parameters or variant.")


@dataclass(frozen=True)
class TrustState:
    trust: float = 1.0
    label: str = "Stable"
    bad: int = 0
    good: int = 0
    confirmed: bool = False
    onsets: tuple = ()
    allowance: float = 4.0


def evidence_category(h, root, peer, history, clipped, usable, options):
    """Strict priority and tie rules of draft equation (evidence)."""
    if not usable:
        return "N", False, False
    if h is None or not math.isfinite(h):
        raise FloatingPointError("Nonfinite or missing reference loss.")
    if any(v is not None and (not math.isfinite(v) or not -1 <= v <= 1)
           for v in (root, peer, history)):
        raise FloatingPointError("Invalid directional evidence.")
    unusual = clipped or any(v is not None and v < 0 for v in (root, peer, history))
    discordant = any(v is not None and v < 0 for v in (root, peer))
    current_geometry = root is not None or peer is not None
    delta = options.relative_loss_tolerance
    if h > options.severe_loss_multiplier * delta:
        code = "X"
    elif h > delta and discordant:
        code = "B"
    elif h < -delta or (h <= delta and not unusual and current_geometry):
        code = "G"
    else:
        code = "U"
    return code, unusual, discordant


def transition(old, code, round_number, options):
    """Pure scalar transition; no tensor, data, scenario or intent access."""
    if code not in {"N", "X", "B", "G", "U"}:
        raise ValueError("Unknown evidence category.")
    if options.variant == "ambiguous_as_adverse" and code == "U":
        code = "B"
    k, window = options.confirmation_rounds, options.episode_window_rounds
    bad = min(k, old.bad + 1) if code in {"B", "X"} else 0
    good = min(k + window, old.good + 1) if code == "G" else 0
    confirmed = code in {"B", "X"} and (old.confirmed or bad >= k or code == "X")
    onsets = tuple(t for t in old.onsets if round_number-window+1 <= t <= round_number)
    new_episode = confirmed and not old.confirmed
    if new_episode:
        onsets += (round_number,)
    decision_m = 0 if options.variant == "no_repeat_memory" else len(onsets)
    trust, label = old.trust, old.label
    if code == "X":
        trust, label, provisional = 0., "Penalised", 0.
    elif code == "B" and confirmed:
        trust = min(trust, options.penalty_cap)
        label, provisional = "Penalised", trust
    elif code in {"B", "U"}:
        label, provisional = "Uncertain", min(trust, options.uncertain_cap)
    elif code == "G":
        eligible = good >= k + decision_m and options.variant != "no_recovery"
        if trust < 1 and eligible:
            trust = min(1., trust + options.restoration_increment / (1 + decision_m))
            label = "Stable" if trust == 1 else "Recovering"
        else:
            label = "Stable" if trust == 1 else "Penalised"
        provisional = trust
    else:
        provisional = 0.
    allowance = old.allowance
    if options.variant == "no_finite_allowance":
        q = provisional
    elif code in {"B", "U", "X"}:
        q = min(provisional, allowance)
        allowance = max(0., allowance-q)
    elif code == "G":
        q, allowance = provisional, options.non_supportive_allowance
    else:
        q = 0.
    new = TrustState(trust, label, bad, good, confirmed, onsets, allowance)
    return new, q, {"decision_evidence": code, "q_provisional": provisional,
                    "new_episode": new_episode, "episode_count": len(onsets),
                    "restoration_episode_count": decision_m}


def add_delta(broadcast, delta):
    result, offset = {}, 0
    for name, reference in broadcast.items():
        count = reference.numel()
        result[name] = (reference.double() + delta[offset:offset+count].reshape(reference.shape)).to(reference.dtype)
        offset += count
    if offset != delta.numel() or not all(torch.isfinite(v).all() for v in result.values()):
        raise FloatingPointError("Invalid reconstructed model.")
    return result


@torch.inference_mode()
def clipped_geometry(points, options):
    norms = torch.linalg.vector_norm(points, dim=1)
    if not torch.isfinite(norms).all():
        raise FloatingPointError("Update norm overflow.")
    radius = options.clip_multiplier * statistics.median(norms.tolist())
    if not math.isfinite(radius):
        raise FloatingPointError("Clipping radius overflow.")
    scales = torch.zeros_like(norms)
    nonzero = norms > options.norm_epsilon
    if radius > options.norm_epsilon:
        scales[nonzero] = (radius / norms[nonzero]).clamp(max=1)
    clipped = points * scales[:, None]
    clipped_norms = torch.linalg.vector_norm(clipped, dim=1)
    usable = clipped_norms > options.norm_epsilon
    units = torch.zeros_like(points)
    units[usable] = clipped[usable] / clipped_norms[usable, None]
    pair_cos = (units @ units.T).clamp(-1, 1)
    peers = []
    for i in range(len(points)):
        values = [float(pair_cos[i, j]) for j in range(len(points)) if j != i and usable[j]]
        peers.append(statistics.median(values) if usable[i] and values else None)
    return norms, radius, scales, clipped, clipped_norms, usable, units, peers


class FedReTrustAggregator:
    name = "FedReTrust"

    def __init__(self, reference, development_seed, sample_counts, options=None,
                 root_options=None, submission_counts=None):
        self.options = FedReTrustConfig() if options is None else options
        self.root_options = FLTrustConfig() if root_options is None else root_options
        if not isinstance(self.options, FedReTrustConfig) or not isinstance(self.root_options, FLTrustConfig):
            raise TypeError("Use typed policy and root configurations.")
        if self.root_options.global_learning_rate != 1:
            raise ValueError("Only FedReTrust's server rate may scale its aggregate.")
        self.sample_counts = dict(sample_counts)
        self.submission_counts = dict(sample_counts if submission_counts is None else submission_counts)
        if (not self.sample_counts or set(self.sample_counts) != set(self.submission_counts)
            or any(type(i) is not int or type(n) is not int or n <= 0 for mapping in
                   (self.sample_counts, self.submission_counts) for i, n in mapping.items())):
            raise ValueError("Expected server-known positive counts for identical integer IDs.")
        if reference is None and self.options.variant != "clipped_FedAvg":
            raise ValueError("A reserved server reference is required.")
        self.reference, self.development_seed = reference, development_seed
        self.name = "FedReTrust" if self.options.variant == "full" else f"FedReTrust_{self.options.variant}"
        self.settings = {
            **asdict(self.options), "root_options": asdict(self.root_options),
            "development_seed": development_seed, "server_known_sample_counts": self.sample_counts,
            "expected_submission_counts": self.submission_counts,
            "root_reference": copy.deepcopy(reference.metadata) if reference is not None else None,
            "rule_version": "mathematical_draft_v0.1",
            "implementation_sha256": file_sha256(__file__),
            "reference_helper_sha256": file_sha256(Path(__file__).with_name("fltrust.py")),
            "delta_convention": "trained_minus_broadcast; add",
            "coefficient_convention": "effective_coefficients_on_raw_client_deltas",
            "server_rate_location": "internal; also sets maximum-share probe; never use external adapter",
            "sample_denominator": "fixed original server-known counts; no trust-mass normalization",
        }
        self.reset()

    def reset(self):
        self.last_round, self.layout = 0, None
        self.states = {i: TrustState(allowance=self.options.non_supportive_allowance) for i in self.sample_counts}
        self.direction_memory = None

    def _reference_evidence(self, broadcast, clipped, p, usable, round_number):
        """Root training and independent full-reference maximum-share probes."""
        start = time.perf_counter()
        root, metrics = train_root(broadcast, self.reference, self.root_options,
                                   self.development_seed, round_number)
        root_seconds = time.perf_counter()-start
        losses = [None] * len(clipped)
        with torch.random.fork_rng(devices=[]), torch.inference_mode():
            model = MnistCNN()
            images, targets = self.reference.dataset.tensors
            model.eval()

            def loss_at(state):
                model.load_state_dict(state, strict=True)
                total = 0.
                # Fixed deterministic batches reduce peak reference memory.
                for offset in range(0, len(images), 100):
                    y = targets[offset:offset+100]
                    value = nn.functional.cross_entropy(model(images[offset:offset+100]), y, reduction="sum")
                    if not torch.isfinite(value):
                        raise FloatingPointError("Nonfinite reference probe.")
                    total += float(value)
                return total / len(images)

            baseline = loss_at(broadcast)
            for i in range(len(clipped)):
                if usable[i]:
                    losses[i] = loss_at(add_delta(broadcast, self.options.server_learning_rate*p[i]*clipped[i]))
        metrics = {**metrics, "root_training_seconds": root_seconds,
                   "reference_forward_examples": len(images)*(1+int(usable.sum())),
                   "reference_baseline_loss": baseline}
        return root, baseline, losses, metrics

    def aggregate(self, submissions, global_state, round_number):
        if type(round_number) is not int or round_number != self.last_round+1:
            raise ValueError("Rounds must be contiguous from one; reset between runs.")
        ordered = sorted(submissions, key=lambda s: s.client_id)
        ids = tuple(s.client_id for s in ordered)
        if len(set(ids)) != len(ids) or set(ids) != set(self.sample_counts):
            raise ValueError("Submission membership must match the known clients.")
        if any(type(s.num_examples) is not int or s.num_examples != self.submission_counts[s.client_id]
               for s in ordered):
            raise ValueError("Submitted sample counts differ from saved training selections.")
        points = torch.stack([flatten_delta(s.state, global_state) for s in ordered])
        layout = tuple((name, tuple(value.shape), str(value.dtype)) for name, value in global_state.items())
        if self.layout is not None and self.layout != layout:
            raise ValueError("Model layout changed.")
        o = self.options
        norms, radius, scales, clipped, cnorms, usable, units, peers = clipped_geometry(points, o)
        p = torch.tensor([self.sample_counts[i] for i in ids], dtype=torch.float64)
        p /= p.sum()
        memory = torch.zeros_like(points) if self.direction_memory is None else self.direction_memory
        root, baseline, losses = torch.zeros_like(points[0]), None, [None]*len(ids)
        metrics = {"root_optimizer_steps": 0, "root_example_visits": 0,
                   "reference_forward_examples": 0, "reference_baseline_loss": None,
                   "root_training_seconds": 0., "reference_skip_reason": "no_usable_updates"}
        if usable.any() and o.variant != "clipped_FedAvg":
            root, baseline, losses, metrics = self._reference_evidence(global_state, clipped, p, usable, round_number)
            if (root.shape != points[0].shape or root.device.type != "cpu"
                    or not torch.isfinite(root).all() or not math.isfinite(baseline) or baseline < 0):
                raise FloatingPointError("Invalid reference outputs.")
        elif o.variant == "clipped_FedAvg":
            metrics["reference_skip_reason"] = "clipped_FedAvg_control"
        root = root.double()
        root_norm = float(torch.linalg.vector_norm(root))
        if not math.isfinite(root_norm):
            raise FloatingPointError("Reference norm overflow.")
        next_states, rows, policy_q = {}, {}, []
        for j, i in enumerate(ids):
            root_cos = float(torch.dot(units[j], root/root_norm).clamp(-1, 1)) if usable[j] and root_norm > o.norm_epsilon else None
            memory_norm = float(torch.linalg.vector_norm(memory[j]))
            hist_cos = float(torch.dot(units[j], memory[j]/memory_norm).clamp(-1, 1)) if usable[j] and memory_norm > o.norm_epsilon else None
            if o.variant == "no_history":
                hist_cos = None
            h = (losses[j]-baseline)/max(baseline, o.loss_epsilon) if usable[j] and o.variant != "clipped_FedAvg" else None
            if o.variant == "clipped_FedAvg":
                code, unusual, discord = "G" if usable[j] else "N", False, False
            else:
                code, unusual, discord = evidence_category(h, root_cos, peers[j], hist_cos, bool(scales[j] < 1), bool(usable[j]), o)
            if o.variant in {"no_history", "clipped_FedAvg"}:
                q = {"G": 1., "U": o.uncertain_cap, "B": o.penalty_cap, "X": 0., "N": 0.}[code]
                new = TrustState(allowance=o.non_supportive_allowance)
                detail = {"decision_evidence": code, "q_provisional": q, "new_episode": False,
                          "episode_count": 0, "restoration_episode_count": 0}
            else:
                new, q, detail = transition(self.states[i], code, round_number, o)
            next_states[i], policy_q = new, policy_q + [q]
            rows[str(i)] = {**asdict(new), **detail, "evidence": code,
                           "persistent_policy_state_enabled": o.variant not in {"no_history", "clipped_FedAvg"},
                           "relative_reference_loss": h, "root_cosine": root_cos,
                           "peer_cosine": peers[j], "history_cosine": hist_cos,
                           "unusual": unusual, "discordant": discord, "policy_multiplier": q,
                           "clip_factor": float(scales[j]), "clipped_update_norm": float(cnorms[j])}
            if o.variant in {"no_history", "no_finite_allowance", "clipped_FedAvg"}:
                rows[str(i)]["allowance"] = None
            if o.variant in {"no_history", "clipped_FedAvg"}:
                rows[str(i)]["trust"] = rows[str(i)]["label"] = None
        qvec = torch.tensor(policy_q, dtype=torch.float64)
        mass = float(torch.dot(p, qvec))
        applied_q = torch.full_like(qvec, mass) if o.variant == "mass_only" else qvec
        coefficients = o.server_learning_rate*p*applied_q*scales
        total = (points*coefficients[:, None]).sum(0)
        state = add_delta(global_state, total)
        next_memory = o.direction_history_decay*memory + (1-o.direction_history_decay)*units
        if o.variant in {"no_history", "clipped_FedAvg"}:
            next_memory = torch.zeros_like(points)
        if not torch.isfinite(next_memory).all() or not torch.isfinite(coefficients).all():
            raise FloatingPointError("Nonfinite policy result.")
        for j, i in enumerate(ids):
            rows[str(i)].update({"applied_multiplier": float(applied_q[j]),
                                 "applied_delta_coefficient": float(coefficients[j]),
                                 "sample_share": float(p[j]), "raw_update_norm": float(norms[j])})
        diagnostics = {**metrics, "clip_radius": radius, "root_update_norm": root_norm,
                       "trust_mass": mass, "clipped_coefficient_mass": o.server_learning_rate*mass,
                       "applied_effective_coefficient_mass": float(coefficients.sum()),
                       "coefficient_convention": "effective_coefficients_on_raw_client_deltas",
                       "persistent_state_tensor_bytes": next_memory.numel()*next_memory.element_size(),
                       "client_metrics": rows}
        # Nothing above mutates persistent state, the broadcast or submissions.
        self.states, self.direction_memory = next_states, next_memory
        self.last_round, self.layout = round_number, layout
        return AggregationResult(state, {i: float(coefficients[j]) for j, i in enumerate(ids)}, diagnostics)
