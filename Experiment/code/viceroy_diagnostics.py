"""Declared server-step controls for diagnosis; not additional main baselines.

The original aggregator determines scores and updates its own history. Only
the applied server update is scaled here, without changing that arithmetic.
"""
from pathlib import Path
import math

import torch

from experiment_core import AggregationResult, file_sha256


class ServerStepControl:
    def __init__(self, base, multiplier=1.0, normalize_mass=False):
        if isinstance(multiplier, bool) or not math.isfinite(multiplier) or multiplier <= 0:
            raise ValueError("Server multiplier must be finite and positive.")
        if type(normalize_mass) is not bool:
            raise ValueError("normalize_mass must be boolean.")
        self.base = base
        self.multiplier = float(multiplier)
        self.normalize_mass = normalize_mass
        suffix = "unit_mass" if normalize_mass else f"server_{multiplier:g}"
        self.name = f"{base.name}_{suffix}_diagnostic"
        self.settings = {
            "diagnostic_control": True, "base_method": base.name, "base_settings": dict(base.settings),
            "server_multiplier": self.multiplier, "normalize_coefficient_mass": normalize_mass,
            "history_policy": "base history sees unscaled local deltas; only applied server step changes",
            "implementation_sha256": file_sha256(Path(__file__)),
        }
        self.reset()

    def reset(self):
        if hasattr(self.base, "reset"):
            self.base.reset()

    @torch.inference_mode()
    def aggregate(self, submissions, global_state, round_number):
        result = self.base.aggregate(submissions, global_state, round_number)
        ordered = sorted(submissions, key=lambda s: s.client_id)
        points = torch.stack([torch.cat([(s.state[k].double() - v.double()).flatten()
                                        for k, v in global_state.items()]) for s in ordered])
        raw = torch.tensor([result.coefficients[s.client_id] for s in ordered], dtype=torch.float64)
        mass = float(raw.sum())
        factor = self.multiplier / mass if self.normalize_mass and mass > 0 else self.multiplier
        applied = raw * factor
        counts = torch.tensor([s.num_examples for s in ordered], dtype=torch.float64)
        average = torch.sum(points * (counts / counts.sum())[:, None], dim=0)
        raw_delta = torch.sum(points * raw[:, None], dim=0)
        delta = torch.sum(points * applied[:, None], dim=0)
        # Reconstruct from client deltas in float64, avoiding an extra rounding
        # through the base's already-cast model when changing its step size.
        if factor == 1:
            state = result.state
        else:
            state, offset = {}, 0
            for name, reference in global_state.items():
                piece = delta[offset:offset + reference.numel()].reshape(reference.shape)
                state[name] = (reference.double() + piece).to(reference.dtype)
                offset += reference.numel()
        norms = torch.linalg.vector_norm(points, dim=1)
        unit = points / torch.where(norms > 0, norms, torch.ones_like(norms))[:, None]
        cosine = (unit @ unit.T).clamp(-1, 1)
        pairs = cosine[torch.triu(torch.ones_like(cosine, dtype=torch.bool), diagonal=1)]
        average_norm, raw_norm, applied_norm = map(float, [torch.linalg.vector_norm(average),
                                      torch.linalg.vector_norm(raw_delta), torch.linalg.vector_norm(delta)])
        direction = float(torch.dot(delta, average) / (applied_norm * average_norm)) if applied_norm > 0 and average_norm > 0 else None
        metrics = {str(s.client_id): {**result.diagnostics.get("client_metrics", {}).get(str(s.client_id), {}),
                        "raw_aggregation_weight": float(raw[i]), "applied_aggregation_weight": float(applied[i])}
                   for i, s in enumerate(ordered)}
        diagnostics = {**result.diagnostics, "raw_coefficient_mass": mass, "applied_multiplier": factor,
            "applied_coefficient_mass": float(applied.sum()), "raw_aggregate_l2_norm": raw_norm,
            "applied_aggregate_l2_norm": applied_norm, "fedavg_delta_l2_norm": average_norm,
            "applied_to_fedavg_norm_ratio": applied_norm / average_norm if average_norm > 0 else None,
            "applied_cosine_to_fedavg": direction,
            "zero_raw_weights": int((raw == 0).sum()),
            "effective_client_count": mass ** 2 / float(raw.square().sum()) if mass > 0 else 0.,
            "mean_pairwise_current_cosine": float(pairs.mean()) if len(pairs) else None,
            "client_metrics": metrics}
        return AggregationResult(state, {s.client_id: float(applied[i]) for i, s in enumerate(ordered)}, diagnostics)
