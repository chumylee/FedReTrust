"""Explicit server learning rates for the four published baseline adapters.

The unchanged base method decides its scores, vector rescaling and history.
This adapter applies a fixed scalar to its update. There is no division by
the observed coefficient mass, clipping, scenario input or new trust rule.
"""
from pathlib import Path
import copy
import math

import torch

from experiment_core import AggregationResult, file_sha256


def server_rate_candidates(method, num_clients, relative_rates=(0.5, 1.0, 2.0)):
    """A declared equal-size grid around a method's algebraic scale anchor."""
    if method not in {"FedAvg", "FLTrust", "RFA", "Viceroy"}:
        raise ValueError("Unknown published baseline.")
    if type(num_clients) is not int or num_clients < 1:
        raise ValueError("Client count must be a positive integer.")
    if not relative_rates or any(isinstance(r, bool) or not math.isfinite(r) or r <= 0
                                 for r in relative_rates):
        raise ValueError("Relative rates must be finite and positive.")
    if len(set(relative_rates)) != len(relative_rates):
        raise ValueError("Do not count duplicate rates as tuning candidates.")
    anchor = 1 / num_clients if method == "Viceroy" else 1.0
    return [{"relative_rate": float(r), "server_learning_rate": anchor * r,
             "anchor": anchor} for r in relative_rates]


class ServerLearningRate:
    """Preserve a baseline rule and apply a declared fixed server step.

    Returned coefficients act on raw trained-minus-broadcast client deltas.
    FLTrust's native trust shares are multiplied by its norm scales first.
    Original native shares/scores remain separately available in diagnostics.
    A rate of one returns the base model exactly; changed rates accumulate
    unrounded client deltas in float64 before one cast to the model dtype.
    """

    def __init__(self, base, server_learning_rate):
        if base.name not in {"FedAvg", "FLTrust", "RFA", "Viceroy"}:
            raise ValueError("This adapter supports only the four audited published baselines.")
        if (isinstance(server_learning_rate, bool) or not math.isfinite(server_learning_rate)
                or server_learning_rate <= 0):
            raise ValueError("Server learning rate must be finite and positive.")
        if base.name == "FLTrust" and base.options.global_learning_rate != 1.0:
            raise ValueError("Set FLTrust's internal global learning rate to one; configure it here only.")
        self.base = base
        self.server_learning_rate = float(server_learning_rate)
        self.name = f"{base.name}_server_lr_{self.server_learning_rate:g}"
        self.settings = {
            "base_method": base.name,
            "base_settings": copy.deepcopy(base.settings),
            "server_learning_rate": self.server_learning_rate,
            "server_update": "broadcast + fixed_rate * base_update",
            "coefficient_convention": "effective_coefficients_on_raw_client_deltas",
            "base_coefficient_convention": ("trust_shares_before_norm_rescaling"
                                             if base.name == "FLTrust" else "coefficients_on_raw_client_deltas"),
            "weight_mass_normalization": False,
            "history_observations": "unscaled_submitted_client_deltas",
            "numerics": "float64_client_delta_accumulation; one_cast; rate_one_preserves_base_state_exactly",
            "implementation_sha256": file_sha256(Path(__file__)),
        }

    def reset(self):
        if hasattr(self.base, "reset"):
            self.base.reset()

    def aggregate(self, submissions, global_state, round_number):
        # FLTrust trains its root model here; inference mode must begin only
        # AFTER the base method has finished any required gradient steps.
        result = self.base.aggregate(submissions, global_state, round_number)
        return self._apply_rate(submissions, global_state, result)

    @torch.inference_mode()
    def _apply_rate(self, submissions, global_state, result):
        ordered = sorted(submissions, key=lambda s: s.client_id)
        if set(result.coefficients) != {s.client_id for s in ordered}:
            raise ValueError("Base coefficients must cover exactly the submitted clients.")
        points = torch.stack([torch.cat([(s.state[k].double() - v.double()).flatten()
                                         for k, v in global_state.items()]) for s in ordered])
        native = torch.tensor([result.coefficients[s.client_id] for s in ordered], dtype=torch.float64)
        metrics = result.diagnostics.get("client_metrics", {})
        if self.base.name == "FLTrust":
            effective = torch.tensor([metrics[str(s.client_id)]["effective_update_coefficient"]
                                      for s in ordered], dtype=torch.float64)
        else:
            effective = native.clone()
        if not torch.isfinite(effective).all() or not (effective >= 0).all():
            raise FloatingPointError("Invalid effective baseline coefficient.")
        applied = effective * self.server_learning_rate
        raw_delta = torch.sum(points * effective[:, None], dim=0)
        applied_delta = torch.sum(points * applied[:, None], dim=0)
        reconstructed, scaled, offset = {}, {}, 0
        for name, reference in global_state.items():
            size = reference.numel()
            raw_piece = raw_delta[offset:offset + size].reshape(reference.shape)
            piece = applied_delta[offset:offset + size].reshape(reference.shape)
            reconstructed[name] = (reference.double() + raw_piece).to(reference.dtype)
            scaled[name] = (reference.double() + piece).to(reference.dtype)
            offset += size
        # This also catches use of trust shares on unnormalized FLTrust deltas.
        for name, value in result.state.items():
            torch.testing.assert_close(reconstructed[name], value, rtol=2e-6, atol=2e-8)
        state = result.state if self.server_learning_rate == 1.0 else scaled
        if not all(torch.isfinite(t).all() for t in state.values()):
            raise FloatingPointError("Non-finite scaled server state.")
        maximum_error = max(float((reconstructed[k].double() - v.double()).abs().max())
                            for k, v in result.state.items())
        client_metrics = {str(s.client_id): {
            **metrics.get(str(s.client_id), {}),
            "base_native_coefficient": float(native[i]),
            "base_effective_delta_coefficient": float(effective[i]),
            "applied_delta_coefficient": float(applied[i]),
        } for i, s in enumerate(ordered)}
        diagnostics = {**result.diagnostics,
            "server_learning_rate": self.server_learning_rate,
            "base_native_coefficient_mass": float(native.sum()),
            "base_effective_coefficient_mass": float(effective.sum()),
            "applied_effective_coefficient_mass": float(applied.sum()),
            "base_update_l2_norm": float(torch.linalg.vector_norm(raw_delta)),
            "applied_update_l2_norm": float(torch.linalg.vector_norm(applied_delta)),
            "base_reconstruction_max_abs_error": maximum_error,
            "coefficient_convention": "effective_coefficients_on_raw_client_deltas",
            "client_metrics": client_metrics,
        }
        return AggregationResult(state, {s.client_id: float(applied[i]) for i, s in enumerate(ordered)}, diagnostics)
