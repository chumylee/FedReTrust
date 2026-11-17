"""Fixed exploratory v0.6: client-specific clipped innovations, then v0.5.

The innovation multiplier 2 is an engineering choice, not a fitted threshold or
a published optimum. Clip21 motivates tracking client-specific estimates while
clipping their changes; this adaptive-radius, server-side local-SGD combination
does not reproduce Clip21/Byz-Clip21 or inherit their convergence/robustness/DP
guarantees. History is untrusted state and can accumulate poisoned updates.

Returned coefficients describe ONLY the current raw-update part. The actual
update also contains a historical residual; use InnovationContributionRecorder
for the complete affine contribution audit.
"""
from pathlib import Path

import torch

from experiment_core import AggregationResult, file_sha256
from fedretrust_v05 import FedReTrustV05
from fedretrust_reference_free import restored_weight
from rfa import RFAConfig, _flatten_submissions, smoothed_weiszfeld


INNOVATION_MULTIPLIER = 2.
COEFFICIENT_CONVENTION = "affine_current_raw_deltas_plus_client_history"


class FedReTrustV06(FedReTrustV05):
    """One declared candidate, with zero cold-start history and no clean warm-up."""

    def __init__(self):
        super().__init__()
        self.name = "FedReTrust-v06"
        self.settings.update({
            "version": "0.6-exploratory",
            "implementation_sha256": file_sha256(Path(__file__)),
            "v05_implementation_sha256": file_sha256(Path(__file__).with_name("fedretrust_v05.py")),
            "innovation_multiplier": INNOVATION_MULTIPLIER,
            "innovation_radius": "2_times_current_cohort_lower_median_innovation_norm",
            "innovation_multiplier_basis": "fixed_engineering_choice_not_fitted_or_theoretically_optimal",
            "innovation_filter": "y_i=h_i+clip(x_i-h_i,tau); all_v05_operations_act_on_double_y_i",
            "history_initialization": "zero_on_first_participation_no_trusted_warmup",
            "history_update": "store_y_i_after_success_even_if_directionally_downweighted; no_update_when_absent",
            "history_trust": "untrusted_tracking_state_not_evidence_of_honesty",
            "zero_median_innovation": "if_median_le_epsilon_bypass_filter_and_log; preserves_minority_change_but_allows_degenerate_attack_bypass",
            "final_aggregation": "unchanged_v05_mass_complement_on_filtered_updates",
            "all_trust_one_behavior": "v05_on_filtered_updates; innovation_clipping_can_still_limit_current_influence",
            "coefficient_convention": COEFFICIENT_CONVENTION,
            "known_risks": "poisoned_history_gradual_attacks_stale_history_and_delayed_honest_drift_or_repair",
            "literature_provenance": ["https://arxiv.org/abs/2305.18929", "https://arxiv.org/abs/2603.23472"],
            "theorem_or_novelty_claim": False,
        })

    def reset(self):
        super().reset()
        self.history = {}
        self.layout_signature = None

    @torch.inference_mode()
    def aggregate(self, submissions, broadcast, round_number):
        if type(round_number) is not int or round_number <= self.last_round:
            raise ValueError("Round numbers must increase; reset between runs.")
        if any(type(s.client_id) is not int for s in submissions):
            raise ValueError("Stable integer client identities are required.")
        raw = _flatten_submissions(submissions, broadcast)
        signature = tuple((name, tuple(t.shape), str(t.dtype)) for name, t in broadcast.items())
        if self.layout_signature is not None and signature != self.layout_signature:
            raise ValueError("Model layout changed; reset the method before a new model.")
        cfg = self.options
        previous = torch.stack([self.history.get(s.client_id, torch.zeros_like(raw[i]))
                                for i, s in enumerate(submissions)])
        if previous.shape != raw.shape or not torch.isfinite(previous).all():
            raise FloatingPointError("Invalid stored client update history.")
        innovations = raw - previous
        innovation_norms = torch.linalg.vector_norm(innovations, dim=1)
        if not torch.isfinite(innovation_norms).all():
            raise FloatingPointError("Non-finite innovation norm.")
        median_innovation = float(innovation_norms.median())
        radius = INNOVATION_MULTIPLIER * median_innovation
        if not torch.isfinite(torch.tensor(radius, dtype=torch.float64)):
            raise FloatingPointError("Non-finite innovation radius.")
        bypass = median_innovation <= cfg.epsilon
        factors = torch.ones_like(innovation_norms)
        if not bypass:
            factors = (radius / innovation_norms.clamp_min(cfg.epsilon)).clamp(max=1.)
            factors[innovation_norms <= cfg.epsilon] = 1.
        points = previous + factors[:, None] * innovations
        # Preserve the exact current update where the filter is inactive.
        points = torch.where((factors == 1.)[:, None], raw, points)
        if not torch.isfinite(points).all():
            raise FloatingPointError("Non-finite filtered update.")

        # The following v0.5 calculation is unchanged, with double-precision
        # filtered deltas replacing raw deltas. There is no intermediate model
        # reconstruction/cast and no scenario or evaluator input.
        base = torch.tensor([s.num_examples for s in submissions], dtype=torch.float64)
        base /= base.sum()
        norms = torch.linalg.vector_norm(points, dim=1)
        cap = cfg.norm_cap * float(norms.median())
        scales = (cap / norms.clamp_min(cfg.epsilon)).clamp(max=1.)
        scales[norms <= cfg.epsilon] = 1.
        bounded = points * scales[:, None]
        anchor, robust_weights, _ = smoothed_weiszfeld(bounded, base, RFAConfig())
        anchor_norm = float(torch.linalg.vector_norm(anchor))
        cosines = ((points @ anchor) / (norms * anchor_norm).clamp_min(cfg.epsilon)).clamp(-1., 1.)
        median_cosine = float(cosines.median())
        evidence = (cosines / max(cfg.alignment_full * median_cosine, cfg.epsilon)).clamp(0., 1.)
        fallback = anchor_norm <= cfg.epsilon or median_cosine <= cfg.epsilon
        updated = dict(self.weights)
        q = torch.ones_like(base)
        trust_mass = None
        if not fallback:
            q = torch.tensor([restored_weight(self.weights.get(s.client_id, 1.), float(evidence[i]), cfg)
                              for i, s in enumerate(submissions)], dtype=torch.float64)
            trust_mass = float(torch.dot(base, q))
            if trust_mass <= cfg.epsilon:
                fallback = True
            else:
                updated.update({s.client_id: float(q[i]) for i, s in enumerate(submissions)})
                robust_complement_mass = max(0., 1. - trust_mass)
                direct_coefficients = base * q * scales
                rfa_coefficients = robust_complement_mass * robust_weights * scales
                if bool(torch.all(q == 1.)):
                    robust_complement_mass = 0.
                    direct_coefficients = base * q / trust_mass * scales
                    rfa_coefficients = torch.zeros_like(base)
                coefficients = direct_coefficients + rfa_coefficients
        if fallback:
            direct_coefficients = torch.zeros_like(base)
            rfa_coefficients = robust_weights * scales
            coefficients = rfa_coefficients
            robust_complement_mass = 1.
        delta = (points * coefficients[:, None]).sum(0)
        if not torch.isfinite(delta).all():
            raise FloatingPointError("Non-finite v0.6 aggregate.")
        state, offset = {}, 0
        for name, reference in broadcast.items():
            piece = delta[offset:offset + reference.numel()].reshape(reference.shape)
            state[name] = (reference.double() + piece).to(reference.dtype)
            if not torch.isfinite(state[name]).all():
                raise FloatingPointError("Non-finite reconstructed model.")
            offset += reference.numel()
        current_coefficients = coefficients * factors
        historical_coefficients = coefficients * (1. - factors)
        next_history = {**self.history, **{s.client_id: points[i].detach().clone()
                                         for i, s in enumerate(submissions)}}
        metrics = {str(s.client_id): {
            "raw_submitted_update_l2_norm": float(torch.linalg.vector_norm(raw[i])),
            "filtered_update_l2_norm": float(norms[i]),
            "previous_tracked_update_l2_norm": float(torch.linalg.vector_norm(previous[i])),
            "innovation_l2_norm": float(innovation_norms[i]), "innovation_factor": float(factors[i]),
            "history_initialized_before_round": s.client_id in self.history,
            "anchor_cosine": float(cosines[i]),
            "evidence_weight": None if fallback else float(evidence[i]),
            "restored_weight": None if fallback else float(q[i]),
            "clip_factor": float(scales[i]), "sample_share": float(base[i]),
            "filtered_effective_coefficient": float(coefficients[i]),
            "filtered_direct_effective_coefficient": float(direct_coefficients[i]),
            "filtered_rfa_effective_coefficient": float(rfa_coefficients[i]),
            "current_delta_coefficient": float(current_coefficients[i]),
            "historical_coefficient": float(historical_coefficients[i]),
            "anchor_coefficient": float(robust_weights[i]),
        } for i, s in enumerate(submissions)}
        info = {
            "coefficient_convention": COEFFICIENT_CONVENTION,
            "returned_coefficients_scope": "current_raw_delta_part_only; historical_residual_is_additional",
            "client_metrics": metrics, "reference_forward_examples": 0,
            "anchor_l2_norm": anchor_norm, "norm_cap": cap,
            "median_cosine": median_cosine, "bounded_rfa_fallback": fallback,
            "trust_mass": trust_mass, "robust_complement_mass": robust_complement_mass,
            "innovation_multiplier": INNOVATION_MULTIPLIER,
            "median_innovation_l2_norm": median_innovation, "innovation_radius": radius,
            "zero_median_innovation_bypass": bypass,
            "history_clients": len(next_history),
            "history_tensor_bytes": sum(v.numel() * v.element_size() for v in next_history.values()),
            "history_bytes_scope": "CPU_float64_tensor_payload_only_excludes_python_overhead_and_temporary_tensors",
        }
        # Commit all persistent state only after every output has been validated.
        self.weights, self.history = updated, next_history
        self.last_round, self.layout_signature = round_number, signature
        return AggregationResult(state, {s.client_id: float(current_coefficients[i])
                                       for i, s in enumerate(submissions)}, info)
