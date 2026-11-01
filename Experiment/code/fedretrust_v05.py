"""Fixed v0.5 candidate: v0.3 evidence with a mass-preserving RFA complement.

For capped updates x_i, sample shares b_i, v0.3 trust q_i, and its RFA
anchor a, return sum(b_i*q_i*x_i) + (1-sum(b_i*q_i))*a. Only the final
aggregation changes. All evidence, clipping, restoration and fallback settings
are inherited unchanged from the selected v0.3 configuration.

This is an exploratory combination, not an inherited RFA robustness theorem.
A rejected client can still have indirect influence through the RFA anchor.
When every q_i is one, the rule cannot improve on v0.3's attack response.
"""
from pathlib import Path

import torch

from experiment_core import AggregationResult, file_sha256
from fedretrust_directional import DirectionalConfig, DirectionalFedReTrust
from fedretrust_reference_free import restored_weight
from rfa import RFAConfig, _flatten_submissions, smoothed_weiszfeld


class FedReTrustV05(DirectionalFedReTrust):
    """One fixed candidate; no independent configuration or parameter sweep."""

    def __init__(self):
        super().__init__(DirectionalConfig())
        self.name = "FedReTrust-v05"
        self.settings.update({
            "version": "0.5-exploratory",
            "final_aggregation": "sample_trust_mass_plus_missing_mass_times_capped_RFA",
            "all_trust_one_behavior": "exact_v03_arithmetic_path",
            "rejected_client_influence": "zero_direct_but_possible_indirect_RFA_influence",
            "implementation_sha256": file_sha256(Path(__file__)),
            "v03_implementation_sha256": file_sha256(Path(__file__).with_name("fedretrust_directional.py")),
        })

    @torch.inference_mode()
    def aggregate(self, submissions, broadcast, round_number):
        if type(round_number) is not int or round_number <= self.last_round:
            raise ValueError("Round numbers must increase; reset between runs.")
        points = _flatten_submissions(submissions, broadcast)
        cfg = self.options
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
                    # Mathematically m=1 here. Preserve v0.3 bit-for-bit even
                    # when normalizing heterogeneous counts leaves rounding.
                    robust_complement_mass = 0.
                    direct_coefficients = base * q / trust_mass * scales
                    rfa_coefficients = torch.zeros_like(base)
                coefficients = direct_coefficients + rfa_coefficients
        if fallback:
            # Match v0.3's bounded RFA fallback and do not update memory from
            # undefined geometry or a degenerate trust mass.
            direct_coefficients = torch.zeros_like(base)
            rfa_coefficients = robust_weights * scales
            coefficients = rfa_coefficients
            robust_complement_mass = 1.
        delta = (points * coefficients[:, None]).sum(0)
        if not torch.isfinite(delta).all():
            raise FloatingPointError("Non-finite v0.5 aggregate.")
        state, offset = {}, 0
        for name, reference in broadcast.items():
            piece = delta[offset:offset + reference.numel()].reshape(reference.shape)
            state[name] = (reference.double() + piece).to(reference.dtype)
            if not torch.isfinite(state[name]).all():
                raise FloatingPointError("Non-finite reconstructed model.")
            offset += reference.numel()
        metrics = {str(s.client_id): {
            "anchor_cosine": float(cosines[i]),
            "evidence_weight": None if fallback else float(evidence[i]),
            "restored_weight": None if fallback else float(q[i]),
            "clip_factor": float(scales[i]), "sample_share": float(base[i]),
            "direct_effective_coefficient": float(direct_coefficients[i]),
            "rfa_effective_coefficient": float(rfa_coefficients[i]),
            "anchor_coefficient": float(robust_weights[i]),
        } for i, s in enumerate(submissions)}
        info = {
            "coefficient_convention": "effective_coefficients_on_raw_client_deltas",
            "client_metrics": metrics, "reference_forward_examples": 0,
            "anchor_l2_norm": anchor_norm, "norm_cap": cap,
            "median_cosine": median_cosine, "bounded_rfa_fallback": fallback,
            "trust_mass": trust_mass, "robust_complement_mass": robust_complement_mass,
        }
        self.weights, self.last_round = updated, round_number
        return AggregationResult(state, {s.client_id: float(coefficients[i]) for i, s in enumerate(submissions)}, info)
