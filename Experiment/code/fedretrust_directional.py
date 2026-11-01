"""Development-only v0.3: direction weighting without the v0.2 distance gate.

Uses only submitted parameters/counts and scalar configuration. No trusted
data, scenario labels, evaluator results, or known-attacker counts are inputs.
This is an experimental combination, without a robustness or novelty claim.
"""
from dataclasses import asdict, dataclass
from pathlib import Path
import math

import torch

from experiment_core import AggregationResult, file_sha256
from fedretrust_reference_free import restored_weight
from rfa import _flatten_submissions, smoothed_weiszfeld, RFAConfig


@dataclass(frozen=True)
class DirectionalConfig:
    unit_anchor: bool = False
    alignment_full: float = .25
    norm_cap: float = 6.
    restoration_step: float = .5
    use_restoration: bool = True
    epsilon: float = 1e-12

    def __post_init__(self):
        if type(self.unit_anchor) is not bool or type(self.use_restoration) is not bool:
            raise ValueError("Boolean switches required.")
        for name, value in asdict(self).items():
            if name not in ("unit_anchor", "use_restoration"):
                if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                    raise ValueError("Positive finite numeric settings required.")
        if self.alignment_full > 1 or self.restoration_step > 1:
            raise ValueError("Alignment and restoration fractions must not exceed one.")


CANDIDATE_CONFIGS = {
    "FedReTrust-Direction-loose": DirectionalConfig(),
    "FedReTrust-Direction-strict": DirectionalConfig(alignment_full=.5),
    "FedReTrust-Unit-direction": DirectionalConfig(unit_anchor=True),
}


class DirectionalFedReTrust:
    def __init__(self, options=None):
        self.options = options or DirectionalConfig()
        self.name = next((name for name, cfg in CANDIDATE_CONFIGS.items() if cfg == self.options),
                         "FedReTrust-Direction-custom")
        self.settings = {
            **asdict(self.options), "version": "0.3-exploratory",
            "trusted_reference_examples": 0, "server_learning_rate": 1.,
            "anchor": "three_step_RFA_on_unit_updates" if self.options.unit_anchor else "three_step_RFA_on_capped_updates",
            "sample_count_assumption": "authentic_counts; no_sybil_defence",
            "secure_aggregation_implemented": False,
            "implementation_sha256": file_sha256(Path(__file__)),
            "rfa_implementation_sha256": file_sha256(Path(__file__).with_name("rfa.py")),
            "restoration_implementation_sha256": file_sha256(Path(__file__).with_name("fedretrust_reference_free.py")),
        }
        self.reset()

    def reset(self):
        self.weights = {}
        self.last_round = 0

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
        robust_delta, robust_weights, _ = smoothed_weiszfeld(bounded, base, RFAConfig())
        if cfg.unit_anchor:
            directions = points / norms.clamp_min(cfg.epsilon)[:, None]
            anchor, _, _ = smoothed_weiszfeld(directions, base, RFAConfig())
        else:
            anchor = robust_delta
        anchor_norm = float(torch.linalg.vector_norm(anchor))
        cosines = ((points @ anchor) / (norms * anchor_norm).clamp_min(cfg.epsilon)).clamp(-1., 1.)
        median_cosine = float(cosines.median())
        # Opposition is now sufficient for zero current evidence; a malicious
        # update cannot evade the rule just by remaining close to the centre.
        evidence = (cosines / max(cfg.alignment_full * median_cosine, cfg.epsilon)).clamp(0., 1.)
        fallback = anchor_norm <= cfg.epsilon or median_cosine <= cfg.epsilon
        updated = dict(self.weights)
        q = torch.ones_like(base)
        if not fallback:
            q = torch.tensor([restored_weight(self.weights.get(s.client_id, 1.), float(evidence[i]), cfg)
                              for i, s in enumerate(submissions)], dtype=torch.float64)
            mass = float(torch.dot(base, q))
            if mass <= cfg.epsilon:
                fallback = True
            else:
                updated.update({s.client_id: float(q[i]) for i, s in enumerate(submissions)})
                # Normalize before clipping so the norm bound is not undone.
                coefficients = base * q / mass * scales
        if fallback:
            # A unit-direction anchor is not a parameter delta. Use the
            # independently computed bounded RFA delta and preserve memory.
            coefficients = robust_weights * scales
        delta = (points * coefficients[:, None]).sum(0)
        if not torch.isfinite(delta).all():
            raise FloatingPointError("Non-finite directional aggregate.")
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
        } for i, s in enumerate(submissions)}
        info = {"coefficient_convention": "effective_coefficients_on_raw_client_deltas",
                "client_metrics": metrics, "reference_forward_examples": 0,
                "anchor_l2_norm": anchor_norm, "norm_cap": cap,
                "median_cosine": median_cosine, "bounded_rfa_fallback": fallback}
        self.weights, self.last_round = updated, round_number
        return AggregationResult(state, {s.client_id: float(coefficients[i]) for i, s in enumerate(submissions)}, info)
