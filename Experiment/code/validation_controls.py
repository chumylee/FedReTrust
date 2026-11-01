"""Direct v0.3 component controls; the frozen candidate is never modified."""
from pathlib import Path
import torch
from experiment_core import AggregationResult, file_sha256
from fedretrust_directional import DirectionalConfig, DirectionalFedReTrust
from fedretrust_reference_free import restored_weight
from rfa import _flatten_submissions, smoothed_weiszfeld, RFAConfig

NO_MEMORY = 'FedReTrust-v03-no-restoration'
NO_CLIP = 'FedReTrust-v03-no-clipping'
CAPPED_RFA = 'FedReTrust-v03-capped-RFA'
CONTROLS = (NO_MEMORY, NO_CLIP, CAPPED_RFA)


class NoRestoration(DirectionalFedReTrust):
    def __init__(self):
        super().__init__(DirectionalConfig(use_restoration=False))
        self.name = NO_MEMORY
        self.settings = {**self.settings, 'control': 'current_evidence_only',
                         'control_source_sha256': file_sha256(__file__)}


class GeometricControl:
    """Either disable clipping throughout v0.3, or always aggregate capped RFA."""
    def __init__(self, method):
        if method not in (NO_CLIP, CAPPED_RFA):
            raise ValueError('Unknown geometric control.')
        self.name = method
        self.options = DirectionalConfig()
        self.settings = {**DirectionalFedReTrust().settings, 'control': method,
            'control_source_sha256': file_sha256(__file__),
            'anchor': 'three_step_RFA_on_raw_updates' if method == NO_CLIP else 'three_step_RFA_on_capped_updates',
            'clipping_enabled': method != NO_CLIP,
            'final_weighting': 'directional_with_restoration' if method == NO_CLIP else 'capped_RFA_only'}
        self.reset()

    def reset(self):
        self.weights, self.last_round = {}, 0

    @torch.inference_mode()
    def aggregate(self, submissions, broadcast, round_number):
        if type(round_number) is not int or round_number <= self.last_round:
            raise ValueError('Rounds must increase; reset between runs.')
        points = _flatten_submissions(submissions, broadcast)
        cfg = self.options
        base = torch.tensor([s.num_examples for s in submissions], dtype=torch.float64)
        base /= base.sum()
        norms = torch.linalg.vector_norm(points, dim=1)
        cap = cfg.norm_cap * float(norms.median())
        scales = torch.ones_like(norms)
        if self.name == CAPPED_RFA:
            scales = (cap / norms.clamp_min(cfg.epsilon)).clamp(max=1.)
            scales[norms <= cfg.epsilon] = 1.
        anchor, robust_weights, _ = smoothed_weiszfeld(points * scales[:, None], base, RFAConfig())
        anchor_norm = float(torch.linalg.vector_norm(anchor))
        cosines = ((points @ anchor) / (norms * anchor_norm).clamp_min(cfg.epsilon)).clamp(-1., 1.)
        median_cosine = float(cosines.median())
        evidence = (cosines / max(cfg.alignment_full * median_cosine, cfg.epsilon)).clamp(0., 1.)
        fallback = anchor_norm <= cfg.epsilon or median_cosine <= cfg.epsilon
        updated = dict(self.weights)
        q = None
        if self.name == CAPPED_RFA:
            coefficients = robust_weights * scales
        else:
            if not fallback:
                q = torch.tensor([restored_weight(self.weights.get(s.client_id, 1.), float(evidence[i]), cfg)
                                  for i, s in enumerate(submissions)], dtype=torch.float64)
                mass = float(torch.dot(base, q))
                if mass <= cfg.epsilon:
                    fallback = True
                else:
                    updated.update({s.client_id: float(q[i]) for i, s in enumerate(submissions)})
                    coefficients = base * q / mass
            if fallback:
                coefficients = robust_weights
        delta = (points * coefficients[:, None]).sum(0)
        if not torch.isfinite(delta).all():
            raise FloatingPointError('Non-finite control aggregate.')
        state, offset = {}, 0
        for name, reference in broadcast.items():
            state[name] = (reference.double() + delta[offset:offset + reference.numel()].reshape(reference.shape)).to(reference.dtype)
            if not torch.isfinite(state[name]).all():
                raise FloatingPointError('Non-finite reconstructed state.')
            offset += reference.numel()
        used_direction = self.name == NO_CLIP and not fallback
        metrics = {str(s.client_id): {'anchor_cosine': float(cosines[i]),
            'evidence_weight': float(evidence[i]) if used_direction else None,
            'restored_weight': float(q[i]) if used_direction else None,
            'clip_factor': float(scales[i]), 'sample_share': float(base[i])}
            for i, s in enumerate(submissions)}
        info = {'coefficient_convention': 'effective_coefficients_on_raw_client_deltas',
            'client_metrics': metrics, 'reference_forward_examples': 0, 'reference_backward_examples': 0,
            'anchor_l2_norm': anchor_norm, 'norm_cap': cap if self.name == CAPPED_RFA else None,
            'median_cosine': median_cosine, 'bounded_rfa_fallback': bool(fallback and self.name == NO_CLIP),
            'clipping_enabled': self.name == CAPPED_RFA,
            'always_use_rfa': self.name == CAPPED_RFA}
        self.weights, self.last_round = updated, round_number
        return AggregationResult(state, {s.client_id: float(coefficients[i]) for i, s in enumerate(submissions)}, info)


def make_control(method):
    if method == NO_MEMORY:
        return NoRestoration()
    return GeometricControl(method)
