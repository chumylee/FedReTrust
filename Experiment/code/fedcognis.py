"""FedCognis trust-aggregation adaptation, not the full QSA/SALSTM system.

Eq. 20 supplies exponential distance evidence and EMA trust; Eq. 15 supplies
sample-count times trust aggregation. Main uses the displayed L2 norm. The
prose's squared-distance interpretation is an explicit sensitivity setting.
All submitted clients are authenticated participants in this threat model;
authentication is not an oracle for detecting their poisoned models.
"""
from dataclasses import dataclass, asdict
import math
import torch
from experiment_core import AggregationResult
from fltrust import flatten_delta


@dataclass(frozen=True)
class FedCognisConfig:
    alpha: float = .5
    beta: float = .5
    distance_power: int = 1
    initial_trust: float = 1.

    def __post_init__(self):
        for name in ('alpha', 'beta'):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(name + ' must be finite and in [0,1].')
        if type(self.distance_power) is not int or self.distance_power not in (1, 2):
            raise ValueError('distance_power must be 1 or 2.')
        if not math.isfinite(self.initial_trust) or not 0 < self.initial_trust <= 1:
            raise ValueError('Initial trust must be finite and in (0,1].')


class FedCognisAggregator:
    name = 'FedCognis-aggregation'

    def __init__(self, options=None, name=None):
        self.options = options or FedCognisConfig()
        if not isinstance(self.options, FedCognisConfig):
            raise TypeError('Explicit FedCognisConfig required.')
        if name is not None: self.name = name
        self.settings = {
            'config': asdict(self.options),
            'paper': 'https://doi.org/10.32604/cmc.2025.066898',
            'implementation_status': 'documented_trust_aggregation_adaptation_not_full_FedCognis',
            'distance': 'raw_full_parameter_delta_L2' if self.options.distance_power == 1 else 'raw_full_parameter_delta_squared_L2',
            'anchor': 'current_broadcast_from_which_local_training_started',
            'trust_timing': 'update_current_trust_before_current_aggregation_as_in_Algorithm1',
            'weighting': 'sample_count_times_updated_trust_normalized_over_all_submissions',
            'hard_trust_filter': False, 'server_reference_examples': 0,
            'calibration_runs': 0, 'authentication': 'all_participating_identities_assumed_valid_including_poisoners',
            'QSA_implemented': False, 'SALSTM_implemented': False,
            'numerical_policy': 'logspace_EMA_and_normalization_prevent_exponential_underflow',
            'missing_client_policy': 'no_trust_change_without_a_fresh_submission',
        }
        self.reset()

    def reset(self):
        self.log_trust = {}
        self.last_round = 0

    @torch.inference_mode()
    def aggregate(self, submissions, global_state, round_number):
        ordered = sorted(submissions, key=lambda s: s.client_id)
        ids = [s.client_id for s in ordered]
        if not ids or len(set(ids)) != len(ids) or any(s.num_examples <= 0 for s in ordered):
            raise ValueError('Distinct clients with positive sample counts required.')
        if type(round_number) is not int or round_number <= self.last_round:
            raise ValueError('Rounds must increase; reset between experiments.')
        deltas = torch.stack([flatten_delta(s.state, global_state) for s in ordered])
        norms = torch.linalg.vector_norm(deltas, dim=1)
        penalty = self.options.beta * norms.pow(self.options.distance_power)
        if not torch.isfinite(norms).all() or not torch.isfinite(penalty).all():
            raise FloatingPointError('Non-finite update distance or penalty.')
        signal = -penalty
        previous = torch.tensor([self.log_trust.get(i, math.log(self.options.initial_trust)) for i in ids], dtype=torch.float64)
        alpha = self.options.alpha
        if alpha == 0: updated = signal
        elif alpha == 1: updated = previous
        else: updated = torch.logaddexp(previous + math.log(alpha), signal + math.log1p(-alpha))
        sizes = torch.tensor([s.num_examples for s in ordered], dtype=torch.float64)
        if not torch.isfinite(sizes).all(): raise ValueError('Non-finite sample counts.')
        weights = torch.softmax(updated + sizes.log(), dim=0)
        total = (weights[:, None] * deltas).sum(0)
        state, offset = {}, 0
        for name, tensor in global_state.items():
            size = tensor.numel()
            state[name] = (tensor.double() + total[offset:offset + size].reshape(tensor.shape)).to(tensor.dtype)
            offset += size
        metrics = {str(cid): {
            'update_l2_norm': float(norms[i]), 'distance_power': self.options.distance_power,
            'distance_penalty': float(penalty[i]), 'instantaneous_trust': float(signal[i].exp()),
            'log_instantaneous_trust': float(signal[i]), 'previous_trust': float(previous[i].exp()),
            'trust': float(updated[i].exp()), 'log_trust': float(updated[i]),
            'excluded': False, 'effective_update_coefficient': float(weights[i]),
        } for i, cid in enumerate(ids)}
        self.log_trust.update({cid: float(updated[i]) for i, cid in enumerate(ids)})
        self.last_round = round_number
        return AggregationResult(state, dict(zip(ids, weights.tolist())), {
            'coefficient_convention': 'effective_coefficients_on_raw_client_deltas',
            'client_order': ids, 'client_metrics': metrics,
            'reference_forward_examples': 0, 'reference_backward_examples': 0,
            'hard_trust_filter': False, 'authentication_assumption': 'valid_participating_identities',
            'trust_updated_before_aggregation': True,
        })
