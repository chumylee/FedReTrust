"""Documented reconstruction of Xiao's FLAURA, Scientific Reports (2026).

Algorithm 1 is authoritative here where the displayed prose equations disagree:
global exponent has denominator 2*sigma_g**2; cutoff is gamma*mean(trust).
Unspecified kernel, bandwidth, knee solver and hyperparameters are explicit in
the study protocol. This is not author code or a reproduction of paper scores.
"""
from dataclasses import dataclass, asdict
import hashlib
import math
import numpy as np
import torch

from experiment_core import AggregationResult, FedAvgAggregator, MnistCNN
from fltrust import flatten_delta
from rfa import smoothed_weiszfeld, RFAConfig
from scenario_utils import ScenarioDataset


@dataclass(frozen=True)
class AuxiliaryReference:
    images: torch.Tensor
    metadata: dict


def build_auxiliary_reference(data, per_class=10):
    """Select from reserved server data only; labels are used only for sampling."""
    if type(per_class) is not int or per_class < 1:
        raise ValueError('Positive per-class count required.')
    reserved = list(data.manifest['indices']['server_reference'])
    forbidden = set(data.manifest['indices']['development_validation'])
    forbidden.update(i for group in data.manifest['indices']['clients'].values() for i in group)
    if len(reserved) != len(set(reserved)) or set(reserved) & forbidden:
        raise ValueError('Auxiliary source overlaps clients or evaluation.')
    selected = []
    for digit in range(10):
        candidates = sorted(i for i in reserved if int(data.pool.targets[i]) == digit)
        if len(candidates) < per_class:
            raise ValueError('Insufficient reserved examples for a class.')
        selected.extend(candidates[:per_class])
    selected.sort()
    norm = data.manifest['normalization']
    view = ScenarioDataset(data.pool, selected, norm['mean'][0], norm['std'][0], seed=0)
    images = torch.stack([view[i][0] for i in range(len(view))]).detach().clone()
    return AuxiliaryReference(images, {
        'examples': len(selected), 'original_indices': selected, 'index_space': 'official_train',
        'class_counts': {str(k): per_class for k in range(10)},
        'tensor_sha256': hashlib.sha256(images.numpy().tobytes()).hexdigest(),
        'partition_sha256': data.manifest_sha256, 'disjoint_from_clients_and_validation': True,
        'labels_used': 'stratified_selection_only_not_scoring_or_training',
        'selection': 'first_sorted_reserved_indices_per_class',
    })


@dataclass(frozen=True)
class FLAURAConfig:
    sigma_g: float = 1.0  # supplied by separately frozen clean calibration
    alpha: float = 0.5
    beta: float = 0.5
    gamma: float = 0.8
    batch_size: int = 100
    median_iterations: int = 200
    median_smoothing: float = 1e-8
    median_tolerance: float = 1e-9
    epsilon: float = 1e-12

    def __post_init__(self):
        for key in ('sigma_g', 'median_smoothing', 'epsilon'):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(key + ' must be finite and positive.')
        for key in ('alpha', 'beta', 'gamma'):
            if not math.isfinite(getattr(self, key)) or not 0 <= getattr(self, key) <= 1:
                raise ValueError(key + ' must be in [0,1].')
        for key in ('batch_size', 'median_iterations'):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(key + ' must be a positive integer.')
        if not math.isfinite(self.median_tolerance) or self.median_tolerance < 0:
            raise ValueError('Invalid median tolerance.')


@torch.inference_mode()
def extract_representations(submissions, images, batch_size=100):
    if images.device.type != 'cpu' or images.ndim != 4 or not torch.isfinite(images).all() or not len(images):
        raise ValueError('Expected finite auxiliary image tensor.')
    # Model construction must not advance the training RNG stream.
    with torch.random.fork_rng(devices=[]):
        model = MnistCNN().eval()
    outputs = []
    for item in submissions:
        model.load_state_dict(item.state, strict=True)
        blocks = [model.classifier[:3](model.features(x)).double()
                  for x in images.split(batch_size)]
        outputs.append(torch.cat(blocks))
    result = torch.stack(outputs)
    if not torch.isfinite(result).all():
        raise FloatingPointError('Non-finite penultimate representation.')
    return result


def representation_center(representations, options):
    flat = representations.reshape(len(representations), -1)
    center, _, diagnostics = smoothed_weiszfeld(flat, torch.ones(len(flat)), RFAConfig(
        max_iterations=options.median_iterations, smoothing=options.median_smoothing,
        relative_tolerance=options.median_tolerance))
    distances = torch.linalg.vector_norm(flat - center, dim=1)
    return center, distances, diagnostics


@torch.inference_mode()
def pairwise_mmd(representations, epsilon=1e-12, bandwidth_squared=None):
    """Biased empirical RBF MMD (the RKHS norm, not its square).

    One shared median-positive squared-distance bandwidth over all pooled PLR
    rows. The paper leaves the kernel/bandwidth unspecified. Diagonals are
    included, as in the empirical mean embeddings in Eq.8.
    """
    r = representations.detach().double()
    if r.ndim != 3 or 0 in r.shape or not torch.isfinite(r).all():
        raise ValueError('Expected finite client x auxiliary x feature tensor.')
    n, m, _ = r.shape
    flat = r.flatten(0, 1)
    norms = flat.square().sum(1)
    squared = (norms[:, None] + norms[None, :] - 2 * (flat @ flat.T)).clamp_min(0)
    if bandwidth_squared is None:
        upper = squared[torch.triu_indices(len(flat), len(flat), offset=1).unbind()]
        positive = upper[upper > epsilon]
        bandwidth_squared = float(positive.median()) if len(positive) else 1.0
    if not math.isfinite(bandwidth_squared) or bandwidth_squared <= 0:
        raise ValueError('Kernel bandwidth must be finite and positive.')
    kernels = torch.exp(-squared / (2 * bandwidth_squared))
    means = kernels.reshape(n, m, n, m).mean(dim=(1, 3))
    within = means.diag()
    mmd2 = within[:, None] + within[None, :] - 2 * means
    mmd2[mmd2.abs() <= epsilon] = 0
    distances = mmd2.clamp_min(0).sqrt()
    distances.fill_diagonal_(0)
    return distances, {'kernel': 'RBF', 'bandwidth_squared': bandwidth_squared,
                       'bandwidth_rule': 'pooled_positive_pairwise_squared_distance_median',
                       'estimator': 'biased_empirical_mean_embedding_norm'}


def adaptive_neighbours(distances, client_ids, epsilon=1e-12):
    """Operational Eq.11 knee: largest positive adjacent jump before the gap.

    Equal-gap/flat profiles retain all peers; boundary ties are all retained.
    This explicit finite-difference choice is not claimed to be author code.
    """
    ids = list(client_ids)
    if distances.shape != (len(ids), len(ids)) or len(set(ids)) != len(ids):
        raise ValueError('Neighbour IDs and distance matrix disagree.')
    if not torch.isfinite(distances).all() or (distances < 0).any():
        raise ValueError('Invalid MMD distances.')
    result = []
    for i in range(len(ids)):
        peers = sorted((j for j in range(len(ids)) if j != i), key=lambda j: (float(distances[i, j]), ids[j]))
        values = np.array([float(distances[i, j]) for j in peers])
        gaps = np.diff(values)
        if len(gaps) and float(gaps.max() - gaps.min()) > epsilon:
            boundary = int(gaps.argmax())
            cutoff = values[boundary]
            peers = [j for j in peers if float(distances[i, j]) <= cutoff + epsilon]
        result.append(peers)
    return result


def trust_weights(global_scores, local_scores, previous, options):
    g, l, previous = [torch.as_tensor(v, dtype=torch.float64) for v in (global_scores, local_scores, previous)]
    if not g.numel() or g.ndim != 1 or l.shape != g.shape or previous.shape != g.shape:
        raise ValueError('Trust vectors must align.')
    if any(not torch.isfinite(v).all() or ((v < 0) | (v > 1)).any() for v in (g, l, previous)):
        raise ValueError('Scores must lie in [0,1].')
    instantaneous = options.alpha * g + (1 - options.alpha) * l
    updated = options.beta * previous + (1 - options.beta) * instantaneous
    threshold = options.gamma * float(updated.mean())
    retained = updated >= threshold
    mass = torch.where(retained, updated, 0.)
    weights = mass / mass.sum() if float(mass.sum()) > options.epsilon else torch.zeros_like(mass)
    return instantaneous, updated, threshold, retained, weights


def weighted_update(submissions, broadcast, weights):
    deltas = torch.stack([flatten_delta(s.state, broadcast) for s in submissions])
    total = (deltas * weights[:, None]).sum(0)
    state, offset = {}, 0
    for name, tensor in broadcast.items():
        size = tensor.numel()
        state[name] = (tensor.double() + total[offset:offset + size].reshape(tensor.shape)).to(tensor.dtype)
        offset += size
    return state


class FLAURAAggregator:
    name = 'FLAURA-reconstruction'

    def __init__(self, reference, options, name=None):
        if not isinstance(options, FLAURAConfig) or not isinstance(reference, AuxiliaryReference):
            raise TypeError('Explicit config and copied auxiliary reference required.')
        self.reference = AuxiliaryReference(reference.images.detach().clone(), dict(reference.metadata))
        self.options = options
        if name is not None:
            self.name = name
        self.settings = {'config': asdict(options), 'reference': dict(reference.metadata),
                         'paper': 'https://doi.org/10.1038/s41598-026-50985-2',
                         'implementation_status': 'documented_Algorithm1_reconstruction',
                         'plr': 'classifier.2_post_ReLU_128', 'global_denominator': '2*sigma_g**2',
                         'hard_threshold': 'gamma*mean(updated_trust)', 'weighting': 'trust_only',
                         'knee': 'largest_adjacent_gap_flat_profile_all_peers',
                         'kernel': 'RBF_biased_MMD_pooled_median_bandwidth',
                         'rebalancing': 'not_implemented_unspecified_and_absent_from_Algorithm1'}
        self.reset()

    def reset(self):
        self.trust = {}
        self.last_round = 0

    @torch.inference_mode()
    def aggregate(self, submissions, global_state, round_number):
        ordered = sorted(submissions, key=lambda s: s.client_id)
        ids = [s.client_id for s in ordered]
        if not ids or len(set(ids)) != len(ids) or any(s.num_examples <= 0 for s in ordered):
            raise ValueError('Distinct submissions and positive sample counts required.')
        if type(round_number) is not int or round_number <= self.last_round:
            raise ValueError('Rounds must strictly increase; reset between runs.')
        # Validate even parameters not used by PLR extraction (including head).
        for s in ordered:
            flatten_delta(s.state, global_state)
        representations = extract_representations(ordered, self.reference.images, self.options.batch_size)
        _, distances, median = representation_center(representations, self.options)
        global_scores = torch.exp(-.5 * (distances / self.options.sigma_g).square())
        mmd, kernel = pairwise_mmd(representations, self.options.epsilon)
        neighbours = adaptive_neighbours(mmd, ids, self.options.epsilon)
        local_scores = torch.tensor([math.exp(-float(mmd[i, js].mean())) if js else 1.
                                     for i, js in enumerate(neighbours)], dtype=torch.float64)
        previous = [self.trust.get(cid, 1.) for cid in ids]
        instant, updated, threshold, retained, weights = trust_weights(global_scores, local_scores, previous, self.options)
        state = weighted_update(ordered, global_state, weights)
        self.trust.update({cid: float(updated[i]) for i, cid in enumerate(ids)})
        self.last_round = round_number
        metrics = {str(cid): {
            'global_score': float(global_scores[i]), 'local_score': float(local_scores[i]),
            'instantaneous_trust': float(instant[i]), 'previous_trust': previous[i], 'trust': float(updated[i]),
            'plr_center_distance': float(distances[i]), 'neighbours': [ids[j] for j in neighbours[i]],
            'excluded': bool(not retained[i]), 'effective_update_coefficient': float(weights[i]),
        } for i, cid in enumerate(ids)}
        return AggregationResult(state, dict(zip(ids, weights.tolist())), {
            'coefficient_convention': 'effective_coefficients_on_raw_client_deltas',
            'client_metrics': metrics, 'threshold': threshold, 'kernel': kernel,
            'median_solver': median, 'mmd_distances': mmd.tolist(), 'client_order': ids,
            'retained_clients': [cid for i, cid in enumerate(ids) if retained[i]],
            'reference_forward_examples': len(ids) * len(self.reference.images),
            'reference_backward_examples': 0, 'fallback_reason': 'zero_retained_trust' if weights.sum() == 0 else None,
        })


class PLRCalibrationObserver:
    """Unmodified FedAvg trajectory; reports geometry for offline scale choice."""
    name = 'FedAvg-PLR-calibration'

    def __init__(self, reference):
        self.reference = reference
        self.settings = {'reference': reference.metadata, 'training': 'original_FedAvg',
                         'purpose': 'clean_geometry_only_no_accuracy_selection'}

    def aggregate(self, submissions, global_state, round_number):
        ordered = sorted(submissions, key=lambda s: s.client_id)
        r = extract_representations(ordered, self.reference.images)
        _, distances, median = representation_center(r, FLAURAConfig())
        result = FedAvgAggregator().aggregate(ordered, global_state, round_number)
        result.diagnostics.update({
            'coefficient_convention': 'effective_coefficients_on_raw_client_deltas',
            'reference_forward_examples': len(ordered) * len(self.reference.images),
            'plr_center_distances': distances.tolist(), 'median_solver': median,
        })
        return result
