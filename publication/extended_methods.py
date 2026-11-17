"""Publication baseline extensions; original experiment sources are unchanged."""
from dataclasses import asdict, dataclass
import math

import numpy as np
import torch

import study
from flaura import (AuxiliaryReference, FLAURAConfig, FLAURAAggregator,
                    adaptive_neighbours, pairwise_mmd, representation_center,
                    trust_weights, weighted_update)
from experiment_core import AggregationResult
from fltrust import flatten_delta


def auxiliary_reference(data):
    """At most 100 balanced examples from the existing reserved root split."""
    x, labels = (value.detach().cpu() for value in data['root'])
    per_class = min(100 // data['classes'], int(torch.bincount(labels).min()))
    positions = torch.cat([torch.where(labels == c)[0][:per_class]
                           for c in range(data['classes'])]).sort().values
    ids = [data['plan']['root_indices'][i] for i in positions.tolist()]
    clients = {i for group in data['plan']['client_indices'].values() for i in group}
    if clients.intersection(ids):
        raise ValueError('Auxiliary examples overlap training clients.')
    return AuxiliaryReference(x[positions].clone(), {
        'original_indices': ids, 'examples': len(ids), 'per_class': per_class,
        'source': 'existing_reserved_training_root_split',
        'labels_used': 'balanced_selection_only', 'test_access': False,
        'disjoint_from_clients': True,
    })


class Representations:
    def __init__(self, factory, reference, device):
        # Model initialization must not consume the experiment RNG.
        with torch.random.fork_rng(devices=[]):
            self.model = factory().to(device).eval()
        self.x = reference.images.to(device)

    @torch.inference_mode()
    def __call__(self, submissions):
        outputs = []
        for submission in submissions:
            self.model.load_state_dict(submission.state)
            blocks = []
            for x in self.x.split(100):
                if hasattr(self.model, 'classifier'):
                    features = self.model.classifier[:-1](self.model.features(x))
                else:
                    features = self.model.network[:-1](x)
                blocks.append(features.detach().cpu().double())
            outputs.append(torch.cat(blocks))
        result = torch.stack(outputs)
        if not torch.isfinite(result).all():
            raise FloatingPointError('Non-finite penultimate representations.')
        return result


class PublicationFLAURA(FLAURAAggregator):
    """Same documented reconstruction, generalized to both publication CNNs."""
    def __init__(self, factory, reference, device, sigma):
        super().__init__(reference, FLAURAConfig(sigma_g=sigma))
        self.extract = Representations(factory, reference, device)
        self.settings.update({
            'plr': 'post_ReLU_before_final_linear_layer',
            'calibration': 'six_clean_FedAvg_rounds_on_separate_seed_per_dataset',
            'implementation_sha256': study.file_sha256(__file__),
            'original_flaura_sha256': study.file_sha256(study.CODE / 'flaura.py'),
        })
        self.diagnostics = []

    @torch.inference_mode()
    def aggregate(self, submissions, broadcast, round_number):
        ordered = sorted(submissions, key=lambda s: s.client_id)
        ids = [s.client_id for s in ordered]
        if not ids or len(set(ids)) != len(ids) or any(s.num_examples <= 0 for s in ordered):
            raise ValueError('Distinct clients with positive sample counts required.')
        if round_number <= self.last_round:
            raise ValueError('Rounds must strictly increase.')
        for item in ordered:
            flatten_delta(item.state, broadcast)
        representations = self.extract(ordered)
        _, distances, median = representation_center(representations, self.options)
        global_scores = torch.exp(-.5 * (distances / self.options.sigma_g).square())
        mmd, kernel = pairwise_mmd(representations, self.options.epsilon)
        neighbours = adaptive_neighbours(mmd, ids, self.options.epsilon)
        local_scores = torch.tensor([math.exp(-float(mmd[i, js].mean())) if js else 1.
                                     for i, js in enumerate(neighbours)], dtype=torch.float64)
        previous = [self.trust.get(cid, 1.) for cid in ids]
        _, updated, threshold, retained, weights = trust_weights(
            global_scores, local_scores, previous, self.options)
        state = weighted_update(ordered, broadcast, weights)
        self.trust.update({cid: float(updated[i]) for i, cid in enumerate(ids)})
        self.last_round = round_number
        info = {'round': round_number, 'client_order': ids, 'trust': updated.tolist(),
                'weights': weights.tolist(), 'retained': retained.tolist(),
                'global_scores': global_scores.tolist(), 'local_scores': local_scores.tolist(),
                'threshold': threshold, 'kernel': kernel, 'median_solver': median}
        self.diagnostics.append(info)
        return AggregationResult(state, dict(zip(ids, weights.tolist())), info)


class CalibrationObserver:
    settings = {'purpose': 'clean_geometry_only_no_evaluation_or_accuracy_selection'}

    def __init__(self, factory, reference, device):
        self.extract = Representations(factory, reference, device)
        self.distances = []

    def aggregate(self, submissions, broadcast, round_number):
        ordered = sorted(submissions, key=lambda s: s.client_id)
        _, distances, _ = representation_center(self.extract(ordered), FLAURAConfig())
        self.distances.extend(distances.tolist())
        return study.FedAvgAggregator().aggregate(ordered, broadcast, round_number)


def nnm_coordinate_median(vectors, faulty):
    """NNM (n-f nearest, including self), then coordinate median (even midpoint)."""
    if vectors.ndim != 2 or len(vectors) < 1 or not 0 <= faulty < len(vectors) / 2:
        raise ValueError('Invalid NNM cohort or fault bound.')
    if not torch.isfinite(vectors).all():
        raise FloatingPointError('Non-finite NNM input.')
    distances = torch.cdist(vectors, vectors)
    neighbours = distances.argsort(dim=1, stable=True)[:, :len(vectors) - faulty]
    mixed = torch.stack([vectors[indices].mean(dim=0) for indices in neighbours])
    ordered = mixed.sort(dim=0).values
    n = len(ordered)
    return (ordered[(n - 1) // 2] + ordered[n // 2]) / 2


@dataclass(frozen=True)
class ByzConfig:
    learning_rate: float = 1.
    beta: float = .1
    beta_hat: float = 1.
    tau: float = 1.

    def __post_init__(self):
        if any(not math.isfinite(v) or v <= 0 for v in asdict(self).values()):
            raise ValueError('Byz-Clip21 settings must be finite and positive.')
        if self.beta > 1 or self.beta_hat > 1:
            raise ValueError('Momentum averaging factors cannot exceed one.')


class ByzClip21State:
    """Algorithm 1, no DP noise. Zero initial state; gradients, never SGD deltas.

    All clients execute this recursion, including clients with poisoned labels.
    This models data poisoning, not arbitrary Byzantine message construction.
    """
    def __init__(self, clients, dimension, options=None):
        self.options = options or ByzConfig()
        self.clients = tuple(sorted(clients))
        self.v = torch.zeros(len(clients), dimension, dtype=torch.float64)
        self.g = torch.zeros_like(self.v)
        self.m = torch.zeros_like(self.v)
        self.aggregate = torch.zeros(dimension, dtype=torch.float64)
        self.faulty_bound = len(clients) // 3

    def observe(self, gradients):
        if gradients.shape != self.v.shape or not torch.isfinite(gradients).all():
            raise FloatingPointError('Invalid Byz-Clip21 gradient matrix.')
        o = self.options
        self.v = (1 - o.beta) * self.v + o.beta * gradients
        innovation = self.v - self.g
        norms = innovation.norm(dim=1)
        clipped = innovation * (o.tau / norms.clamp_min(1e-30)).clamp(max=1)[:, None]
        self.g = self.g + o.beta_hat * clipped
        self.m = self.m + o.beta_hat * clipped  # sigma_omega = 0
        self.aggregate = nnm_coordinate_median(self.m, self.faulty_bound)
        return {'gradient_norms': gradients.norm(dim=1).tolist(),
                'innovation_norms': norms.tolist(),
                'clipped_clients': int((norms > o.tau).sum()),
                'aggregate_norm': float(self.aggregate.norm())}


def full_gradient(model, x, y):
    """One complete local data pass at a fixed model; no optimizer step."""
    model.train()
    model.zero_grad(set_to_none=True)
    for xb, yb in zip(x.split(64), y.split(64)):
        loss = torch.nn.functional.cross_entropy(model(xb), yb, reduction='sum') / len(y)
        loss.backward()
    result = torch.cat([p.grad.detach().cpu().double().flatten() for p in model.parameters()])
    if not torch.isfinite(result).all():
        raise FloatingPointError('Non-finite full local gradient.')
    return result
