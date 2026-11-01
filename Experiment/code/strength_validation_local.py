"""Fixed per-client contribution curves; client data stays inside evaluation."""
import time
import torch
from torch.utils.data import DataLoader
from experiment_core import MnistCNN
from client_validation_local import _validate_models, _validate_views, _loss_vectors, _gain_summary
from aggregate_validation_local import build_aggregate_counterfactuals

SCALES = (0., .25, .5, 1.)
SEMANTICS = {
    'model': 'G_i(s) = w + sum_{j!=i} 0.1*delta_j + s*0.1*delta_i',
    'gain': 'loss(G_i(0))-loss(G_i(s)); accuracy gain has the reverse order',
    'weights': 'Only one client changes at a time; other nine keep weight 0.1; no renormalization.',
    'zero': 'An identical-model zero action has exactly zero utility and is never selected from a claimed report.',
    'scope': 'Counterfactual contribution curves, not an aggregate applying all chosen scales together.'}


@torch.inference_mode()
def build_strength_models(broadcast, submissions):
    full, without = build_aggregate_counterfactuals(broadcast, submissions)
    models = {0.: without, .25: {i: {} for i in range(10)},
              .5: {i: {} for i in range(10)}, 1.: {i: full for i in range(10)}}
    for name, ref in broadcast.items():
        origin = ref.detach().double()
        contributions = torch.stack([.1*(submissions[i][name].detach().double()-origin) for i in range(10)])
        total = contributions.sum(0)
        for s in (.25, .5):
            for cid in range(10):
                value = (origin + (total-contributions[cid]+s*contributions[cid])).to(ref.dtype)
                if not torch.isfinite(value).all():
                    raise FloatingPointError('Non-finite scaled model.')
                models[s][cid][name] = value
    return models


def _named_summary(summary):
    result = {}
    for key, value in summary.items():
        if key.startswith('baseline_'):
            key = 'without_client_' + key[len('baseline_'):]
        elif key.startswith('candidate_'):
            key = 'scaled_' + key[len('candidate_'):]
        result[key] = value
    return result


def evaluate_strength_models(models, report_view, audit_view, config, validator_id):
    if set(models) != set(SCALES):
        raise ValueError('Exactly the four declared scales are required.')
    for states in models.values():
        _validate_models(models[1.][0], states, validator_id)
    # Scale 1 is shared; ensure it really is the same state for every candidate.
    if any(not torch.equal(v, models[1.][0][k]) for state in models[1.].values() for k, v in state.items()):
        raise ValueError('Full aggregate must be identical for all candidates.')
    _validate_views(report_view, audit_view)
    if type(config.evaluation_batch_size) is not int or config.evaluation_batch_size < 1:
        raise ValueError('Invalid evaluation batch size.')
    batch_size = min(128, config.evaluation_batch_size)
    started, outputs, batches_used = time.perf_counter(), {}, 0
    with torch.random.fork_rng(devices=[]), torch.inference_mode():
        model = MnistCNN(); model.eval()
        for role, view in (('report', report_view), ('audit', audit_view)):
            loader = DataLoader(view, batch_size=batch_size, shuffle=False, num_workers=0,
                                generator=torch.Generator().manual_seed(0))
            batches = [(x.detach().clone(), y.detach().clone()) for x, y in loader]
            if any(x.device.type != 'cpu' or y.device.type != 'cpu' for x,y in batches):
                raise ValueError('Client-local evaluation requires CPU batches.')
            targets = torch.cat([y for _, y in batches])
            if len(targets) != len(view): raise ValueError('Incomplete local view.')
            flipped_targets = 9-targets
            def evaluate(state):
                nonlocal batches_used
                model.load_state_dict(state, strict=True)
                logits = torch.cat([model(x) for x, _ in batches])
                batches_used += len(batches)
                return (_loss_vectors(logits, targets),
                        _loss_vectors(logits, flipped_targets) if role == 'report' else None)
            full = evaluate(models[1.][0])
            rows = []
            for cid in range(10):
                without = evaluate(models[0.][cid])
                measured = {0.: without, 1.: full}
                for scale in (.25, .5): measured[scale] = evaluate(models[scale][cid])
                for scale in SCALES:
                    current = measured[scale]
                    row = {'candidate_id': cid, 'validator_id': validator_id, 'scale': scale,
                           **_named_summary(_gain_summary(without[0], current[0], targets))}
                    if role == 'report':
                        summary = _named_summary(_gain_summary(without[1], current[1], flipped_targets))
                        row.update({'flipped_'+k: v for k,v in summary.items() if k != 'examples'})
                    rows.append(row)
            outputs[role+'_rows'] = rows
    outputs['cost'] = {'validator_id': validator_id, 'models_per_role': 31,
        'report_examples': len(report_view), 'audit_examples': len(audit_view),
        'report_forward_examples': 31*len(report_view), 'audit_forward_examples': 31*len(audit_view),
        'total_forward_examples': 31*(len(report_view)+len(audit_view)),
        'full_aggregate_models_per_role': 1, 'omission_models_per_role': 10,
        'fractional_models_per_role': 20, 'forward_batches': batches_used,
        'evaluation_batch_size': batch_size, 'optimizer_steps': 0, 'additional_training_examples': 0,
        'elapsed_seconds': time.perf_counter()-started}
    return outputs


def verify_endpoint_rows(rows, previous):
    """Require fresh 100% predictions and statistics to equal the frozen pilot."""
    current = {(r['candidate_id'],r['validator_id']):r for r in rows if r['scale']==1.}
    expected = {(r['candidate_id'],r['validator_id']):r for r in previous}
    if len(current)!=100 or len(previous)!=100 or set(current)!=set(expected):
        raise ValueError('Incomplete endpoint comparison.')
    for key, old in expected.items():
        new = current[key]
        for name, value in old.items():
            if name == 'evaluation_semantics': continue
            target = name.replace('full_aggregate_', 'scaled_')
            if new[target] != value:
                raise AssertionError(f'100% endpoint changed: {key}, {name}')
    zero = [r for r in rows if r['scale']==0.]
    if len(zero)!=100: raise ValueError('Missing zero controls.')
    for row in zero:
        for key, value in row.items():
            if key.endswith('_gain') and value != 0.:
                raise AssertionError('Zero action must have exactly zero utility.')
            if key.endswith('_gains') and any(x is not None and x != 0. for x in value):
                raise AssertionError('Zero action must have exactly zero class utility.')
    return 100
