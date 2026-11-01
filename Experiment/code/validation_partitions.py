"""Generate the ten predeclared partitions without training or evaluation."""
from dataclasses import asdict
from pathlib import Path
import copy
import json
import numpy as np
from torchvision import datasets
from experiment_core import file_sha256, write_json
from non_iid_partitions import LabelSkewConfig, prepare_label_skew, class_counts, distribution_metrics
from research_runs import canonical_hash


def build_manifest(original, labels, block, regime, protocol_sha256):
    if regime not in ('IID', 'non-IID'):
        raise ValueError('Unknown regime.')
    original = copy.deepcopy(original)
    source = original['indices']
    pool = sorted(i for values in source['clients'].values() for i in values)
    if len(pool) != 53700 or len(set(pool)) != 53700:
        raise ValueError('Expected the declared original client pool.')
    seed = block['iid_partition_seed' if regime == 'IID' else 'non_iid_partition_seed']
    if regime == 'IID':
        order = np.random.default_rng(seed).permutation(pool)
        clients = {str(i): order[i * 5370:(i + 1) * 5370].tolist() for i in range(10)}
        generation = {'method': 'one_permutation_equal_chunks', 'partition_seed': seed}
    else:
        clients, generation = prepare_label_skew(labels, source['clients'], LabelSkewConfig(partition_seed=seed))
    if sorted(i for group in clients.values() for i in group) != pool or any(len(v) != 5370 for v in clients.values()):
        raise AssertionError('New partition changed client-pool membership or counts.')
    original['indices']['clients'] = clients
    original['purpose'] = 'Frozen v0.3 replication partition; no model outcomes used in generation.'
    original['settings'] = {**original['settings'], 'partition_seed': seed,
                            'partition_type': 'iid' if regime == 'IID' else 'balanced_dirichlet_label_skew'}
    if regime == 'non-IID':
        original['settings']['dirichlet_alpha'] = .5
    original['generation'] = {'protocol_sha256': protocol_sha256, 'block': block['block'], 'regime': regime,
        'generator_sha256': file_sha256(__file__), 'label_skew_source_sha256': file_sha256(Path(__file__).with_name('non_iid_partitions.py')),
        'algorithm': generation}
    original.pop('partition_id', None)
    original['partition_id'] = 'mnist_validation_' + ('iid' if regime == 'IID' else 'noniid') + '_' + canonical_hash(original)[:12]
    return original


def validate_manifest(manifest, source):
    for name in ('server_reference', 'development_validation', 'reserved_final_test'):
        if manifest['indices'][name] != source['indices'][name]:
            raise AssertionError('Reserved membership changed: ' + name)
    old = sorted(i for v in source['indices']['clients'].values() for i in v)
    new = sorted(i for v in manifest['indices']['clients'].values() for i in v)
    if new != old or len(set(new)) != len(new):
        raise AssertionError('Client pool changed.')
    flat = new + source['indices']['server_reference'] + source['indices']['development_validation']
    if sorted(flat) != list(range(60000)):
        raise AssertionError('Training assignments must be disjoint and complete.')


def prepare(root, protocol_path, destination):
    root, destination = Path(root), Path(destination)
    p = json.loads(Path(protocol_path).read_text())
    source_path = root / p['partition_design']['source']
    if file_sha256(source_path) != p['partition_design']['source_sha256']:
        raise ValueError('Original partition changed.')
    source = json.loads(source_path.read_text())
    for name in ('train-images-idx3-ubyte', 'train-labels-idx1-ubyte'):
        if file_sha256(root / 'data/MNIST/raw' / name) != source['dataset_files'][name]['raw_sha256']:
            raise ValueError('Training data changed.')
    labels = datasets.MNIST(root=root / 'data', train=True, download=False).targets.numpy()
    records = []
    for block in p['blocks']:
        for regime in ('IID', 'non-IID'):
            manifest = build_manifest(source, labels, block, regime, file_sha256(protocol_path))
            validate_manifest(manifest, source)
            # Recompute to verify generation is deterministic before freezing it.
            if build_manifest(source, labels, block, regime, file_sha256(protocol_path)) != manifest:
                raise AssertionError('Nondeterministic partition generation.')
            path = root / 'data/partitions' / manifest['partition_id'] / 'indices.json'
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                if json.loads(path.read_text()) != manifest:
                    raise ValueError('Preserve changed existing partition.')
            else:
                write_json(path, manifest)
            counts = class_counts(labels, manifest['indices']['clients'])
            records.append({'block': block['block'], 'regime': regime, 'partition_id': manifest['partition_id'],
                'path': str(path.relative_to(root)), 'sha256': file_sha256(path),
                'class_counts': counts.tolist(), 'distribution': distribution_metrics(counts)})
    result = {'protocol_sha256': file_sha256(protocol_path), 'generator_sha256': file_sha256(__file__),
        'partitions': records, 'new_partitions': 10, 'model_training': False, 'official_test_loaded': False,
        'all_reserved_sets_preserved': True, 'deterministic_regeneration_checked': True}
    destination.mkdir(parents=True, exist_ok=True)
    out = destination / 'partitions.json'
    if out.exists() and json.loads(out.read_text()) != result:
        raise ValueError('Existing partition inventory differs; preserve and investigate.')
    write_json(out, result)
    return result
