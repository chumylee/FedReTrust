"""Check local MNIST training bytes against the frozen partition fingerprints."""
import argparse
import json
from pathlib import Path
from common import ROOT, RESULTS, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'Experiment/data',
                        help='Directory containing MNIST/raw; no downloads are performed.')
    args = parser.parse_args()
    protocol = json.loads((RESULTS / 'protocol_and_scenarios.json').read_text())
    for part in protocol['partitions']:
        path = ROOT / 'Experiment' / part['path']
        if digest(path) != part['sha256']:
            raise ValueError('Partition differs: ' + part['path'])
    manifest = json.loads((ROOT / 'Experiment' / protocol['partitions'][0]['path']).read_text())
    for name in ('train-images-idx3-ubyte', 'train-labels-idx1-ubyte'):
        if digest(args.data_dir / 'MNIST/raw' / name) != manifest['dataset_files'][name]['raw_sha256']:
            raise ValueError('MNIST training file differs: ' + name)
    print(json.dumps({'status': 'verified', 'paired_partitions': 10, 'training_files': 2,
                      'official_test_opened': False}, indent=2))


if __name__ == '__main__':
    main()
