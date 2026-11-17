"""Replay the four-dataset paper using its original, unmodified metric runners."""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys

import torch
from torchvision.datasets import MNIST
from torchvision.datasets.utils import check_integrity

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('auto', 'mps', 'cpu', 'cuda'), default='auto')
    args = parser.parse_args()
    if args.device not in ('auto', 'mps') or not torch.backends.mps.is_available():
        parser.error('The recorded paper replay requires Apple Silicon/MPS. '
                     'Use reproduce.py to verify the saved results on any platform, '
                     'or run.py without --paper for the older portable training protocol.')
    evidence = HERE / 'evidence/full'
    destination = HERE / 'results/full'
    if not destination.exists():
        shutil.copytree(evidence, destination)
    if (destination / 'manifest.json').read_bytes() != (evidence / 'manifest.json').read_bytes():
        parser.error('results/full belongs to a different run. Use a fresh clone for the paper replay.')
    data = HERE / 'data/mnist'
    if not all(check_integrity(data / filename, digest) for filename, digest in MNIST.resources):
        MNIST(str(data), train=True, download=True)
        for filename, _ in MNIST.resources:
            shutil.copyfile(data / 'MNIST/raw' / filename, data / filename)
    print('Replaying 240 paper jobs: 4 datasets x 4 methods x 3 conditions x 5 seeds. '
          'Verified completed jobs are resumed. Frozen evidence stays in publication/evidence.', flush=True)
    for runner in ('table2_metrics.py', 'clean_metrics.py', 'mnist_study.py'):
        subprocess.run([sys.executable, str(HERE / runner)], cwd=HERE.parent, check=True)
    subprocess.run([sys.executable, str(HERE / 'reproduce.py'),
                    '--evidence', str(HERE / 'results'),
                    '--output', str(HERE.parent / 'generated/publication-replay'),
                    '--check-confusions'], cwd=HERE.parent, check=True)


if __name__ == '__main__':
    main()
