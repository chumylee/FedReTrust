"""Paths and output isolation for submission-only commands."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'Experiment/outputs/final_results/20261031_v06'
TABLES = RESULTS / 'tables'


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def new_output(kind, name):
    """Create only a new direct child of the package's disposable output area."""
    if not name or name in {'.', '..'} or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in name):
        raise ValueError('Output name must contain only letters, digits, underscores or hyphens.')
    area = ROOT / 'generated'
    area.mkdir(exist_ok=True)
    if area.is_symlink() or area.resolve().parent != ROOT:
        raise ValueError('Generated output area must be local, not a symlink.')
    path = area / f'{kind}-{name}'
    path.mkdir(exist_ok=False)
    return path
