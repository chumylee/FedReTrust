"""Verify the portable package; this is not the full 13 GB archive verifier."""
import json
from pathlib import Path
from common import ROOT, digest


def checked_path(relative):
    path = (ROOT / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(ROOT):
        raise ValueError('Path outside submission: ' + relative)
    return path


def verify():
    manifest = json.loads((ROOT / 'provenance/package_manifest.json').read_text())
    for relative, expected in manifest['files_sha256'].items():
        if digest(checked_path(relative)) != expected:
            raise ValueError('Package hash mismatch: ' + relative)
    mapping = json.loads((ROOT / 'provenance/source_map.json').read_text())
    for record in mapping['files']:
        if digest(checked_path(record['destination'])) != record['sha256']:
            raise ValueError('Copied source differs: ' + record['destination'])
    frozen = json.loads((ROOT / 'Experiment/outputs/research_freeze/20261031_v06_final/research_freeze.json').read_text())
    for relative, expected in frozen['scientific_sources_sha256'].items():
        if digest(checked_path(relative)) != expected:
            raise ValueError('Scientific source differs: ' + relative)
    if len(frozen['scientific_sources_sha256']) != 73:
        raise ValueError('Expected 73 scientific sources.')
    return {'status': 'verified', 'packaged_files': len(manifest['files_sha256']),
            'byte_identical_copies': sum(not r.get('transformation') for r in mapping['files']),
            'document_only_transformations': sum(bool(r.get('transformation')) for r in mapping['files']),
            'frozen_scientific_sources': 73,
            'scope': 'Portable package only; omitted raw run artifacts were not verified.'}


if __name__ == '__main__':
    print(json.dumps(verify(), indent=2))
