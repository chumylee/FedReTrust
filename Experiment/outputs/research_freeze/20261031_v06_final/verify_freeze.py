"""Read-only integrity check for the final project research selection.

Uses only the Python standard library. It never trains or rewrites evidence.
"""
from pathlib import Path
import hashlib
import json

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[3]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def project_path(relative):
    candidate = (PROJECT / relative).resolve()
    if Path(relative).is_absolute() or not candidate.is_relative_to(PROJECT):
        raise ValueError('Artifact must remain inside the project: ' + relative)
    return candidate


def verify():
    freeze = json.loads((HERE / 'research_freeze.json').read_text())
    assert digest(HERE / 'research_freeze.json') == json.loads((HERE / 'freeze_digest.json').read_text())['sha256']
    for name, expected in freeze['supporting_files_sha256'].items():
        assert digest(project_path(name)) == expected, name
    sources = freeze['scientific_sources_sha256']
    for name, expected in sources.items():
        assert digest(project_path(name)) == expected, name
    for name, expected in freeze['scientific_snapshot_sha256'].items():
        assert digest(project_path(name)) == expected, name
    evidence = json.loads((HERE / 'evidence_manifest.json').read_text())
    for name, expected in evidence['study_and_report_artifacts_sha256'].items():
        assert digest(project_path(name)) == expected, name
    identities, raw_count = set(), 0
    for record in evidence['records']:
        job = record['job']
        identity = tuple(job[k] for k in ('method', 'regime', 'block', 'scenario'))
        assert identity not in identities, identity
        identities.add(identity)
        for name, expected in record['artifact_sha256'].items():
            assert digest(project_path(record['run_path'] + '/' + name)) == expected, name
            raw_count += 1
    assert len(identities) == freeze['authoritative_run_count'] == 880
    assert raw_count == freeze['raw_artifact_count'] == 11440
    assert len(sources) == freeze['scientific_source_count'] == 73
    selection = json.loads((HERE / 'selection.json').read_text())
    assert selection['selected_method'] == freeze['selected_method'] == 'FedReTrust-v06'
    assert selection['full_success_targets_met'] is False
    return {'status': 'verified', 'selected_method': selection['selected_method'],
        'scientific_sources': len(sources), 'authoritative_runs': len(identities),
        'raw_artifact_hashes': raw_count,
        'study_and_report_artifacts': len(evidence['study_and_report_artifacts_sha256']),
        'research_freeze_sha256': digest(HERE / 'research_freeze.json')}


if __name__ == '__main__':
    print(json.dumps(verify(), indent=2))
