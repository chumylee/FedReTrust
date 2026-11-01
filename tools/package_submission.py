"""Seal the documented submission files and create a ZIP, excluding local outputs."""
import argparse
import json
from pathlib import Path
import zipfile
from common import ROOT, digest, new_output, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', default='release', help='Unique archive output label.')
    args = parser.parse_args()
    mapping = json.loads((ROOT / 'provenance/source_map.json').read_text())
    files = set()
    for record in mapping['files']:
        path = (ROOT / record['destination']).resolve()
        if not path.is_relative_to(ROOT) or digest(path) != record['sha256']:
            raise ValueError('Copied file changed: ' + record['destination'])
        files.add(path)
    freeze = json.loads((ROOT / 'Experiment/outputs/research_freeze/20261031_v06_final/research_freeze.json').read_text())
    for relative, expected in freeze['scientific_sources_sha256'].items():
        if digest(ROOT / relative) != expected:
            raise ValueError('Frozen scientific source changed: ' + relative)
    for name in ('README.md', 'CHANGELOG.md', 'THIRD_PARTY_NOTICES.md', '.gitignore',
                 'requirements-demo.txt', 'requirements.txt', 'provenance/source_map.json',
                 'provenance/publication.json', 'provenance/step5_checks.json',
                 'provenance/step6_checks.json'):
        files.add(ROOT / name)
    for folder, pattern in [('tools', '*.py'), ('tests', '*.py'), ('docs', '*.md')]:
        files.update((ROOT / folder).glob(pattern))
    # Resolve and validate every allowlisted file; never sweep the full workspace.
    for path in files:
        if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(ROOT):
            raise ValueError('Invalid package file: ' + str(path))
    manifest = {'schema_version': 1, 'scope': 'Portable submission, not full raw archive',
                'files_sha256': {str(p.relative_to(ROOT)): digest(p) for p in sorted(files)}}
    out = new_output('archive', args.name)
    manifest_path = ROOT / 'provenance/package_manifest.json'
    write_json(manifest_path, manifest)
    files.add(manifest_path)
    archive = out / 'fedretrust-v06-submission.zip'
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as handle:
        for path in sorted(files):
            handle.write(path, 'fedretrust/' + path.relative_to(ROOT).as_posix())
    summary = {'files': len(files), 'uncompressed_bytes': sum(p.stat().st_size for p in files),
               'archive_bytes': archive.stat().st_size, 'archive_sha256': digest(archive),
               'archive': str(archive.relative_to(ROOT))}
    write_json(out / 'archive_summary.json', summary)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
