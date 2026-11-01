"""Verify completed follow-up artifacts and the executed presentation."""
from pathlib import Path
import json
import shutil
import nbformat
import pandas as pd
from flaura_second_seed import ROOT, STUDY, verify, records
from directional_study import checked_result
from experiment_core import file_sha256, write_json


def check():
    assert json.loads((STUDY / 'status.json').read_text())['status'] == 'complete'
    protocol = verify(STUDY)
    assert len(records(STUDY)) == 6
    manifest = json.loads((STUDY / 'result_manifest.json').read_text())
    assert len(manifest['records']) == 132 and len(manifest['new_records']) == 6
    for r in manifest['records']:
        checked_result(r)
        for name, digest in r['source_sha256'].items():
            assert file_sha256(ROOT / 'code' / name) == digest
    summary = json.loads((STUDY / 'analysis/summary.json').read_text())
    assert summary['source_sha256'] == file_sha256(ROOT / 'code/flaura_second_seed.py')
    assert len(summary['matching_checks']) == 132 and len(summary['pre_event_checks']) == 88
    assert summary['exact_first_round_updates'] == 60 and summary['audited_new_rounds'] == 144
    assert summary['maximum_reconstruction_error'] == 0
    audit = json.loads((STUDY / 'access_audit.json').read_text())
    assert not audit['official_test_used'] and audit['new_calibrations'] == 0
    assert audit['server_auxiliary_images'] == 100
    assert audit['totals'] == dict(client_updates=1440, optimizer_steps=11520,
        train_forward_examples=737280, reference_forward_examples=144000,
        development_task_forward_examples=450000, checkpoint_replay_forward_examples=36000)
    decision = json.loads((STUDY / 'decision.json').read_text())
    for name, digest in decision['evidence_sha256'].items():
        assert file_sha256(STUDY / name) == digest
    tests = (STUDY / 'preflight_tests.log').read_text()
    assert 'Ran 279 tests' in tests and tests.rstrip().endswith('OK')
    notebook_path = ROOT / 'code/27_flaura_second_seed.ipynb'
    notebook = nbformat.read(notebook_path, as_version=4); nbformat.validate(notebook)
    cells = [c for c in notebook.cells if c.cell_type == 'code']
    assert len(cells) == 4 and [c.execution_count for c in cells] == [1, 2, 3, 4]
    assert not any(o.output_type == 'error' for c in cells for o in c.outputs)
    assert sum('image/png' in o.get('data', {}) for c in cells for o in c.outputs) == 2
    names = ['flaura_seed2_comparison', 'flaura_seed2_trajectories']
    figures = [STUDY / 'analysis/figures' / (n + ext) for n in names for ext in ('.png', '.pdf')]
    assert all(p.stat().st_size > 10000 for p in figures)
    visual = json.loads((STUDY / 'visual_review.json').read_text())
    assert visual['passed'] and visual['reviewed_figures'] == names
    for p in figures:
        assert file_sha256(p) == visual['figure_sha256'][p.name]
    rows = {p.name: len(pd.read_csv(p, low_memory=False)) for p in (STUDY / 'analysis').glob('*.csv')}
    assert rows['local_training.csv'] == 31680 and rows['raw_task_metrics.csv'] == 9900
    assert rows['trust.csv'] == 8640
    for src, dest in [('/tmp/flaura-seed2-run.log', 'coordinator.log'), ('/tmp/flaura-seed2-notebook.log', 'notebook_build.log')]:
        shutil.copy2(src, STUDY / dest)
    tracked = [ROOT / 'README.md', notebook_path, Path(__file__), ROOT / 'code/flaura_second_seed_notebook.py']
    tracked += [p for p in STUDY.iterdir() if p.is_file() and p.name not in ('verification.json', 'coordinator.lock')]
    tracked += [p for p in (STUDY / 'analysis').rglob('*') if p.is_file()]
    result = {'status': 'passed', 'new_runs': 6, 'checked_run_artifacts': 132,
        'preflight_tests': 279, 'frozen_sources': len(protocol['source_sha256']),
        'unchanged_parent_artifacts': len(protocol['parent_artifact_sha256']),
        'matched_input_pairs': 132, 'exact_pre_event_trajectories': 88,
        'exact_first_round_client_updates': 60, 'audited_new_rounds': 144,
        'maximum_aggregate_reconstruction_error': summary['maximum_reconstruction_error'],
        'maximum_median_iterations': summary['maximum_median_iterations'],
        'executed_notebook_cells': 4, 'embedded_png_outputs': 2, 'exported_figures': 4,
        'visually_reviewed_figures': 2, 'official_test_used': False, 'workload': audit['totals'],
        'table_rows': rows, 'artifact_sha256': {str(p.relative_to(ROOT)): file_sha256(p) for p in tracked}}
    write_json(STUDY / 'verification.json', result)
    return result


if __name__ == '__main__':
    print(json.dumps(check(), indent=2))
