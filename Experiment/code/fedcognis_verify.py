"""Verify completed FedCognis study, interpretation and executed report."""
from pathlib import Path
import json
import shutil
import nbformat
import pandas as pd
from experiment_core import file_sha256, write_json
from directional_study import checked_result
from fedcognis_study import ROOT, STUDY, verify, records


def check():
    state = json.loads((STUDY / 'status.json').read_text())
    assert state['status'] == 'complete' and state['new_runs'] == 70
    protocol = verify(STUDY)
    new = records(STUDY); assert len(new) == 70
    manifest = json.loads((STUDY / 'result_manifest.json').read_text())
    all_records = manifest['records'] + manifest['context_records']
    assert len(all_records) == 424
    for r in all_records:
        checked_result(r)
        for name, digest in r['source_sha256'].items():
            assert file_sha256(ROOT / 'code' / name) == digest
    summary = json.loads((STUDY / 'analysis/summary.json').read_text())
    assert summary['analysis_source_sha256'] == file_sha256(ROOT / 'code/fedcognis_analysis.py')
    assert len(summary['matching_checks']) == 424 and all(x['matched'] for x in summary['matching_checks'])
    assert len(summary['pre_event_checks']) == 356 and all(x['exact'] for x in summary['pre_event_checks'])
    assert summary['exact_first_round_client_updates_vs_fedavg'] == 700
    assert summary['audited_new_rounds'] == 1680 and summary['maximum_reconstruction_error'] <= 1e-6
    audit = json.loads((STUDY / 'access_audit.json').read_text())
    assert audit['official_test_used'] is False and audit['defence_uses_evaluation_data'] is False
    assert audit['server_reference_examples'] == audit['calibration_runs'] == 0
    assert audit['totals'] == dict(client_updates=16800, optimizer_steps=134400,
        train_forward_examples=8601600, reference_forward_examples=0,
        development_task_forward_examples=5250000, checkpoint_replay_forward_examples=420000)
    decision = json.loads((STUDY / 'decision.json').read_text())
    for name, digest in decision['evidence_sha256'].items():
        assert file_sha256(STUDY / name) == digest
    tests = (STUDY / 'preflight_tests.log').read_text()
    assert 'Ran 291 tests' in tests and tests.rstrip().endswith('OK')
    notebook_path = ROOT / 'code/28_fedcognis_comparison.ipynb'
    notebook = nbformat.read(notebook_path, as_version=4); nbformat.validate(notebook)
    cells = [c for c in notebook.cells if c.cell_type == 'code']
    assert len(cells) == 7 and [c.execution_count for c in cells] == list(range(1, 8))
    assert not any(o.output_type == 'error' for c in cells for o in c.outputs)
    assert sum('image/png' in o.get('data', {}) for c in cells for o in c.outputs) == 4
    names = ['fedcognis_honest_trajectories', 'fedcognis_label_flip', 'fedcognis_attack_stops', 'fedcognis_influence']
    figures = [STUDY / 'analysis/figures' / (name + ext) for name in names for ext in ('.png', '.pdf')]
    assert all(p.is_file() and p.stat().st_size > 10000 for p in figures)
    review = json.loads((STUDY / 'visual_review.json').read_text())
    assert review['passed'] is True and review['reviewed_figures'] == names
    for p in figures: assert file_sha256(p) == review['figure_sha256'][p.name]
    rows = {p.name: len(pd.read_csv(p, low_memory=False)) for p in (STUDY / 'analysis').glob('*.csv')}
    assert rows['local_training.csv'] == 101760 and rows['raw_task_metrics.csv'] == 31800
    assert rows['trust.csv'] == 16800
    for source, destination in [('/tmp/fedcognis-run.log', 'coordinator.log'), ('/tmp/fedcognis-notebook.log', 'notebook_build.log')]:
        shutil.copy2(source, STUDY / destination)
    tracked = [ROOT / 'README.md', notebook_path, Path(__file__), ROOT / 'code/fedcognis_notebook.py']
    tracked += [p for p in STUDY.iterdir() if p.is_file() and p.name not in ('verification.json', 'coordinator.lock')]
    tracked += [p for p in (STUDY / 'analysis').rglob('*') if p.is_file()]
    result = {'status': 'passed', 'preflight_tests': 291, 'new_runs': 70, 'checked_artifact_bundles': 424,
        'frozen_sources': len(protocol['source_sha256']), 'unchanged_parent_artifacts': len(protocol['parent_artifact_sha256']),
        'matched_input_pairs': 424, 'exact_pre_event_trajectories': 356, 'exact_first_round_client_updates': 700,
        'audited_new_rounds': 1680, 'maximum_reconstruction_error': summary['maximum_reconstruction_error'],
        'executed_notebook_cells': 7, 'embedded_png_outputs': 4, 'visually_reviewed_figures': 4, 'exported_figures': 8,
        'server_reference_examples': 0, 'official_test_used': False, 'workload': audit['totals'],
        'table_rows': rows, 'artifact_sha256': {str(p.relative_to(ROOT)): file_sha256(p) for p in tracked}}
    write_json(STUDY / 'verification.json', result)
    return result


if __name__ == '__main__': print(json.dumps(check(), indent=2))
