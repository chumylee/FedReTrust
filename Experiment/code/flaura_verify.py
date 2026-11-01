"""Verify completed FLAURA artifacts and the executed report; no model inference."""
from pathlib import Path
import json
import shutil
import nbformat
import pandas as pd
from experiment_core import file_sha256, write_json
from flaura_study import ROOT, STUDY, records, verify
from directional_study import checked_result


def check():
    status = json.loads((STUDY / 'status.json').read_text())
    assert status['status'] == 'complete' and status['new_runs'] == 68
    protocol = verify(STUDY)
    for name, digest in protocol['source_sha256'].items():
        assert file_sha256(STUDY / 'source_snapshot' / name) == digest
    new = records(STUDY)
    assert len(new) == 68
    manifest = json.loads((STUDY / 'result_manifest.json').read_text())
    all_records = manifest['records'] + manifest['context_records'] + manifest['calibration_records']
    assert len(all_records) == 380
    for record in all_records:
        checked_result(record)
        for name, digest in record['source_sha256'].items():
            assert file_sha256(ROOT / 'code' / name) == digest
    summary = json.loads((STUDY / 'analysis/summary.json').read_text())
    assert len(summary['matching_checks']) == 378
    assert all(x['matched'] for x in summary['matching_checks'])
    assert all(x['exact'] for x in summary['pre_event_checks'])
    assert summary['exact_first_round_client_updates_vs_fedavg'] == 660
    assert summary['analysis_source_sha256'] == file_sha256(ROOT / 'code/flaura_analysis.py')
    errors, iterations, terminations = [], [], {}
    for record in new:
        diagnostics = json.loads((ROOT / record['run_path'] / 'aggregation_diagnostics.json').read_text())
        assert len(diagnostics) == record['job']['rounds']
        for row in diagnostics:
            values = row['values']
            errors.append(values['contribution_measurements']['reconstruction_max_abs_error'])
            solver = values['median_solver']
            iterations.append(solver['iterations_used'])
            terminations[solver['termination']] = terminations.get(solver['termination'], 0) + 1
    assert len(errors) == 1596 and max(errors) <= 1e-6
    audit = json.loads((STUDY / 'access_audit.json').read_text())
    assert not audit['official_test_used'] and audit['server_auxiliary_examples'] == 100
    expected = dict(client_updates=15960, optimizer_steps=127680, train_forward_examples=8171520,
                    reference_forward_examples=1596000, development_task_forward_examples=4992000,
                    checkpoint_replay_forward_examples=408000)
    assert audit['totals'] == expected
    decision = json.loads((STUDY / 'decision.json').read_text())
    for name, digest in decision['evidence_sha256'].items():
        assert file_sha256(STUDY / name) == digest
    tests = (STUDY / 'preflight_tests.log').read_text()
    assert 'Ran 279 tests' in tests and tests.rstrip().endswith('OK')
    notebook_path = ROOT / 'code/26_flaura_comparison.ipynb'
    notebook = nbformat.read(notebook_path, as_version=4)
    nbformat.validate(notebook)
    cells = [c for c in notebook.cells if c.cell_type == 'code']
    assert len(cells) == 7 and [c.execution_count for c in cells] == list(range(1, 8))
    assert not any(o.output_type == 'error' for c in cells for o in c.outputs)
    images = sum('image/png' in o.get('data', {}) for c in cells for o in c.outputs)
    assert images == 4
    names = ['flaura_honest_trajectories', 'flaura_label_flip', 'flaura_attack_stops', 'flaura_influence']
    figures = [STUDY / 'analysis/figures' / (name + extension) for name in names for extension in ('.png', '.pdf')]
    assert all(p.is_file() and p.stat().st_size > 10000 for p in figures)
    # This acknowledgement is written only after visual review of all four PNGs.
    visual = json.loads((STUDY / 'visual_review.json').read_text())
    assert visual['reviewed_figures'] == names and visual['passed'] is True
    for p in figures:
        assert visual['figure_sha256'][p.name] == file_sha256(p)
    table_rows = {p.name: len(pd.read_csv(p, low_memory=False)) for p in (STUDY / 'analysis').glob('*.csv')}
    assert table_rows['local_training.csv'] == 90720
    assert table_rows['raw_task_metrics.csv'] == 28350
    assert table_rows['trust.csv'] == 15840
    for origin, destination in [('/tmp/flaura-run.log', 'coordinator.log'), ('/tmp/flaura-notebook.log', 'notebook_build.log')]:
        shutil.copy2(origin, STUDY / destination)
    tracked = [ROOT / 'README.md', notebook_path, Path(__file__), ROOT / 'code/flaura_notebook.py']
    tracked += [p for p in STUDY.iterdir() if p.is_file() and p.name not in ('verification.json', 'coordinator.lock')]
    tracked += [p for p in (STUDY / 'analysis').rglob('*') if p.is_file()]
    report = dict(status='passed', preflight_tests=279, new_runs=68, checked_artifact_bundles=380,
                  frozen_source_snapshots=len(protocol['source_sha256']), matched_input_pairs=378,
                  exact_pre_event_trajectories=len(summary['pre_event_checks']),
                  exact_first_round_client_updates=660, audited_new_rounds=len(errors),
                  maximum_aggregate_reconstruction_error=max(errors), maximum_median_iterations=max(iterations),
                  median_terminations=terminations, executed_notebook_cells=7, embedded_png_outputs=4,
                  visually_reviewed_figures=4, exported_figures=8, table_rows=table_rows,
                  official_test_used=False, new_run_workload=audit['totals'],
                  artifact_sha256={str(p.relative_to(ROOT)): file_sha256(p) for p in tracked})
    write_json(STUDY / 'verification.json', report)
    return report


if __name__ == '__main__':
    print(json.dumps(check(), indent=2))
