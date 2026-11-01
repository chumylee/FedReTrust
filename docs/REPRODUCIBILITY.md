# Reproducibility and limits

## Portable, supported commands

Run from the submission root. Every output label must be new.

```sh
python3 -B tools/verify_submission.py
python3 -B tools/reproduce_results.py --name review
python3 -B tools/audit_saved_evidence.py
python3 -B -m unittest discover -s tests -p 'test_*.py'
```

With the ML dependencies installed:

```sh
python -B tools/demo.py --name review
python -B -m unittest discover -s Experiment/code/tests -p 'test_fedretrust_v0*.py'
```

Some broader baseline tests instantiate RFA, FLTrust and Viceroy, whose
unchanged constructors hash three cached paper PDFs. Run
`python3 -B tools/prepare_reference_papers.py --download` before those tests.
The PDFs are untracked and verified against the frozen originals. The v0.6
demo and targeted v0.5/v0.6 tests do not need them.

The verifier checks the submission manifest, every copied source/evidence hash,
and all 73 frozen scientific source hashes. The new reproduction command uses
the saved per-block scalar measurements, calculates sample standard deviations
and `mean ± 2.7764451051977987 * SD / sqrt(5)`, and compares every summary with
the frozen tables within an absolute tolerance of 1e-9. Paired intervals use
block differences with matching training seeds; regimes remain separate.
The saved-evidence audit cross-checks run, setting and data-access identities;
v0.6's 40-run coverage; unused official test/evaluation data as reported by
all 880 access records; and all six failed follow-up gates.

`n=5` counts blocks, not rounds or clients. Missing cells remain `NA/not_tested`.
The original 45/51 checks and all six failures stay unchanged. Reproduction
outputs are disposable derivatives, not replacements for the authoritative data.

## What this package reproduces

| Level | Included capability | Limitation |
| --- | --- | --- |
| Source integrity | Exact scientific source hashes and copied evidence hashes | Manifest consistency is not independent scientific confirmation |
| Saved summaries | All 1,298 summary rows, 636 paired comparisons and 110 main cells | Starts from block metrics, not raw client/round logs |
| Mechanism execution | Actual v0.6, small synthetic inputs, affine diagnostics | No MNIST training or accuracy claim |
| Full original evidence audit | Historical verifier and evidence manifest retained | Needs omitted raw archive and snapshots |
| Full retraining | Historical orchestration sources retained | Not a portable one-command rebuild; no 880-run rerun performed |

The frozen final figures are included as saved artifacts. The portable HTML view
uses recalculated numbers and those unchanged figures. Notebook 34 is supplied
with executed outputs for reading; its builder expects omitted full-archive
inputs. Do not execute it as the portable reproduction command.

## Full archive verification (owner's workspace only)

From the original project root, with the original raw evidence still present:

```sh
python3 -B Experiment/outputs/research_freeze/20261031_v06_final/verify_freeze.py
```

This hashes 11,440 raw artifacts representing 880 distinct runs, the scientific
sources, historical snapshots and indexed study/report artifacts. It can take
time because the raw archive is about 13 GB. The portable package deliberately
does not claim to have those bytes. Retained original manifests reference some
omitted paths; use `tools/verify_submission.py` for the portable package.

## Verified on 1 November 2026 and remaining limits

The original full-archive verifier passed: 73 scientific sources, 880 run
identities, 11,440 raw artifact hashes and 184 study/report artifacts. The
submission ZIP was extracted outside the project and verified; its saved
summaries, synthetic demo and relevant implementation/scenario tests ran there.
Three historical baseline-paper PDFs were downloaded separately and verified
against their frozen hashes before the broader baseline tests. See
`VERIFICATION_20261101.md` for the checks and the initial failures they exposed.

The isolated run reused the existing Python 3.14 environment and dependencies.
An independent fresh installation, cross-platform numerical identity, full
retraining and checkpoint replay have not been established here.

All study blocks informed development; no independent confirmation is claimed.
The official test set remains unevaluated. v0.6 recovery, component ablations,
adaptive attacks and broader datasets were not tested. No new demo/reproduction
output may be added to study means, gates or the 880-run count.
