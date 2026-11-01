# Reproducibility verification — 1 November 2026

The original research workspace and its frozen evidence were read only during
this review. No new study trajectories were trained or added to reported means.

## Checks that passed

| Check | Result |
| --- | --- |
| Original `verify_freeze.py` | 73 frozen scientific source files, 880 distinct runs, 11,440 raw artifact hashes and 184 study/report artifact hashes verified |
| Submission ZIP in a temporary folder outside the project | Package manifest, 239 byte-identical copied files, one documented notebook path change and 73 frozen scientific source hashes verified |
| Saved-measurement reproduction in the temporary folder | 1,298 five-block summaries, 636 paired contrasts and 110 main-table cells match frozen values at absolute tolerance 1e-9 |
| Saved evidence audit | 880 unique run/access/settings identities; 40 v0.6 runs covering eight regime/scenario cells with five blocks each; 45 passed and six failed follow-up checks retained |
| Declared data access in 880 saved access records | No official test use or evaluator data sent to defences; all v0.6 runs record zero server reference images |
| Synthetic algorithm demonstration in the temporary folder | Unchanged v0.6 and its contribution recorder executed; output separate from study evidence |
| Portable guard tests and selected existing tests | 110 checks passed: five new guards, 24 v0.5/v0.6, ten scenario, 16 RFA, 25 scenario-metric, 12 FLTrust and 18 Viceroy |
| Three baseline-paper reference PDFs | Downloaded from the pinned public URLs, checked against frozen SHA-256 values, then used for broad baseline tests |

Initial isolated RFA/scenario-metric tests found four errors because historical
baseline constructors hash local paper PDFs excluded from the public ZIP. The
new `tools/prepare_reference_papers.py` restores exactly those verified runtime
inputs without editing the scientific code or adding the PDFs to Git. The
previously failing suites passed after this documented setup step.

The five-block sample supports exploratory intervals only. All blocks were
known development conditions. The original 25/27 failed v0.6 pilot remains
failed; 45/51 follow-up checks indicate partial success. No v0.6 recovery,
component ablation, official test, new dataset or adaptive-attack result exists.

The portable results command recalculates summaries from saved block metrics;
it does not independently extract every scalar from the 13 GB raw archive.
The full archive verifier establishes byte integrity, not independent
confirmation of the scientific conclusions. The temporary-folder execution
used the existing Python 3.14 dependencies; a fresh clean-machine install,
cross-platform match, checkpoint replay and 880-run retraining were not done.

The release archive remains small because it excludes raw MNIST, checkpoints,
large logs and the three cached paper PDFs. `tools/check_data.py` verifies the
included ten final partition files and externally supplied MNIST training bytes.
Generated demo and reproduction outputs remain outside frozen study evidence.
