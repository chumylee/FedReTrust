# Additional publication baselines

Run from the existing `fedretrust` repository:

```sh
python3 publication/run.py --extended
```

This extension requires the saved `publication/results/full` results from the
original comparison. It uses those exact data plans, initial models, library
versions and device. It does not rerun or overwrite the original methods.
The extension is currently intended for the existing Mac study, not an empty
clone without its saved results. Use `--extended --smoke` for a separate six-round
implementation check. Repeating a command resumes verified completed jobs.

Follow a running full comparison from another terminal:

```sh
tail -f publication/results/extended_baselines/run.log
```

Results are saved under `publication/results/extended_baselines/`:

- `progress.json`: completed job count and last recorded job.
- `results_table.csv`: running summary, with planned/completed/diverged counts.
- `results_by_seed.csv`: each seed's mean accuracy, macro precision and macro recall.
- `runs/<dataset>/<seed>/<method>/<condition>/per_round.csv`: round measurements.
- `confusions.npz` and `diagnostics.json` alongside each run: confusion counts and method diagnostics.
- `development/<dataset>/`: every calibration/tuning candidate, selected settings and data plan.
- `manifest.json` and `source_snapshot/`: protocol and exact code used.
- `completed.json`: written only after all planned jobs are recorded; inspect divergence count too.

Generate the side-by-side comparison with the four existing Table 2 methods:

```sh
publication/.venv/bin/python publication/compare_extended.py
```

Partial results are marked with their completion counts. Do not interpret a
six-round smoke test or an incomplete seed average as a publication result.

## Frozen evaluation plan

Two additional methods × three datasets × five paired seeds × three conditions
= **90 new comparison jobs**, each with 48 communication rounds. Conditions are
clean, honest input drift, and label flipping, in separate runs. Existing onset,
drift ramp, affected clients, models, samples, initializations and evaluation
definitions are retained. Metrics average rounds 5–48. Macro metrics use the
same confusion-based definitions and zero-division rule as Table 2.

Before comparison, each dataset uses a different development seed (first
evaluation seed + 50,000). Only official training data are used for development
scoring. The official test split is never scored to select settings.

## FLAURA reconstruction

Source: https://www.nature.com/articles/s41598-026-50985-2

This extends the project's earlier documented Algorithm 1 reconstruction.
It is **not author code or a reproduction of the paper's published scores**.
The paper does not fully specify all implementation settings. Choices are:

- Penultimate features are the final hidden ReLU activations: 128-dimensional
  for Fashion-MNIST/EMNIST and 32-dimensional for HAR.
- Auxiliary inputs come only from the existing reserved root split, disjoint
  from clients: 100 Fashion-MNIST, 94 EMNIST, and 96 HAR examples. Labels are used
  for balanced selection, not trust scoring or training. This is extra server
  data access relative to FedReTrust.
- Six clean FedAvg development rounds calibrate a fixed sigma as the median
  of all client-to-geometric-median representation distances. No test scores or
  attack/drift outcomes are used. This is an explicit clean-development privilege.
- Alpha=0.5, beta=0.5, gamma=0.8. Use Algorithm 1's Gaussian denominator
  `2*sigma^2` and threshold `gamma*mean(trust)`.
- RBF MMD uses a pooled median positive squared-distance bandwidth. The knee
  is the largest adjacent gap, with ties retained and flat profiles keeping peers.
- Trust-only weights follow Algorithm 1. The prose mentions rebalancing without
  an operational formula; it is not invented here.
- Local training remains the original SGD protocol. No changes to FedReTrust.

## Byz-Clip21-SGD2M without DP noise

Source: https://arxiv.org/html/2603.23472v2 (Algorithm 1 and Appendix F).
The paper links https://anonymous.4open.science/r/ByzClip21SGD2M; that repository
could not be accessed during this implementation. This is an equation-based
reconstruction, not verified author code.

This is a **gradient-based baseline with a declared protocol difference**:

- Each client computes its mean gradient over its entire local sample set at
  the same shared model. Microbatches of 64 accumulate that gradient. There
  are no local optimizer steps. This matches one local data pass per round,
  but not the optimizer trajectory of the local-SGD baselines.
- Use persistent client momentum, clipped innovation/error-feedback state,
  server state, then nearest-neighbour mixing (NNM) and coordinate median.
- NNM retains `n-f` nearest vectors; `f=floor(n/3)` is fixed even during clean
  and honest-drift runs. This gives the method a declared fault upper bound;
  it never receives affected-client identities. Even coordinate medians use
  the midpoint of the central two values; distance ties use stable client order.
- DP noise is zero; beta=0.1 and beta_hat=1. There is no privacy claim.
- All buffers start at zero. Algorithm 1 applies the previous aggregate before
  collecting new gradients, so its first model update is a no-op. At 48 rounds
  there are 48 gradient passes and 47 potentially nonzero model updates.
- Label-poisoned clients use the same recursion with flipped labels. This is
  data poisoning, not a reproduction of arbitrary Byzantine message attacks.
- A prespecified clean development grid tests learning rates 0.01, 0.1, 1, 10
  and clipping radii 0.1, 1 for 24 rounds (eight candidates per dataset).
  Choose the lowest final cross-entropy on the reserved development root set.
  All candidates, including failures, are retained. This is additional labeled
  development access; no runtime reference data are supplied to this method.

The extension therefore adds 18 FLAURA calibration rounds and 576 Byz-Clip21
development rounds before/during the 90 comparison jobs. Settings are frozen
per dataset before that dataset's evaluation begins. Full comparison results
must be reviewed for convergence, fairness and failures before inclusion in the
paper. They cannot establish superiority over the authors' original protocols.

## Validation

```sh
cd publication
.venv/bin/python -m unittest test_extended -v
```

Tests cover independently calculated NNM/median and clipping recurrence,
full-gradient equivalence without local parameter updates, all three model
feature extractors, exact agreement with the previous FLAURA implementation,
auxiliary-data separation, Byz-Clip21's update ordering, and resume protection.
