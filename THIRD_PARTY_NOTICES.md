# Attribution and third-party notices

FedReTrust's implementation and experimental combination should not be described
as an independently established novel aggregation principle or as inheriting
theoretical guarantees from the methods that motivated it.

- **RFA:** the geometric-median aggregation foundation. The preserved upstream
  files and BSD license are under `Experiment/resources/rfa_reference/`.
- **MNIST-C:** shot-noise and stripe recipes adapted from Google Research,
  copyright 2018 Google LLC, Apache-2.0. The pinned source, full license and
  adaptation manifest are under `Experiment/resources/mnist_c_reference/`.
- **Viceroy:** preserved upstream source, MIT license and adaptation notes under
  `Experiment/resources/viceroy_reference/`.
- **FLAURA and FedCognis:** reconstructed/adapted comparators, not claims to
  reproduce the complete author systems. Source manifests under the matching
  reference directories identify the articles and interpretation choices.
- **Clip21 / Byz-Clip21 motivation:** v0.6 tracks and clips changes in per-client
  update estimates. Its server-side adaptive-radius/local-SGD combination is
  not a reproduction of those methods and inherits no claimed convergence,
  Byzantine-robustness or differential-privacy theorem. The exact cited sources
  remain in `fedretrust_v06.py` and the frozen candidate settings.

MNIST is obtained separately via torchvision; dataset access details and exact
training-byte fingerprints are in `docs/DATA_AND_SETUP.md`.

Copied paper/article images are deliberately not distributed here. The preserved
reference manifests can mention those omitted research-cache files. Their
absence does not affect the portable commands or the scientific implementation.

No new license is assigned to the student's original code by this packaging step.
