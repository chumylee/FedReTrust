# FedReTrust publication artifact

Companion code and saved evidence for **FedReTrust: Balancing Attack Protection
and Honest Client Adaptation in Federated Learning as a Service**.

The paper evaluates **FedReTrust, RFA, FLTrust and adapted Viceroy** on
**MNIST, Fashion-MNIST, EMNIST Balanced and Smartphone HAR** under clean
training, persistent label poisoning and gradual honest input drift.
`FedReTrust-v06` is the implementation identifier for the method called
FedReTrust in the paper.

## Reproduce the paper table without training

Git and Python 3.10 or newer are sufficient; no GPU, ML packages or dataset
downloads are needed for this command:

```sh
git clone https://github.com/chumylee/FedReTrust.git
cd FedReTrust
python3 publication/reproduce.py
```

This validates the recorded file hashes, matched initial models, original
accuracy curves and per-round metric means. It then rebuilds the table and
paired differences in `generated/publication/`. On Windows use
`py -3 publication/reproduce.py`.

- [View the complete paper table](tables/results_table.md)
- [Download table values as CSV](tables/results_table.csv)
- [Inspect the 240 individual job summaries](tables/results_by_seed.csv)
- [Paired differences and nominal confidence intervals](tables/paired_differences.csv)
- [Validation report](tables/validation.json)

There are **240 recorded jobs, 236 completed and four diverged**. Viceroy
diverged for Fashion-MNIST seed 1111201 in all three conditions and for HAR
seed 921202 under honest drift. Failed runs and partial measurements are
retained; their incomplete trajectories are excluded from the reported means.
The corresponding table entries use four completed repeats instead of five.

## What the metrics mean

Each job trains for 48 communication rounds. We average each metric over
rounds 5–48 within a completed run, then average those run means. The reported
values are percentages, not final-round scores. Precision and recall use
**macro averaging** with zero for undefined class scores. The table uses the
short labels Precision and Recall.

Clean and poisoning results use unchanged test inputs with correct test labels.
Honest-drift results mix unchanged and shifted predictions according to the
round's drift fraction, and weight true classes using the affected clients'
mean class profile. Thus honest-drift scores describe the affected clients.
The image test subsets are balanced, so macro recall equals accuracy for clean
training and poisoning. HAR retains its original test-class imbalance.

Five seeds are matched across methods on initial models, client examples and
affected-client assignments. The paired intervals use Student's t with four
degrees of freedom when all five pairs are complete. They are nominal 95%
intervals, without a multiple-comparison correction; incomplete pairs have no
reported interval. The per-round CSVs and confusion matrices support checking
all three metrics independently.

## Train the paper experiments again

The exact recorded protocol uses Apple Silicon **MPS**, Python **3.14.8**,
PyTorch **2.13.0** and NumPy **2.5.2**. From the repository root on an Apple
Silicon Mac, run:

```sh
python3 publication/run.py --paper
```

The launcher accepts Python 3.12–3.14 and can bootstrap Python 3.12 through
Conda when needed. For the closest reproduction use the recorded Python
3.14.8 environment. It prepares pinned dependencies and reference PDFs,
downloads verified public datasets, copies the original plans into the live
results area, and executes the three metric runners. This is **240 training
jobs** on a fresh clone. Completed jobs are verified and resumed after an
interruption. Keep the Mac awake; allow several hours depending on hardware.
Backend and software differences can change training results.

Progress is printed to the terminal every eight rounds. To also retain a log:

```sh
python3 -u publication/run.py --paper 2>&1 | tee publication/replay.log
```

The live outputs go to `publication/results/`; the verified replay table goes
to `generated/publication-replay/`. The frozen evidence in `publication/evidence/`
is preserved. Run `python3 publication/reproduce.py --evidence publication/results
--output generated/publication-replay --check-confusions` on one line to
recheck a replay with a Python environment that has NumPy installed.

To independently recalculate the *saved* paper metrics from confusion counts:

```sh
python3 publication/run.py --setup-only
publication/.venv/bin/python publication/reproduce.py --check-confusions
```

The default table-reproduction command works on any platform. Exact training
replay intentionally keeps the original MPS backend. The older portable
runner, `python3 publication/run.py`, supports CPU, CUDA and MPS but runs the
original 225-job, three-dataset study including FedAvg; it is not the paper's
four-dataset metric replay. `--smoke` runs a short check of that older protocol.
Use a separate clone for that study to keep its live output distinct.

## Code and evidence map

| File or folder | Purpose |
| --- | --- |
| [reproduce.py](reproduce.py) | Rebuild and validate the paper table and paired comparisons. |
| [run.py](run.py), [replay_paper.py](replay_paper.py) | Dependency setup and one-command paper replay. |
| [study.py](study.py) | Shared local training and aggregation loop; records non-finite failures. |
| [data.py](data.py) | Pinned datasets, client splits, normalisation and intervention plans. |
| [table2_metrics.py](table2_metrics.py) | Poisoning/drift replay and confusion-based scoring. |
| [clean_metrics.py](clean_metrics.py) | Clean metric replay for Fashion-MNIST, EMNIST and HAR. |
| [mnist_study.py](mnist_study.py) | MNIST runs for all three conditions. |
| [FedReTrust implementation](../Experiment/code/fedretrust_v06.py) | Evaluated FedReTrust algorithm. |
| [RFA](../Experiment/code/rfa.py), [FLTrust](../Experiment/code/fltrust.py), [Viceroy](../Experiment/code/viceroy.py) | Baseline implementations and settings. |
| [evidence/full](evidence/full) | Original three-dataset study, including FedAvg, seed plans and provenance. |
| [evidence/table2_metrics](evidence/table2_metrics) | 120 poisoning/drift metric jobs for the three non-MNIST datasets. |
| [evidence/clean_metrics](evidence/clean_metrics) | 60 clean metric jobs for those same datasets. |
| [evidence/mnist_publication](evidence/mnist_publication) | 60 MNIST metric jobs. |
| [artifact_manifest.json](artifact_manifest.json) | File hashes for this publication artifact. |

Each metric run includes `result.json`, `per_round.csv` and `confusions.npz`.
Source snapshots and manifests preserve the code and settings used at training
time. There are no trained-model checkpoints: the retained evidence consists
of measurements, confusion counts, data plans and source code. Dataset archives,
Python environments, local logs and caches are excluded from Git.

Images use ten clients with 512 examples each and Dirichlet label skew
(alpha 0.5). HAR uses 19 training subjects and nine separate test subjects.
FLTrust receives disjoint, balanced server reference data; FedReTrust does not.
The adapted Viceroy implementation aggregates local model deltas with its raw
weighted sum and a unit server step. These implementation choices and the
observed divergences limit claims about the original Viceroy protocol.

The separate [exploratory baseline extension](EXTENDED_BASELINES.md) contains
FLAURA and Byz-Clip21 reconstructions. Those methods are not included in the
paper table. Their code is provided separately from the reported comparison.

Dataset sources: [MNIST](https://www.tensorflow.org/datasets/catalog/mnist),
[Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist),
[EMNIST Balanced](https://www.nist.gov/itl/products-and-services/emnist-dataset)
([pinned archive mirror](https://huggingface.co/datasets/Royc30ne/emnist-balanced/tree/e6fd3608dc5d46960a89a423472f67c9c0f9ec31)),
and [UCI Smartphone HAR](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones).
Third-party implementation attribution remains in [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).
