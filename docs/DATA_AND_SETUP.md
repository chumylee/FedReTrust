# Data and environment

## Two installation levels

The portable integrity check and table reproduction need only Python 3.10+.
They do not import NumPy, PyTorch, pandas or a notebook runtime.

The aggregation demo and existing implementation tests use `requirements-demo.txt`.
Use Python 3.14 to match the recorded major/minor version; the study used 3.14.8
on macOS arm64 with CPU execution. `requirements.txt` additionally includes the
plotting/notebook-format dependencies used by historical source modules.
`Experiment/requirements.lock.txt` is the preserved historical environment list,
not a newly validated cross-platform lockfile. It includes macOS-specific tools.

```sh
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-demo.txt
```

On Windows, environment activation uses `.venv\Scripts\activate` instead.
Historical study orchestration also imports Unix-specific `fcntl`/`resource`;
Windows execution of those scripts has not been established. Do not silently
substitute dependency versions and claim exact research reproduction.

## MNIST

Neither the saved-results command nor the synthetic demo needs MNIST downloads.
For inspection or later separately labelled reproduction, the original data are
the official MNIST 60,000-image training split. The saved indices reserve:

- 53,700 examples for ten client populations (5,370 each);
- 6,000 development-evaluation examples;
- 300 clean server-reference examples, unused by FedReTrust v0.6.

Training selects 512 images per client, and evaluation selects 1,000 balanced
development images per task. Ten saved five-block IID/non-IID partitions are
included, along with the earlier partition metadata. Do not regenerate or tune
the partitions. The final protocol lists their exact IDs and SHA-256 hashes.

To obtain the dataset using the pinned torchvision loader, from the package root:

```sh
python -c 'from torchvision.datasets import MNIST; MNIST(root="Experiment/data", train=True, download=True)'
python -B tools/check_data.py
```

Torchvision's downloader may cache both the training and official test files,
even with `train=True`; downloading bytes is not evaluation. The study and our
checker use only the training images/labels. No command here evaluates the
official test set. The inspected loader lists the mirror
`https://ossci-datasets.s3.amazonaws.com/mnist/`; downloads require network access.
The downloader and a fresh install have not been tested by this packaging step.

Three historical baseline constructors also hash locally cached published paper
PDFs. They are omitted from Git and the ZIP. Restore the exact verified bytes
before running broad baseline tests:

```sh
python3 -B tools/prepare_reference_papers.py --download
```

If you already have the original research cache, use
`--from-directory /path/to/original/Experiment/resources` instead. The helper
checks the frozen SHA-256 of each PDF, refuses changed bytes and never touches
training/evaluation data. Downloading requires access to the publisher/arXiv
URLs listed in `Experiment/resources/README.md`.

Expected uncompressed files under `Experiment/data/MNIST/raw`:

| File | SHA-256 |
| --- | --- |
| `train-images-idx3-ubyte` | `ba891046e6505d7aadcbbe25680a0738ad16aec93bde7f9b65e87a2fc25776db` |
| `train-labels-idx1-ubyte` | `65a50cbbf4e906d70832878ad85ccda5333a97f0f4c3dd2ef09a8a9eef7101c5` |

Existing cached data can be checked without copying it into the submission:

```sh
python3 -B tools/check_data.py --data-dir /path/to/original/Experiment/data
```

## Exact executed protocol

All ten clients participate each round. Each uses one local epoch, batch size 64,
SGD learning rate 0.01 and momentum 0.9. The CNN has 421,642 parameters. Each run
has 24 rounds with four clean initial rounds; interventions operate in rounds
5–24. v0.6 uses the same settings in IID and non-IID.

Noise is severity-3 shot noise. Stripes invert the leftmost/rightmost seven
columns of selected 28×28 images; the changed-image fraction rises over eight
rounds. Honest changes keep labels. Label flipping uses `9 - label` for affected
clients 0–2. Scenario truth stays in the simulator, not the aggregation interface.

Full settings: `Experiment/outputs/final_results/20261031_v06/protocol_and_scenarios.json`
and `v06_exact_settings.json` in the same directory.
