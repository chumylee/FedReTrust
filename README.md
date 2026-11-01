# FedReTrust v0.6 — COMP6002 submission code

FedReTrust is an experimental federated aggregation method combining client
update history, directional weighting, trust restoration and an RFA complement.
This package contains the final v0.6 implementation, earlier methods, frozen
development results and small runnable demonstrations. It is a separate
submission copy; the original research archive is preserved.

**Finding:** v0.6 improved non-IID label-flip protection over v0.5, at a cost to
IID stripe adaptation. RFA retained better mean attack protection; v0.6 retained
better honest-stripe accuracy. The follow-up passed **45/51** engineering checks,
not all of them. This is not a universal winner or an established novelty claim.

## Start here — no dataset or ML installation needed

Run from this repository's root with Python 3.10 or newer:

```sh
python3 -B tools/verify_submission.py
python3 -B tools/reproduce_results.py --name first
```

Open `generated/reproduction-first/results.html` in a browser. The command
recalculates 1,298 five-block summaries, 636 paired comparisons and 110 main-table
cells and checks them against frozen values. It does not train models. Use a new
`--name` on subsequent runs: existing outputs are never overwritten.

## Run the actual aggregation code

The recorded research environment was Python **3.14.8**, macOS arm64, CPU.
The direct dependencies below match that environment. A fresh dependency
installation and other platforms have not yet been verified.

```sh
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-demo.txt
python -B tools/demo.py --name first
python -B -m unittest discover -s Experiment/code/tests -p 'test_fedretrust_v0*.py'
```

The demo runs the unchanged v0.6 algorithm on small synthetic vectors and prints
clipping, trust and historical contribution diagnostics. It is a mechanism
demonstration, not MNIST training, measured accuracy or evidence of model recovery.
Its trace is saved separately in `generated/demo-first/trace.json`.

## What to read

| Purpose | Entry point |
| --- | --- |
| Final algorithm | [fedretrust_v06.py](Experiment/code/fedretrust_v06.py) |
| Earlier v0.5 comparison | [fedretrust_v05.py](Experiment/code/fedretrust_v05.py) |
| Algorithm explanation and module map | [Architecture](docs/ARCHITECTURE.md) |
| Exact data and setup instructions | [Data and environment](docs/DATA_AND_SETUP.md) |
| Reproduction levels and limitations | [Reproducibility](docs/REPRODUCIBILITY.md) |
| Practical demo walkthrough | [Demo guide](docs/DEMO.md) |
| Included and omitted artifacts | [Packaging decisions](docs/PACKAGING.md) |
| Every source-to-copy mapping and hash | [Source map](provenance/source_map.json) |
| Submission-only changes | [Change log](CHANGELOG.md) |
| Upstream attribution | [Third-party notices](THIRD_PARTY_NOTICES.md) |
| GitHub publication and access | [Publication](docs/PUBLICATION.md) |
| Final executed notebook (read-only historical output) | [Notebook 34](Experiment/code/34_fedretrust_final_results.ipynb) |

## Evidence scope

- **880 distinct frozen runs** across the historical study and follow-ups.
  v0.6 covers **40 runs: five blocks × two regimes × four scenarios**.
- Keep IID and non-IID separate. Use matched block differences and nominal 95%
  t intervals. All five blocks were known development conditions.
- Accuracy is on reserved development images. The official test set was not
  evaluated. Honest noise/stripe scores concern affected honest clients.
- Report absolute attacked accuracy together with own-clean-relative damage.
- FLAURA/FedCognis lack five-block noise measurements. Missing means not tested.
- Earlier v0.3 recovery/component findings are not v0.6 recovery/ablation results.
- The failed v0.6 pilot (25/27) and all six failed follow-up checks are retained.

Original notebooks and study orchestration scripts retain their historical
paths and assumptions. They are included for inspection, not as the portable
entry points. Follow the commands above rather than running every notebook or
relaunching the full study.

Repository: [chumylee/FedReTrust](https://github.com/chumylee/FedReTrust).
The submission package is public. Repository access without signing in was
verified on 1 November 2026. See `provenance/publication.json` for the recorded
upload and access checks.
