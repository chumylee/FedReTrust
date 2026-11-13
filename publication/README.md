# FedReTrust publication experiments

Install Git and Python 3.12–3.14. On a machine with an NVIDIA GPU, check that the
current NVIDIA driver is installed and that `nvidia-smi` lists the GPU. The
launcher installs the [PyTorch 2.13 CUDA 13.0 build](https://pytorch.org/get-started/previous-versions/)
on supported NVIDIA systems. It can also run on Apple Silicon or CPU.

Clone the repository and enter the folder:

```sh
git clone https://github.com/chumylee/FedReTrust.git
cd FedReTrust
```

Run this one command from that folder:

```sh
python3 publication/run.py
```

On Windows PowerShell, use `py -3 publication/run.py`. The first run installs
Python dependencies and downloads the datasets and baseline references. The
terminal prints the device, current job and round. Allow several hours and
keep the computer awake. If interrupted, run the same command to resume from
the last completed job. For a short end-to-end check before the full run,
append `--smoke`. Its results go to a separate folder.

When the terminal says **Complete**, open
[publication/results/full/results_table.csv](results/full/results_table.csv).
The same folder contains:

| File | Contents |
| --- | --- |
| `results_table.csv` | One row per dataset and method. |
| `results_by_seed.csv` | The individual repeated results behind those means. |
| `paired_differences.csv` | FedReTrust v0.6 minus each comparator, in percentage points, with nominal paired 95% intervals. |
| `runs/` | Per-round measurements and method settings for each job. |
| `manifest.json` and `plans/` | Protocol, software versions, code hashes, data splits and affected clients. |

Send Chuma **`publication/FedReTrust-results.zip`** after the run. The launcher
creates it from the completed result files. You can also upload this ZIP in a
GitHub issue or send it directly. If you have write access to the repository,
these commands put just the ZIP on a results branch:

```sh
git switch -c results/deepak
git add -f publication/FedReTrust-results.zip
git commit -m "Add publication experiment results"
git push -u origin results/deepak
```

## What runs

The full run has 225 jobs: Fashion-MNIST, EMNIST Balanced and Smartphone HAR,
each with FedAvg, RFA, FLTrust, Viceroy and FedReTrust v0.6. Each combination
runs five matched seeds under clean training, honest input drift and label
poisoning. Every job has **48 rounds**, twice the previous 24, with one local
SGD epoch per round. The intervention runs from round 5 through round 48. Honest
drift keeps the eight-round ramp and uses stripes for images and a 30-degree
sensor rotation for HAR. GPU acceleration applies to training and evaluation.
The original aggregation rules operate on CPU model updates.

Images use 10 clients with 512 examples each and Dirichlet label skew
(alpha 0.5). The image plans keep the previous client samples and test subsets.
FLTrust uses clean, class-balanced training examples disjoint from those client
samples (300 for Fashion-MNIST and 329 for EMNIST). HAR reserves two training
participants and samples 300 clean windows from them for FLTrust. All five HAR
methods use the same other 19 training participants, with six affected clients
per seed. HAR evaluation uses the 2,947 windows of the nine official test
participants. FLTrust starts its server model from the round's broadcast and
uses eight SGD batches of 64 at the dataset's client learning rate. Viceroy
uses the project's local-SGD adaptation with its original raw weighted sum.

The C, P and H columns represent clean accuracy, poisoning accuracy and honest
drift accuracy. H combines per-class accuracies on clean and shifted test
inputs according to the affected clients' class profiles and the drift
schedule. The primary results average rounds 5–48, then the five seeds. The
table also gives round-24 and final-round values. Compare the five methods
within this study. The longer reporting window and new HAR split change the
protocol relative to the earlier paper results.

Dataset sources: [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist),
[EMNIST Balanced mirror](https://huggingface.co/datasets/Royc30ne/emnist-balanced/tree/e6fd3608dc5d46960a89a423472f67c9c0f9ec31),
and [UCI Smartphone HAR](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones).
Downloads are checked against the files used in the previous experiments.
