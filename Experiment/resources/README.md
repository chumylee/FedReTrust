# Comparison algorithm papers

The new experiment will evaluate FedReTrust against four published comparison
methods: **FedAvg, FLTrust, Viceroy and RFA**. Their papers are saved here as PDFs.
This gives five methods in the main evaluation, including FedReTrust.

Downloaded on 10 October 2026. Each PDF was checked for readability and its
first-page title or identifying content.

## FedAvg

- **Paper:** Communication-Efficient Learning of Deep Networks from Decentralized Data.
- **Authors:** H. Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson and Blaise Aguera y Arcas.
- **Publication:** AISTATS 2017, PMLR 54:1273-1282.
- **Local PDF:** [FedAvg_McMahan_2017.pdf](FedAvg_McMahan_2017.pdf)
- **Paper page:** https://proceedings.mlr.press/v54/mcmahan17a.html
- **PDF source:** https://proceedings.mlr.press/v54/mcmahan17a/mcmahan17a.pdf

## FLTrust

- **Paper:** FLTrust: Byzantine-robust Federated Learning via Trust Bootstrapping.
- **Authors:** Xiaoyu Cao, Minghong Fang, Jia Liu and Neil Zhenqiang Gong.
- **Publication:** NDSS Symposium 2021.
- **Local PDF:** [FLTrust_Cao_2021.pdf](FLTrust_Cao_2021.pdf)
- **Paper page:** https://www.ndss-symposium.org/ndss-paper/fltrust-byzantine-robust-federated-learning-via-trust-bootstrapping/
- **PDF source:** https://www.ndss-symposium.org/wp-content/uploads/ndss2021_6C-2_24434_paper.pdf
- **DOI:** https://doi.org/10.14722/ndss.2021.24434

Notebook 05 implements the aggregation rule in section IV-B, equations (2)-(4),
and Algorithm 2. The common experiment's training budget, server-reference
size, descent-delta sign convention and numerical edge-case policies are
documented in `../README.md`. This is a paper-based implementation for matched
experiments rather than a claim to reproduce the paper's complete results.

## Viceroy

- **Paper:** Attacks against Federated Learning Defense Systems and their Mitigation.
- **Authors:** Cody Lewis, Vijay Varadharajan and Nasimul Noman.
- **Publication:** Journal of Machine Learning Research 24(30):1-50, 2023.
- **Local PDF:** [Viceroy_Lewis_2023.pdf](Viceroy_Lewis_2023.pdf)
- **Paper page:** https://jmlr.org/papers/v24/22-0014.html
- **PDF source:** https://jmlr.org/papers/volume24/22-0014/22-0014.pdf

Notebook 07 implements Algorithm 1 and section 5.1, using the author's modified
FoolsGold scoring function. `viceroy_reference/` preserves unmodified source
files and the MIT licence at commit `3f726ca60ee8a975140a5a6a575351bf0e603e47`.
Its source manifest records retrieval URLs and SHA-256 values. Initialization,
first-round handling, paper/code parameter conventions, raw-sum aggregation,
numerical corner cases and the local-SGD/IID adaptation are documented in
[viceroy_reference/ADAPTATION_NOTES.md](viceroy_reference/ADAPTATION_NOTES.md).
The original reference training stack is not imported into the experiment.

- **Author code:** https://github.com/codymlewis/viceroy
- **Pinned rule:** https://github.com/codymlewis/viceroy/blob/3f726ca60ee8a975140a5a6a575351bf0e603e47/fl/server/viceroy.py

## RFA

- **Paper:** Robust Aggregation for Federated Learning.
- **Authors:** Krishna Pillutla, Sham M. Kakade and Zaid Harchaoui.
- **Publication:** IEEE Transactions on Signal Processing 70:1142-1154, 2022.
- **Local PDF:** [RFA_Pillutla_2022.pdf](RFA_Pillutla_2022.pdf)
- **Saved version:** Open author manuscript, arXiv:1912.13445v2, revised 17 January 2022; includes appendices. This is the arXiv manuscript rather than the publisher-formatted journal PDF.
- **Paper page:** https://arxiv.org/abs/1912.13445
- **PDF source:** https://arxiv.org/pdf/1912.13445v2
- **Journal DOI:** https://doi.org/10.1109/TSP.2022.3153135

Notebook 06 implements the joint weighted geometric-median rule using the
paper's Algorithms 1-2 and Appendix D.2. The shared MNIST protocol, solver
budget/initialization accounting and secure-aggregation limitation are recorded
in `../README.md`. This is a matched baseline implementation, not a reproduction
of the paper's complete experiments.

`rfa_reference/` contains the authors' original NumPy aggregation methods within
an unmodified copy of `models/model.py`, the BSD-2-Clause licence and a pinned
SHA-256 source manifest. The unit checks extract only those pure numeric methods;
they do not import the TensorFlow module.

- **Author code:** https://github.com/krishnap25/RFA
- **Pinned source:** https://github.com/krishnap25/RFA/blob/40945d27a5246b2c82fd37bd453db4f5bc049952/models/model.py
- **Local notes:** [rfa_reference/README.md](rfa_reference/README.md)

## Evaluation scope update - 10 October 2026

The submitted proposal also included the project's exploratory validation-loss
trust heuristic. It is excluded from the new experiment's primary comparison
because no dedicated published source was identified for its exact rule. The
primary comparison will prioritise independently published methods when
assessing FedReTrust's benefits and trade-offs.

Separate ablation experiments will examine the effects of FedReTrust's history
and recovery components. Matched controls will examine the effect of update
clipping. These experiments explain the proposed policy's behaviour and
complement the comparisons against published methods.

Record this change from the submitted proposal in the final methodology. Until
the experiments are complete, the scope can be described as follows:

> The exploratory validation-loss heuristic is excluded from the primary
> comparison to prioritise established published methods. FedReTrust will be
> evaluated against FedAvg, FLTrust, Viceroy and RFA, with separate ablation
> experiments examining its components and matched controls for update clipping.

## Scenario reference code

`mnist_c_reference/` contains a pinned copy of Google Research's MNIST-C
corruption source and Apache-2.0 licence, with SHA-256 fingerprints and
adaptation notes in `source_manifest.json`. Notebook 04 uses two selected
recipes: shot noise and stripes. It generates variants of the project's saved
training/development splits; it does not load the precomputed MNIST-C test set.

- **Paper:** [MNIST-C: A Robustness Benchmark for Computer Vision](https://arxiv.org/abs/1906.02337), Norman Mu and Justin Gilmer, 2019.
- **Code:** [google-research/mnist-c, pinned commit bba57e4](https://github.com/google-research/mnist-c/blob/bba57e4ccc282f106907c5239958e72298451ea7/corruptions.py).

See `../README.md` for the exact schedules, attack-paper references and the
differences between these controlled scenarios and the source experiments.

## Additional literature reviewed in Step 4

The focused novelty review on 11 October 2026 also considered FedDrift,
FedCognis, SSMTD, RECESS, FL-LTD, FLDetector and FedRecover. Their primary-source
URLs, publication status, inspected sections and relevance are recorded in
[`literature_sources.json`](../outputs/research_gap_review/20261011/attempt_0001/literature_sources.json).
The additional sources were read online; this folder's four comparison PDFs
remain the performance-baseline paper set. Literature inclusion does not add a
baseline or validate the previously removed local loss heuristic.
