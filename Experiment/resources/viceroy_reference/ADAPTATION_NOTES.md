# Viceroy reference and common-experiment adaptation

Source: Cody Lewis, Vijay Varadharajan and Nasimul Noman, *Attacks against
Federated Learning Defense Systems and their Mitigation*, JMLR 24(30):1-50
(2023), especially Algorithm 1 and section 5.1 (pages 25-26), section 5.2
(page 27) and Appendix D (pages 44-45).

- Paper: https://jmlr.org/papers/v24/22-0014.html
- Author repository: https://github.com/codymlewis/viceroy
- Pinned commit: `3f726ca60ee8a975140a5a6a575351bf0e603e47`
- Retrieval URLs, UTC date and SHA-256 values: `source_manifest.json`.
- Original author licence: `LICENSE.md` (MIT, copyright Cody Lewis, 2023).

The original README and source files are unmodified. These notes are separate
so that all downloaded-file fingerprints remain valid. Our numeric FedScale
is adapted from the author's MIT-licensed implementation. The experiment uses
our PyTorch implementation in `../../code/viceroy.py` and does not import the
downloaded JAX training modules or install their legacy requirements.

## Rule and update order

For a fixed set of clients, let `d_i` be the current submitted update,
`H_i` the previous history and `R_i` the previous reputation:

```text
c = FedScale(d)
h_old = FedScale(H)
R_new = clip(R + eta/2 * (1 - 2*abs(h_old - c)), 0, 1)
H_new = omega * H + d
h_new = FedScale(H_new)
p = R_new * h_new + (1 - R_new) * c
global_new = global + sum_i(p_i * d_i)
```

`FedScale` acts on the whole set of client vectors, not independently on each
vector. Its arithmetic is the author's stateless modified FoolsGold: pairwise
cosine similarities, diagonal subtraction, similarity pardoning, raw weights
`1 - max_similarity`, clamping, division by the largest raw weight, replacement
of exact unit weights by 0.99, and the logit transform plus 0.5, clamped to [0,1].
All model parameters are included in a single vector. There is no loss-based
heuristic, feature selection, clean-root reference, sample-count factor, oracle
attacker count, vector clipping or additional FoolsGold history update.

The old history scores determine the reputation change. The newly updated
history scores determine the historical contribution to the final blend.
Substituting one for the other changes the algorithm. Reputation is clipped to
[0,1], following Algorithm 1 and author code. The nearby prose has an inconsistent
range and clip expression; it is not used to override the executable rule.

## Initialization and parameters

The author code initializes zero histories and reputations of **1**, and skips
the reputation update in the first round. We preserve these details, which are
not fully specified by Algorithm 1 alone. First-round histories are the current
updates, so their FedScale scores also determine first-round coefficients.

We fix `omega = sys.float_info.epsilon**(1/56)` (approximately 0.5254) and
`eta = 0.2`, before viewing outcomes. Appendix D defines `eta = 2/tau_1` with
`tau_1 = 10`. The author constructor instead names a parameter `tau_1 = 5`
and computes `eta = 1/tau_1`. Both give the same numeric value 0.2. We expose
omega and eta directly to avoid confusing these parameter conventions.

With no new contribution, 56 decay steps shrink an old vector by machine
epsilon; this is not literal mathematical zero or a guarantee of attack removal.
The six-round development runs are much shorter than this horizon. A maximum
positive reputation change is 0.1 per round, so ten best-case updates would
raise a reputation from zero to one, although our initial reputations are one.

## Explicit differences from the original training experiment

Our common runner uses ten IID MNIST clients, the shared CNN, 512 fixed examples
per client, one epoch of local SGD (eight minibatches), learning rate 0.01,
momentum 0.9, six rounds, seed 20261014 and three held-out development tasks.
Its signed update is `d_i = trained_model_i - broadcast`, which is added to the
global model. No further server optimizer or learning-rate factor is applied.

The authors used a JAX federated-SGD pipeline, 100 clients and 5,000 rounds,
different model architectures and attack/data settings. Their pinned `main.py`
uses Optax SGD with learning rate 0.01. `fl/utils/network.py` returns sums of
raw gradients from local steps, and `fl/server/server.py` applies a weighted
gradient sum through the server optimizer. Our trained deltas already contain
the local optimizer steps and momentum; they are not those raw gradient sums.
Changing to local model deltas is an explicit common-harness adaptation.

The original code uses Dirichlet alpha=0.5 for Viceroy/FoolsGold/CONTRA and
alpha=1000 for the other main methods on MNIST/CIFAR-10. We keep the common
IID split to avoid comparing different client data. Similar honest IID updates
can receive low FoolsGold-derived weights, so this setting is consequential.
Future non-IID checks must use the same non-IID partitions across all methods.

**No coefficient normalization is added.** Algorithm 1 and the author's server
use a raw weighted sum. `sum(p)` need not equal one. It changes the update size
relative to FedAvg/RFA, which use normalized weights. FLTrust also normalizes
its shares but rescales vectors using its root update. A normalized Viceroy
variant would be a separately declared control, not the literal rule used here.
Any final robustness comparison needs optimization-scale sensitivity checks;
these short runs cannot attribute all differences to filtering quality.

We track histories by stable client ID and require all ten clients every round.
Submissions are sorted by ID; changed membership, changed parameter layouts or
noncontiguous rounds fail explicitly. Partial participation, asynchronous
updates and restoring a running server from a checkpoint are not implemented.
Every scenario calls reset before observing data. Full final histories and
reputations are saved separately from inference model weights.

## Numerical policies

- Work in float64, then cast the returned model to its original dtype.
- Clamp computed cosines to [-1,1] to remove floating-point overshoot. Zero
  vectors have zero cosine similarities, matching the reference convention.
- If the maximum raw FedScale weight is at most 1e-12, return zero scales.
  Identical nonzero directions otherwise cause division by zero in the author
  implementation. This declared corner-case policy is not a new defence.
- If the final coefficient total is zero, keep the broadcast model while still
  updating reputations and histories. Do not fall back to FedAvg.
- Fail on incompatible/nonfinite inputs without committing new defence state.

## Verification scope

`test_viceroy.py` verifies pinned file hashes, extracts the inspected author
numeric functions and server methods using Python's AST, removes only JAX's
JIT decorator and supplies NumPy arrays plus an independent NumPy cosine
calculation. It checks FedScale on random/clustered vectors and twenty stateful
rounds against the author's update order, histories, reputations and raw sum.
It does not claim to execute the original JAX/sklearn runtime or reproduce
the authors' complete training pipeline. Analytic cases independently check
raw summation, duplicates, zero updates, initialization, state reset and identity
tracking. The explicitly undefined reference corner is tested against our
declared finite policy rather than claiming equality to author NaNs.

The experiment notebook verifies the shared data/model/settings/schedules,
fresh FedAvg reproduction, current/history blend, per-round reputation changes,
coefficient totals, final defence snapshots and exact restored model logits.
Simulator truth is joined only after training. Reputation is an interpolation
factor based on score consistency, not a probability of honesty or an accurate
classification of malicious intent. Two equally low scores can increase
reputation while the client's aggregation weight remains low.
