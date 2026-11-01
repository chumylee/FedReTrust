# RFA reference code

This directory preserves the paper authors' original TensorFlow repository's
`models/model.py` at commit `40945d27a5246b2c82fd37bd453db4f5bc049952`, along
with its BSD-2-Clause licence and SHA-256 manifest. The source is unmodified.

Repository: https://github.com/krishnap25/RFA

`test_rfa.py` extracts only four inspected, pure NumPy methods of `ServerModel`:
`weighted_average_oracle`, `geometric_median_update`, `l2dist` and
`geometric_median_objective`. It does not import the downloaded module, install
TensorFlow, or execute its training classes. The reference comparison uses the
same three refinements and smoothing `1e-6`, with early stopping disabled in both
implementations, to verify that the joint aggregation calculations agree.

Our implementation is written in PyTorch from the paper's Algorithms 1-2 and
Appendix D.2. The original source defaults to four refinements, smoothing `1e-5`
and stopping on the unsmoothed objective. Our experiment uses three refinements,
smoothing `1e-6`, and the relative change in the **smoothed** objective specified
in Appendix D.2, with tolerance `1e-6`. The initial sample-weighted mean is
recorded as one extra averaging operation, giving up to four in total. The
paper discusses three iterations as well as a three-oracle-call budget; we
explicitly account for initialization rather than claiming three total calls.

This is a single-process numeric implementation. It does not implement the
paper's cryptographic secure-average protocol or measure network traffic.
