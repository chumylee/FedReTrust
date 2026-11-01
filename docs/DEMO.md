# Practical demonstration

This is a runnable algorithm demonstration plus a saved-results walkthrough.
It does not train a new model or change the frozen research. The presentation
timing, rehearsal and backup screen recording remain later tasks.

## 1. Execute the actual algorithm

```sh
python -B tools/demo.py --name live
```

Two deliberately small, separately reset fixtures use ten clients and two-entry
model updates. This makes the actual implementation's decisions easy to inspect.

- **History filtering:** seven clients submit `[1, 0]`, three submit `[100, 0]`.
  The innovation radius is 2; the applied global update is `[1.3, 0]` despite
  full directional trust. The next round reverses those three raw submissions;
  inspect how stored history can keep the filtered updates aligned.
- **Directional trust:** in a fresh method instance, seven updates point along
  `[1, 0]`, three oppose them. Then all ten submit aligned updates twice. Inspect
  the trust sequence and RFA complement. The printed zero-median bypass is part
  of the actual rule and must not be hidden when explaining this toy fixture.

The trace records current and historical contribution terms and the recorder's
reconstruction diagnostics. A nonzero historical coefficient can multiply a
zero vector at cold start; distinguish coefficients from actual contributions.
The labelled fixtures do not prove that the method identifies malicious intent,
nor do they establish v0.6 model recovery after an MNIST attack.

## 2. Walk through the measured results

```sh
python3 -B tools/reproduce_results.py --name live
```

Open `generated/reproduction-live/results.html` and explain:

1. Non-IID absolute attacked accuracy: v0.5 77.35%, v0.6 79.20%, RFA 81.29%.
2. Non-IID affected honest-stripe accuracy: v0.6 80.81%, RFA 77.47%.
3. IID stripe cost: v0.5 82.04% versus v0.6 80.92%.
4. Paired intervals, five known development blocks, the 45/51 partial success,
   and the original failed pilot. This is a measured compromise.

## Recovery if the live environment fails

Keep the generated HTML and JSON trace available offline. They are saved
walkthrough artifacts, not a substitute for the required backup video. Record
the working demonstration during the presentation-preparation stage. No video
or rehearsal is claimed by this submission packaging step.
