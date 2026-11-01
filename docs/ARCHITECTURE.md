# Algorithm and source map

The server receives stable client IDs, example counts, submitted model states,
the broadcast model and round number. Scenario truth and development evaluation
are not inputs to FedReTrust. Reference-image access is zero.

## v0.6 processing

For raw update `x_i`, the stored per-client vector is `h_i`, initially zero.
The innovation is `x_i - h_i`. Clip its norm to twice the cohort's lower median
innovation norm, giving `y_i = h_i + clip(x_i - h_i, radius)`. A zero/near-zero
median bypasses this filter and is logged; this is an explicit limitation.

Cap the norm of `y_i` at six times the lower median filtered-update norm. Call
the resulting update `z_i`. RFA computes an approximate geometric-median anchor
`a` using up to three smoothed Weiszfeld iterations on these capped updates.

Let `c_i` be cosine alignment of filtered update `y_i` with `a`. With positive,
nondegenerate geometry, evidence is `clip(c_i / (0.25 * median(c)), 0, 1)` with
the implementation's epsilon safeguards. Trust is
`q_i = min(evidence_i, previous_q_i + 0.5)`; a new client's previous trust is 1.
Restriction is immediate; restoration is bounded per participating round.

With normalized sample shares `b_i`, the final update is:

```text
sum_i(b_i * q_i * z_i) + (1 - sum_i(b_i * q_i)) * a
```

Degenerate geometry or negligible trust mass falls back to capped RFA without
updating trust memory. Vector history still records each successful filtered
update, including downweighted clients. The history is untrusted state and may
contain poisoning. With full trust the final combination is a sample-weighted
average of filtered, capped updates; history filtering can still be active.

Actual contributions include current raw deltas and historical residuals.
Returned current-update coefficients alone cannot reconstruct the model update.
`v06_recording.py` audits both terms without changing the aggregate.

## Modules

| Module under `Experiment/code` | Responsibility |
| --- | --- |
| `fedretrust_v06.py` | History filtering followed by the unchanged v0.5 rule |
| `fedretrust_v05.py` | Trust-weighted mass plus RFA complement |
| `fedretrust_directional.py` | Directional configuration and earlier v0.3 method |
| `fedretrust_reference_free.py` | Shared restoration recurrence |
| `rfa.py` | RFA solver and validated submission flattening |
| `experiment_core.py` | CNN, local SGD, data loading, aggregation interface, recording |
| `scenario_utils.py` | Image corruptions and label-flip dataset views |
| `scenario_metrics.py` | Final 24-round schedules and legitimate-task weighting |
| `v06_recording.py` | Full affine contribution diagnostics |
| `v06_pilot.py`, `v06_followup.py` | Historical study orchestration; requires full archive |
| `fltrust.py`, `viceroy.py`, `flaura.py`, `fedcognis.py` | Comparator implementations/adaptations |

The 24-round final schedule comes from `scenario_metrics.longer_scenarios`, not
the older provisional `scenario_utils.scenario_catalog`. Follow the frozen
`protocol_and_scenarios.json` when explaining the executed study.

The remaining modules and notebooks reflect earlier development stages. Their
presence is provenance, not permission to conflate their outcomes with v0.6.
