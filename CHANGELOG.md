# Submission packaging changes — 1 November 2026

The research selection remains FedReTrust v0.6 (`0.6-exploratory`). No training,
aggregation equation, parameter, seed, partition, scenario, target, failure or
reported study measurement has been changed.

1. Copied all 73 frozen scientific files byte for byte into a separate tree,
   retaining their relative locations and imports. Supporting Python modules
   and existing tests are retained to avoid breaking the historical import graph.
2. Added a concise v0.6 README and documentation, replacing the old chronological
   README as the submission entry point. The original README remains untouched
   in the research workspace.
3. Added standard-library tools for portable hash verification, saved-block
   summary reproduction and training-data fingerprint checks.
4. Added a synthetic-vector demonstration that calls the unchanged algorithm
   and recorder. Its outputs are explicitly separate from the 880-run evidence.
5. Copied the final tables, figures, protocol/selection metadata, failure history,
   partition indices, final executed notebook and upstream license/provenance
   files. Omitted full raw logs, checkpoints, raw MNIST data, cached paper/article
   copies, environments and private handoff instructions.
6. Replaced one absolute workspace fallback in the copied notebook's data-path
   setup cell with a relative path. All other cell source and every saved output
   are unchanged. The source map retains both hashes and the exact transformation;
   the original is untouched. No numerical calculation was edited.
7. Added explicit dependency, packaging, publication and reproducibility notes.
   Study-wide refactoring is deliberately avoided to preserve numerical behavior.

The copied source hash check provides exact source equivalence. Command smoke
checks belong to Step 5; full archive integrity and broader reproducibility
verification are Step 6. No independent clean-machine installation is claimed.
