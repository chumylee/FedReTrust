# Inclusion decisions

This copy is intended for code assessment and saved-artifact reproduction. The
original frozen workspace is the full research archive and remains unchanged.

| Artifact | Decision | Reason |
| --- | --- | --- |
| All 73 frozen scientific sources | Include unchanged | Exact numerical source preservation |
| Supporting Python modules and existing tests | Include | Preserve local import dependencies and development provenance |
| Final result tables, figures, settings and failed gates | Include unchanged | Assessable findings, missingness and uncertainty |
| Research freeze and evidence manifests | Include | Traceability to full archive; hashes do not replace omitted files |
| Final executed notebook | Include, one data-path fallback made relative | Convenient inspection of unchanged saved outputs |
| Partition indices | Include unchanged | Exact data assignments without bundling raw images |
| Raw MNIST images and labels | Exclude; document retrieval and hashes | Downloadable dataset; not needed for portable reproduction/demo |
| Model checkpoints | Exclude | Demo uses the actual algorithm on small vectors; no checkpoint replay claim |
| Full client/round logs and source snapshots | Exclude | About 13 GB archive remains locally available and indexed |
| Cached papers/articles/screenshots | Exclude; retain source manifests | Reference links and attribution suffice for the code package |
| Licensed upstream source and notices | Include | Preserve attribution for adapted code |
| Earlier executed notebooks | Exclude | Historical modules remain; final outputs are the entry point |
| Original long README and chat handoffs | Exclude | Stale instructions and local paths are unsuitable as submission guidance |
| Environments, caches and generated demos/reproductions | Exclude from Git/archive | Rebuild locally; no addition to study evidence |

`provenance/source_map.json` records source-relative and destination-relative
paths, checksums, sizes and the single document-only transformation. New tools
and documents appear in `provenance/package_manifest.json`. The manifest excludes
itself, `.git`, environments, caches and disposable `generated` outputs.

The source-map entries do not claim all copied historical orchestrators work
without the omitted archive. Their constants and hash chains still describe
the original research layout. Use the documented portable tools for assessment.

No project-management snapshots, student contributions, signatures, supervisor
confirmations, repository publication or submission events have been fabricated.

After an intentional documentation/publication update, reseal and create a ZIP:

```sh
python3 -B tools/package_submission.py --name release
python3 -B tools/verify_submission.py
```

Use a new archive name each time. Packaging uses an explicit file allowlist,
checks all copied scientific/evidence files before sealing, and excludes Git
metadata, downloaded MNIST, environments and generated outputs. This updates
only the submission manifest; it never changes the original research freeze.
