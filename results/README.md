# Results

The measurements cited by the README and `docs/evaluation.md`. Paths inside
these files are relative to the repository root; analyzed upstream sources
appear under `.deps/<project>-<version>/`, where `scripts/prepare_targets.py`
puts them.

## `real-projects/`: the 0.3.0 scan

- `cuda-samples.sarif`, `faiss.sarif`, `rmm.sarif`: SARIF 2.1.0, one run per
  parsed translation unit, with the analyzer's `codeFlows`. The first run of
  each file carries the project's coverage totals in `properties.coverage`.
- `triage.csv`: one row per alert (36), with the file relative to the project
  root, line, rule, level, message, category and the reason for the category.
  Rows match SARIF results by file and line.
- `triage-0.2.0.csv`: the 76 alerts of version 0.2.0 with their judgment and
  root-cause family. `review_scope` says whether a row was triaged against the
  source and analyzer path (23 rows) or carries the judgment of an earlier
  triage of the same site, matched by a semantic fingerprint with the source
  bytes checked unchanged (53 rows). The 0.2.0 SARIF is not included.
- `summary.json`: per-project alert counts by category and coverage totals
  (translation units, parsed, failed, functions that hit the node budget,
  names of unmodeled CUDA calls), taken from the scan's coverage records. No
  script in this repository regenerates it, and the per-TU coverage records and
  logs it was taken from are not included.

## `holdout-first/`: the first, frozen holdout run (version 0.1.0)

- `out.sarif`, `coverage.json`, `metrics.json`: the run's output and its
  evaluation against `corpus/holdout/labels.json` (25/1/0/23).
- `first-result.json`: start and finish times, the budget, and the hashes of
  the freeze, the labels, the native executable and the 1,423 header and
  library files the run depended on. It names the dependency manifest it was
  given (`results/holdout-dependencies.json`); that manifest is not included,
  but its entries are the `dependencyFiles` listed here.
- `triage.json`: the one false positive, written after the run without
  changing labels or rules, and the hashes of the run's artifacts.

These records, the freeze and its first-run marker originally held absolute
paths of the analysis machine. Those paths were made repository-relative for
publication and the hashes that pin the edited files were recomputed; see
`corpus/holdout/README.md` for exactly what the hashes cover. Evaluating the
published `out.sarif` and `coverage.json` against the labels again with
`python3 -m eval --root .` reproduces the totals and per-case results in
`metrics.json`. (`eval` refuses a `holdout` split without a matching freeze,
and the freeze's rule hashes describe the 0.1.0 code, so this check needs a
copy of the labels with another `split` value.)

## `regression/`

Seed and holdout metrics of version 0.3.0. The holdout has been seen, so it
was scored as a development set (`split` is `dev`), not as a held-out result.

## `probes/completion.json`

Output of `probes/completion.py` on an RTX 4090 with driver 595.99: how long
`cudaMemcpyAsync` D2H takes to return behind a 20 ms kernel, for pinned and
pageable host memory, five runs each.
