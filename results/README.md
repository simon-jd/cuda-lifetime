# Results

The measurements cited by the README and `docs/evaluation.md`. Paths inside
these files are relative to the repository root; analyzed upstream sources
appear under `.deps/<project>-<version>/`, where `scripts/prepare_targets.py`
puts them.

## `real-projects/`: the 0.4.0 scan

Produced by `scripts/analyze_parallel.py --jobs 10` per project and published
with `scripts/summarize_scan.py`, which also checks that the triage matches
the scan's alerts one to one.

- `cuda-samples.sarif`, `faiss.sarif`, `rmm.sarif`: SARIF 2.1.0, one run per
  parsed translation unit, with the analyzer's `codeFlows`. Each result's rule,
  level, memory kind and semantic ID are SARIF fields. The first run of each
  file carries the project's coverage totals in `properties.coverage`.
- `triage.csv`: one row per alert (36), with the file relative to the project
  root, line, rule, level, message, category and the reason for the category.
  The alerts of 0.4.0 are those of 0.3.0, so the triage is the 0.3.0 one.
- `triage-0.2.0.csv`: the 76 alerts of version 0.2.0 with their judgment and
  root-cause family. `review_scope` says whether a row was triaged against the
  source and analyzer path (23 rows) or carries the judgment of an earlier
  triage of the same site, matched by a semantic fingerprint with the source
  bytes checked unchanged (53 rows). The 0.2.0 SARIF is not included.
- `summary.json`: per-project alert counts by category and level, the 0.2.0
  counts, and coverage: translation units, parsed, failed, top-level functions
  analyzed and those that exhausted the node budget, wall-clock timeouts,
  coverage gaps, unmodeled CUDA API functions, and wall and per-TU times.
- `timings.csv`: per TU, the analysis time, peak memory, top-level functions
  analyzed and those over budget. Times were measured with ten TUs analyzed
  at a time.

The per-TU logs and coverage records of the scan are not included.

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

Seed and holdout metrics of version 0.4.0 from `tests/run_seed.py`. The
holdout has been seen, so it was scored as a development set (`split` is
`dev`), not as a held-out result.

## `probes/completion.json`

Output of `probes/completion.py` on an RTX 4090 with driver 595.99: how long
`cudaMemcpyAsync` D2H takes to return behind a 20 ms kernel, for pinned and
pageable host memory, five runs each.
