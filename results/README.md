# Results

The measurements cited by the README and `docs/evaluation.md`. Paths inside
these files are relative to the repository root; analyzed upstream sources
appear under `.deps/<project>-<version>/`, where `scripts/prepare_targets.py`
puts them.

## `real-projects/`: the 0.5.0 scan

Produced by `scripts/analyze_parallel.py --jobs 10` per project and published
with `scripts/summarize_scan.py`, which also checks that the triage matches
the scan's alerts one to one.

- `cuda-samples.sarif`, `faiss.sarif`, `rmm.sarif`: SARIF 2.1.0, one run per
  parsed translation unit, with the analyzer's `codeFlows`. Each result's rule,
  level, memory kind and semantic ID are SARIF fields. The first run of each
  file carries the project's coverage totals in `properties.coverage`.
- `triage.csv`: one row per alert (46), with the file relative to the project
  root, line, rule, level, message, category and the reason for the category.
  36 rows are the 0.3.0 triage (0.3.0 and 0.4.0 reported the same alerts);
  the 10 faiss CL2 rows that 0.5.0 adds were triaged against the source.
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

## `mutants/`: the cuda-samples mutants with version 0.5.0

The commands, SARIF, coverage and per-TU logs of analyzing the three original
cuda-samples files and five mutated variants in `corpus/mutants/`, and
`comparison.json`, the comparison with their predetermined oracle sites
(`eval.mutation_report`). Paths were made relative to the repository root;
`corpus/mutants/README.md` explains the results.

## `real-bugs/results.json`

Output of `scripts/run_real_bugs.py` for the 13 cases in `corpus/real-bugs/`:
for the buggy and the fixed revision of each, whether the TU parsed (or the
first errors), its diagnostics, functions analyzed and over budget, coverage
gaps, the generated summaries loaded (`summaries`: each helper TU, its
status and the functions summarized), and whether a diagnostic is on the
labelled lines. Paths are relative to the case's checked-out revision.

## `real-bugs/ctu-trial.json`

The attempt to reach cuDF's `cuda_memcpy_async_impl` with CSA's
cross-translation-unit analysis instead: the configuration, the 19
`ExplicitInstantiation` import errors and the importer assertion it ended in.

## `real-bugs/determinism.json`

Three reruns of `cudf-string-scalar-source` with the 0.5.0 build: per run,
the buggy revision's diagnostic lines and the fixed revision's diagnostic
count, and whether the runs match each other and `results.json` (they do,
apart from timings).

## `real-bugs-heldout/results.json`

The same runner and 0.5.0 build on the held-out cases in
`corpus/real-bugs-heldout/` (`--cases` and `--commands` pointing there):
one record per case with both revisions' diagnostics, coverage and loaded
summaries, and the dependency commits used.

## `baselines/`

Clang 23.1.3's stock analyzer and clang-tidy on the same inputs as the
checker. `seed-*.json` and `holdout-*.json` score `scripts/baseline.py`
reports (in `reports/`, paths made relative) against the corpus labels with
`scripts/score_baseline.py`, listing the warnings behind each hit.
`real-bugs-*.json` and `real-bugs-heldout-*.json` are
`scripts/run_real_bugs.py --tool csa|clang-tidy` records: per revision, the
number of diagnostics and those that touch the labelled file.

## `real-bugs/parquet-budget.json`

Why `cudf-parquet-string-offsets` is missed: the same TU at 2,000,000 nodes,
and the bug's pattern on its own in a 10-line function.

## `regression/`

Seed and holdout metrics from `tests/run_seed.py`. The holdout has been
seen, so it was scored as a development set (`split` is `dev`), not as a
held-out result. The files were written by 0.4.0; `reruns.json` records
rerunning both corpora with the 0.5.0 build and with the two refactored
builds after it (effects dispatched through an enum, clang-format): for each,
the native binary's hash, the totals, the metrics file's hash and a digest
of every SARIF result. All three builds produce byte-identical metrics,
equal to these files, and identical results.

## `probes/stale-read.json`

Output of `probes/stale_read.py` on the same GPU: for pinned and pageable host
memory, the values `probes/stale_read.cu` read from a `cudaMemcpyAsync` D2H
destination before and after synchronizing the stream (20 runs each), and the
checker's reports on that source.

## `probes/sanitizer.json`

Output of `probes/sanitizer.py`: for each scenario of `stale_read.cu` and
`lifetime.cu`, the error count and first reports of each compute-sanitizer
tool and the program's own output, and the checker's reports on both sources.

## `probes/completion.json`

Output of `probes/completion.py` on an RTX 4090 with driver 595.99: how long
`cudaMemcpyAsync` D2H takes to return behind a 20 ms kernel, for pinned and
pageable host memory, five runs each.
