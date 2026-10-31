# Independent evaluation

This package only reads human labels, SARIF, and coverage artifacts. It does not
import or call the checker. From the repository root:

```sh
python -m eval --labels corpus/seed/labels.json --sarif out.sarif \
  --coverage coverage.json --out metrics.json
```

Each positive label matches at most one diagnostic by `(ruleId, file, line)`.
Duplicates are false positives. URI decoding, `file://` URIs, SARIF artifact
indices, URI bases, and relative paths are normalized before matching. Metrics
cover CL1–CL4 in labelled files; diagnostics outside those files and informational
notes are counted separately. Severity mismatches are reported separately and
fail the evaluation. An absent precision/recall denominator is JSON `null`.

`tus.analyzed` (or `tus.parsedFiles`) must list analyzed source filenames. A parsed
count alone cannot establish coverage. Failures, translation-unknown files,
skipped files, `budgetExhausted`, and `coverageUnknown` files are excluded from certified coverage,
even when also listed as analyzed. Positive misses remain FN; un-analyzed
negative cases are uncovered, never TN. Unmodeled API counts remain in the metric
report and prevent an overall pass, as does an explicit `complete: false` from
the coverage producer. Global incompleteness or unmodeled call counts also prevent
crediting silent negatives as TN, since these aggregate fields do not establish
which paths were fully covered. Every negative label means no error or
warning for that rule anywhere in its source file. Exit codes are 0 for a fully
matching covered corpus, 1 for mismatches/incomplete coverage, and 2 for invalid
artifacts.

Labels use `schemaVersion: 1`, a `split`, and `cases`, each containing `id`,
`file`, `ruleId`, positive `line`, `expect` (`warning` or `none`), optional `level`,
and nonempty `semanticIds`. Do not change labels in response to analyzer output.

To record actual current artifact bytes once:

```sh
python -m eval freeze --labels corpus/holdout/labels.json --out freeze.json \
  --rule-file docs/semantics.md --rule-file checker/AsyncLifetime.cpp
python -m eval --labels corpus/holdout/labels.json --freeze freeze.json \
  --sarif out.sarif --coverage coverage.json
```

Use the actual rule source filename in the command. A holdout run requires a
freeze manifest with rule hashes. Existing freezes cannot be overwritten; any
change to labels, sources, or recorded rules fails verification. Hashes establish
integrity after the recorded timestamp; they do not independently prove when a
human wrote the labels, independence from development, or a prior rule freeze.

Actual one-pass holdout execution uses `tests/run_holdout.py`, which requires a
larger freeze than the minimal evaluation-only example above. Its
`required_rule_files(root, native)` function enumerates the executable, checker,
driver, translator, evaluator, models, declaration headers, semantic document,
environment lock, build file, and execution harness that must be frozen. It also
requires a frozen dependency manifest with this format:

```json
{"schemaVersion": 1, "files": [{"file": "/absolute/library-or-header", "sha256": "64 lowercase hex digits"}]}
```

Every listed file is hashed in chunks before and after analysis. The dependency
manifest must itself appear in the rule freeze. The harness requires explicit
`--labels`, `--freeze`, `--native`, `--dependencies`, and a new `--out` directory.
It writes a first-attempt marker beside the freeze before invoking the analyzer;
an aborted or incomplete attempt is retained and cannot be overwritten by
selecting another output directory. Protocol tests use temporary invented files,
not the real holdout. Creating this harness does not freeze or execute a corpus.
