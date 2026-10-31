# Holdout corpus

The 48 translation units and 49 labels here were written by hand from
`docs/semantics.md` as positive/negative pairs. Their author had previously
run the checker on the separate seed and development sets. Sources and labels
were fixed before the rule freeze and before any of these files was run
through the checker; the labels were not changed after the first run, including
for the one false positive it produced. This is a held-out corpus in that
sense, not a blinded external study: the cases come from the same semantics
document as the rules.

The 24 semantic pairs cover stream/event transitivity, event snapshots, host
lifetime, aliases, copy ranges, default-stream kinds and explicit completion.
Some positive files have more than one rule label; every expected primary
location is annotated. They are static fixtures, not GPU-runtime evidence. As
with the seed fixtures, compilation uses `corpus/include/cuda_runtime.h` for
runtime declarations; the real-toolkit kernel pairs live in `../dev/`.

## Records

- `authorship.json`: written with the corpus, before the freeze. It records
  the authoring time, the hash of `docs/semantics.md` at that time, and the
  hashes of the labels and every source. `"executionStatus": "not-run"` here
  and in `labels.json` describes the corpus when it was written.
- `freeze.json`: the rule freeze recorded immediately before the first run. It
  hashes the labels, the 48 sources, and the native executable, checker,
  driver, translator, evaluator, models, declaration header, semantics
  document, environment lock and build file as they were then (version
  0.1.0).
- `freeze.json.first-run.json`: the first-attempt marker written by
  `tests/run_holdout.py` when that run started. The run's outputs are in
  `results/holdout-first/` and are reported in `docs/evaluation.md`.

The sources match their hashes in `authorship.json` and `freeze.json`
byte for byte. When the repository was prepared for publication, the
provenance note (`labelOrigin`) in `labels.json` was reworded and absolute
paths in the first-run records were made relative to the repository root; the
label entries themselves (ids, files, lines, expected rules and levels) are
unchanged. The hashes of those edited files, and of the records that pin them,
were recomputed then, so they cover the files as published here, not the
bytes of the original records. The hashes of rule files in `freeze.json` are
those of the 0.1.0 code: most of those files have changed since and no longer
match, and the 0.1.0 sources are not part of this repository. Likewise,
`docs/semantics.md` has been revised since `authorship.json` recorded its
hash.
