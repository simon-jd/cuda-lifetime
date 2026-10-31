# Reproducible cuda-samples v13.2 mutations

`originals/` preserves three complete upstream files byte for byte, including
their NVIDIA copyright/license headers. `generated/` contains five full-source
variants. The upstream checkout under `.deps/` is read-only to this workflow.

| Variant | Original | Predetermined semantic expectation |
|---|---|---|
| `delete_final_sync` | simpleMultiCopy | CL1 error: final pinned D2H output can be read without completion. |
| `sync_after_host_read` | simpleMultiCopy | CL1 error: completion moved after `test()` cannot protect its read. |
| `record_before_launch` | simpleMultiCopy | CL4 warning: the stop-event capture precedes the timed kernel. |
| `launch_on_legacy_stream` | streamOrderedAllocation | CL3 error: a legacy launch does not cover allocations on the nonblocking stream. |
| `pageable_synchronized_control` | asyncAPI | No CL1: the allocation becomes pageable while completion checks remain. Its matching deallocator becomes `free`. |

These are semantic expectations authored before analyzer execution, not measured
outcomes. Three of the variants mark their change with a
`cuda-lifetime mutation:` comment. `manifest.json` records upstream tag/URLs, original and output hashes,
each unique text substitution and its intermediate hashes, primary oracle
locations, assumptions, possible secondary findings, and analysis limitations.
Each replacement must match exactly once. The tool refuses changed source
hashes, ambiguous replacements, writes into the source tree, and overwriting
modified generated files. Repeat generation preserves the first timestamp.

Generate or reproduce the recorded bytes:

```sh
python3 -m eval.mutate --recipes corpus/mutants/recipes.json \
  --source-root .deps/cuda-samples-13.2 --out corpus/mutants
python3 -m eval.mutate --replay-manifest corpus/mutants/manifest.json \
  --source-root .deps/cuda-samples-13.2 --out /tmp/cuda-lifetime-mutant-replay
python3 -m eval.mutation_report --manifest corpus/mutants/manifest.json \
  --sarif results/mutants/out.sarif \
  --coverage results/mutants/coverage.json \
  --native-logs results/mutants/out.logs \
  --out results/mutants/comparison.json
```

The report expects an analysis of `originals/` and `generated/` in
`results/mutants/`: its SARIF, coverage report and per-TU native logs.

The static comparison must analyze originals and variants with identical CUDA
headers, host flags, model files, node limits, and wall-clock budgets. A missing
finding in an incompletely analyzed file is unverified, not a passing negative.
These whole samples contain global arrays, wrappers, and loops: coverage may be
insufficient to establish the expected source fact. The required-site oracle is
not a fully labelled precision benchmark; separately report other diagnostics.

The pageable variant is a negative control. Removing only asyncAPI's polling
loop would leave its checked `cudaEventElapsedTime` result as completion evidence
on successful continuation, so that would not justify inventing a CL1 defect.
None of this workflow executes GPU code or changes the withheld holdout corpus.

The hashes in `manifest.json` cover the files as published here; they were
recomputed when the marker comments were renamed for publication, which
changed no code and no line numbers.

The one recorded run of these variants (its outputs are not included in this
repository) found the original `simpleMultiCopy` and its three variants all
failing the same host `max` overload check; the other four files parse. That
run recorded incomplete coverage, four unverified required-site misses
(including a native crash on the changed launch stream), and unexpected CL1
findings in the synchronized pageable control. It does not establish a passing
mutation score, and no mutation score is claimed for this checker. The
predetermined oracle was not changed after that run; the only later change to
the variants is the marker-comment rename noted above.
