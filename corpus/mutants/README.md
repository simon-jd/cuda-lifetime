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

Generate or reproduce the recorded bytes, analyze the eight files and compare
the findings with the oracle:

```sh
python3 -m eval.mutate --recipes corpus/mutants/recipes.json \
  --source-root .deps/cuda-samples-13.2 --out corpus/mutants
python3 -m eval.mutate --replay-manifest corpus/mutants/manifest.json \
  --source-root .deps/cuda-samples-13.2 --out /tmp/cuda-lifetime-mutant-replay
./cuda-lifetime analyze --compdb results/mutants/compile_commands.json \
  --native build/cuda-lifetime-native --sarif mutants-run/out.sarif --coverage mutants-run/coverage.json
python3 -m eval.mutation_report --manifest corpus/mutants/manifest.json \
  --sarif mutants-run/out.sarif --coverage mutants-run/coverage.json \
  --native-logs mutants-run/out.logs --out mutants-run/comparison.json
```

`results/mutants/compile_commands.json` holds the commands used: real CUDA
13.2 headers, host-only Clang, cuda-samples' `Common/` includes and no header
profile. The report can also be recomputed from the published records with
`--paths-root .`, since their paths are relative to the repository root.

The static comparison must analyze originals and variants with identical CUDA
headers, host flags, model files, node limits, and wall-clock budgets. A missing
finding in an incompletely analyzed file is unverified, not a passing negative.
These whole samples contain global arrays, wrappers, and loops: coverage may be
insufficient to establish the expected source fact. The required-site oracle is
not a fully labelled precision benchmark; separately report other diagnostics.

The pageable variant is a negative control. Removing only asyncAPI's polling
loop would leave its checked `cudaEventElapsedTime` result as completion evidence
on successful continuation, so that would not justify inventing a CL1 defect.
None of this workflow executes GPU code or changes the holdout corpus.

The hashes in `manifest.json` cover the files as published here; they were
recomputed when the marker comments were renamed for publication, which
changed no code and no line numbers.

## Results of version 0.5.0

`results/mutants/` has the analysis of the three originals and five variants
with version 0.5.0 (SARIF, coverage, the per-TU coverage records the
comparison reads) and the oracle comparison.
Version 0.4.0's run failed to parse `simpleMultiCopy.cu` and its three
variants: Clang's CUDA headers declare `max(float, float)` for the device
only, where nvcc's `crt/math_functions.hpp` declares it for host and device,
and the sample calls it from host code (lines 148, 243 and 245).
`include/host_max.h` adds the host overloads nvcc has; every command,
originals and variants alike, force-includes it
(`results/mutants/compile_commands.json`). All eight files now parse:

| Variant | Expected | Result | Why |
|---|---|---|---|
| `delete_final_sync` | CL1 error, line 361 | unverified miss | the copies are queued on `stream[i]`, a global array the checker cannot resolve to a stream, so they are not modeled (the coverage report counts 3 `cudaMemcpyAsync` and 2 launch visits with an unknown stream); `main`, `test` and `init` exhaust the node budget |
| `sync_after_host_read` | CL1 error, line 362 | unverified miss | as above |
| `record_before_launch` | CL4 warning, line 214 | unverified miss | as above; the `cudaEventRecord` is also on an unknown stream |
| `launch_on_legacy_stream` | CL3 error, line 83 | found | coverage incomplete: two functions exhausted the node budget |
| `pageable_synchronized_control` | no CL1 | none reported, unverified | coverage incomplete: `main` exhausted the node budget |

The three simpleMultiCopy misses are now analyzed misses rather than parse
failures: the checker identifies a stream from a handle value it has seen
created or passed, and an element of a global array indexed by a loop
variable is neither. For `launch_on_legacy_stream` the checker also reports
the three `cudaFreeAsync` calls that follow the misplaced launch (lines 85, 86
and 88), which the oracle lists as possible additional findings. The
originals produce no findings. No variant has complete coverage, so none of
this is a passing mutation score, and none is claimed.

Version 0.4.0 gave the same results for `launch_on_legacy_stream` and the
control, with the three simpleMultiCopy files failing to parse. The first
run of these variants, with an earlier version of the checker, had the same
three parse failures; its fourth miss was a crash of the native tool on
`launch_on_legacy_stream`, and it reported CL1 on the pageable control. Its
outputs are not included. The predetermined oracle was not changed after
any run; the only later changes are the marker-comment rename noted above
and the forced host header.
