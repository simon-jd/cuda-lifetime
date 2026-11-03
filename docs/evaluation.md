# Evaluation

## Labelled corpora

Each case is a small C++17 program using the declarations in
`corpus/include/cuda_runtime.h`, labelled from the CUDA semantics
([semantics.md](semantics.md)) before the checker ran on it. Matching is exact
and one-to-one on rule, file and primary line; a duplicate diagnostic counts as
a false positive, a negative label applies to its rule throughout the file, and
a file whose coverage is incomplete cannot earn a true negative.

These corpora test the checker against the semantics it was written from: the
cases are short (7 to 18 lines each), use a declaration stub instead of
the CUDA headers, and were written from the same semantics document as the
rules. They show that the rules do what the document says, not how the checker
fares on code written by others; that is what the real-project scan below is
for.

**Seed** (`corpus/seed/`): ten positive/negative pairs, one per rule family,
used during development. 10 TP, 0 FP, 0 FN, 10 TN with versions 0.4.0 and
0.5.0, whose metrics files are byte-identical
(`results/regression/seed-metrics.json`, `results/regression/reruns.json`).

**Holdout** (`corpus/holdout/`): 48 translation units and 49 labels covering
host accesses and lifetimes, aliases and byte ranges, synchronization branches,
stream-ordered allocation, event snapshots and default-stream behavior. The
labels, sources and the checker were frozen (`corpus/holdout/freeze.json`)
before the first run, and the harness (`tests/run_holdout.py`) allows only one
first run per freeze. That run used version 0.1.0 of the checker. Its sources
are not part of this repository; `freeze.json` records the hashes of the
checker, driver, model and declaration files as they were then, and most of
those files have changed since. The first run:

| Rule | TP | FP | FN | TN |
|---|---:|---:|---:|---:|
| CL1 | 6 | 1 | 0 | 5 |
| CL2 | 6 | 0 | 0 | 6 |
| CL3 | 7 | 0 | 0 | 6 |
| CL4 | 6 | 0 | 0 | 6 |
| Total | 25 | 1 | 0 | 23 |

The false positive is `h04_branch_completion_good.cpp:9`:
`if (finish) cudaStreamSynchronize(s); if (!finish) cudaDeviceSynchronize();`
before a read of the pinned destination. The checker forks each CUDA call into
success and failure, and ends the failure path of a call whose status is
discarded. It treated a call in an `if` body as having its status used, so the
failure path of `cudaDeviceSynchronize`, which synchronizes nothing, reached
the read. Artifacts: `results/holdout-first/` (see
[results/README.md](../results/README.md) for what each file records).

Version 0.3.0 fixed that (a call in statement position, including an `if` or
loop body, discards its status; `tests/native/checked_return_helpers.cpp`
covers both forms) and changed other rules after the holdout was seen.
Versions 0.4.0 and 0.5.0 score 25 TP, 0 FP, 0 FN, 24 TN on the holdout, as
0.3.0 did (`results/regression/holdout-metrics.json`); because the set has been seen,
that is a regression result, not a held-out score. The first, frozen run above
remains the holdout result.

**Mutants** (`corpus/mutants/`): five variants of three whole cuda-samples
files, each with a predetermined expectation (four seeded defects and one
negative control). With version 0.5.0, and a forced header that gives host
code the `max(float, float)` overload nvcc's headers have, all eight files
parse. The seeded CL3 defect in `launch_on_legacy_stream` is found; the three
variants of `simpleMultiCopy.cu` are missed because their copies are queued on
elements of a global stream array, which the checker cannot identify, and
three functions exhaust the node budget; the control has no CL1 but cannot be
credited because `main` exhausted the node budget (`results/mutants/`,
[corpus/mutants/README.md](../corpus/mutants/README.md)). No mutation score is
claimed.

## Real projects

`scripts/prepare_targets.py` downloads and builds cuda-samples 13.2, faiss
1.15.1 (GPU enabled) and RMM 26.10; `cuda-lifetime translate` converts their
nvcc databases into Clang host-only commands. The results depend on the header
profiles in [compatibility-profile.md](compatibility-profile.md): cuda-samples
and faiss were translated with `profiles/cccl-510cc9d-host-headers.json`, which
puts a CCCL development snapshot (upstream commit `510cc9d`, not a release)
ahead of the CUDA 13.2 headers, and RMM with `profiles/host-headers.json`,
which keeps RMM's own CCCL. With the CUDA 13.2 headers alone, most faiss CUDA
sources do not parse under Clang.

The scan covers every configured command of cuda-samples and RMM and the
`faiss/gpu/` module, at 225,000 nodes and 180 s per TU, with the models in
`models/`. cuda-samples' own configure step leaves out samples whose optional
dependencies (graphics, MPI, FreeImage, NvSCI) are missing on the build host,
so those samples are not among its 233 commands.

| Project | TUs | Parsed | Alerts, 0.2.0 | Alerts, 0.3.0 | Alerts, 0.4.0 | Alerts, 0.5.0 |
|---|---:|---:|---:|---:|---:|---:|
| cuda-samples 13.2 | 233 | 223 | 32 | 7 | 7 | 7 |
| faiss 1.15.1, `faiss/gpu/` | 190 | 189 | 42 | 29 | 29 | 39 |
| RMM 26.10 | 54 | 54 | 2 | 0 | 0 | 0 |

The ten cuda-samples parse failures are nine `min`/`max` overload errors (for
example in `simpleMultiCopy.cu` and `transpose.cu`) and one unsupported
initialization of a `__shared__` variable (`globalToShmemAsyncCopy.cu`);
faiss's one is a `min` overload error in `GpuDistance.cu`.

Parsed is not the same as fully explored. `scripts/summarize_scan.py` writes
the per-project totals of the 0.5.0 scan to `results/real-projects/summary.json`
and the per-TU figures to `results/real-projects/timings.csv`:

| Project | Top-level functions analyzed | Exhausted the node budget | Hit the 180 s limit | Wall time, 10 processes (0.4.0) | Per-TU time, median / max |
|---|---:|---:|---:|---:|---:|
| cuda-samples 13.2 | 837 | 385 (46%) | 0 | 81 s (97 s) | 2.4 s / 58 s |
| faiss 1.15.1, `faiss/gpu/` | 720 | 232 (32%) | 0 | 146 s (166 s) | 4.0 s / 122 s |
| RMM 26.10 | 233 | 60 (26%) | 0 | 18 s (20 s) | 1.8 s / 16 s |

A function that exhausts the budget was explored only partly, so a TU with one
is never counted as clean. The wall times are for one run with ten TUs
analyzed at a time on a 32-thread machine (the 0.5.0 run shared it with one
more analyzer process, the real-bug runner); the per-TU times include that
concurrency. The differences from 0.4.0 are run-to-run variation, not a
speed-up: the 0.5.0 checker does strictly more work per call. The scans use
the shipped models only, without a summary pass over each project's TUs. `unmodeledCudaFunctions` lists the CUDA Runtime and Driver API
functions the analysis met but does not model (122 in cuda-samples, mostly
driver API calls); a function counts as a CUDA API call only if it is declared
in one of the toolkit's API headers.

### What changed from 0.2.0 to 0.3.0

Every 0.2.0 alert had been sorted into a root-cause family
(`results/real-projects/triage-0.2.0.csv`: 45 false positives, 31 with
insufficient evidence, none confirmed). 23 of the 76 rows were triaged against
the source and the analyzer path; the other 53 carry the judgment of an earlier
triage of the same sites, matched by a semantic fingerprint with the source
bytes checked unchanged (column `review_scope`). By family:

| Family (0.2.0 triage) | 0.2.0 | 0.3.0 | Change that addresses it |
|---|---:|---:|---|
| Relies on pageable copies being synchronous | 22 | 18 | none; these are real patterns (below). The 4 gone were repeat reports at calls CSA now inlines |
| Free or read after a blocking `cudaMemcpy` | 13 | 0 | synchronous `cudaMemcpy` modeled |
| Event of unknown history (object member, container, IPC handle, array) | 16 | 0 | only events created on the path are known unrecorded |
| Call with a body, or a pointer-attribute query, reported as opaque | 12 | 0 | opaque-call rule at PostCall for calls CSA did not inline; `address_only` models |
| Captured graph work treated as running | 6 | 0 | stream capture modeled |
| Event wait reported for later work on the waiting stream | 5 | 1 | waits skip work the waiter is already ordered after |
| Double buffer with symbolic offsets | 1 | 1 | none |
| Wrapper CSA did not inline | 1 | 1 | none |
| Total | 76 | 21 | |

The 0.3.0 scan reports 15 more alerts at sites 0.2.0 did not: 6 in
`watershedSegmentationNPP.cpp`, whose `cudaMemcpy2DAsync` copies were not
modeled before, and 9 in faiss, where paths 0.2.0 reported at a call it
mistook for opaque are now reported at a later call (1 portability finding and
the 8 `getDeviceForAddress` false positives below).

### From 0.3.0 to 0.4.0

Version 0.4.0 reports the same 36 alerts as 0.3.0, with the same file, line,
rule, level and message, so `results/real-projects/triage.csv` carries over
unchanged. Its changes show in the coverage records instead:

- Six cuda-samples functions that 0.3.0 counted as unmodeled CUDA calls
  because of their names (`cudaGraphsManual`, `cudaGraphsUsingStreamCapture`,
  `cudaDeviceInit`, `cuGet`, `cuRand`, `cuEqual`) are no longer counted;
  cuda-samples' unmodeled-call visits drop from 70,412 to 60,135.
- RMM's one `cudaMemcpyDefault` copy, whose operands are function parameters
  of unknown memory kind, is now a recorded coverage gap rather than an
  unknown copy kind. In cuda-samples and faiss the direction modeling added no
  alerts and no coverage gaps; the native fixtures
  `tests/native/memcpy_default*.cpp` exercise it.
- Rule, level, memory kind and semantic ID are SARIF fields written by the
  native consumer; they used to be parsed out of the message text.

### From 0.4.0 to 0.5.0

Version 0.5.0 reports the 36 alerts of 0.4.0 unchanged and 10 more, all in
faiss and all from the new host-container rule: a `std::vector` destroyed
while a `cudaMemcpyAsync` from or into its buffer is still queued, with no
synchronization in between. Six are H2D copies from temporaries in `IVFBase`
and `IVFFlat` (`copyFrom`, `toDeviceTemporary`, `DeviceVector::append`)
destroyed at the end of their scope; four are vectors filled by a D2H in
`copyToHost` or `copyToVector`, returned by value and destroyed by the
caller. All ten are portability findings, like the 22 below: correct only
while copies to and from pageable memory are synchronous. CSA does not elide
the named return value of `copyToHost` and `copyToVector`, so the rule
passes a moved vector's buffer on to the vector it moves into; otherwise the
local would be reported at their `return`. cuda-samples' 7 alerts are unchanged, and RMM
still has none: modeling `cudaMemcpyBatchAsync` and RMM's `memcpy_async`
direction drops RMM's unmodeled-call visits from 273 to 33, but their
operands inside RMM are of unknown memory kind.

### The 46 alerts

None of the 46 is a confirmed defect, and all are warnings or notes, so
`--errors-only` reports nothing on these projects.
`results/real-projects/triage.csv` has each alert with its reason; the SARIF
files with full paths are beside it.

| Category | Count | What it is |
|---|---:|---|
| Portability | 32 | Host code reads, writes or frees ordinary `new[]`/`malloc`/`std::vector` memory right after `cudaMemcpyAsync` with no synchronization in between: 29 in faiss (`HostTensor` copies used by CPU training and index remapping, H2D copies from temporaries in `IVFBase`, the 10 vectors above), 3 in cuda-samples (an error path that frees copy sources). |
| Informational | 3 | `cudaFree` on that same error path relies on its implicit synchronization (CL3 note). |
| False positive | 11 | 8 from faiss's `getDeviceForAddress`, whose result the analyzer cannot relate to the pointer, so it explores copy directions that cannot happen; 1 double buffer whose halves do not overlap but have symbolic offsets; 1 wrapper CSA evaluated without inlining; 1 event wait for work whose buffers the waiter never touches. |

The portability findings are correct today only because the driver makes
pageable copies synchronous, which CUDA permits but does not promise
("might be synchronous with respect to host"). `probes/completion.py` measures
it on the analysis machine (RTX 4090, driver 595.99, five runs each, in
`results/probes/completion.json`): with a 20 ms kernel queued ahead,
`cudaMemcpyAsync` D2H returns in 0.007 ms for pinned memory, before the kernel
finishes, and after 18.4 ms for pageable memory, once the copy is done. The
probe measures device-to-host copies only; some of the portability findings
free the source of a host-to-device copy. Most faiss alerts carry
`memoryKind=unknown`, so "pageable" there is the triage's reading of the
source, not something the analyzer established. The same faiss code would race
if the buffers were pinned or registered.

## Fixed bugs from public projects

`corpus/real-bugs/` has 13 stream-ordering and lifetime bugs that cuDF,
RMM, llama.cpp, cuda-samples and three smaller projects fixed in host code, each
labelled from its fix's diff before the checker ran: eight are bugs the rules
describe (six CL2, one CL1, one CL4), five are of kinds the checker does not
claim to find. `scripts/run_real_bugs.py` analyzed the buggy and the fixed
revision of each affected TU (`results/real-bugs/results.json`); the labels
did not change between versions.

These 13 cases are a development set. Version 0.4.0 was run on them first,
and the changes listed below were made to analyze them; version 0.5.0 was
then scored on the same cases. Its 3 of 13 therefore shows what the new
features do on the bugs they were built for, not how the checker does on
bugs it has not seen.

| | Cases | 0.4.0 found | 0.5.0 found | Missed | Not analyzable |
|---|---:|---:|---:|---:|---:|
| Within the rules | 8 | 0 | 3 | 4 | 1 |
| Outside the rules | 5 | 0 | 0 | 3 | 2 |

Version 0.5.0 finds 3 of the 8 in-scope bugs (3 of 13 overall), and each
fix silences its report: cuDF's scoped `single_node` passed to
`set_element_async`, and the staging and regex-program `host_vector`s
destroyed with a copy pending. Four changes, each made for these cases,
made that possible:

- **The crash.** The native tool aborted on cuDF's parquet `reader_impl.cpp`
  where `clang++ --analyze` did not because the checker, not CSA, read the
  store for every variable a function body names, including the constant
  locals of a generic lambda's pattern, whose initializers are
  value-dependent; stock CSA never evaluates template patterns. Those
  variables are now skipped (`tests/native/dependent_locals.cpp`).
- **Helpers in other TUs.** The copies are queued in RMM's and cuDF's
  libraries (`rmm::detail::memcpy_async`, `cuda_memcpy_async_impl`), and the
  fixes wait in RMM's `cuda_stream_view::synchronize`. CSA's CTU analysis
  could not import them (`results/real-bugs/ctu-trial.json`); summaries
  generated from those TUs could ([models.md](models.md#generated-summaries)).
- **Containers.** The buffers belong to `host_vector`s reached through
  `host_span` conversions; the owner's destruction is now a CL2 warning.
- **Headers.** A patched copy of cuDF's `type_dispatcher.hpp` (Clang rejects
  its specializations in CUDA mode) makes the three cuDF `.cu` TUs parse, and
  a stand-in for gpe's generated `gpe_kernels.h` makes its TU parse.

What still misses, in scope: the parquet bug, whose pattern is reported in
isolation but whose function exhausts its budget even at 2,000,000 nodes
(`results/real-bugs/parquet-budget.json`); cuDF's `string_scalar` copy (through RMM's
`device_buffer`) from the caller's string, which no lifetime in the TU ends; gpe's staging ring,
copied with driver API calls across calls to `dispatch`; and RMM's
`merge_lists`, whose defect (an event not re-recorded after a wait) only
matters to waits in later calls, which a per-path checker cannot relate. The
Arrow case now parses past the dispatcher but needs `nanoarrow.h`, which
cuDF's build downloads and was not fetched. The string-scalar TU also gets
four CL2 warnings off its labelled line, copies from by-value parameters and
a temporary that end while queued (portability findings, pageable memory).
[corpus/real-bugs/README.md](../corpus/real-bugs/README.md) lists every case
with its reason.

### Held-out fixed bugs

`corpus/real-bugs-heldout/` has ten bugs mined after 0.5.0 was released,
from fix commits not used in its development, all of kinds the rules
describe (five CL1, four CL2, one CL3; NCCL, cuVS, cuML, RAFT, cuGraph,
Warp). The labels were committed before any run, then the compile commands
(from compiler-only syntax checks), then the results of the same 0.5.0
build (`results/real-bugs-heldout/results.json`):

| | Cases | Found | Fix removes the report | Missed | Not analyzable |
|---|---:|---:|---:|---:|---:|
| Held out, within the rules | 10 | 4 | 2 | 5 | 1 |
| Development set, within the rules | 8 | 3 | 3 | 4 | 1 |

The two found bugs whose fixes are not recognized synchronize through
`raft::interruptible::synchronize`, which passes `cudaStreamQuery` to a
helper as a function pointer; the checker does not resolve the indirect
call, so the fixed revisions are reported again, false positives. The five
misses are in functions that exhaust the node budget (cuVS's `serialize`
and test bodies, cuML's `TSNE_fit`, RAFT's MST test, Warp's
`wp_memtile_device`), and the RAPIDS copies go through `cudaMemcpyDefault`
with operands of unknown memory kind, whose direction the checker leaves
unmodeled. One TU does not parse (CCCL 3.1 under Clang 23). The per-case
reasons are in
[corpus/real-bugs-heldout/README.md](../corpus/real-bugs-heldout/README.md).

## Stock analyzer and clang-tidy

The same translation units, with the same flags, through Clang 23.1.3's
stock static analyzer (`clang++ --analyze`, default checkers) and clang-tidy
23.1.3 (`clang-analyzer-*` and `bugprone-*`, headers included, the projects'
own `.clang-tidy` files overridden). Neither has CL1–CL4 rules, so matching
ignores the rule: on the corpora, any warning on a label's file and line
counts (`scripts/score_baseline.py`); on the fixed bugs, any diagnostic on a
site line, as for cuda-lifetime (`scripts/run_real_bugs.py --tool csa` and
`--tool clang-tidy`). Results are in `results/baselines/`.

| | cuda-lifetime 0.5.0 | Stock CSA | clang-tidy |
|---|---|---|---|
| Seed, TP/FP/FN/TN | 10/0/0/10 | 2/2/8/8 | 2/2/8/8 |
| Holdout (seen, regression), TP/FP/FN/TN | 25/0/0/24 | 4/4/21/20 | 4/4/21/20 |
| Development bugs within the rules, found of 8 | 3 | 0 | 0 |
| Held-out bugs, found of 10 | 4 | 0 | 1 |

Every corpus hit of both baselines is `deadcode.DeadStores` ("value stored
during its initialization is never read") on a line that the positive and
the negative case of a pair share, so each true positive comes with a false
positive; none is about the asynchronous lifetime. clang-tidy's one held-out
hit (Warp) and one hit on a development case outside the rules (cuda-samples
`systemWideAtomics`) are `bugprone-signed-bitwise` and
`bugprone-narrowing-conversions` warnings that happen to sit on a site line
and remain after the fix. Neither baseline reports anything about a host
access or lifetime racing an asynchronous copy; that is not what their
checks look for, so this shows the gap the rules fill rather than a
competing detector. The baselines fail to parse the same TUs cuda-lifetime
does (cuVS's CCCL 3.1 revision, the MPI and nanoarrow cases), and
clang-tidy crashes on the cuVS one.

## What compute-sanitizer sees

compute-sanitizer is not a baseline for these rules. It checks the device
side of the executions it observes: kernel and copy accesses to memory that
is out of bounds, freed or not yet allocated, races in shared and global
memory between device threads, and misuse of device synchronization. The
rules are about host code racing work that is still in flight, which is not
a device access. The comparison shows what each tool sees in the same
programs, not which is better; neither covers the other's ground.

`probes/sanitizer.py` runs every compute-sanitizer tool (memcheck, memcheck
with `--track-stream-ordered-races all`, racecheck, synccheck, initcheck;
version 2026.1.1 on the RTX 4090) on each scenario of `probes/stale_read.cu`
and `probes/lifetime.cu`, and the checker on both sources
(`results/probes/sanitizer.json`):

| Scenario | cuda-lifetime | compute-sanitizer |
|---|---|---|
| read a pinned D2H destination before synchronizing (`stale_read pinned`, `lifetime pinned`; the read saw the stale value) | CL1 error | nothing |
| the same with pageable memory | CL1 warning | nothing |
| `cudaFreeHost` while a D2H into the buffer is pending (`lifetime free-host`) | CL2 warning | nothing |
| `cudaFree` of a buffer a running kernel writes (`lifetime free`) | CL3 note | nothing |
| a kernel on another stream uses a `cudaMallocAsync` allocation with no dependency (`lifetime ordered-bad`) | CL3 error | memcheck: 2 errors (invalid global write), with or without `--track-stream-ordered-races all` |
| the same with an event dependency (`lifetime ordered-good`) | nothing | nothing |

compute-sanitizer reports the one scenario with a device-side fault, where
the kernel writes memory its stream is not ordered after the allocation of.
The host-side races are outside what it instruments: a host read of an
in-flight D2H destination, or a host free under a pending copy, is not a
device access, so it passes silently even in the run where the stale value
was read. The checker reports the five scenarios that depend on timing or
on behavior CUDA does not promise (pageable copies being synchronous,
`cudaFree` synchronizing), at the level the rules give them, and nothing on
the ordered one, from the source alone. It sees nothing of what
compute-sanitizer checks inside kernels: out-of-bounds or uninitialized
accesses and data races between device threads are outside its rules. The
seed and holdout cases cannot be run: they are functions against a
declaration stub, without `main`.

## Reproduce

Build the checker and run the corpora:

```sh
ctest --test-dir build --output-on-failure
python3 tests/run_seed.py --native build/cuda-lifetime-native --out results/seed
```

The fixed bugs, development and held-out, with cuda-lifetime and the two
baselines (LLVM 23.1.3 release binaries for the baselines):

```sh
for set in real-bugs real-bugs-heldout; do
  python3 scripts/run_real_bugs.py --cases corpus/$set/cases.json \
    --commands corpus/$set/commands.json --clones .deps/recall \
    --native build/cuda-lifetime-native --out results/$set-run.json
  for tool in csa clang-tidy; do
    python3 scripts/run_real_bugs.py --tool $tool --clang $LLVM/bin/clang++ \
      --tidy $LLVM/bin/clang-tidy --cases corpus/$set/cases.json \
      --commands corpus/$set/commands.json --clones .deps/recall \
      --out results/$set-$tool.json
  done
done
python3 scripts/baseline.py --compdb results/seed/compile_commands.json \
  --mode analyze --clang $LLVM/bin/clang++ --out results/seed-csa
python3 scripts/score_baseline.py --labels corpus/seed/labels.json \
  --report results/seed-csa/report.json --out results/seed-csa.json
```

Build the three projects, then translate each nvcc database into
`targets/<project>/clang/`, which `scripts/analyze_targets.py` reads. First set
`ccclRoot` in `profiles/cccl-510cc9d-host-headers.json` to an unmodified
extract of the CCCL archive pinned in
[compatibility-profile.md](compatibility-profile.md). The output directories
must be new or empty.

```sh
python3 scripts/prepare_targets.py --build --jobs 8
for project in cuda-samples faiss; do
  ./cuda-lifetime translate --compdb targets/$project/build --out targets/$project/clang \
    --clang /opt/llvm-build/bin/clang++ --cuda-path /usr/local/cuda-13.2 \
    --gcc-toolchain /usr --compatibility-profile profiles/cccl-510cc9d-host-headers.json
done
./cuda-lifetime translate --compdb targets/rmm/build --out targets/rmm/clang \
  --clang /opt/llvm-build/bin/clang++ --cuda-path /usr/local/cuda-13.2 \
  --gcc-toolchain /usr --compatibility-profile profiles/host-headers.json
python3 scripts/analyze_targets.py --native build/cuda-lifetime-native \
  --toolchain-kind source-assertions --out results/real-projects-run
python3 scripts/summarize_scan.py --scan results/real-projects-run \
  --triage results/real-projects/triage.csv \
  --previous-triage results/real-projects/triage-0.2.0.csv --previous-version 0.2.0 \
  --out results/real-projects-summary
python3 probes/completion.py --out results/probes/completion.json
```

`analyze_targets.py` restricts faiss to `faiss/gpu/` and runs one TU at a time
per project. The committed results were produced with
`scripts/analyze_parallel.py --jobs 10` on databases translated with the same
profiles (faiss already restricted to its `faiss/gpu/` entries), which runs the
same per-TU analysis in parallel and records the wall time;
`summarize_scan.py` reads either layout and refuses a triage that does not
match the scan's alerts.
