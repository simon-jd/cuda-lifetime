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
used during development. 10 TP, 0 FP, 0 FN, 10 TN.

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

Version 0.3.0 fixes that (a call in statement position, including an `if` or
loop body, discards its status; `tests/native/checked_return_helpers.cpp`
covers both forms) and changes other rules after the holdout was seen. On the
holdout it scores 25 TP, 0 FP, 0 FN, 24 TN
(`results/regression/holdout-metrics.json`); because the set has been seen,
that is a regression result, not a held-out score.

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

| Project | TUs | Parsed | Alerts, 0.2.0 | Alerts, 0.3.0 |
|---|---:|---:|---:|---:|
| cuda-samples 13.2 | 233 | 223 | 32 | 7 |
| faiss 1.15.1, `faiss/gpu/` | 190 | 189 | 42 | 29 |
| RMM 26.10 | 54 | 54 | 2 | 0 |

The ten cuda-samples parse failures are nine `min`/`max` overload errors (for
example in `simpleMultiCopy.cu` and `transpose.cu`) and one unsupported
initialization of a `__shared__` variable (`globalToShmemAsyncCopy.cu`);
faiss's one is a `min` overload error in `GpuDistance.cu`.

Parsed is not the same as fully explored. The analysis records, per TU, the
functions that exhausted the node budget and the CUDA calls that are not
modeled; those per-TU coverage records are not included in this repository.
`results/real-projects/summary.json` has the per-project totals:

- `functionsOverBudget` (385, 232 and 60) counts functions that hit the node
  budget. The total number of functions analyzed was not recorded, so these
  are not fractions of anything.
- `unmodeledCudaFunctions` lists the names of unmodeled calls. They are found
  by name (`cuda` or `cu` followed by an uppercase letter), so the lists also
  contain project functions such as cuda-samples' `cudaGraphsManual` and
  `cuGet`; most real entries for cuda-samples are driver API calls.

### What changed

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

### The 36 alerts of 0.3.0

None of the 36 is a confirmed defect, and all are warnings or notes, so
`--errors-only` reports nothing on these projects.
`results/real-projects/triage.csv` has each alert with its reason; the SARIF
files with full paths are beside it.

| Category | Count | What it is |
|---|---:|---|
| Portability | 22 | Host code reads, writes or frees ordinary `new[]`/`malloc` memory right after `cudaMemcpyAsync` with no synchronization in between: 19 in faiss (`HostTensor` copies used by CPU training and index remapping, H2D copies from temporaries in `IVFBase`), 3 in cuda-samples (an error path that frees copy sources). |
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

## Reproduce

Build the checker and run the corpora:

```sh
ctest --test-dir build --output-on-failure
python3 tests/run_seed.py --native build/cuda-lifetime-native --out results/seed
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
python3 probes/completion.py --out results/probes/completion.json
```

`analyze_targets.py` restricts faiss to `faiss/gpu/` and runs one TU at a time
per project. The committed SARIF files were produced with
`scripts/analyze_parallel.py --jobs 10` on databases translated with the same
profiles (faiss already restricted to its `faiss/gpu/` entries); it runs the
same per-TU analysis in parallel and merges the output.
