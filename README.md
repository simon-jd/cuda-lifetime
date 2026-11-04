# cuda-lifetime: a Clang Static Analyzer checker for CUDA asynchronous lifetimes

CUDA host code queues copies, kernel launches and frees that run later, in
stream order. Reading a copy's destination, overwriting its source or
freeing its memory before that work is complete is a race that tests rarely
show. `cuda-lifetime` is an out-of-tree, path-sensitive checker for the
Clang Static Analyzer that tracks each stream's pending work along every
path through host code and reports a SARIF path from the queueing call to
the conflicting access.

**Scope:** host code, one translation unit at a time; LLVM/Clang 23.1.3 and
CUDA 13.2; 29 fixed bugs from public CUDA projects, two small synthetic
labelled corpora, and 477 TUs of cuda-samples 13.2, faiss 1.15.1 and RMM 26.10.

| Set | Role | Positives | Found | Fix silences it | Stock CSA found | clang-tidy found |
|---|---|---:|---:|---:|---:|---:|
| Fixed bugs held out from 0.6.0 | held out, one run | 6 | 2 | 2 | not run | not run |
| Fixed bugs held out from 0.5.0 | development set v2 | 10 | 6 | 3 | 0 | 1 |
| First fixed-bug set, within the rules | development set | 8 | 3 | 3 | 0 | 0 |
| Holdout corpus (synthetic), 0.1.0's first run | held out; freeze not verifiable from git | 25 | 25 | – | 4 | 4 |

Version 0.6.0 unless noted, from `results/real-bugs-heldout-0.6.0/results.json`,
`results/real-bugs-heldout/results-0.6.0.json`, `results/real-bugs/results-0.6.0.json`,
`results/holdout-first/metrics.json` and `results/baselines/`. The first and
last rows are held out; 0.6.0 was developed on the second row's bugs and 0.5.0
on the third's, so those rows are development results. "Fix silences it" is
the runner's `fixSilencesOnSiteLines`. None of the baselines' hits is about the
race (they are dead-store, narrowing and signed-bitwise warnings), and on the
holdout each baseline also makes 4 false reports (0.1.0: 1). On the three
projects 0.6.0 reports 46 alerts and no confirmed defect
(`results/real-projects/summary.json`).

## Limitations

- **Low recall on unseen bugs:** 2 of 6. Two held-out TUs do not parse under
  Clang, and two bugs are event dependencies across translation units.
- **No confirmed defect in real code:** of 46 alerts, 32 are portability
  findings, 3 notes and 11 false positives; 26–38% of each project's
  top-level functions exhaust the node budget.
- **Fixes not recognized:** on development set v2, 3 of the 6 found bugs are
  still reported after their fix.
- **Scope:** one host thread, device and TU; no driver API, device code or
  graph contents; a `cudaMemcpyDefault` copy with an operand of unknown
  memory kind is left out, as a coverage gap.
- **The synthetic corpora** were written from the same semantics document as
  the rules, so they test conformance, not recall.

[How it works](#how-it-works) · [design trade-offs](docs/design-tradeoffs.md) ·
[full results and version history](docs/evaluation.md) · [result files](results/README.md) ·
[reproducing](docs/evaluation.md#reproduce) · [modeling limits](#modeling-limits)

## A bug it finds

cuCollections' `dynamic_map::insert`, one of the six held-out bugs, copied a
submap's success counter to the host and used it at once
(`include/cuco/detail/dynamic_map.inl` at `ad9c478`, lines 172–179, condensed):

```cpp
std::size_t h_num_successes;
cudaMemcpyAsync(&h_num_successes, submap_num_successes_[submap_idx],
                sizeof(atomic_ctr_type), cudaMemcpyDeviceToHost, stream);
submaps_[submap_idx]->size_ += h_num_successes;  // line 178: CL1
size_ += h_num_successes;                         // line 179: CL1
```

The checker reports both reads as CL1 ("Host reads the destination before
asynchronous D2H completion"), with the copy on line 173 on the path, and the
same two reports in `erase` (lines 221–227). They are warnings: a stack
variable is pageable memory, and current drivers happen to finish such a copy
before `cudaMemcpyAsync` returns. The fix (`71bcfa4`) synchronizes the stream
after each copy, and the fixed revision has no report in the file.

With pinned memory the same read is a CL1 error:

```cpp
int early_read(void *device, cudaStream_t stream) {
  void *host;
  cudaMallocHost(&host, 4);
  cudaMemcpyAsync(host, device, 4, cudaMemcpyDeviceToHost, stream);
  return *static_cast<int *>(host);  // CL1 error: read before the D2H completes
}
```

On an RTX 4090, [`probes/stale_read.cu`](probes/stale_read.cu) reads the stale
value this way in 20 of 20 runs, against 0 of 20 with pageable memory.

## Rules

| Rule | Finds | Severity |
|---|---|---|
| CL1 | host access to a buffer an asynchronous copy still uses: a read of a D2H destination, a write of an H2D source, the buffer passed to a call the analyzer cannot see into, or stored where it escapes the path | `error` for pinned memory with exact byte overlap; `warning` for pageable or unknown memory, partial overlap, and opaque calls |
| CL2 | host storage released or leaving scope with work pending: `cudaFreeHost`, `free`, `delete`, destructors, stack buffers at function return | `warning`, `error` for known pinned stack storage |
| CL3 | stream-ordered allocations used before their allocation is ordered, freed on a stream that does not cover a pending use, used after `cudaFreeAsync`; `cudaFree` relying on its implicit synchronization | `error` with known streams, `note` for the implicit-synchronization reliance |
| CL4 | a wait on an event that was created but never recorded, or whose capture predates producer work the waiter is not otherwise ordered after | `warning` |

The policy behind each rule, with its CUDA documentation reference, is in
[docs/semantics.md](docs/semantics.md), and the main design decisions, with
their alternatives and costs, in [docs/design-tradeoffs.md](docs/design-tradeoffs.md).
Warnings are shown by default; `--errors-only` hides them.

## How it works

- **Stream model.** Each stream has a sequence number; a queued operation
  records its stream, sequence, buffer region, byte range, direction and memory
  kind (pinned, pageable, device, stream-ordered, unknown). Synchronization
  retires operations up to the point it proves complete: `cudaStreamSynchronize`
  and successful `cudaStreamQuery` for the stream and everything it waits on,
  `cudaEventSynchronize` for the event's capture, `cudaDeviceSynchronize` for all.
  `cudaStreamWaitEvent` orders device work only and completes nothing for the host.
- **Default streams.** The legacy default stream waits for, and is waited on by,
  blocking streams; non-blocking and per-thread default streams are independent
  (`CUDA_API_PER_THREAD_DEFAULT_STREAM` and `-fgpu-default-stream` are honored).
- **Synchronous copies.** `cudaMemcpy` and `cudaMemcpy2D` with host memory on
  either side return only after earlier default-stream work (and work on the
  streams it waits for) is complete, so they retire it; device-to-device copies
  and `cudaMemset` on device memory do not.
- **Graph capture.** Work enqueued on a capturing stream, or on a stream that
  joins the capture through an event, is recorded into a graph rather than
  executed, so it creates no pending operation. Graph contents and launches are
  reported as a coverage gap.
- **Calls the analyzer cannot see.** A call CSA inlines is checked access by
  access. For a call it evaluates conservatively, the checker computes at
  `PreCall` which pending buffers the arguments reach, directly or through
  pointers stored in memory they reach, and reports at `PostCall` only when the
  call was not inlined. A buffer reached through a pointer-to-const parameter
  can only be read, which conflicts only with a pending D2H.
- **Return values.** Each modeled call forks success and failure paths. A
  checked failure that ends in a `noreturn` helper (`exit`, `abort`, CUDA
  samples' `checkCudaErrors`) ends the path; ignored results are assumed to
  succeed, and that assumption is recorded.
- **Wrappers.** Project functions that wrap CUDA calls can be summarized in a
  JSON model with a source reference ([docs/models.md](docs/models.md));
  `models/` has the faiss and RMM ones used below.
- **Helpers in other translation units.** `cuda-lifetime-native --summarize`
  writes such models from a helper's own TU: the copy a function's body
  queues on its parameters, or the synchronization every path through it
  performs, composed through the calls it makes. Bodies CSA declines to
  inline in the analyzed TU (container methods) get the same summary at the
  call. A helper that polls a function its caller passes (RAFT's
  interruptible wait hands it `cudaStreamQuery`) is summarized as calling
  that parameter. CSA's own cross-TU analysis was tried and fails on
  libstdc++ code
  ([docs/design-tradeoffs.md](docs/design-tradeoffs.md#helpers-in-other-translation-units-generated-summaries-not-ctu)).
- **The node budget.** A path ends where it can only reach a throw or a call
  that never returns; shared_ptr reference counting, iostreams and locks are
  not inlined unless a pending buffer reaches them; and the CUDA-calling
  callees of a function whose budget ran out are analyzed on their own
  (fallback entry points). `--node-profile` records where each top-level
  function's exploded-graph nodes are
  ([docs/design-tradeoffs.md](docs/design-tradeoffs.md#spending-the-node-budget-where-cuda-state-can-change)).
- **Host containers.** The buffer `data()` returns from `std::vector`,
  `std::basic_string` or `thrust::host_vector` (and classes derived from
  them, like cuDF's `host_vector`), or from a view of one, belongs to the
  container: destroying it with a copy pending is CL2. Moving the container
  moves the buffer.

SARIF output carries the analyzer's own path (`codeFlows`), including the
"queued here" event for the pending operation. A per-TU coverage report lists
parse failures, unmodeled CUDA calls, exhausted node budgets and unknown
effects; a TU with incomplete coverage is never counted as clean.

## Results

**Labelled corpora.** Every case is a small C++ program with the CUDA runtime
declarations, labelled from the CUDA semantics before the checker ran on it.

| Corpus | Cases | TP | FP | FN | TN |
|---|---:|---:|---:|---:|---:|
| seed (development pairs) | 20 | 10 | 0 | 0 | 10 |
| holdout (48 TUs, 49 labels, first run; freeze not verifiable from git, see below) | 49 | 25 | 1 | 0 | 23 |

The holdout numbers are those of the first run after the freeze. The freeze
cannot be checked against this repository's history: the corpus,
`corpus/holdout/freeze.json` (frozen at 2026-10-14T06:57Z) and the first-run
record (started 28 s later) all arrived in the single initial commit of
2026-10-30, so the order rests on the timestamps and hashes in those files,
not on commits. Its one false positive
exposed a checker bug: in `if (finish) cudaStreamSynchronize(s); if (!finish)
cudaDeviceSynchronize();` the calls in the `if` bodies were treated as having
their status checked, so their failure paths, which synchronize nothing,
continued to the read. With that fixed, this version scores 25/0/0/24 on the
holdout, but the set has been seen, so that is a regression result, not a
held-out score.

**Fixed bugs.** Each bug is labelled from its fix commit (rule, file, site
lines) before any analyzer runs on it, and `scripts/run_real_bugs.py`
analyzes the buggy and the fixed revision of its translation unit. A bug is
found when the buggy revision has a CL1–CL4 report on a site line. The fix
silences it when the fixed revision has none on those lines, as the fix's
diff moved them (`fixSilencesOnSiteLines`). Three sets were labelled in
turn:

- 13 bugs from cuDF, RMM, llama.cpp and other projects, 8 of them of the kinds
  the rules describe. Version 0.4.0 found none; 0.5.0 was developed on them.
- 10 bugs from NCCL, cuVS, cuML, RAFT, cuGraph and Warp, labelled before 0.5.0
  ran on them. Version 0.6.0 was then developed on them.
- 6 bugs from cuCollections, cuOpt, spark-rapids-jni, instant-ngp and
  GROMACS, projects no earlier search used, labelled before 0.6.0 ran on them
  once.

The per-case reasons are in the corpus READMEs
([first](corpus/real-bugs/README.md), [second](corpus/real-bugs-heldout/README.md),
[third](corpus/real-bugs-heldout-0.6.0/README.md)), and how each version
scored and what changed between them in
[docs/evaluation.md](docs/evaluation.md#fixed-bugs-from-public-projects).
Clang's stock analyzer and clang-tidy, run on the same units with the same
flags, report nothing about these races: on the corpora and the first two
bug sets their only hits on labelled lines are dead-store, narrowing and
signed-bitwise warnings, present in the fixed code too
([docs/evaluation.md](docs/evaluation.md#stock-analyzer-and-clang-tidy)).

compute-sanitizer is a dynamic, device-side tool, so it complements the
checker rather than serving as a baseline for it: on the same stale-read
program and five timing-dependent lifetime scenarios it reports the one
device-side fault, a kernel using a stream-ordered allocation before it is
ordered, and none of the host reads or frees racing an in-flight copy,
which are not device accesses
([docs/evaluation.md](docs/evaluation.md#what-compute-sanitizer-sees)).
CuSan, a dynamic host/device race detector built on ThreadSanitizer, reports
the four host reads racing a D2H copy, but not the frees with work pending
or the stream-ordered allocation race
([docs/evaluation.md](docs/evaluation.md#cusan)).

**Real projects.** Every configured host compilation command of cuda-samples
13.2 (233; samples whose optional dependencies were missing are not
configured), faiss 1.15.1's GPU module (190) and RMM 26.10 (54), at a budget of
225,000 nodes and 180 s per TU. cuda-samples and faiss were translated with a
header profile that puts a CCCL development snapshot ahead of CUDA 13.2's own
CCCL headers ([docs/compatibility-profile.md](docs/compatibility-profile.md)); without
it most faiss CUDA sources do not parse. Each alert was triaged against the
source (`results/real-projects/triage.csv`):

| Project | TUs (parsed) | Alerts | Portability | Informational | False positive |
|---|---:|---:|---:|---:|---:|
| cuda-samples 13.2 | 233 (223) | 7 | 3 | 3 | 1 |
| faiss 1.15.1, GPU module | 190 (189) | 39 | 29 | 0 | 10 |
| RMM 26.10 | 54 (54) | 0 | 0 | 0 | 0 |
| Total | 477 | 46 | 32 | 3 | 11 |

All 46 are warnings or notes. In the 0.6.0 scan 397 of 1,034, 234 of 731 and
61 of 237 top-level functions exhaust the node budget, so these projects are
only partly explored. The summed per-TU analysis time is 586 s, 1,232 s and
126 s, and the wall time with six analyzer processes 105 s, 211 s and 24 s
(per-TU times in `results/real-projects/timings.csv`). Earlier versions
reported 76 alerts (0.2.0), 36 (0.3.0 and 0.4.0) and the same 46 (0.5.0);
[docs/evaluation.md](docs/evaluation.md#real-projects) has the per-version
counts, the modeling gaps behind the drop after 0.2.0, and the 0.2.0 triage.

**What remains is mostly one pattern.** 29 faiss reports and 3 cuda-samples
reports are host code that reads or frees ordinary `new[]`/`malloc`/`std::vector` memory right
after a `cudaMemcpyAsync`, with no synchronization in between. That is correct
only because copies to and from pageable memory are synchronous on current
drivers, which the CUDA documentation allows but does not promise ("might be
synchronous"). On an RTX 4090 with driver 595.99, `probes/completion.py` shows
the difference (`results/probes/completion.json`, five runs each):

| `cudaMemcpyAsync` D2H behind a 20 ms kernel | call returns after | kernel done when it returns |
|---|---:|---|
| pinned host memory | 0.007 ms | no |
| pageable host memory | 18.4 ms | yes |

The same faiss code would race if those buffers were pinned (for example
through `cudaHostRegister` for faster transfers). `probes/stale_read.cu` shows
what that race looks like: it reads the destination of a `cudaMemcpyAsync` D2H
before synchronizing the stream, with a 20 ms kernel that writes 42 queued
ahead of the copy. cuda-lifetime reports the read as a CL1 error for pinned
memory and a CL1 warning for pageable memory, and on the RTX 4090
(`results/probes/stale-read.json`, 20 runs each):

| Host memory | value read before `cudaStreamSynchronize` | after |
|---|---|---|
| pinned (`cudaMallocHost`) | -1, the stale sentinel, in 20 of 20 runs | 42 |
| pageable (`malloc`) | 42 in 20 of 20 runs | 42 |

The 11 false positives are
listed in [docs/evaluation.md](docs/evaluation.md); 8 of them come from one
faiss helper, `getDeviceForAddress`, whose result the analyzer cannot relate
to the pointer, so it explores copy directions that cannot happen.

## Build and run

LLVM/Clang 23.1.3 development libraries, CUDA 13.2, CMake and Ninja, Python 3.12.

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_DIR=/path/to/llvm/lib/cmake/llvm -DClang_DIR=/path/to/llvm/lib/cmake/clang
cmake --build build
ctest --test-dir build --output-on-failure            # lit fixtures, native checks, seed corpus
python3 -m unittest discover -s tests -p 'test_*.py'  # driver, translator, evaluator

./cuda-lifetime translate --compdb /project/build --out /project/build-clang \
  --cuda-path /usr/local/cuda-13.2 --clang /path/to/llvm/bin/clang++
./cuda-lifetime analyze --compdb /project/build-clang --model models \
  --sarif out.sarif --coverage coverage.json
```

The GitHub Actions workflow (`.github/workflows/tests.yml`) runs the Python
unit tests on Python 3.12 and 3.13. They use fake native executables, so CI
does not cover the checker itself; the checker, its lit fixtures and the seed
corpus need LLVM 23.1.3 and run under CTest.

`analyze` exits with 2 when coverage is incomplete. `scripts/analyze_parallel.py`
runs the same analysis with several processes; `scripts/prepare_targets.py`
downloads and builds the three projects above; `docker/Dockerfile` builds the
whole toolchain from source. [docs/install.md](docs/install.md) has the details,
[docs/flag-map.md](docs/flag-map.md) the nvcc-to-Clang translation,
[docs/compatibility-profile.md](docs/compatibility-profile.md) the optional
header profile for CCCL-heavy projects (`translate --compatibility-profile`,
used for the cuda-samples and faiss results), and
[docs/evaluation.md](docs/evaluation.md#reproduce) the steps behind the
results above.

## Modeling limits

- One host thread, one device and one translation unit at a time; no driver
  API (`cu*`), no device-code analysis, no CUDA Graph contents. Other
  translation units are seen only through generated summaries of the helper
  TUs named for a run (copies and synchronizations only, not frees or
  event records); the whole-project scans below use none.
- CSA explores bounded paths: loops and large functions exhaust budgets
  (reported per function), and calls it does not inline lose their effects.
  Accesses on a branch that can only throw are not checked, and the deleter
  a shared_ptr's last reference runs is not followed. A stream looked up
  again through a virtual call (RAFT's `resource::get_cuda_stream`) gets a
  new value each time, so `raft::resource::sync_stream(res)` is not known to
  cover a copy queued on an earlier lookup.
- Accesses through global variables inside opaque callees are not reported;
  buffer ranges with symbolic offsets count as overlapping.
- Device-to-device races between streams are reported only for stream-ordered
  allocations and event captures, not for ordinary device buffers.
- A `cudaMemcpyDefault` copy takes its direction from the memory kinds the
  analyzer knows for its operands. An operand of unknown kind, such as a
  pointer parameter of the analyzed function, is left out and recorded as a
  coverage gap, so code that passes UVA pointers around is mostly unchecked.
- RMM copies only through `rmm::detail::memcpy_async`, which calls
  `cudaMemcpyAsync` with `cudaMemcpyDefault` on its parameters, or
  `cudaMemcpyBatchAsync`. `models/rmm.json` now passes it as a
  `cudaMemcpyDefault` copy, but inside RMM its operands are mostly pointers
  of unknown memory kind (device buffers, caller memory), so most RMM copies
  are still coverage gaps, and RMM's 0 alerts do not mean verified code.
- Host containers are recognized by name, their memory is taken as pageable
  (so these reports are warnings even for pinned vectors), and only
  destruction or a move ends a buffer, not `clear` or `resize`.
- The labelled corpora were written from the same semantics document as the
  rules; they test conformance to it, not recall on code written by others.

## Layout

| Path | What it is |
|---|---|
| `checker/` | the CSA checker (`AsyncLifetime.cpp`) and JSON model loader |
| `tool/` | native driver with the SARIF consumer, and the Python CLI and runner |
| `translate/` | nvcc to Clang compilation-database translation |
| `eval/` | label-based evaluation, independent of the checker code |
| `corpus/` | seed and holdout corpora with labels; CUDA declarations for them |
| `tests/` | native fixtures, driver and translator tests, protocol checks |
| `models/` | wrapper models for faiss and RMM |
| `probes/` | GPU probes of runtime behavior |
| `scripts/` | target preparation, parallel and whole-project analysis |
