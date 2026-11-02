# cuda-lifetime: a Clang Static Analyzer checker for CUDA asynchronous lifetimes

**The problem.** CUDA host code queues work and returns immediately:
`cudaMemcpyAsync`, kernel launches and `cudaFreeAsync` take effect later, in
stream order. Code that reads a destination buffer, overwrites a source, or
frees memory before that work is known to be complete has a race that tests
rarely show, because the outcome depends on timing and, for pageable memory, on
driver behavior.

**The approach.** `cuda-lifetime` is an out-of-tree, path-sensitive checker
for the Clang Static Analyzer (LLVM 23.1.3). Along each path through host code
it tracks every stream's queue, the buffers pending operations use, what each
synchronization proves complete and what each event captures, and reports a
SARIF path from the queueing call to the conflicting access. A Python driver
translates an nvcc `compile_commands.json` into Clang host-only commands, runs
one analyzer process per translation unit and reports coverage gaps instead of
counting them as clean.

**What it found: 0 confirmed defects.** On 477 host translation units of
cuda-samples 13.2, faiss 1.15.1's GPU module and RMM 26.10, version 0.5.0
reports 46 alerts, all warnings or notes:

- **32 portability findings**, 29 of them in faiss: host code that reads or
  frees ordinary `malloc`/`new[]`/`std::vector` memory right after
  `cudaMemcpyAsync`, with no synchronization in between. It works only
  because current drivers make pageable copies synchronous, which CUDA
  permits but does not promise. Pin the same buffers, a common optimization,
  and the copy no longer waits: reading a pinned D2H destination early,
  [`probes/stale_read.cu`](probes/stale_read.cu) got the stale value in 20 of
  20 runs on an RTX 4090, against 0 of 20 with pageable memory.
- 3 informational notes and 11 false positives, 8 of them from one faiss
  helper whose result the analyzer cannot relate to the pointer.

**On 13 real fixed bugs it finds 3, but these 13 are a development set.**
Of 13 stream-ordering bugs that cuDF, RMM, llama.cpp and other projects
fixed, eight are of the kinds the rules describe. Version 0.4.0 found none
of them; version 0.5.0 was built to analyze these cases, and it finds three,
each silenced by its fix. The 0.5.0 features came from these 13: copy and
synchronization summaries generated from RMM's and cuDF's own translation
units, ownership of a host container's buffer, a fix for a crash in the
tool, and a patched cuDF header. So 3 of 13 is a development result, not a
measure of how the checker does on bugs it has not seen. Of the other five
in-scope cases, one does not parse (a missing dependency), one
exhausts the node budget, and three need what a per-TU, per-path checker
does not see: the caller's storage, driver API copies across calls, an
event's role across calls ([docs/evaluation.md](docs/evaluation.md#fixed-bugs-from-public-projects)).
compute-sanitizer is a dynamic, device-side tool, so it complements the
checker rather than serving as a baseline for it: on the same stale-read
program and five timing-dependent lifetime scenarios it reports the one
device-side fault, a kernel using a stream-ordered allocation before it is
ordered, and none of the host reads or frees racing an in-flight copy,
which are not device accesses
([docs/evaluation.md](docs/evaluation.md#what-compute-sanitizer-sees)).
On its hand-written labelled corpora it scores 10/0/0/10 (seed) and 25/1/0/23
(holdout, first run after its freeze) as TP/FP/FN/TN. Large parts of the scanned code
were only partly explored, and several CUDA idioms are not modeled; see
[Limitations](#limitations) before reading the zero as "no races".

```cpp
int early_read(void *device, cudaStream_t stream) {
  void *host;
  cudaMallocHost(&host, 4);
  cudaMemcpyAsync(host, device, 4, cudaMemcpyDeviceToHost, stream);
  return *static_cast<int *>(host);  // CL1 error: read before the D2H completes
}
```

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
  call. CSA's own cross-TU analysis was tried and fails on libstdc++ code
  ([docs/design-tradeoffs.md](docs/design-tradeoffs.md#helpers-in-other-translation-units-generated-summaries-not-ctu)).
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

**Real projects.** Every configured host compilation command of cuda-samples
13.2 (233; samples whose optional dependencies were missing are not
configured), faiss 1.15.1's GPU module (190) and RMM 26.10 (54), at a budget of
225,000 nodes and 180 s per TU. cuda-samples and faiss were translated with a
header profile that puts a CCCL development snapshot ahead of CUDA 13.2's own
CCCL headers ([docs/compatibility-profile.md](docs/compatibility-profile.md)); without
it most faiss CUDA sources do not parse. Each of these alerts was triaged
against the source. None of the previous version's 76 alerts was a
confirmed defect either (45 false positives, 31 with insufficient evidence);
23 of those were triaged against the source and 53 carry over the judgment of
an earlier triage of the same sites, matched by fingerprint:

| Project | TUs (parsed) | Alerts, 0.2.0 | Alerts, 0.4.0 | Alerts, 0.5.0 | Portability | Informational | False positive |
|---|---:|---:|---:|---:|---:|---:|---:|
| cuda-samples 13.2 | 233 (223) | 32 | 7 | 7 | 3 | 3 | 1 |
| faiss 1.15.1, GPU module | 190 (189) | 42 | 29 | 39 | 29 | 0 | 10 |
| RMM 26.10 | 54 (54) | 2 | 0 | 0 | 0 | 0 | 0 |
| Total | 477 | 76 | 36 | 46 | 32 | 3 | 11 |

Versions 0.3.0 and 0.4.0 reported the same 36 alerts; 0.5.0 adds 10 faiss
CL2 warnings from the host-container rule, `std::vector`s destroyed with a
copy from or into them still queued, all portability findings. With ten analyzer
processes the 0.5.0 scan takes 81 s, 146 s and 18 s of wall time (0.4.0: 97 s,
166 s, 20 s), and 385 of 837,
232 of 720 and 60 of 233 top-level functions exhaust the node budget, so
these projects are only partly explored ([docs/evaluation.md](docs/evaluation.md#real-projects);
per-TU times in `results/real-projects/timings.csv`).

The alerts that went away after 0.2.0 came from five modeling gaps, each fixed by a rule
change: blocking `cudaMemcpy` was not modeled (13), events with no history on
the path were treated as never recorded (16), calls CSA inlines or that only
query pointer attributes were reported as opaque (12), captured graph work was
treated as running (6), and waits were reported for work already ordered (4).
Four more were repeat reports of the pageable pattern below at calls CSA now
inlines. [docs/evaluation.md](docs/evaluation.md) has the per-family table.

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

## Limitations

- One host thread, one device and one translation unit at a time; no driver
  API (`cu*`), no device-code analysis, no CUDA Graph contents. Other
  translation units are seen only through generated summaries of the helper
  TUs named for a run (copies and synchronizations only, not frees or
  events); the whole-project scans below use none.
- CSA explores bounded paths: loops and large functions exhaust budgets
  (reported per function), and calls it does not inline lose their effects.
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
