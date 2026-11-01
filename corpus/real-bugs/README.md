# Fixed bugs from public CUDA projects

`cases.json` lists 13 stream-ordering and lifetime bugs that public CUDA
projects fixed in host code, each with the fix commit, its parent (the buggy
revision), the affected file and the lines involved. They were found by
searching commit messages and diffs for fixes that add a stream, event or
device synchronization, or move a free or read after one, in the histories
of cuDF, RMM, llama.cpp, faiss, cuda-samples, nccl-tests, CUDALibrarySamples,
darknet, Caffe, tiny-cuda-nn and TensorRT, and with GitHub's commit search
across all public repositories. Most matches were races inside kernels, new
code rather than fixes, or driver workarounds; those are not listed.

Each label was written from the fix's message and diff before the checker ran
on either revision:

- `inRuleScope` says whether the bug is one the rules describe (a host access
  or host lifetime end racing an asynchronous copy, a stream-ordered
  allocation, or an event wait that does not cover a producer), with the rule.
  Eight cases are; five are real stream-ordering bugs of kinds the checker does
  not claim to find (device-side ordering between streams, managed memory,
  buffers handed to MPI or a NIC), kept to show what falls outside it.
- `siteLines` are lines of the parent revision: the queueing call and the
  access or lifetime end it races with.

A case counts as found when the parent revision has a CL1–CL4 diagnostic on
one of those lines (primary location or any code-flow location in that file),
and the fix silences it when the fixed revision has none there.

## Results

`scripts/run_real_bugs.py` produced `results/real-bugs/results.json` with
version 0.4.0 (the compile flags are in `commands.json`). No case was found.

| Case | Rule | Result | Why |
|---|---|---|---|
| `cudf-json-scoped-host-value` | CL2 | not analyzable | Clang rejects cuDF's `type_dispatcher.hpp` specializations in CUDA mode (20 errors) |
| `cudf-device-view-staging` | CL2 | not analyzable | same |
| `cudf-arrow-stream-chunk` | CL2 | not analyzable | same |
| `cudf-parquet-string-offsets` | CL2 | not analyzable | the native tool aborts on a Clang assertion (`EvaluateAsInt` on a value-dependent expression); `clang++ --analyze` on the same command does not |
| `gpe-staging-slot-reuse` | CL1 | not analyzable | needs `gpe_kernels.h`, which the project's build generates |
| `cudf-string-scalar-source` | CL2 | missed | the copy is queued inside `rmm::device_buffer`'s constructor, defined in RMM's own library, and the source string belongs to the caller; 107 of 226 functions exhausted the node budget |
| `cudf-regex-program` | CL2 | missed | the copy goes through `cudf::detail::cuda_memcpy_async_impl`, defined in another TU, so no pending copy exists; the host vector's buffer is not related to the vector object; `reprog_device::create` itself exhausted the node budget |
| `rmm-merge-lists-event` | CL4 | missed | the event lives in a member structure, so its history is unknown to the checker (a coverage gap, not a report) |
| `llama-set-tensor-visibility` | outside | missed | device-side ordering between streams |
| `llama-copy-stream` | outside | missed | device-side ordering between streams |
| `samples-system-atomics` | outside | missed | managed memory (`cudaMallocManaged` is unmodeled) and all 4 functions exhausted the node budget |
| `csc-pingpong-mpi-send` | outside | not analyzable | `mpi.h` is not installed |
| `cail-staged-copy` | outside | not analyzable | `mpi.h` is not installed |

Of the eight bugs within the rules, five could not be analyzed and three were
analyzed and missed. Every in-scope bug here goes through a library
abstraction (RMM and cuDF copy helpers, containers, events held in objects,
Arrow chunks, a staging ring kept across calls) that the checker models only
where its body is in the same translation unit, or not at all.
