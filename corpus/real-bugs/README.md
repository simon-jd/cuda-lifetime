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
and the fix silences it when the fixed revision has none there. The runner
records that as `fixSilencesOnSiteLines`, with the site lines moved to the
fixed revision by its diff (`fixedSiteLines`), next to a coarser
`fixSilences`: no diagnostic with a matched report's rule and message
anywhere in the labelled file.

## Results

`scripts/run_real_bugs.py` produced `results/real-bugs/results.json` with
version 0.5.0. The compile flags are in `commands.json`, together with what
the runner adds per case: a patched copy of cuDF's `type_dispatcher.hpp`
searched first (Clang rejects its specializations in CUDA mode), a stand-in
for gpe's generated `gpe_kernels.h`, and the helper translation units whose
generated summaries are loaded (RMM's `cuda_memcpy.cpp`,
`cuda_stream_view.cpp` and `device_buffer.cpp`, cuDF's `cuda_memcpy.cu`).
The labels are those written before the first run. Three bugs are found,
and their fixes silence the reports (`fixSilencesOnSiteLines` and
`fixSilences` are both true for all three).

This is a development set: version 0.4.0 ran on these cases first, and the
summaries, container ownership, crash fix and header patches of 0.5.0 were
written to analyze them. The three finds show those changes working on the
cases they were made for; they are not a held-out result.

| Case | Rule | Result | Why |
|---|---|---|---|
| `cudf-json-scoped-host-value` | CL2 | found, fix silences | `set_element_async`, a `device_uvector` method CSA does not inline, is summarized from its body as a copy from its `value` argument (through `rmm::detail::memcpy_async`, summarized from RMM's `cuda_memcpy.cpp`); the block-scoped `single_node` is reported when the function returns, with the call on line 225 on the path |
| `cudf-device-view-staging` | CL2 | found, fix silences | with the dispatcher patch the TU parses; `cuda_memcpy_async_impl`'s summary queues the copy from the `host_vector` staging buffer, whose destruction at the return is reported (line 89); the fix's `cuda_memcpy` waits through `cuda_stream_view::synchronize`, summarized from RMM |
| `cudf-string-scalar-source` | CL2 | missed | the copy source is the caller's string: nothing in this TU ends it, and `device_buffer`'s copy takes its stream from a member, so no summary of RMM's `device_buffer.cpp` covers it (107 of 226 functions exhausted the node budget). The buggy revision has four other CL2 warnings, off the labelled line: `set_value` and two constructors return while a copy from a by-value parameter is queued, and a temporary source ends with its full expression (reported inside CCCL's `any_resource` destructor that ends it); all four are portability findings, sources in pageable memory. The fixed revision reports none of them, every time: three reruns gave identical results (`results/real-bugs/determinism.json`), so the silence is not nondeterminism. The lines they are on are unchanged, but the fix changes more of `scalar.cpp` than the labelled lines (two `fixed_point_scalar`/`fixed_width_scalar` constructors now copy their `device_scalar` through `data.value(stream)`, which synchronizes) and two lines of `scalar.hpp`; which of these changes silences the reports was not established (107 of 226 functions exhaust the budget in both) |
| `cudf-parquet-string-offsets` | CL2 | missed | analyzable since the crash fix. The pattern (a pinned `host_vector` converted to a `host_span` for `make_device_uvector_async`, destroyed at the end of its block) is reported in a 10-line function, but `decode_page_data` still has work left at 2,000,000 nodes and no explored path reaches the block's end with the copy pending (`results/real-bugs/parquet-budget.json`) |
| `cudf-regex-program` | CL2 | found, fix silences | the `host_vector` holding the program converts to a `host_span` for `cuda_memcpy_async`, so the copy's source belongs to it; its destruction at the return is reported (line 117) |
| `cudf-arrow-stream-chunk` | CL2 | not analyzable | the dispatcher patch fixes the CUDA-mode errors; the TU then needs `nanoarrow/nanoarrow.h`, a dependency cuDF's build downloads, which was not fetched |
| `gpe-staging-slot-reuse` | CL1 | missed | analyzable with the `gpe_kernels.h` stand-in; the copies are driver API calls (`cuMemcpyHtoDAsync`), which the checker does not model, and the slot is reused across calls to `dispatch` |
| `rmm-merge-lists-event` | CL4 | missed | the defect is a missing re-record: after the wait, the event keyed to the merged blocks no longer covers the donor stream's work, which matters only to waits in later calls. A per-path checker sees nothing wrong in `merge_lists`, and the event's history is in other member functions |
| `llama-set-tensor-visibility` | outside | missed | device-side ordering between streams |
| `llama-copy-stream` | outside | missed | device-side ordering between streams |
| `samples-system-atomics` | outside | missed | managed memory (`cudaMallocManaged` is unmodeled) and all 4 functions exhausted the node budget |
| `csc-pingpong-mpi-send` | outside | not analyzable | `mpi.h` is not installed |
| `cail-staged-copy` | outside | not analyzable | `mpi.h` is not installed |

Of the eight bugs within the rules, three are found, four are analyzed and
missed, and one cannot be analyzed. Version 0.4.0 found none: five could not
be analyzed (one because the tool crashed) and three were missed.

## Version 0.6.0

`results/real-bugs/results-0.6.0.json` reruns the 13 cases with 0.6.0: the
same three are found and silenced by their fixes, and every case has the
same buggy-revision diagnostics in its labelled file, and the same number of
fixed-revision diagnostics, as with 0.5.0. One fewer function of
`llama-set-tensor-visibility` exhausts its budget (55 of the buggy
revision's, against 56).
