# Fixed bugs held out from 0.5.0 (development set v2)

`cases.json` lists 10 stream-ordering and lifetime bugs that public CUDA
projects fixed in host code, none of which was used while version 0.5.0 was
developed. The 13 cases in [`../real-bugs/`](../real-bugs/README.md) drove
0.5.0's changes and are a development set; these were the held-out set for
0.5.0. Version 0.6.0 was developed on them, so for 0.6.0 they are
development set v2, and [`../real-bugs-heldout-0.6.0/`](../real-bugs-heldout-0.6.0/README.md)
is its held-out set. The 0.5.0 results below are the held-out score; the
0.6.0 results at the end are a development result.

**How they were found.** After 0.5.0 was released, the histories of NCCL,
cuVS, cuML, RAFT, cuGraph, Warp and other CUDA projects were searched for
commits whose message or diff adds a stream, event or device
synchronization, or moves a free, read or destruction after one, next to an
asynchronous copy, a `set_value_async`/`device_scalar` initialization or a
stream-ordered free. Each match's diff was read. Races inside kernels, new
code rather than fixes, unmerged branches, device-side ordering between
streams and fixes of the 13 development cases were dropped. Of the
in-scope fixes found, three were left out before any labelling as unlikely
to compile here (not tried): cuGraph `fe2883ff` (2020) and cuML `51d73d8e`
(2019) predate the CUDA 13.2 and CCCL headers available by years, and cuML
`24bf99b2` needs the external cumlprims package. Two real host-side bugs
outside the rules (AMReX `ca99c228`, a buffer freed while a kernel reads it;
DALI `195bc26`, nvJPEG rewriting a pinned buffer) were not included. The ten
kept are all of the kinds the rules describe: five CL1 (a host read or write
of a pending copy's buffer), four CL2 (a host source destroyed with its copy
pending) and one CL3 (a stream-ordered free on a stream that does not cover
a kernel's use). Eight go through RAPIDS wrappers (`raft::copy`,
`rmm::device_scalar`, `set_value_async`), one calls `cudaMemcpyAsync`
directly (NCCL) and one frees through Warp's allocator.

**Protocol.** Each label (rule, file, lines, a one-sentence description)
was written from the fix commit's message and diff, and this file was
committed before the checker ran on either revision of any case. The
compile commands are added afterwards in `commands.json`, from what each
translation unit needs to parse (checked with the compiler alone), not from
checker output. The matching rule is the development set's: a case is found
when the buggy revision has a CL1–CL4 diagnostic on one of its `siteLines`
in the labelled file, and the fix silences it when the fixed revision has
none there (`fixSilencesOnSiteLines`, on the site lines as the fix moved
them). Labels are not changed after the first run.

## Results of version 0.5.0

`scripts/run_real_bugs.py --cases corpus/real-bugs-heldout/cases.json
--commands corpus/real-bugs-heldout/commands.json` produced
`results/real-bugs-heldout/results.json` with the 0.5.0 build used for the
development set, after the labels and the commands were committed. Nine of
the ten translation units parse. The checker reports **4 of the 10** on a
labelled line of the buggy revision; in 2 of those 4 the fix removes the
report (`fixSilencesOnSiteLines`), and in the other 2 the fixed revision
gets the same report, a false positive.

| Case | Rule | Result | Why |
|---|---|---|---|
| `nccl-ep-mask-clean` | CL2 | found; the fix removes it | the vector's destruction at the return (line 3154) is reported with the copy on line 3150 on the path. Both revisions also have a CL2 at line 1252, off the labelled lines (not triaged), which is why the runner's coarse `fixSilences` (no report with the same rule and message anywhere in the file) is false while `fixSilencesOnSiteLines` is true |
| `cuml-cd-alpha-temporary` | CL2 | found, fix silences | a CL2 at `cdFit`'s return (line 306) has the copy from the temporary behind `cublas_alpha(1.0, stream)` (line 216) on its path: the checker ends the temporary's storage with the function, not with the full expression. The fix removes the copy. Two CL1 reports elsewhere in `cdFit` (line 162 and in RAFT's `operators.hpp`) are in both revisions (not triaged) |
| `raft-device-reference-read` | CL1 | found; fix not recognized | CL1 and CL2 on `return v` after `update_host`. The fix synchronizes with `raft::interruptible::synchronize`, which calls `cudaStreamQuery` through a function pointer (`synchronize_impl(cudaStreamQuery, stream, ...)`); the checker does not resolve that call, so it reports the fixed revision the same way |
| `cugraph-bfs-source-temporary` | CL2 | found; fix not recognized | the device scalar's copy from the converted temporary on line 141 is on the path of seven CL2 reports at the test's returns (lines 213 to 259), where the checker ends the temporary's storage; the fix's `handle.sync_stream()` also ends in `raft::interruptible::synchronize`, so the fixed revision keeps them |
| `cuvs-vamana-serialize-dataset` | CL1 | missed | the one top-level function analyzed (`serialize`) exhausts the node budget; `raft::copy` copies with `cudaMemcpyDefault`, and an operand of unknown memory kind (the host matrix's heap storage, the device dataset) leaves its direction unmodeled |
| `cuvs-all-neighbors-host-dataset` | CL1 | missed | the test's constructor, body and destructor exhaust the node budget; the same `cudaMemcpyDefault` gap |
| `cuml-tsne-min-max-source` | CL1 | missed | `min_max` is reached only from `TSNE_fit` and `TSNE_fit_sparse`, both of which exhaust the node budget |
| `raft-mst-edge-count` | CL2 | missed | the test functions that call `solve` exhaust the node budget; one CL2 at line 326, on a `raft::launch_kernel` call off the labelled lines (not triaged) |
| `warp-memtile-scratch-free` | CL3 | missed | `wp_memtile_device` exhausts the node budget (28 of 188 functions do) |
| `cuvs-binary-quantizer-threshold` | CL1 | not analyzable | the revision builds against CCCL 3.1, whose `cuda/std/string_view` declares host-only deduction guides that Clang 23 rejects in CUDA mode; RMM 25.12 does not compile against the newer CCCL of the RMM 26.10 build |

On the development set the same version finds 3 of 13 (3 of 8 within the
rules) with every fix silencing its report. On these ten, all within the
rules, it finds 4 and recognizes 2 of the 4 fixes. The misses are mostly
budget exhaustion in large RAPIDS and Warp translation units and copies
through `cudaMemcpyDefault` with operands whose memory kind is unknown; the
two unrecognized fixes are one synchronization idiom, RAFT's interruptible
wait.

## Development set v2: version 0.6.0

These ten cases drove 0.6.0's changes: the node profile of the budget
misses (`--node-profile`), the waits through RAFT's interruptible
synchronize, the headers that make the CCCL 3.1 case parse, RMM 26's
out-of-line stream accessors, and the budget changes in
[design-tradeoffs.md](../../docs/design-tradeoffs.md#spending-the-node-budget-where-cuda-state-can-change).
The labels are unchanged; `commands.json` gained a header patch and an
OpenMP stand-in for `cuvs-binary-quantizer-threshold` after the 0.5.0 run.
`results/real-bugs-heldout/results-0.6.0.json` is the 0.6.0 run, and
`nodes-before.json` and `nodes-0.6.0.json` are node-profile runs (node
reclamation off, so every exploded-graph node created is counted) of the
checker before the budget changes (commit `9bfcdb4`: 0.5.0's analysis with
the node accounting and the function-pointer waits, and the CCCL commands)
and of 0.6.0. The "before" seconds are the 0.5.0 run's
(`results.json`); both columns add the buggy and the fixed revision.
"Fix silences" is `fixSilencesOnSiteLines` in the two result files.

| Case | 0.5.0 | 0.6.0 | Nodes (millions), before → 0.6.0 | Seconds, 0.5.0 → 0.6.0 | Functions over budget / analyzed (buggy), 0.5.0 → 0.6.0 |
|---|---|---|---:|---:|---:|
| `nccl-ep-mask-clean` | found, fix silences | found, fix silences | 4.01 → 3.71 | 24 → 17 | 4/24 → 4/24 |
| `cuvs-vamana-serialize-dataset` | missed | missed | 0.76 → 0.76 | 19 → 20 | 1/1 → 1/1 |
| `cuvs-binary-quantizer-threshold` | not analyzable | found | 5.93 → 9.03 | – → 158 | – → 10/28 |
| `cuvs-all-neighbors-host-dataset` | missed | missed | 2.29 → 6.25 | 59 → 97 | 3/24 → 5/35 |
| `cuml-tsne-min-max-source` | missed | found | 1.55 → 4.65 | 113 → 152 | 2/2 → 6/10 |
| `cuml-cd-alpha-temporary` | found, fix silences | found, fix silences | 6.46 → 6.06 | 132 → 136 | 8/10 → 7/24 |
| `raft-mst-edge-count` | missed | missed | 2.74 → 2.35 | 97 → 90 | 3/12 → 3/19 |
| `raft-device-reference-read` | found | found, fix silences | 11.65 → 11.45 | 270 → 277 | 13/63 → 13/64 |
| `cugraph-bfs-source-temporary` | found | found | 4.56 → 4.54 | 166 → 172 | 6/29 → 6/32 |
| `warp-memtile-scratch-free` | missed | missed | 22.23 → 22.20 | 86 → 100 | 28/188 → 28/188 |

Version 0.6.0 finds **6 of the 10** (0.5.0: 4). In 3 of the 6 the fixed
revision has no report on the labelled lines (`fixSilencesOnSiteLines`):
`cuml-cd-alpha-temporary` and `raft-device-reference-read` (also by the
runner's coarse `fixSilences`) and `nccl-ep-mask-clean`, whose fixed
revision keeps a CL2 off the labelled lines (line 1252, not triaged). The
other three fixes are not recognized:

- `cuvs-binary-quantizer-threshold`: the fixed revision reports the same
  read (line 453; see "Parsed" below).
- `cuml-tsne-min-max-source`: the labelled CL1, the host write to the copy's
  source, is gone, but a CL2 at `min_max`'s return (line 152) keeps the
  copy from `max_init` (line 139) on its path.
- `cugraph-bfs-source-temporary`: seven CL2 reports at the test's returns
  keep the device scalar's copy, now from `source` on line 142, on their
  path; the fix's `handle.sync_stream()` does not retire it.

- **Waits recognized.** `raft-device-reference-read`'s fix waits with
  `raft::interruptible::synchronize`, which polls `cudaStreamQuery` through a
  function pointer; the summary of the polling helper resolves it. That
  alone was not enough: the copy is queued on `stream.value()`, which RMM 26
  defines in `cuda_stream_view.cpp`, so each call had returned a fresh value;
  the accessor now returns the view's handle.
- **Parsed.** `cuvs-binary-quantizer-threshold` parses with a patched copy of
  CCCL 3.1's `cuda/std/string_view` (its deduction guides are marked
  `__host__`, which nvcc accepts and Clang 23 rejects; a deduction guide has
  no body, so the patch drops the annotation) and a stand-in `omp.h`
  declaring the four OpenMP routines RAFT's headers call. The read on line
  447 is found, but the fixed revision reports the same read (line 453): the
  copy is queued on `raft::resource::get_cuda_stream(res)` and the fix waits
  with `raft::resource::sync_stream(res)`, which looks the stream up again
  through a virtual call on RAFT's resource registry, so the two lookups
  give different stream values.
- **Found through a fallback entry point.** `cuml-tsne-min-max-source`'s
  `min_max` is never inlined from `TSNE_fit`, whose budget runs out; it
  launches a kernel itself, so it was analyzed as an entry point of its own and reports the host write on line 136.
- **Still missed.** `cuvs-vamana-serialize-dataset`: `serialize` exhausts its
  budget in nested loops over the graph and in mdspan and resource code;
  CSA inlined `serialize_dataset` on some path, so it is not a fallback entry
  point, and its `raft::copy` passes `cudaMemcpyDefault` with operands of
  unknown memory kind. `cuvs-all-neighbors-host-dataset` and
  `raft-mst-edge-count`: the test bodies exhaust their budgets, and the
  functions with the bug call RAFT's copy rather than the CUDA API, so they
  are not fallback entry points. `warp-memtile-scratch-free`: Warp's
  functions call the driver API through function pointers with checked
  failure paths that print and continue, and the 28 functions over budget
  are unchanged.

The node counts show where the changes act. In the budget-exhausting
functions of the RAPIDS misses, the frames of RAFT's error branches
(exception construction, stack-trace collection, message formatting) were
among the twelve largest of each profile before, with 7,500 to 23,800 of
about 380,000 nodes, and are absent from them in 0.6.0; the shared_ptr
frames among the twelve largest of cuVS's `serialize` held 69,600 nodes
before and 28,800 in 0.6.0 (`profile.byFrame` in the two files). Those functions still exhaust the budget, so
their node counts do not fall; a case's total rises where fallback entry
points add functions (`cuml-tsne-min-max-source`, `cuvs-all-neighbors-host-dataset`,
`cuvs-binary-quantizer-threshold`). Over the ten cases 0.6.0 takes 1,219 s
against 966 s for 0.5.0; 158 s of the difference is the TU that now parses,
and the rest is mostly fallback entry points (the build without them took
987 s).
