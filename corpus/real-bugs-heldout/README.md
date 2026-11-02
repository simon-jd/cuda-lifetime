# Held-out fixed bugs

`cases.json` lists 10 stream-ordering and lifetime bugs that public CUDA
projects fixed in host code, none of which was used while version 0.5.0 was
developed. The 13 cases in [`../real-bugs/`](../real-bugs/README.md) drove
0.5.0's changes and are a development set; these are the held-out set.

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
none there. Labels are not changed after the first run.

## Results of version 0.5.0

`scripts/run_real_bugs.py --cases corpus/real-bugs-heldout/cases.json
--commands corpus/real-bugs-heldout/commands.json` produced
`results/real-bugs-heldout/results.json` with the 0.5.0 build used for the
development set, after the labels and the commands were committed. Nine of
the ten translation units parse. The checker reports **4 of the 10** on a
labelled line of the buggy revision; in 2 of those 4 the fix removes the
report, and in the other 2 the fixed revision gets the same report, a false
positive.

| Case | Rule | Result | Why |
|---|---|---|---|
| `nccl-ep-mask-clean` | CL2 | found; the fix removes it | the vector's destruction at the return (line 3154) is reported with the copy on line 3150 on the path. Both revisions also have a CL2 at line 1252, off the labelled lines (not triaged), which is why the runner's coarse `fixSilences` (no report with the same rule and message anywhere in the file) is false |
| `cuml-cd-alpha-temporary` | CL2 | found, fix silences | a CL2 at `cdFit`'s return (line 306) has the copy from the temporary behind `cublas_alpha(1.0, stream)` (line 216) on its path: the checker ends the temporary's storage with the function, not with the full expression. The fix removes the copy. Two CL1 reports elsewhere in `cdFit` (line 162 and in RAFT's `operators.hpp`) are in both revisions (not triaged) |
| `raft-device-reference-read` | CL1 | found; fix not recognized | CL1 and CL2 on `return v` after `update_host`. The fix synchronizes with `raft::interruptible::synchronize`, which calls `cudaStreamQuery` through a function pointer (`synchronize_impl(cudaStreamQuery, stream, ...)`); the checker does not resolve that call, so it reports the fixed revision the same way |
| `cugraph-bfs-source-temporary` | CL2 | found; fix not recognized | the device scalar's copy from the converted temporary on line 141 is on the path of eight CL2 reports at the test's returns (lines 213 to 259), where the checker ends the temporary's storage; the fix's `handle.sync_stream()` also ends in `raft::interruptible::synchronize`, so the fixed revision keeps them |
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
