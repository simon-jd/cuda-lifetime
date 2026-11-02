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
