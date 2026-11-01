# Fixed bugs from public CUDA projects

`cases.json` lists 13 host-side stream-ordering and lifetime bugs that public
CUDA projects fixed, each with the fix commit, its parent (the buggy
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
