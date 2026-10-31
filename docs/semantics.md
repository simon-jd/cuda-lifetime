# CUDA 13.2.2 semantic contract

This table defines the diagnostic policy, not a proof that an asynchronous operation
is still running at a particular host instruction. A report identifies a path with
no established ordering or completion guarantee. CSA explores bounded paths and
does not prove absence of defects. Runtime experiments cannot strengthen the
documented guarantee.

| ID | Rule / modeled fact | Evidence and severity |
|---|---|---|
| S1 | CL1: pinned D2H destination read / H2D source write before completion | [API synchronization, asynchronous transfers](https://docs.nvidia.com/cuda/archive/13.2.2/cuda-runtime-api/api-sync-behavior.html#api-sync-behavior__memcpy-async). Pinned evidence must come from an allocation or registration on the current path. `error` for a missing completion guarantee on that path. |
| S2 | CL1: pageable or unknown memory / opaque call receiving an outstanding buffer | Same API section explicitly permits synchronous behavior for pageable transfers. `warning`, never `error`. An opaque call is one CSA evaluated without inlining; it is reported when its arguments reach a pending buffer, directly or through stored pointers. A pointer-to-const parameter only reads, which conflicts only with a pending D2H. |
| S3 | Retirement only through covering completion | [Stream synchronize](https://docs.nvidia.com/cuda/archive/13.2.2/cuda-runtime-api/group__CUDART__STREAM.html), [event synchronize](https://docs.nvidia.com/cuda/archive/13.2.2/cuda-runtime-api/group__CUDART__EVENT.html), [device synchronize](https://docs.nvidia.com/cuda/archive/13.2.2/cuda-runtime-api/group__CUDART__DEVICE.html). A stream wait orders device work; it alone never proves host completion. |
| S4 | CL2: outstanding host storage released or leaving scope | [Pinned allocation, registration and release](https://docs.nvidia.com/cuda/archive/13.2.2/cuda-runtime-api/group__CUDART__MEMORY.html). `cudaFreeHost` synchronization is unspecified: always contract `warning`. Known pinned automatic storage leaving its scope can be `error`; opaque destructors remain contractual. |
| S5 | CL3: stream-ordered pointer use before allocation is ordered | [Memory pools, detailed description and cudaMallocAsync](https://docs.nvidia.com/cuda/archive/13.2.2/cuda-runtime-api/group__CUDART__MEMORY__POOLS.html). Allocation must precede access in stream execution order; known stream-ordered memory supports `error`. Unknown stream identity cannot establish a definite missing dependency. |
| S6 | CL3: free on another stream does not cover pending use | [cudaFreeAsync](https://docs.nvidia.com/cuda/archive/13.2.2/cuda-runtime-api/group__CUDART__MEMORY__POOLS.html). All uses must precede free in execution order. `error` only with established stream-ordered memory and known ordering evidence. |
| S7 | CL3: synchronous cudaFree of an asynchronous allocation | [cudaFree](https://docs.nvidia.com/cuda/archive/13.2.2/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1ga042655cbbf3408f01061652a075e094). This call does not implicitly synchronize stream-ordered allocations. Outstanding use supports `error`. |
| S8 | CL4: empty or stale event capture | [cudaEventRecord](https://docs.nvidia.com/cuda/archive/13.2.2/cuda-runtime-api/group__CUDART__EVENT.html#group__CUDART__EVENT_1gf4fcb74343aa689f4159791967868446). Record replaces the capture; a wait takes a snapshot at the time it is issued. Before the first record, capture is empty. These waits are legal API calls, so a standalone suspicious wait is contract `warning`; it also explains a CL1/CL3 path. Only an event created on the path is known to be unrecorded; an event read from an object, a container or another process has unknown history and is not reported. Later producer work is reported only if the waiter is not already ordered after it. |
| S9 | Legacy default stream orders known blocking streams | [Stream synchronization behavior](https://docs.nvidia.com/cuda/archive/13.2.2/cuda-runtime-api/stream-sync-behavior.html). Nonblocking streams are excluded. Per-thread default streams are not legacy default streams. Unknown handles must not be assumed distinct or blocking. |
| S10 | Ordinary cudaFree / stream and event destruction | [Memory API](https://docs.nvidia.com/cuda/archive/13.2.2/cuda-runtime-api/group__CUDART__MEMORY.html), [stream API](https://docs.nvidia.com/cuda/archive/13.2.2/cuda-runtime-api/group__CUDART__STREAM.html), [event API](https://docs.nvidia.com/cuda/archive/13.2.2/cuda-runtime-api/group__CUDART__EVENT.html). Ordinary cudaFree permits implicit synchronization; it is not a definite lifetime violation. Destroy does not imply that queued work vanished. Never generate an error merely because these calls precede an explicit sync. |

Two further modeled facts:

- **Synchronous copies and memset** ([api-sync-behavior](https://docs.nvidia.com/cuda/archive/13.2.2/cuda-runtime-api/api-sync-behavior.html)).
  `cudaMemcpy`/`cudaMemcpy2D` run on the default stream; with host memory on
  either side they return after the copy and earlier default-stream work (and
  the work of streams the legacy default stream waits for) is complete, or, for
  a pageable H2D source, once it is staged. Device-to-device copies do not
  synchronize. `cudaMemset` is asynchronous except on pinned host memory.
- **Stream capture** ([stream capture](https://docs.nvidia.com/cuda/archive/13.2.2/cuda-c-programming-guide/index.html#creating-a-graph-using-stream-capture)).
  Between `cudaStreamBeginCapture` and `cudaStreamEndCapture`, work queued on
  the capturing stream, or on a stream that joined the capture by waiting on an
  event recorded in it, is recorded into a graph and does not run. Graph
  contents and `cudaGraphLaunch` effects are not modeled and are reported as a
  coverage gap.

All APIs are modeled in a single host thread, on one device, within a TU. Success
and failure return paths are kept separate. Wrapper summaries are user
assumptions and are listed in the coverage report. Warnings are reported by
default; `--errors-only` hides them and records how many were hidden.
