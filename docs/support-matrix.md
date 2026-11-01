# Support and coverage boundaries

The checker is out of tree, uses public Clang interfaces, and analyzes host
code in one translation unit at a time. The S1–S10 policy is in
[semantics.md](semantics.md); measured results are in [evaluation.md](evaluation.md).

| Area | Implemented behavior | Boundary |
| --- | --- | --- |
| Toolchain | LLVM/Clang 23.1.3, C++17 checker, CUDA 13.2.2, Python 3.12+ | Clang C++ ABI is version specific. The development override does not certify another LLVM release. |
| Execution scope | One host thread and one device; CSA explores symbolic paths and inline callees | No inter-TU, host concurrency, multi-device, driver API, CUDA Graph contents, cooperative launch, or device-body analysis. Stream capture is recognized: captured work is not treated as running. |
| CL1 host access | Pending copies retain direction, memory kind and byte range. Pinned/registered host accesses can be errors; unknown/pageable behavior is contract-level. Calls CSA does not inline are checked against the pending buffers their arguments reach; pointer-to-const parameters only read. Synchronous `cudaMemcpy` and `cudaMemset` complete or queue work as documented. `cudaMemcpyDefault` copies take their direction from the operands' known memory kinds. | Symbolic offsets/lengths and opaque aliases cannot establish exact overlap; these do not justify a definite error. A `cudaMemcpyDefault` operand of unknown memory kind is a coverage gap. |
| CL2 lifetime | Stack lifetime end, direct receiver destruction, `cudaFreeHost`, and unregister | Container `data()` to owner lifetime relations are not generally modeled. `cudaFreeHost` behavior remains a contract warning without the required runtime evidence. |
| CL3 device allocation | Stream-ordered allocation, kernel/copy use, async free, completion, cross-stream dependencies | Kernel operand recovery is limited to recoverable pointer regions and stream expressions. Unrecoverable expressions are reported as unmodeled. |
| CL4 event capture | Empty captures of events created on the path, stale captures, re-recording, waits and transitive dependency snapshots; events of unknown history are a coverage gap, not a report | A stream wait orders later device work; it does not itself complete work for the host. |
| Streams | Null/default mode, reserved legacy/per-thread handles, created blocking/nonblocking streams, constraint-equal aliases | Unknown flags or possibly equal handles do not establish independent streams or guaranteed implicit ordering. |
| CUDA returns | Explicitly checked success/failure paths; query success establishes completion | Discarded return values are assumed successful. This assumption is written into coverage and SARIF metadata. |
| Ordinary free | Pending stream-ordered use before `cudaFree` can be an error; ordinary allocation/free implicit completion is informational | Runtime version and allocation provenance affect which contract applies. |
| Wrappers | Strict JSON effect schema, argument positions, source rationale and assumption flag | Every external model is a user assumption. An available body is analyzed instead of adding a second summary. Opaque unsupported wrappers remain outside the claim. |
| AST pre-scan | Traverses source definitions, direct callees, constructors/destructors, macro expansion and explicit template roots | Indirect/virtual calls retain CSA. Global/static destruction and header-defined explicit template roots with CUDA effects carry `coverageUnknown` when CSA exploration is not guaranteed. |
| Budget | Per-function exploded-node budget plus CSA remaining-work reporting | A parsed TU may still be incomplete due to node/loop/aborted-block limits. |
| Diagnostics | Actual CSA `PathDiagnostic` events serialized through a public custom consumer and LLVM `SarifDocumentWriter`; the consumer writes each report's rule (CL1–CL4), level, memory kind and semantic ID as SARIF fields | LLVM 23's stock exporter asserts for an externally registered checker; this project does not patch LLVM. Warnings are shown by default; `--errors-only` hides them and records how many were hidden. |
| Failure reporting | Parse failures, translation unknown, unmodeled calls, budget exhaustion, malformed native output and changed native binary invalidate complete coverage | A zero diagnostic count is never a negative result when relevant coverage is incomplete. |

`modeledCallVisits`, `unmodeledCudaCalls`, `pointerEscapes` and the separate
`pendingMemberCallUses`, `pendingArgumentUses`, `pendingPointerEscapes` count CSA visits,
not distinct source locations. A single source call may be visited on multiple
paths. `tus.parsed` is a compiler-success count, not a count of completely
explored programs. Headers and external calls can contain behavior beyond the
supported scope even when the TU itself has no modeled CUDA operations.

The initial faiss model covers `faiss::gpu::synchronizeAllDevices` under the
single-device assumption. The RMM model describes
`rmm::detail::memcpy_async` as a `cudaMemcpyDefault` copy. A stream passed as
a wrapper whose only member is the handle (`rmm::cuda_stream_view`,
`cuda::stream_ref`) is decoded to that handle; any other stream object remains
unmodeled. These files are not comprehensive models of either library;
generated summaries ([models.md](models.md#generated-summaries)) cover
helpers whose bodies are available.

The Ubuntu 24.04 container is the reference platform. The local Ubuntu 26.04.1
host is outside the archived CUDA 13.2.2 Linux guide's support list; host results
remain separately identified. Runtime probes require an actual GPU and driver.
Compilation or successful static regression cannot substitute for those probes.

The optional [header compatibility profile](compatibility-profile.md) maps CCCL-heavy projects onto Clang-compatible headers.
