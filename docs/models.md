# Wrapper model format

`--model DIR` loads every `.json` file in that directory. Keep reports and other
JSON artifacts elsewhere: an invalid model file rejects the invocation instead
of partially trusting its effects. A file contains a versioned object with a
`models` array (a bare array is also accepted by the current loader).

```json
{
  "schemaVersion": 1,
  "models": [{
    "function": "project::wait_for_stream",
    "aliases": [],
    "effect": "sync_stream",
    "args": {"stream": 0},
    "assumption": true,
    "rationale": "The wrapper calls cudaStreamSynchronize on its first argument.",
    "doc": {
      "url": "https://example.org/project/source",
      "anchor": "wrapper-definition",
      "cudaVersion": "13.2.2"
    }
  }]
}
```

This is a format example; its URL is a placeholder, not evidence for a real
wrapper. Use the actual checked implementation and document the scope in which
its summary is valid. Function and alias names must be nonempty and unique
across the whole directory. Arguments are zero-based integer positions in the
call's explicit argument list, not including a C++ receiver. Wrong argument
counts produce an unmodeled call rather than inventing an argument value.

| Effect | Required argument keys | Optional keys used by the checker |
| --- | --- | --- |
| `enqueue_copy` | `stream`, and `dst` or `src` | `size` (bytes), `kind` (CUDA memcpy kind), `count` (entries of a batched copy); absence means the direction/extent may be unknown. A side that is not an argument (a receiver's member buffer) is left out. |
| `launch` | `stream` | Other call arguments are inspected as possible device operands. |
| `malloc_host`, `malloc_device` | `ptr` (output pointer) | None. |
| `host_register` | `ptr` | `size` (bytes); an unknown span does not certify unrelated storage. |
| `host_unregister`, `free_host`, `free_device` | `ptr` | None. |
| `malloc_async` | `ptr` (output pointer), `stream` | None. |
| `free_async` | `ptr`, `stream` | None. |
| `sync_device` | None | None. |
| `sync_stream` | `stream` | None. |
| `sync_event` | `event` | None. |
| `event_elapsed` | `start`, `end` | `ms` (output duration). Only a successful status proves completion of the two event captures. |
| `event_record` | `event`, `stream` | None. |
| `stream_wait_event` | `stream`, `event` | None. |
| `stream_create` | `ptr` (output handle) | `flags`; unknown flags cannot prove a blocking stream. |
| `event_create` | `ptr` (output handle) | `flags`; known DisableTiming events cannot produce a successful elapsed-time result. |
| `destroy` | None | Legal deferred stream/event destruction; does not establish host completion. |
| `copy` | `dst`, `src` | `size`, `kind`. A synchronous copy on the default stream: with host memory on either side it completes earlier default-stream work. |
| `memset`, `memset_async` | `ptr` (and `stream` for `memset_async`) | `size`. Synchronous only for pinned host memory. |
| `capture_begin`, `capture_end` | `stream` | `ptr` (output graph) for `capture_end`. Work queued on a capturing stream is recorded, not executed. |
| `graph_launch` | `stream` | Queues work with unknown buffers. |
| `address_only` | None | The function inspects a pointer value (for example its attributes) and never dereferences it; it is not reported as an opaque access. |

Two optional fields cover wrappers whose arguments are not the operands
themselves. `constants` fixes `kind`, `size` or `stream` to an integer, for a
wrapper that always passes `cudaMemcpyDefault` (4) or a fixed size.
`elements` lists arguments that are arrays whose first element is the
operand, as for `cudaMemcpyBatchAsync`; only that entry is modeled, and a
`count` other than one is recorded as a coverage gap.

The loader requires `aliases`, `args`, and a `doc` object with nonempty `url` and
`anchor` plus `cudaVersion: "13.2.2"`. It rejects unknown effects, missing
required arguments, non-integer/negative indices, duplicate names, malformed
JSON, invalid schema versions, and non-boolean assumption flags.

Every external summary is recorded as a **user assumption**, even when a file
sets an assumption flag to false. When a wrapper body is available, the checker
lets CSA analyze that body rather than applying a summary in addition to it.
Models do not certify that opaque C++ stream or pointer wrappers can be decoded;
unrecoverable values remain in coverage telemetry. The shipped faiss/RMM models
are intentionally narrow; see [the support matrix](support-matrix.md).

Adding a model changes analysis semantics. Include all model files in a frozen
evaluation manifest, review its source rationale, and evaluate it on independent
development cases before a held-out run. Do not add a synchronization effect
merely to suppress a warning.

## Generated summaries

`cuda-lifetime-native FILE --summarize OUT.json -- FLAGS` parses one
translation unit and writes, in this format, a summary for each externally
visible function it defines whose body queues a copy on its own parameters.
The summary is composed from the AST: CUDA calls, loaded models (pass
`--model` to build on summaries of other TUs) and the bodies of other
functions in the TU, following a parameter through casts, wrapper accessors
(`stream.value()`), its address and array elements. It is a "may" summary:
a copy on any path counts. A body that synchronizes anything, or whose copies
forward different parameters, gets none; a call that never returns (declared
`noreturn`, or every path throws) is ignored, since its effects never precede
the caller's code. Functions already modeled, and names defined more than
once in the TU (overloads), are left out. Each entry carries
`"generated": true`, the definition's file and line as its `doc`, and is
recorded as a user assumption like any other model.

Inside the analyzed TU, a body CSA declines to inline (a method of a
container class in a header, or a call over an inlining limit) is summarized
the same way at the call, and the summary is applied after CSA's
conservative evaluation; `assumptions` in the coverage report lists it as
`generated summary: NAME`.
