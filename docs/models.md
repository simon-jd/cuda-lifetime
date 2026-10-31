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
| `enqueue_copy` | `dst`, `src`, `stream` | `size` (bytes), `kind` (CUDA memcpy kind); absence means the direction/extent may be unknown. |
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
