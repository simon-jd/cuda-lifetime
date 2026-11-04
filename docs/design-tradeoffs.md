# Design trade-offs

Most of what cuda-lifetime reports, and most of what it misses, follows from
thirteen choices: analyzing host code path by path inside CSA, judging copies by
CUDA's documented contract rather than by what today's driver does, treating
opaque calls and wrapper models with suspicion, spending CSA's node budget
where CUDA state can change, and refusing to call a translation unit clean
when its coverage is incomplete. The sections below
take them one at a time, with the options that were passed over and the
alerts or blind spots each choice causes. The numbers come from
[evaluation.md](evaluation.md): the labelled corpora and the scan of 477 host
translation units of cuda-samples 13.2, faiss 1.15.1's GPU module and RMM
26.10.

## A path-sensitive checker in the Clang Static Analyzer

**Decision.** The rules are a checker for the Clang Static Analyzer (CSA),
which explores symbolic paths through host code and carries the state of
streams, pending operations and events along each one.

**Alternatives.** A dynamic tool that observes executions (CuSan instruments
programs and uses ThreadSanitizer; compute-sanitizer's racecheck looks at
device code); AST matchers (clang-tidy); a dataflow analysis over LLVM IR.

**Why.** These races depend on timing and on driver behavior, so the
executions a test runs rarely show them, and a dynamic tool sees only the
paths a run executes. CuSan, which tracks happens-before through
ThreadSanitizer, does report a host read racing a copy whatever the timing,
but on the probe programs it reports none of the frees with work pending
or the stream-ordered allocation race
([evaluation.md](evaluation.md#cusan)).
Whether a read is safe depends on whether a synchronization covering the
copy happened on that path, which a flow-insensitive matcher cannot tell.
CSA supplies the paths, the symbolic values for buffers, streams and events,
and a report path from the queueing call to the access, which becomes the
SARIF `codeFlows`.

**Cost.** CSA explores bounded paths: loops and large functions exhaust the
node budget (reported per function), and what CSA does not inline loses its
effects. The analysis is one translation unit, one host thread and one device
at a time.

## Out of tree, against one LLVM release

**Decision.** The checker is a plugin built against the public C++
interfaces of LLVM/Clang 23.1.3, with its own driver and SARIF consumer.

**Alternatives.** Contribute it to LLVM; fork LLVM.

**Why.** It ships and changes independently of LLVM's release cycle, and users
need only LLVM's development libraries, not a patched compiler.

**Cost.** Clang's C++ interfaces have no stable ABI, so the build is pinned to
23.1.3. Two bugs in LLVM 23's SARIF export (the stock consumer asserts on an
externally registered checker; the writer sorts artifacts without updating
their indices) are worked around locally instead of being fixed upstream
([upstream.md](upstream.md)).

## Severity from the evidence on the path

**Decision.** A report is an `error` only when the path proves the facts the
contract needs: pinned memory from an allocation or registration on the path
and an exact byte overlap. Pageable or unknown memory, partial overlaps and
calls the analyzer cannot see into give a `warning`.

**Alternatives.** One severity; errors for every missing synchronization.

**Why.** The CUDA documentation lets copies to and from pageable memory be
synchronous ("might be synchronous"), and on current drivers they are: with
a 20 ms kernel queued ahead, a pageable D2H `cudaMemcpyAsync` returned after
18.4 ms with the kernel done, a pinned one after 0.007 ms. Code that reads
pageable memory right after the copy works today, so calling it an error
would be wrong; calling it nothing would hide a race that appears as soon as
the buffer is pinned.

**Cost.** 32 of the 46 real-project alerts are this portability pattern, 29
of them in faiss. Users who want only definite defects pass `--errors-only`
and lose those warnings, which the run records as hidden.

## The documented contract, not observed behavior

**Decision.** Each rule cites the CUDA 13.2.2 documentation for what it
assumes ([semantics.md](semantics.md)), and runtime probes are used to
explain reports, never to strengthen a guarantee.

**Alternatives.** Model what the driver does on the analysis machine.

**Why.** Driver behavior is not a contract: a pageable buffer can become
pinned through `cudaHostRegister`, and a driver can change how it stages
copies. A rule built on a measurement would be silently wrong on the next
system.

**Cost.** The checker reports code that works on every driver today, and the
reader has to understand why (the README and evaluation explain the probe).

## Opaque calls: reach at PreCall, report at PostCall if not inlined

**Decision.** Before a call, the checker computes which pending buffers its
arguments reach, directly or through pointers stored in memory they reach.
After the call it reports only if CSA evaluated the call without inlining it.
A pointer-to-const parameter is treated as a read, which conflicts only with
a pending D2H.

**Alternatives.** Report every call that receives a pending buffer; rely on
CSA's pointer-escape callbacks.

**Why.** CSA inlines many calls, and inside an inlined call each access is
checked on its own, so reporting the call as well duplicated reports (12 of
the previous version's 76 alerts came from calls with bodies, or from pointer
attribute queries, reported as opaque). The escape callback also fires for
globals and other regions the call's arguments never reach.

**Cost.** Accesses through global variables inside an opaque callee are not
reported, and a wrapper CSA evaluates without inlining is still a source of
false positives (1 of the 11).

## Success and failure paths for every modeled call

**Decision.** Each modeled CUDA call forks a success and a failure path. A
checked failure that ends in a `noreturn` helper (`exit`, `abort`,
`checkCudaErrors`) ends the path; a call whose status is discarded, including
one in statement position such as an `if` body, is assumed to succeed, and the
assumption is recorded in the coverage report and SARIF.

**Alternatives.** Assume every call succeeds; explore failure everywhere.

**Why.** Real code checks errors, and a failed synchronization proves
nothing, so a path that continues after an unchecked failure would report
races that need the call to fail silently. The holdout's single false
positive was exactly this: calls in `if` bodies were taken as checked, and
their failure paths reached the read.

**Cost.** More paths per function, so the node budget runs out sooner, and
correctness rests on the recorded assumption that ignored results succeed.

## Wrapper models in JSON, each recorded as an assumption

**Decision.** Project functions that wrap CUDA calls can be described in a
JSON model (effect, argument positions, rationale, a source reference),
loaded strictly: a malformed model rejects the whole run
([models.md](models.md)).

**Alternatives.** Hard-code known wrappers; source annotations; infer
summaries automatically.

**Why.** Libraries wrap the runtime (faiss's device synchronization, RMM's
`memcpy_async`), and the wrapper's body is often in another translation unit.
A model is a claim about someone else's code, so it carries its evidence and
is listed in every coverage report.

**Cost.** A model can be wrong, and a synchronization effect added to silence
a warning would hide real races; models need review and evaluation on
independent cases before a held-out run. The shipped faiss and RMM models are
deliberately narrow.

## Helpers in other translation units: generated summaries, not CTU

**Decision.** A copy helper defined in another translation unit is described
by a summary generated from that unit (`cuda-lifetime-native --summarize`):
the copy its body queues on its own parameters, composed through the calls it
makes, written in the model format and loaded with `--model`. Bodies in the
analyzed unit that CSA declines to inline get the same summary at the call.

**Alternatives.** CSA's cross-translation-unit analysis, which imports the
helper's definition on demand through an external-definition map and inlines
it; hand-written models for each helper; path-sensitive summaries computed by
running the checker itself on the helper.

**Why.** CTU was tried first, on the cuDF regex case
(`results/real-bugs/ctu-trial.json`). Clang 23's ASTImporter does not support
the `ExplicitInstantiation` node that libstdc++'s `extern template`
declarations produce, so importing `cuda_memcpy_async_impl` from its unit
reported 19 import errors (which the tool must count as a failed TU) and then
aborted on an importer assertion. Even when the import works, CTU only helps
where CSA inlines the call, which it does not do for container methods or in
functions that already exhaust their node budget, and the imported body still
passes `cudaMemcpyDefault` on pointers whose memory kind is unknown inside it.
A summary is cheap (one parse of the helper's unit, no path exploration),
survives those cases, and is a reviewable claim in the same format as a
hand-written model. Path-sensitive summaries would be more precise but need
the checker to treat parameters' unknown memory symbolically, which its state
model does not do.

A body that polls a function its caller passes (RAFT's
`interruptible::synchronize` hands `cudaStreamQuery` or `cudaEventQuery` to
`synchronize_impl`, which calls it in a loop until it stops returning
`cudaErrorNotReady`) is summarized as calling that parameter with the
arguments it forwards; the summary resolves once the caller names the
function, statically or as the argument's value at the call. CSA needs this
because the polling loop reaches its block-visit limit on the not-ready
path, and CSA then replays the call without inlining and stops inlining the
helper for the rest of the translation unit
(`tests/native/interruptible_wait.cpp`). A const accessor without parameters
on a single-member stream or event wrapper, declared but defined in another
unit (RMM 26's `cuda_stream_view::value()`), returns that member
(`tests/native/handle_accessor.cpp`); before 0.6.0 each call returned a fresh
value, so a copy on `stream.value()` never matched a wait on the same view.

**Cost.** Summaries are flow-insensitive "may" claims: a copy on any path
counts, so a helper that copies only on some paths makes every call look
pending. A body that synchronizes anything gets no copy summary, and one whose
paths copy different parameters gets none either. Besides copies, only a
stream, device or event synchronization on every path that returns is
summarized, not frees or event records, and a summary carries no memory
kind: it creates a pending operation only when the caller knows the operand
is host memory. A wait through a function pointer is resolved only when the
pointer is a parameter that the caller sets to a modeled query or
synchronization. Which helper units to summarize must be named (`summarize`
in `corpus/real-bugs/commands.json`); the whole-project scans do not run a
summary pass.

## Spending the node budget where CUDA state can change

**Decision.** Three changes in 0.6.0 keep CSA's budget of 225,000 nodes per
top-level function from going to code that cannot change what the checker
reports. A path ends on entering a block from which every path reaches a
throw or a call that never returns. Standard-library internals that branch
on values no path determines (shared_ptr reference counts, iostream state,
mutexes) are evaluated as CSA evaluates a call it does not inline, unless a
pending buffer reaches the call. And when a top-level function exhausts its
budget, the header functions it calls, directly or through other project
functions, that themselves call the CUDA API, a model or a kernel become
entry points of their own if CSA never inlined them, up to 16 per
translation unit.

**Alternatives.** A larger budget; lower inlining limits (stack depth,
inlinable size) or no inlining of the standard library for every call; loop
widening; summaries of each RAFT and RMM helper.

**Why.** The node profile (`--node-profile`, which attributes every node of
a top-level function to the function, the direct callee and the line it is
in) of the development cases that exhausted their budgets showed where the
nodes went. On every checked CUDA call, RAFT's error branch formats a
message with streams and collects a stack trace in a loop before it throws,
and CSA ends the path only at the throw; 7,500 to 23,800 of the roughly
380,000 nodes of each such function were in those frames, and none are
after the change. Each shared_ptr copy and release splits a path on the
count and on whether the process is threaded, and RAFT's resource lookups
copy several per call (in cuVS's `serialize`, 69,600 nodes before and 28,800
after). CSA stops at a throw anyway, so ending the path earlier loses only the
accesses between the branch and the throw, and a library call that a pending
buffer reaches is still inlined. A larger
budget only moves the problem (cuDF's parquet case still exhausts at
2,000,000 nodes), and global inlining limits would also cut the wrappers that
queue copies. Fallback entry points recover a callee the exhausted caller never
reached: cuML's `TSNE::min_max`, whose report was the one budget miss these
changes turned into a find.

**Cost.** Accesses on a branch that can only throw are not checked
(`tests/native/throw_paths.cpp`), and the deleter that a shared_ptr's last
reference runs is not followed. A fallback entry point starts with nothing
known about its parameters, so its reports depend only on what it allocates
and queues itself, and analyzing it costs budget too: on development set v2 the
build without fallback entry points took 987 s and 0.6.0 1,219 s, 158 s of
it on a translation unit that now parses
([corpus/real-bugs-heldout/README.md](../corpus/real-bugs-heldout/README.md#development-set-v2-version-060)).
On the three projects 131, 11 and 2 fallback entry points were analyzed
(cuda-samples, faiss, RMM) without adding an alert, and the summed per-TU
analysis time still fell, from 718, 1,363 and 137 s with 0.5.0 to 586, 1,232
and 126 s. The functions that still exhaust their
budget mostly spend it in mdspan extent arithmetic, RAFT's resource
registry, gtest's message streams and nested loops over host data, and
Warp's driver calls through function pointers with checked failure paths
that print and continue.

## Host containers own their buffer

**Decision.** A pointer that `data()` returns from `std::vector`,
`std::basic_string`, `thrust::host_vector` or a class derived from one is
host memory owned by that object, and so is the buffer of a view
(`std::span`, `string_view`, cuDF's `host_span`) built from it. When the
owner's destructor runs while an asynchronous operation still uses the
buffer, that is a CL2 warning.

**Alternatives.** Let CSA inline container methods
(`c++-container-inlining`) and follow the allocator's `new`/`delete`;
annotate container types in models.

**Why.** CSA does not inline methods of classes with `begin()` defined in
headers, so `v.data()` is a fresh symbol unrelated to `v`, and a vector going
out of scope released nothing the checker could see. Inlining every
container would push the large functions that already exhaust their budget
further over it, and cuDF's pinned vectors allocate through an RMM resource
whose body is elsewhere. Naming the owner types is a small, checkable claim.

**Cost.** The owner list is by name; other containers are not covered. The
memory kind is taken as pageable, so these reports are warnings even for
cuDF's pinned vectors. Only destruction ends the buffer: `clear`,
`resize` or move assignment, which can also release it, are not modeled.

## Coverage gaps are reported, never counted as clean

**Decision.** Each translation unit gets a coverage record: parse failures,
unmodeled CUDA calls, exhausted budgets, unknown effects, events of unknown
history. `analyze` exits with 2 when coverage is incomplete, and such a unit
cannot earn a true negative in the evaluation.

**Alternatives.** Report findings only.

**Why.** A static analyzer that skips code silently overstates safety; zero
alerts means nothing without the code that was actually explored.

**Cost.** Large projects always have gaps (the driver API in cuda-samples,
budget exhaustion), so the exit status alone cannot gate CI; the coverage
report has to be read.

## nvcc databases translated to Clang host-only commands

**Decision.** A translator maps each nvcc command to a Clang host-only command
flag by flag ([flag-map.md](flag-map.md)). A command with any argument it
cannot account for is excluded and reported, not guessed.

**Alternatives.** Require projects to build with Clang CUDA; analyze nvcc's
preprocessed output.

**Why.** Most CUDA projects build with nvcc, and Clang's CUDA host-only mode
parses the same sources with the macros (`__CUDACC__`, the per-thread default
stream) the analysis depends on.

**Cost.** Some units do not parse: 223 of 233 in cuda-samples (missing
libraries or headers) and 189 of 190 in faiss (one generated unit). CCCL-heavy
projects need an optional header compatibility profile.

## Labels before runs, and a holdout scored once

**Decision.** Corpus cases are labelled from the CUDA semantics before the
checker runs on them. The holdout's sources, labels and checker were frozen,
and its first run is the reported score (25 TP, 1 FP, 0 FN, 23 TN). Real
alerts are triaged by hand against the source.

**Alternatives.** Tune on the evaluation set; report the best run.

**Why.** Labels written after seeing output follow the checker. The frozen
holdout exposed a real checker bug, and the fixed version's 25/0/0/24 is
reported as a regression result, since the set had been seen.

**Cost.** The corpora are small (20 seed cases, 49 holdout labels), and triage
is manual. Of version 0.2.0's 76 real-project alerts none was a
confirmed defect, and of the current 46, 32 are portability findings, 3
informational and 11 false positives.
