# Design trade-offs

Most of what cuda-lifetime reports, and most of what it misses, follows from
ten choices: analyzing host code path by path inside CSA, judging copies by
CUDA's documented contract rather than by what today's driver does, treating
opaque calls and wrapper models with suspicion, and refusing to call a
translation unit clean when its coverage is incomplete. The sections below
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
executions a test runs rarely show them, and a dynamic tool sees only those.
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

**Cost.** 22 of the 36 real-project alerts are this portability pattern, 19
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
is manual. Of the previous version's 76 real-project alerts none was a
confirmed defect, and of the current 36, 22 are portability findings, 3
informational and 11 false positives.
