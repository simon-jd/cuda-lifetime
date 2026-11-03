# LLVM interfaces, SARIF workarounds and related tools

cuda-lifetime is an out-of-tree consumer of the public LLVM/Clang 23.1.3 C++
interfaces. It does not modify LLVM; its native driver (`tool/main.cpp`)
registers the checker and its own SARIF consumer with an otherwise unmodified
Clang Static Analyzer.

## Two SARIF export problems in LLVM 23.1.3

**The stock CSA SARIF consumer cannot describe an external checker.** It looks
up each checker's description in the built-in checker table (`Checkers.inc`)
and reaches `llvm_unreachable` for a checker registered from outside that
table, such as this one. `tool/SarifConsumer.cpp` therefore builds the SARIF
document itself, using the public `SarifDocumentWriter` object model and the
analyzer's own `PathDiagnostic`s, so results and `codeFlows` are the ones CSA
produced.

**`SarifDocumentWriter` reorders artifacts without updating references.** It
assigns artifact indices in insertion order, then serializes the artifact
array sorted by URI, so `artifactLocation.index` values in results and code
flows can point at the wrong file. The consumer checks that the written
indices form a complete permutation and restores the artifact array to index
order; result messages, paths and source coordinates are left exactly as CSA
emitted them. If the indices are not a valid permutation, the TU's coverage is
marked incomplete instead. The native test `tests/native/artifact_order_z.cpp`
reports through a header, `artifact_order_a.h`, whose name sorts before the
main file's, and checks that every index still points at the right artifact.

## Related tools

The [Clang Static Analyzer checker list](https://clang.llvm.org/docs/analyzer/checkers.html)
and `clang/include/clang/StaticAnalyzer/Checkers/Checkers.td` at 23.1.3 list
no checker for CUDA stream lifetimes. [CuSan](https://github.com/tudasc/cusan)
instruments programs and uses ThreadSanitizer to observe host/device races
dynamically, and compute-sanitizer's racecheck examines device code. A limited
web search (October 2026) found no equivalent public static checker for host
code; such a search cannot show that none exists.

## Compiler baselines

[clang-tidy's CUDA notes](https://github.com/llvm/llvm-project/blob/llvmorg-23.1.3/clang-tools-extra/docs/clang-tidy/index.rst)
describe removing nvcc flags and adding host-only and CUDA-path arguments.
`scripts/baseline.py` runs Clang (syntax check or the stock analyzer) and
clang-tidy on the same translated database as cuda-lifetime, so both see the
same host definitions and include paths. Compare their raw diagnostics, not only return codes; the baseline
script does not map their rule names onto CL1–CL4. `scripts/score_baseline.py`
scores a report against corpus labels, and `scripts/run_real_bugs.py --tool`
runs both on the fixed bugs; the results, none of which concern an
asynchronous lifetime, are in
[evaluation.md](evaluation.md#stock-analyzer-and-clang-tidy).
