# nvcc → Clang host compilation mapping

The translator uses only the Python standard library (Python 3.12+). It processes
each compilation command independently. A command with any unsupported argument
has status `translation-unknown` and is excluded from the output database, while
its full evidence stays in `translation-report.json`. `translated` means all
arguments were accounted for; it does not mean the TU parsed or was analyzed.

```sh
python3 -m translate translate --compdb build --out build-clang \
  --clang .deps/LLVM-23.1.3-Linux-X64/bin/clang++ --cuda-path /usr/local/cuda-13.2 \
  --gcc-toolchain "$PWD/.deps/gcc13/usr"
python3 -m translate macros --compdb build-clang
```

The standalone CLI exits 2 for incomplete translation or macro verification,
1 for an unreadable/invalid database, and 0 for the requested completed operation.
Compiler builtin differences are retained but the macro CLI's success gate checks
project macros, `__CUDACC__`, and `CUDA_API_PER_THREAD_DEFAULT_STREAM`.

Version 0.2.0 adds opt-in `--compatibility-profile PATH` to both translate CLIs.
The [profile guide](compatibility-profile.md) describes the small reusable
templates, automatic resolved lock, complete official CCCL snapshot pin, and
actual GCC support-header discovery. Without the option, the mapping and
two-file output below remain unchanged. Profile translation uses a new/empty
output directory and records original, derived reference and Clang commands
separately; its fixed compiler discovery queries are explicit opt-in behavior.

Python API:

```python
from translate import translate_compdb, macro_diff_compdb
report = translate_compdb("build", "build-clang", cuda_path="/usr/local/cuda-13.2",
                          clang="/absolute/path/to/clang++", gcc_toolchain="/opt/gcc13/usr")
macro_report = macro_diff_compdb("build-clang", timeout=30)
```

`translate_compdb` accepts paths as strings or `Path` objects. Its output directory
contains `compile_commands.json` and `translation-report.json`. Macro collection
adds `macro-diff.json` and updates each translation entry's `macroDiff`. Relative
command directories resolve against the input database directory. Relative
response paths, including nested references, resolve against the command's
working directory. No shell is executed; `arguments` takes precedence over
`command`. POSIX command quoting is supported; Windows/MSVC command syntax,
shell operators and unknown command wrappers are rejected.

| Input | Output / disposition |
| --- | --- |
| `@file`, `--options-file file`, `--options-file=a,b`, `-optf` | Recursively expand POSIX tokens, preserve ordering, reject missing files, malformed quoting, cycles and nesting over 32. |
| `nvcc`, `g++`, `c++`, `gcc`, `cc`, `clang++`, `clang` | Selected Clang driver. `ccache` and `sccache` wrappers are removed with evidence. Version suffixes on host compilers are accepted. |
| `.cu`, `.cuh`, `-x cu` | `-x cuda --cuda-host-only`, plus the supplied `--cuda-path`. Explicit source language takes precedence over extension. `.cpp` remains host C++. |
| CUDA 13 nvcc implicit system include | If the selected toolkit contains `include/cccl`, append that `-isystem` search path after project paths. Locked nvcc `--dryrun` confirms it; Clang's `--cuda-path` alone does not add it. |
| `-D`, `-U`, `-I`, `-isystem`, `-iquote`, `-include`, `-imacros`, `-std` | Preserve argument values and order, including project macros, spaces in paths and macro string literal quotes. Corresponding nvcc long options are normalized. |
| `-Xcompiler=…`, `-Xcompiler …`, `--compiler-options` | Expand comma list before normal host mapping, respecting quoted and escaped commas; nested `-D/-I/-include/-std` remain intact. |
| `--default-stream=legacy/per-thread` | **`-Xclang -fgpu-default-stream=…`**. A compiler probe showed that the driver spelling alone is reported as unused in host-only compilation. Forwarding to cc1 sets both language state and the per-thread runtime macro. |
| `-rdc=true/false`, `--relocatable-device-code` | `-fgpu-rdc` / `-fno-gpu-rdc`, plus define/undefine `__CUDACC_RDC__`. |
| `-dc`, `--device-c` | `-c -fgpu-rdc -D__CUDACC_RDC__=1`. |
| `--expt-relaxed-constexpr` | Clang enables implicit host/device constexpr by default; preserve `__CUDACC_RELAXED_CONSTEXPR__=1`. The nonexistent positive Clang driver switch is not emitted. |
| `--extended-lambda`, `--expt-extended-lambda` | Clang accepts explicit host/device lambdas by default; preserve `__CUDACC_EXTENDED_LAMBDA__=1`. Checked with capture-positive and missing-capture-negative cases. |
| `-G`, `--device-debug` | No device output is needed; preserve `__CUDACC_DEBUG__=1` for host preprocessing. |
| `--generate-code`, `-gencode`, `-arch`, `--gpu-architecture`, `-code` | Validate explicit `sm_XX` / `compute_XX` forms; omit device generation selection with a recorded reason. `native`/`all` remain unknown. `__CUDA_ARCH_LIST__` differences remain visible in macro evidence. |
| `-ccbin`, `--compiler-bindir`, forwarding policy | Remove with reason; the configured Clang driver handles the host frontend. An explicit `gcc_toolchain` / `--gcc-toolchain` preserves the intended GCC header installation in the output command. Without it, the report explicitly marks host header search and builtin equivalence unverified. Unknown forwarded options still fail translation. |
| Explicit `cccl-510cc9d-host-headers` profile | Verify the intact pinned official source tree; prepend its three ordered include roots to both original-driver reference and Clang commands. Preserve all original argv and record this deliberate dependency override. |
| Explicit profile host-header discovery | Query the original actual GCC host executable with `-print-file-name=include`, lock its full returned header directory, and append `-idirafter DIRECTORY` only to Clang. Missing/ambiguous host identity remains unknown. The `host-headers` control profile performs this step without replacing project CCCL. |
| `-Xptxas`, `-Xfatbin`, `-Xnvlink` | Remove device assembler, packaging and linker options with reasons. |
| `-Xcudafe` | Remove recognized numeric diagnostic severity/suppression payloads and display-error-number controls with reasons. Map `--promote_warnings` to `-Werror`. Other frontend options remain unknown because they can change semantics. |
| `--diag-suppress`, `--diag-error`, `--diag-warn` | Remove nvcc diagnostic settings with reasons. |
| `-Werror=all-warnings`, `-Wno-subobject-linkage` | Preserve all-warning error policy as Clang `-Werror`; remove the GCC-specific warning suppression with a reason because Clang has no corresponding diagnostic. |
| Temporary output, parallelism, `-cudart`, device line information | Remove with individually recorded reasons. |
| Reviewed host options | Preserve a finite allowlist for optimization, ABI, exceptions/RTTI, warnings, target, dependency output and compilation output. Unrecognized `-f…`, plugin, linker and preprocessor forwarding options are unknown. |

The mapping follows the [nvcc driver documentation](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html)
and [Clang option reference](https://clang.llvm.org/docs/ClangCommandLineReference.html).
Those current documentation pages are not proof of locked-version equivalence.
The mapping was checked against the locked LLVM 23.1.3 and CUDA 13.2.2 headers
with small compiler probes (their outputs are not included in this repository).
The lambda/constexpr positive and negative cases were checked that way, and so
was the default-stream mapping: `clang++ --cuda-host-only
-fgpu-default-stream=per-thread` reports the option as unused and does not
define the per-thread default-stream macro, which is why the translator
forwards it with `-Xclang`.

The translator records a dialect assumption when nvcc did not enable relaxed
constexpr: Clang's default frontend accepts additional constexpr calls. Turning
off that Clang behavior also breaks its own CUDA runtime headers. Passing these
probes establishes support for the exercised constructs, not equivalence of
every CUDA C++ program. The translator does not define `__NVCC__` or pretend
Clang is a particular nvcc release.

Macro probes run the original compiler with `-E -Xcompiler=-dM` for nvcc (host
compilers use `-E -dM`) and the translated Clang compiler with `-E -dM`. **For CUDA
inputs, the nvcc dump contains `__CUDA_ARCH__`; this public preprocessing phase
does not isolate the host pass.** It therefore produces `project-only`, with
`scope=nvcc-preprocessing-host-unverified`, even when the observed project and
CUDA macros match. Real dumps and their `observedPreprocessorDifferences` and
`observedRequiredDifferences` remain in the report, but the verification gate
does not treat them as host macro parity.

Output and dependency flags are removed to avoid overwriting build products.
Nonzero exit, timeout, missing executable, or an output without macro definitions
also yields `project-only`. In this fallback only explicit `-D/-U` options are
compared, and host builtins, runtime/default-stream and header macros stay
unverified. Commands, return codes, stderr and real macro sets are retained when
available. Ordinary host C++ commands can receive status `compared` after both
real host preprocessors succeed.

With a profile, the reference side is the separately retained derived
NVCC/original-host command with the same candidate CCCL roots, not the unchanged
old-dependency command. Full real outputs and dual-driver `-M` header provenance
are retained and tied to `compatibility-resolved.json`. Mixed candidate/old
headers, missing bindings, drift or failed dependency collection cannot pass the
profile macro gate. CUDA host-pass uncertainty remains nonzero; no identity
macros are invented to obtain parity.

Validation:

```sh
python3 -m unittest discover -s tests -p test_translate.py -v
```
