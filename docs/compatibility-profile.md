# Header compatibility profiles

A profile makes `translate` use a pinned CCCL source tree and the GCC header
directory of each command's own host compiler. The cuda-samples and faiss
results in [evaluation.md](evaluation.md) were translated with the CCCL
profile below and the RMM results with the host-header profile; with the
locked CUDA 13.2 headers alone, most faiss CUDA sources do not parse under
Clang. Translation without a profile keeps the default mapping and runs no
compiler queries. Profiles were added in version 0.2.0.

The public template [profiles/cccl-510cc9d-host-headers.json](../profiles/cccl-510cc9d-host-headers.json)
is small:

```json
{
  "schemaVersion": 1,
  "profile": "cccl-510cc9d-host-headers",
  "ccclRoot": "/opt/cuda-lifetime-compat/candidate-510cc9d/cccl-510cc9de1ea656a453d6072ea3e1f6097410db9a"
}
```

Change only `ccclRoot` to the location of your complete, unmodified extracted
source archive. Relative paths resolve against the template directory. A Git
checkout with extra files, a partial header installation, and a locally patched
tree will not match the pin. The tool neither downloads dependencies nor changes
the toolkit. The fixed source identity is:

| Field | Identity |
|---|---|
| Official commit | `510cc9de1ea656a453d6072ea3e1f6097410db9a` |
| [Source archive](https://codeload.github.com/NVIDIA/cccl/tar.gz/510cc9de1ea656a453d6072ea3e1f6097410db9a) SHA256 | `17bc62b70be51107727a2bc5694693fca1fc73d18c182620522a503c5edfe680` |
| Complete file-tree SHA256 | `62acadd639e08afc90ce4edf83515a27c50e9e9008d5643bbf2e0105210e5baa` |

This is an official development commit snapshot, **not the v3.4.0 release**,
despite its `CCCL_VERSION=3004000` macro. The tree digest is SHA256 of
`json.dumps({relative_file_path: file_sha256, ...}, sort_keys=True).encode()`;
the two official symlink identities are checked separately. The expected archive
pin is recorded, while translation verifies the complete extracted tree itself.

```sh
cuda-lifetime translate --compdb build/compile_commands.json --out build-clang-compat \
  --clang /opt/llvm-build/bin/clang++ --cuda-path /usr/local/cuda-13.2 \
  --gcc-toolchain /usr \
  --compatibility-profile profiles/cccl-510cc9d-host-headers.json
cuda-lifetime macros --compdb build-clang-compat --out build-clang-compat/macros
```

`python3 -m translate translate` accepts the same option. The output directory
must be new or empty. Existing default options still select the Clang executable,
CUDA toolkit and GCC toolchain prefix; they are not guessed from the profile.
The validated cohort used LLVM 23.1.3, CUDA 13.2.2 and actual container GCC 13.3.
The resolved lock records the selected tools, including real paths, bytes and
version output. A different tool installation or project is a new set of inputs,
not automatically covered by the original validation.

For an original dependency control such as RMM, use
[profiles/host-headers.json](../profiles/host-headers.json). This profile discovers
the host support headers but makes **no CCCL replacement**. RMM keeps its own
project-selected CCCL 3.5.0. Do not apply the CCCL replacement globally to unrelated
projects merely because they share a compilation database workflow.

```python
from translate import translate_compdb

report = translate_compdb(
    "build", "build-clang-compat", cuda_path="/usr/local/cuda-13.2",
    clang="/opt/llvm-build/bin/clang++", gcc_toolchain="/usr",
    compatibility_profile="profiles/cccl-510cc9d-host-headers.json",
)
```

The profile prepends `thrust`, `cub`, and `libcudacxx/include`, in that order, to
both the original-driver reference and its Clang translation. Original options,
macros and include ordering remain behind this explicitly declared override.
Known compiler-cache wrapper chains remain ahead of the actual reference driver.
No compiler-identity macros or forced source includes are added by the profile.

The original NVCC command must select an actual GCC executable using `-ccbin` or
`--compiler-bindir`; an implicit selection or a directory-only value is unverified
and remains a reported translation-unknown. For ordinary C/C++, the original GCC
driver supplies the identity. Each actual host driver is queried with precisely
`-print-file-name=include`; the returned absolute directory is validated and
locked. `gcc` and `g++` are tracked separately. Only the Clang command receives
`-idirafter DIRECTORY`, appended at the end. This preserves after-system search
priority and does not replace the explicit `--gcc-toolchain` option. Version
queries use fixed `--version` argv. No shell or user-defined query command runs.

The tool generates these files automatically; users do not write an experimental
manifest or enumerate the database/header files manually:

- `original-compile_commands.json`: exact original database bytes.
- `reference-compile_commands.json`: explicitly derived original NVCC/host
  commands using the candidate headers. It may include a reference for a TU
  whose Clang translation is unknown; report indices show the distinction.
- `compile_commands.json`: accepted Clang commands only.
- `translation-report.json`: schema 2 for profile mode, retaining original,
  expanded, reference and translated argv, original command hashes and indices,
  response-file provenance, reasons and unknown entries.
- `compatibility-resolved.json`: template, current database/source/response/tool
  identities, full candidate and discovered-host-header file inventories,
  symlink paths, fixed-query outputs and per-entry command bindings.

All selected header and recorded input identities are checked before and after
translation. A bad global pin rejects the operation; a per-entry compiler or
discovery failure stays in the report and denominator, excluded from the accepted
database. No-profile reports keep schema 1 and their existing two-file output.
`translate_entry` itself remains a pure argument mapping without subprocesses.

Profile macro collection uses **referenceArguments**, not the original command
that still selects the old dependency. Missing or altered command bindings,
output databases, tools or locked headers reject collection. The legacy
`originalMacros` field is explicitly labelled as the compatibility-derived
reference, and real original argv are retained separately. Full macro stdout,
stderr, actual commands, exit status and hashes are saved.

The lock also binds immutable translation metadata, including language and
original indices. Profile reports cannot silently fall back to the legacy
macro path when their profile metadata is missing. Command-token resolution is
checked again in each recorded working directory, so a changed `PATH` cannot
substitute another compiler. Evidence output paths are checked against bound
inputs, output databases and header trees before any writes, including symlink
and existing hard-link aliases.

The same collection also runs `-M` for both derived drivers and records actual
header paths/hashes. Candidate CCCL headers must come from the declared roots and
match the locked bytes; mixed old/new roots or failed dependency collection are
unverified. The drivers may access different candidate file sets. A successful
dependency probe that accesses no CCCL is not-applicable. The host-only control
retains dependency evidence without claiming a candidate replacement.

`profileApplied` is not syntax success or a header-selection proof. Header
identity is initially not-run and is updated by the real probes. The resolved
lock binds the candidate and GCC support-header trees plus listed direct inputs;
it is not a snapshot of every transitive SDK/STL input. Actual dependency outputs
are retained separately. Version/profile labels do not prove compiler language
equivalence or complete analysis coverage.

CUDA NVCC preprocessing still does not isolate the host pass. Its macro status
remains `project-only / nvcc-preprocessing-host-unverified`, and both macro CLIs
return nonzero even when explicit project macros match and candidate-header
provenance verifies. Host macro comparisons require matching required values;
profile dependency failure also prevents a successful macro command.

Unit regression (mocked compilers; no checker/GPU/holdout):

```sh
python3 -m unittest discover -s tests -p test_compatibility_profile.py -v
```

## Effect on parsing

Host-only syntax checks of every configured command, with the locked CUDA 13.2
headers and with this profile (CCCL commit `510cc9d`, plus the GCC header
directory discovered from each command's own compiler):

| Project | Configuration | All commands parsed | CUDA commands parsed |
| --- | --- | ---: | ---: |
| cuda-samples 13.2 | locked headers | 217/233 | 98/114 (86.0%) |
| cuda-samples 13.2 | profile | 223/233 | 104/114 (91.2%) |
| faiss 1.15.1 | locked headers | 423/854 | 4/182 (2.2%) |
| faiss 1.15.1 | profile | 724/854 | 181/182 (99.5%) |
| RMM 26.10 | either | 53/54 | no `.cu` commands |

The remaining cuda-samples failures are nine `min`/`max` overload errors and one
unsupported shared-variable initialization. 129 faiss host commands target ARM
SVE, which these x86 drivers do not support, and `GpuDistance.cu` has an
overload error. RMM's one failure is a host warning promoted to an error.
Syntax success does not mean complete analyzer coverage.

These counts come from host-only syntax runs; `environment.compatibility.lock`
records them under `qualification`, but the per-command logs are not included
in this repository. They agree with the analysis runs in
[evaluation.md](evaluation.md): with the profile, 223 of cuda-samples' 233
commands parse there too, the ten failures are the same nine overload errors
and one `__shared__` initialization, and the one faiss GPU-module failure is
the same `GpuDistance.cu`.
