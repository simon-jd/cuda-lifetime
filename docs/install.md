# Installation and reproduction

The reproducible environment is defined in `docker/Dockerfile` and
`environment.lock`. The image builds unmodified LLVM 23.1.3 from its verified
source archive, using Release with assertions and clang-tools-extra. CUDA is
installed from NVIDIA's Ubuntu 24.04 repository at toolkit package 13.2.2-1; the
toolkit's own `version.json` is checked. A GPU is unnecessary for static work.

```sh
docker build --build-arg BUILD_JOBS=8 -f docker/Dockerfile -t cuda-lifetime .
docker run --rm cuda-lifetime --help
```

An initial Docker build compiles LLVM and downloads the full CUDA toolkit, so
it takes a while and needs substantial disk space.

For an existing matching toolchain, build the native tool and run regressions:

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_DIR=/opt/llvm-build/lib/cmake/llvm \
  -DClang_DIR=/opt/llvm-build/lib/cmake/clang
cmake --build build -j 8
python3 -m unittest discover -s tests -p 'test_*.py'
ctest --test-dir build --output-on-failure
```

CTest runs the annotated checker fixtures in `tests/native` as lit tests
when it finds `llvm-lit` (an LLVM build tree has it in `bin/`; otherwise set
`-DLLVM_LIT=` to an installed `lit`). Each fixture's `RUN:` lines analyze it
and check the SARIF against its `// CLn level` comments with
`tests/native/verify.py`; one fixture can be run alone with
`llvm-lit -v tests/native/copy_ranges.cpp --param native=build/cuda-lifetime-native`.

The Python package uses the standard library. It can be run directly with
`./cuda-lifetime` or `python3 -m tool.cli`. An installed native executable is
selected with `--native /absolute/path/to/cuda-lifetime-native`; otherwise the
CLI uses `build/cuda-lifetime-native` relative to the working directory. The
native executable normally uses its matching Clang build's resource headers.
If relocating that SDK, supply its actual matching `--resource-dir` to the
native driver. LLVM shared libraries must remain discoverable by the loader.

## Imported target build

The following script downloads pinned, checksum-verified upstream archives,
uses their public CMake interfaces and records exact commands and build logs in
a new timestamped directory. It leaves imported sources unpatched. Existing
source directories are reused only after verifying every regular archive member
against the original bytes; extra local files are not certified by this check.
For a clean reproduction use an empty `.deps` and target build directories.

```sh
python3 scripts/prepare_targets.py --build --jobs 8
python3 scripts/check_translation.py \
  --clang /opt/llvm-build/bin/clang++ --gcc-toolchain /usr \
  --cuda-path /usr/local/cuda-13.2 --host-samples 1000 \
  --out results/translation-feasibility.json
```

`prepare_targets.py` accepts `--cmake`, `--cc`, `--cxx`, `--deps`, `--blas` and
`--lapack` for local installations. GCC must be 13.3 or later in the GCC 13 series;
the development host used a locally unpacked 13.4.0 toolchain. It configures
faiss with GPU enabled, Python/cuVS/MKL/extras disabled, RMM without tests or
benchmarks, and cuda-samples for sm_89. The sample project's own optional
dependency selection remains visible in its configure log; missing graphics,
MPI, FreeImage or NvSCI dependencies exclude those targets. A report over its
exported database is not a claim to have analyzed excluded sample sources.

With the locked CUDA 13.2 headers, many faiss CUDA sources do not parse under
Clang; the optional [header compatibility profile](compatibility-profile.md)
(`translate --compatibility-profile`) fixes most of them. Translation without
the option keeps the default mapping.

## Analysis and baselines

`analyze_targets.py` reads each project's translated database from
`targets/<project>/clang/`, so translate first. The published results use the
CCCL profile for cuda-samples and faiss and the host-header profile for RMM
(set `ccclRoot` in the CCCL template first; see the
[profile documentation](compatibility-profile.md)). `check_translation.py`
above writes a default translation to the same `targets/<project>/clang/`
directories; remove those first, since the output directory must be new or
empty:

```sh
for project in cuda-samples faiss; do
  ./cuda-lifetime translate --compdb targets/$project/build --out targets/$project/clang \
    --clang /opt/llvm-build/bin/clang++ --cuda-path /usr/local/cuda-13.2 \
    --gcc-toolchain /usr --compatibility-profile profiles/cccl-510cc9d-host-headers.json
done
./cuda-lifetime translate --compdb targets/rmm/build --out targets/rmm/clang \
  --clang /opt/llvm-build/bin/clang++ --cuda-path /usr/local/cuda-13.2 \
  --gcc-toolchain /usr --compatibility-profile profiles/host-headers.json
python3 scripts/analyze_targets.py --native build/cuda-lifetime-native \
  --toolchain-kind source-assertions --out results/real-projects-v1
python3 scripts/baseline.py \
  --compdb results/real-projects-v1/faiss/input --mode syntax \
  --clang /opt/llvm-build/bin/clang++ --out results/baseline-faiss-syntax
```

`analyze_targets.py` snapshots the executable, Python driver/packages and models,
records input/source hashes, and refuses to overwrite a previous run. External
SDK/header paths still require the recorded environment; the snapshot alone is
not a portable copy of all dependencies or proof of how a binary was built.
Its scope is every configured
cuda-samples/RMM command and the faiss GPU module, including generated CUDA TUs.
The toolchain label must match the actual selected executable. Baseline modes
`analyze` and `tidy` use the same scoped input database and preserve per-TU logs,
commands, timeouts, elapsed time and peak RSS; use `--tidy` for the exact matching
clang-tidy. Each baseline invocation gets an isolated compilation database for
that exact entry, including sources with multiple configurations. Compare
matched successful TUs separately from failed or timed-out commands. Do not
divide aggregate times from different subsets, job counts or concurrent machine
loads; record those conditions before interpreting relative timings.

Exit status 2 from analysis means coverage is incomplete. Failed parsing,
unmodeled CUDA operations, unsupported roots and budget exhaustion stay visible
alongside findings. Warnings are shown by default; `--errors-only` hides them.
Runtime behavior is measured separately by the [GPU probes](../probes/README.md).
