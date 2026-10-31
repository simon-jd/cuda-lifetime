# Third-party components and imported evidence

This project links to LLVM/Clang and analyzes independently licensed upstream
sources. Imported sources are retained under `.deps/` for local evidence and are
not incorporated into the checker implementation. Generated CUDA sample mutants
retain the original notices in each copied source file.

| Component | Version / source | Upstream license |
| --- | --- | --- |
| LLVM / Clang | llvmorg-23.1.3 | Apache-2.0 with LLVM exceptions; see the imported `LICENSE.TXT`. |
| faiss | v1.15.1 | MIT; see the imported `LICENSE`. |
| RMM | v26.10.00 | Apache-2.0; see the imported license. |
| CUDA samples | v13.2 | NVIDIA's BSD-style redistribution terms in the original files/license. |
| CUDA Toolkit | 13.2.2 | NVIDIA CUDA Toolkit license; the toolkit is installed separately from NVIDIA's repository. |
| Optional CCCL compatibility snapshot | Official development commit `510cc9de1ea656a453d6072ea3e1f6097410db9a` | Component-specific terms: Thrust Apache-2.0 with listed exceptions, libcu++ Apache-2.0 with LLVM exceptions, and CUB BSD-style terms. The complete downloaded tree retains its root and component license files. |
| NVBench | deb95d3da687fb1e57ba65364210e354aa67198c | Apache-2.0; see the imported `LICENSE`. |
| cuCollections | e348af6e42c5c4e3c2f8a214393bb90e811387f8 | Apache-2.0; see the imported `LICENSE`. |

Dependency archives, checksums, target versions and configure/build logs are
evidence of the actual installations. They are not a claim of upstream approval,
contribution or affiliation. CMake-fetched dependencies retain their upstream
license files in each target's build tree. Redistributing an executable or
container also requires retaining the applicable dependency notices and license
terms; this file does not replace those terms.
