#!/usr/bin/env python3
"""Fetch pinned upstream archives and configure reproducible read-only targets.

The network is used only for the pinned archives and upstream CMake dependencies.
No source patch is applied. CUDA samples select sm_89 through a public CMake hook.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import urllib.request
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = {
    "cuda-samples": ("NVIDIA/cuda-samples", "v13.2", "cuda-samples-13.2", ""),
    "faiss": ("facebookresearch/faiss", "v1.15.1", "faiss-1.15.1", ""),
    "rmm": ("rapidsai/rmm", "v26.10.00", "rmm-26.10.00", "cpp"),
}


def fetch_source(name, deps, lock):
    repository, tag, folder, subdirectory = UPSTREAM[name]
    archives = deps / "downloads"
    archives.mkdir(parents=True, exist_ok=True)
    archive = archives / f"{name}-{tag}.tar.gz"
    url = f"https://codeload.github.com/{repository}/tar.gz/refs/tags/{tag}"
    expected = lock["targetArchiveSha256"][name]
    if not archive.exists():
        temporary = archive.with_suffix(".download")
        with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output)
        if hashlib.sha256(temporary.read_bytes()).hexdigest() != expected:
            temporary.unlink()
            raise ValueError(f"archive checksum mismatch: {name}")
        temporary.rename(archive)
    if hashlib.sha256(archive.read_bytes()).hexdigest() != expected:
        raise ValueError(f"archive checksum mismatch: {name}")
    source = deps / folder
    reused = source.exists()
    if not source.exists():
        with tarfile.open(archive) as bundle:
            bundle.extractall(deps, filter="data")
    if not (source / subdirectory / "CMakeLists.txt").is_file():
        raise ValueError(f"archive did not contain expected source: {source}")
    verified = 0
    with tarfile.open(archive) as bundle:
        for member in bundle:
            if member.isfile():
                original = bundle.extractfile(member)
                candidate = deps / member.name
                if not candidate.is_file() or hashlib.sha256(candidate.read_bytes()).digest() != hashlib.sha256(original.read()).digest():
                    raise ValueError(f"imported source differs from the locked archive: {candidate}")
                verified += 1
    return source / subdirectory, {"url": url, "archiveSha256": expected,
                                  "source": str(source), "sourceReused": reused,
                                  "archivedFilesVerified": verified,
                                  "verificationScope": "all regular archive members; extra local files are not certified"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", nargs="+", choices=UPSTREAM, default=list(UPSTREAM))
    parser.add_argument("--deps", type=Path, default=ROOT / ".deps")
    parser.add_argument("--cmake", default="cmake")
    parser.add_argument("--cc", default="gcc-13")
    parser.add_argument("--cxx", default="g++-13")
    parser.add_argument("--cuda-path", type=Path, default=Path("/usr/local/cuda-13.2"))
    parser.add_argument("--blas", help="Optional local BLAS library path")
    parser.add_argument("--lapack", help="Optional local LAPACK library path")
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--out", type=Path, help="new log directory; default is a timestamped results/prepare-targets subdirectory")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("jobs must be positive")
    args.deps = args.deps.resolve()
    lock = json.loads((ROOT / "environment.lock").read_text())
    if lock["cuda"]["version"] != json.loads((args.cuda_path / "version.json").read_text())["cuda"]["version"]:
        parser.error("the selected CUDA toolkit does not match environment.lock")
    cmake_version = subprocess.check_output([args.cmake, "--version"], text=True).splitlines()[0]
    if cmake_version != f"cmake version {lock['cmake']}":
        parser.error("the selected CMake does not match environment.lock")
    gcc_version = subprocess.check_output([args.cxx, "-dumpfullversion"], text=True).strip()
    parts = tuple(int(part) for part in gcc_version.split("."))
    if parts[:1] != (13,) or parts < (13, 3):
        parser.error("GCC 13.3 or later in the GCC 13 series is required")
    logs = args.out or ROOT / "results" / "prepare-targets" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    logs.mkdir(parents=True, exist_ok=False)
    records = []
    for name in args.targets:
        source, provenance = fetch_source(name, args.deps, lock)
        build = ROOT / "targets" / name / "build"
        command = [args.cmake, "-S", str(source), "-B", str(build), "-G", "Ninja",
                   "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
                   "-DCMAKE_C_COMPILER=" + args.cc, "-DCMAKE_CXX_COMPILER=" + args.cxx,
                   "-DCUDAToolkit_ROOT=" + str(args.cuda_path)]
        if name != "rmm":
            command += ["-DCMAKE_CUDA_COMPILER=" + str(args.cuda_path / "bin/nvcc"),
                        "-DCMAKE_CUDA_HOST_COMPILER=" + args.cxx, "-DCMAKE_CUDA_ARCHITECTURES=89"]
        if name == "cuda-samples":
            command += ["-DCMAKE_PROJECT_INCLUDE=" + str(ROOT / "targets/sm89.cmake")]
        elif name == "faiss":
            command += ["-DFAISS_ENABLE_GPU=ON", "-DFAISS_ENABLE_PYTHON=OFF", "-DFAISS_ENABLE_CUVS=OFF",
                        "-DFAISS_ENABLE_MKL=OFF", "-DFAISS_ENABLE_EXTRAS=OFF", "-DFAISS_OPT_LEVEL=generic",
                        "-DBUILD_TESTING=OFF"]
            if args.blas:
                command.append("-DBLAS_LIBRARIES=" + args.blas)
            if args.lapack:
                command.append("-DLAPACK_LIBRARIES=" + args.lapack)
        else:
            command += ["-DBUILD_TESTS=OFF", "-DBUILD_BENCHMARKS=OFF", "-DRMM_NVTX=OFF"]
        record = {"target": name, "provenance": provenance, "cmakeVersion": cmake_version,
                  "gccVersion": gcc_version, "commands": [command]}
        records.append(record)
        (logs / "commands.json").write_text(json.dumps(records, indent=2) + "\n")
        with (logs / f"{name}-configure.log").open("w") as output:
            subprocess.run(command, check=True, stdout=output, stderr=subprocess.STDOUT)
        if args.build:
            command = [args.cmake, "--build", str(build), "-j", str(args.jobs)]
            if name == "faiss":
                command += ["--target", "faiss"]
            record["commands"].append(command)
            (logs / "commands.json").write_text(json.dumps(records, indent=2) + "\n")
            with (logs / f"{name}-build.log").open("w") as output:
                subprocess.run(command, check=True, stdout=output, stderr=subprocess.STDOUT)
        print(name + ": " + str(build / "compile_commands.json"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
