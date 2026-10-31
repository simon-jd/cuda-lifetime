#!/usr/bin/env python3
"""Run stale_read.cu on the GPU and the checker on its source; record both.

The program reads the pinned (or pageable) destination of a cudaMemcpyAsync
D2H before synchronizing the stream, with a 20 ms kernel queued ahead of the
copy. For each memory kind the record lists the value read before and after
synchronization in every run. With --native, it also records what
cuda-lifetime reports for the same source. This observes one GPU and driver;
it is not a CUDA guarantee.

    python3 probes/stale_read.py --ccbin g++-13 --native build/cuda-lifetime-native \\
      --cuda-path /usr/local/cuda-13.2 --out results/probes/stale-read.json
"""
import argparse
import collections
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "probes/stale_read.cu"


def build(nvcc, ccbin, binary):
    binary.parent.mkdir(parents=True, exist_ok=True)
    command = [nvcc, "-std=c++17", "-arch=sm_89", str(SOURCE), "-o", str(binary)]
    if ccbin:
        command[1:1] = ["-ccbin", ccbin]
    subprocess.run(command, check=True)


def observe(binary, memory, repeats):
    runs = []
    for _ in range(repeats):
        done = subprocess.run([str(binary), memory], capture_output=True, text=True, timeout=60)
        if done.returncode:
            raise SystemExit(f"{memory}: exit {done.returncode}\n{done.stdout}{done.stderr}")
        runs.append(json.loads(done.stdout.strip().splitlines()[-1]))
    return {"runs": repeats,
            "beforeSync": dict(sorted(collections.Counter(str(r["beforeSync"]) for r in runs).items())),
            "afterSync": dict(sorted(collections.Counter(str(r["afterSync"]) for r in runs).items())),
            "staleReads": sum(r["beforeSync"] != r["afterSync"] for r in runs)}


def analyze(native, cuda_path):
    with tempfile.TemporaryDirectory() as temp:
        sarif, coverage = Path(temp) / "out.sarif", Path(temp) / "coverage.json"
        done = subprocess.run([native, str(SOURCE), "--sarif", str(sarif), "--coverage", str(coverage), "--",
                               "-x", "cuda", "--cuda-host-only", "--cuda-path=" + cuda_path,
                               "--offload-arch=sm_89", "-std=c++17"], capture_output=True, text=True)
        if done.returncode:
            raise SystemExit(f"checker: exit {done.returncode}\n{done.stderr}")
        results = [r for run in json.loads(sarif.read_text())["runs"] for r in run["results"]]
        lines = SOURCE.read_text().splitlines()
        return [{"ruleId": r["ruleId"], "level": r["level"], "memoryKind": r["properties"]["memoryKind"],
                 "line": r["locations"][0]["physicalLocation"]["region"]["startLine"],
                 "source": lines[r["locations"][0]["physicalLocation"]["region"]["startLine"] - 1].strip(),
                 "message": r["message"]["text"]} for r in results]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--binary", type=Path, default=ROOT / "build/stale-read")
    parser.add_argument("--nvcc", default=shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc")
    parser.add_argument("--ccbin", help="host compiler for nvcc")
    parser.add_argument("--native", help="cuda-lifetime-native executable; also analyze the source")
    parser.add_argument("--cuda-path", default="/usr/local/cuda-13.2")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not args.binary.exists():
        build(args.nvcc, args.ccbin, args.binary)
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                         capture_output=True, text=True, check=True).stdout.strip()
    record = {"gpu": gpu, "source": str(SOURCE.relative_to(ROOT)), "kernelSpinCycles": 50_000_000,
              "sentinel": -1, "written": 42,
              "observations": {memory: observe(args.binary, memory, args.repeats) for memory in ("pinned", "pageable")}}
    if args.native:
        record["checker"] = analyze(args.native, args.cuda_path)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
