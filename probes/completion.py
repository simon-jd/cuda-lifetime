#!/usr/bin/env python3
"""Observe when cudaMemcpyAsync D2H returns for pinned and pageable host memory.

A kernel that spins for about 20 ms writes a value; the copy of that value to
the host is queued right behind it. For each mode the probe reports how long
the copy call blocked and whether the kernel had finished when it returned.
This is an observation of one GPU and driver, not a CUDA guarantee.

    python3 probes/completion.py --out results/probes/completion.json
"""
import argparse
import json
from pathlib import Path
import shutil
import statistics
import subprocess

ROOT = Path(__file__).resolve().parents[1]
MODES = ["pinned-observe", "pageable-observe", "pinned-synchronized", "pageable-synchronized"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--binary", type=Path, default=ROOT / "build/completion-probe")
    parser.add_argument("--nvcc", default=shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc")
    parser.add_argument("--ccbin", help="host compiler for nvcc")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not args.binary.exists():
        args.binary.parent.mkdir(parents=True, exist_ok=True)
        command = [args.nvcc, "-std=c++17", "-arch=sm_89", str(ROOT / "probes/completion_contract.cu"), "-o", str(args.binary)]
        if args.ccbin:
            command[1:1] = ["-ccbin", args.ccbin]
        subprocess.run(command, check=True)
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                         capture_output=True, text=True, check=True).stdout.strip()
    modes = {}
    for mode in MODES:
        runs = []
        for _ in range(args.repeats):
            done = subprocess.run([str(args.binary), mode], capture_output=True, text=True, timeout=60)
            lines = [json.loads(line) for line in done.stdout.splitlines() if line.startswith("{")]
            observation = next((line for line in lines if line.get("observation")), None)
            if done.returncode or observation is None:
                raise SystemExit(f"{mode}: exit {done.returncode}\n{done.stdout}{done.stderr}")
            runs.append(observation)
        modes[mode] = {
            "copyCallMs": [round(r["copyCallMilliseconds"], 3) for r in runs],
            "copyCallMsMedian": round(statistics.median(r["copyCallMilliseconds"] for r in runs), 3),
            "kernelDoneWhenCopyReturned": sorted({r["producerAfterCopy"] == "ready" for r in runs}),
            "valueAfterSync": sorted({r["synchronizedValue"] for r in runs}),
        }
        print(mode, json.dumps(modes[mode]), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"gpu": gpu, "repeats": args.repeats, "kernelSpinCycles": 50_000_000,
                                    "modes": modes}, indent=2) + "\n")


if __name__ == "__main__":
    main()
