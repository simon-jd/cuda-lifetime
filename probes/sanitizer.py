#!/usr/bin/env python3
"""Compare compute-sanitizer with cuda-lifetime on the probe programs.

Runs every compute-sanitizer tool (memcheck, memcheck with
--track-stream-ordered-races all, racecheck, synccheck, initcheck) on each
scenario of stale_read.cu and lifetime.cu, and the checker on both sources,
and records what each reports. Run it under the GPU lock:

    flock /var/tmp/rtx4090.lock python3 probes/sanitizer.py --ccbin g++-13 \\
      --native build/cuda-lifetime-native --out results/probes/sanitizer.json
"""
import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
PROGRAMS = {"stale_read": ("probes/stale_read.cu", ["pinned", "pageable"]),
            "lifetime": ("probes/lifetime.cu", ["pinned", "pageable", "free", "free-host", "ordered-bad", "ordered-good"])}
TOOLS = {"memcheck": ["--tool", "memcheck"],
         "memcheck-stream-ordered": ["--tool", "memcheck", "--track-stream-ordered-races", "all"],
         "racecheck": ["--tool", "racecheck"],
         "synccheck": ["--tool", "synccheck"],
         "initcheck": ["--tool", "initcheck"]}
SUMMARY = re.compile(r"(ERROR|RACECHECK) SUMMARY: (\d+) (error|hazard)")


def sanitize(sanitizer, tool, binary, mode):
    done = subprocess.run([sanitizer, *TOOLS[tool], str(binary), mode], capture_output=True, text=True, timeout=600)
    text = done.stdout + done.stderr
    counts = [int(m.group(2)) for m in SUMMARY.finditer(text)]
    reports = [line.strip("= ").strip() for line in text.splitlines()
               if line.startswith("=========") and re.search(r"error|Error|hazard|Hazard|Invalid|Race|race", line)
               and "SUMMARY" not in line]
    observed = [json.loads(line) for line in done.stdout.splitlines() if line.startswith("{") and "beforeSync" in line
                or line.startswith("{\"immediate\"")]
    return {"returncode": done.returncode, "errors": sum(counts) if counts else None, "reports": reports[:4],
            "programOutput": observed}


def check(native, source, cuda_path):
    with tempfile.TemporaryDirectory() as temp:
        sarif, coverage = Path(temp) / "out.sarif", Path(temp) / "coverage.json"
        done = subprocess.run([native, str(ROOT / source), "--sarif", str(sarif), "--coverage", str(coverage), "--",
                               "-x", "cuda", "--cuda-host-only", "--cuda-path=" + cuda_path, "--offload-arch=sm_89",
                               "-std=c++17"], capture_output=True, text=True)
        if done.returncode:
            raise SystemExit(f"checker failed on {source}:\n{done.stderr}")
        lines = (ROOT / source).read_text().splitlines()
        return sorted(({"ruleId": r["ruleId"], "level": r["level"],
                        "line": r["locations"][0]["physicalLocation"]["region"]["startLine"],
                        "source": lines[r["locations"][0]["physicalLocation"]["region"]["startLine"] - 1].strip(),
                        "message": r["message"]["text"]}
                       for run in json.loads(sarif.read_text())["runs"] for r in run["results"]),
                      key=lambda d: (d["line"], d["ruleId"], d["level"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--nvcc", default=shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc")
    parser.add_argument("--ccbin", help="host compiler for nvcc")
    parser.add_argument("--sanitizer", default=shutil.which("compute-sanitizer") or "/usr/local/cuda/bin/compute-sanitizer")
    parser.add_argument("--native", required=True)
    parser.add_argument("--cuda-path", default="/usr/local/cuda-13.2")
    parser.add_argument("--build", type=Path, default=ROOT / "build")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    version = subprocess.run([args.sanitizer, "--version"], capture_output=True, text=True).stdout.strip().splitlines()[-1]
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                         capture_output=True, text=True, check=True).stdout.strip()
    record = {"gpu": gpu, "computeSanitizer": version, "tools": TOOLS, "programs": {}}
    for name, (source, modes) in PROGRAMS.items():
        binary = args.build / f"sanitizer-{name}"
        command = [args.nvcc, "-std=c++17", "-arch=sm_89", "-lineinfo", str(ROOT / source), "-o", str(binary)]
        if args.ccbin:
            command[1:1] = ["-ccbin", args.ccbin]
        subprocess.run(command, check=True)
        runs = {mode: {tool: sanitize(args.sanitizer, tool, binary, mode) for tool in TOOLS} for mode in modes}
        record["programs"][name] = {"source": source, "checker": check(args.native, source, args.cuda_path), "runs": runs}
        for mode, tools in runs.items():
            print(name, mode, {tool: result["errors"] for tool, result in tools.items()}, flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
