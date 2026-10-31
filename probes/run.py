"""Preserve raw probe/sanitizer evidence; never infer static labels from it."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", default="build/lifetime-probe")
    parser.add_argument("--out", default="results/probes")
    parser.add_argument("--sanitizer", default="compute-sanitizer")
    args = parser.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    records = []
    def run(name, command):
        try:
            r = subprocess.run(command, capture_output=True, text=True, timeout=90)
            result = {"name":name,"command":command,"returncode":r.returncode,"stdout":r.stdout,"stderr":r.stderr}
        except (OSError, subprocess.TimeoutExpired) as exc:
            result = {"name":name,"command":command,"status":"unavailable","error":str(exc)}
        records.append(result)
        (out/(name+'.json')).write_text(json.dumps(result,indent=2)+'\n')
        return result.get("returncode") == 0
    available = run("gpu", ["nvidia-smi"])
    if available:
        for mode in ("pinned", "pageable", "free", "free-host", "ordered-bad", "ordered-good"):
            run(mode, [args.binary, mode])
        for mode in ("ordered-bad", "ordered-good"):
            run("sanitizer-"+mode, [args.sanitizer,"--tool","memcheck","--track-stream-ordered-races","all","--error-exitcode","9",args.binary,mode])
    summary={"status":"executed" if available else "not-measured", "reason":None if available else "NVIDIA driver unavailable", "records":records,
             "limitations":"Observations apply only to the recorded GPU, driver, and executed path. No static labels are changed."}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return 0 if available else 2


if __name__ == "__main__":
    raise SystemExit(main())
