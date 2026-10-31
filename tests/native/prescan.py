#!/usr/bin/env python3
"""Native AST pre-scan check: which TUs are skipped as having no reachable CUDA calls.

Usage: prescan.py NATIVE. Each fixture in prescan/ is analyzed once; the
expectation is (skipped, diagnostics), where diagnostics None means "at least
one diagnostic or an explicit coverage gap".
"""
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent / "prescan"
EXPECTED = {
    "unused_header": (True, 0),
    "used_header": (False, 1),
    "local_destructor": (False, 0),
    "virtual_call": (False, 0),
    "macro_definition": (False, 1),
    "global_destructor": (False, 0),
    "template_instantiation": (False, None),
    "constructor_initializer": (False, 0),
    "member_destructor": (False, 0),
    "base_destructor": (False, 0),
    "global_initializer": (False, None),
    "static_local_destructor": (False, None),
    "header_global_destructor": (False, None),
    "host_template_static": (True, 0),
    "cuda_named_constructor": (True, 0),
    "project_cuda_names_only": (True, 0),
}
MODELED = {"constructor_initializer", "member_destructor", "base_destructor", "local_destructor"}


def main(native):
    failures = []
    with tempfile.TemporaryDirectory() as temp:
        for name, (skipped, diagnostics) in EXPECTED.items():
            sarif, coverage = Path(temp) / (name + ".sarif"), Path(temp) / (name + ".json")
            run = subprocess.run([native, str(ROOT / (name + ".cpp")), "--sarif", str(sarif),
                                  "--coverage", str(coverage), "--", "-std=c++17"],
                                 capture_output=True, text=True)
            if run.returncode:
                failures.append((name, "exit " + str(run.returncode), run.stderr[-500:]))
                continue
            cov = json.loads(coverage.read_text())
            count = sum(len(r["results"]) for r in json.loads(sarif.read_text())["runs"])
            problems = []
            if cov["astScan"]["skippedNoCudaCalls"] != skipped:
                problems.append("skip " + str(cov["astScan"]["skippedNoCudaCalls"]))
            if diagnostics is None:
                if not count and not cov["coverageUnknown"]:
                    problems.append("neither a diagnostic nor a coverage gap")
            elif count != diagnostics:
                problems.append(f"{count} diagnostics")
            if name in MODELED and not cov["modeledCallVisits"]:
                problems.append("no modeled CUDA call was visited")
            if name == "global_destructor" and not (cov["coverageUnknown"] or cov["modeledCallVisits"]):
                problems.append("global destruction neither modeled nor reported")
            if skipped and (cov["coverageUnknown"] or cov["unmodeledCudaCalls"]):
                problems.append("a skipped TU reports coverage gaps")
            if cov["tus"]["parsed"] != 1 or cov["tus"]["failed"]:
                problems.append("not parsed")
            if problems:
                failures.append((name, *problems))
    assert not failures, failures
    print(f"Pre-scan expectations passed for {len(EXPECTED)} fixtures")


if __name__ == "__main__":
    main(sys.argv[1])
