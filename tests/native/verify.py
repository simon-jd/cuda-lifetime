#!/usr/bin/env python3
"""Check one native analysis against the expected diagnostics annotated in its source.

Usage: verify.py SOURCE SARIF COVERAGE [options]. Every source line with a
comment "// CLn error|warning|note" expects exactly one such diagnostic on that
line, and there must be no others. The TU must be parsed and fully explored,
with no unmodeled CUDA calls unless --unmodeled names them.
"""
import argparse
import json
from pathlib import Path
import re
import sys

ANNOTATION = re.compile(r"// (CL[1-4]) (error|warning|note)")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("sarif", type=Path)
    parser.add_argument("coverage", type=Path)
    parser.add_argument("--no-coverage-gaps", action="store_true", help="require an empty coverageUnknown")
    parser.add_argument("--coverage-gap", action="append", default=[], help="require a coverage gap mentioning TEXT")
    parser.add_argument("--unmodeled", action="append", default=[], help="the exact unmodeled CUDA calls expected")
    parser.add_argument("--functions", type=int, help="the number of top-level functions analyzed")
    parser.add_argument("--default-stream", choices=("legacy", "per-thread"))
    args = parser.parse_args(argv)
    expected = sorted((m.group(1), m.group(2), number)
                      for number, line in enumerate(args.source.read_text().splitlines(), 1)
                      if (m := ANNOTATION.search(line)))
    results = [r for run in json.loads(args.sarif.read_text())["runs"] for r in run["results"]]
    actual = sorted((r["ruleId"], r["level"], r["locations"][0]["physicalLocation"]["region"]["startLine"])
                    for r in results)
    cov = json.loads(args.coverage.read_text())
    problems = []
    if actual != expected:
        problems.append(f"diagnostics {actual} != expected {expected}")
    for r in results:
        if not r.get("codeFlows"):
            problems.append(f"line {r['locations'][0]['physicalLocation']['region']['startLine']}: no CSA path")
        if not all(r.get("properties", {}).get(key) for key in ("memoryKind", "semanticId")):
            problems.append(f"{r['ruleId']}: missing memoryKind or semanticId")
    if cov["tus"]["parsed"] != 1 or cov["tus"]["failed"]:
        problems.append("TU not parsed")
    if cov["budgetExhausted"]:
        problems.append(f"budget exhausted: {cov['budgetExhausted']}")
    if sorted(cov["unmodeledCudaCalls"]) != sorted(args.unmodeled):
        problems.append(f"unmodeled CUDA calls {sorted(cov['unmodeledCudaCalls'])} != {sorted(args.unmodeled)}")
    reasons = [entry["reason"] for entry in cov["coverageUnknown"]]
    if args.no_coverage_gaps and reasons:
        problems.append(f"coverage gaps: {reasons}")
    for text in args.coverage_gap:
        if not any(text in reason for reason in reasons):
            problems.append(f"no coverage gap mentions {text!r}: {reasons}")
    if args.functions is not None and cov["functionsAnalyzed"] != args.functions:
        problems.append(f"{cov['functionsAnalyzed']} functions analyzed, expected {args.functions}")
    if args.default_stream and cov["defaultStreamMode"] != args.default_stream:
        problems.append(f"default stream {cov['defaultStreamMode']}, expected {args.default_stream}")
    for problem in problems:
        print(f"{args.source.name}: {problem}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
