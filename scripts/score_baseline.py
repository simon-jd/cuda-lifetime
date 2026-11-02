#!/usr/bin/env python3
"""Score a scripts/baseline.py report against a labelled corpus.

The baselines (Clang's stock analyzer, clang-tidy) have no CL1-CL4 rules, so
the score ignores rules: a positive label is a true positive when the
baseline has any warning on its file and line, and a negative label is a
false positive when it has one there. Warnings elsewhere in labelled files
are counted separately.

    python3 scripts/score_baseline.py --labels corpus/seed/labels.json \\
      --report results/baselines/seed-csa/report.json --out results/baselines/seed-csa.json
"""
import argparse
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
WARNING = re.compile(r"^(.+?):(\d+):\d+: warning: (.*)$")


def warnings(report):
    """(repository-relative file, line, text) of every warning in the report's records."""
    found = []
    for record in report["records"]:
        for line in record.get("diagnosticLines", []):
            match = WARNING.match(line)
            if match:
                path = Path(match.group(1))
                if not path.is_absolute():
                    path = Path(record["workingDirectory"]) / path
                try:
                    path = path.resolve().relative_to(ROOT)
                except ValueError:
                    pass
                found.append((str(path), int(match.group(2)), match.group(3)))
    return found


def score(labels, report):
    found = warnings(report)
    at = {(path, line) for path, line, _ in found}
    totals = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    cases = []
    for case in labels["cases"]:
        hit = (case["file"], case["line"]) in at
        outcome = ("tp" if hit else "fn") if case["expect"] != "none" else ("fp" if hit else "tn")
        totals[outcome] += 1
        cases.append({"id": case["id"], "expect": case["expect"], "outcome": outcome})
    labelled = {(case["file"], case["line"]) for case in labels["cases"]}
    files = {case["file"] for case in labels["cases"]}
    return {"schemaVersion": 1, "mode": report["mode"], "tool": report["toolVersion"].get("stdout", "").splitlines()[:1],
            "matching": "rule-agnostic: any warning on a label's file and line",
            "status": report["summary"], "totals": totals,
            "warnings": len(found),
            "offLabelWarnings": sorted({f"{p}:{n}: {text}" for p, n, text in found if p in files and (p, n) not in labelled}),
            "cases": cases}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = score(json.loads(args.labels.read_text()), json.loads(args.report.read_text()))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"totals": result["totals"], "status": result["status"], "warnings": result["warnings"]}))


if __name__ == "__main__":
    main()
