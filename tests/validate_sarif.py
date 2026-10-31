#!/usr/bin/env python3
"""Independent OASIS schema and source-path checks on saved native SARIF."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

import jsonschema

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eval.metrics import _artifact_file  # noqa: E402


def validate(document, schema, root):
    failures = ["schema " + "/".join(str(v) for v in error.absolute_path) + ": " + error.message
                for error in jsonschema.Draft4Validator(schema, format_checker=jsonschema.FormatChecker()).iter_errors(document)]
    counts = {"diagnostics": 0, "traceLocations": 0, "queueEvidence": 0, "physicalLocations": 0}
    if failures:
        return failures, counts
    for run_index, run in enumerate(document["runs"] or []):
        rules = run["tool"]["driver"].get("rules", [])

        def locations(value):
            if isinstance(value, list):
                for child in value:
                    locations(child)
            elif isinstance(value, dict):
                physical = value.get("physicalLocation")
                if physical:
                    counts["physicalLocations"] += 1
                    try:
                        artifact = physical["artifactLocation"]
                        filename = _artifact_file(artifact, run, root)
                        if "index" in artifact:
                            index = artifact["index"]
                            indexed = _artifact_file(run["artifacts"][index]["location"], run, root)
                            if filename != indexed:
                                raise ValueError("artifact URI/index disagree")
                        source = Path(filename)
                        region = physical.get("region", {})
                        if not source.is_file():
                            raise ValueError("source file does not exist: " + filename)
                        length = len(source.read_text(errors="replace").splitlines())
                        start, end = region.get("startLine", 1), region.get("endLine", region.get("startLine", 1))
                        if not (1 <= start <= end <= max(length, 1)):
                            raise ValueError("source range is outside file: " + filename)
                        if start == end and region.get("endColumn", region.get("startColumn", 1)) < region.get("startColumn", 1):
                            raise ValueError("endColumn precedes startColumn")
                    except (IndexError, KeyError, TypeError, ValueError) as exc:
                        failures.append(f"run {run_index} source location: {exc}")
                for child in value.values():
                    locations(child)

        for index, result in enumerate(run.get("results", [])):
            counts["diagnostics"] += 1
            if "ruleIndex" in result:
                try:
                    rule = rules[result["ruleIndex"]]
                    if "ruleId" in result and result["ruleId"] != rule["id"]:
                        raise ValueError("ruleId/ruleIndex disagree")
                except (IndexError, KeyError, ValueError) as exc:
                    failures.append(f"run {run_index} result {index}: {exc}")
            threads = [thread for flow in result.get("codeFlows", []) for thread in flow.get("threadFlows", [])]
            if not threads or any(not thread.get("locations") for thread in threads):
                failures.append(f"run {run_index} result {index}: missing CSA diagnostic path")
            for thread in threads:
                for location in thread.get("locations", []):
                    counts["traceLocations"] += 1
                    message = location.get("location", {}).get("message", {}).get("text", "")
                    counts["queueEvidence"] += "queued here" in message
            locations(result)
    return failures, counts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sarif", action="append", required=True)
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    schema_path = ROOT / "tests/schema/sarif-schema-2.1.0.json"
    schema = json.loads(schema_path.read_text())
    reports = []
    for filename in args.sarif:
        failures, counts = validate(json.loads(Path(filename).read_text()), schema, args.root)
        reports.append({"file": str(Path(filename).resolve()), "failures": failures, **counts})
    report = {"kind": "independent-oasis-schema-and-source-path-validation",
              "schemaSha256": sha256(schema_path.read_bytes()).hexdigest(), "artifacts": reports,
              "passed": all(not row["failures"] for row in reports)}
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
