#!/usr/bin/env python3
"""Publish a real-project scan: repository-relative SARIF, per-TU timings and summary.json.

The scan directory holds one scripts/analyze_parallel.py output per project
(SCAN/<project>/out.sarif and coverage.json). The triage CSV must have exactly
one row per alert of the scan, matched by project, file, line and rule; the
optional previous triage gives the alert counts of an earlier version; it may
use the 0.2.0 triage's columns (target, judgment) instead of (project, category).

    python3 scripts/summarize_scan.py --scan results/raw/scan-0.4.0 \\
      --triage results/real-projects/triage.csv \\
      --previous-triage results/real-projects/triage-0.2.0.csv --previous-version 0.2.0 \\
      --out results/real-projects
"""
import argparse
import collections
import csv
import json
from pathlib import Path
import statistics
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parents[1]
PROJECTS = {"cuda-samples": ".deps/cuda-samples-13.2/", "faiss": ".deps/faiss-1.15.1/", "rmm": ".deps/rmm-26.10.00/"}


def relative(value, root):
    """A file:// URI or absolute path inside root, as a root-relative path; anything else unchanged."""
    if not isinstance(value, str):
        return value
    path = unquote(urlparse(value).path) if value.startswith("file://") else value
    try:
        return Path(path).resolve().relative_to(root).as_posix()
    except (ValueError, OSError):
        return value


def function(name):
    """The function of an unmodeled-call key such as rmm::detail::memcpy_async:unknown-copy-kind."""
    head, separator, tail = name.rpartition(":")
    return head if separator and tail and not head.endswith(":") and "-" in tail else name


def relativize(node, root):
    if isinstance(node, dict):
        return {key: relative(value, root) if key == "uri" else relativize(value, root) for key, value in node.items()}
    if isinstance(node, list):
        return [relativize(value, root) for value in node]
    return node


def alerts(document, project):
    prefix = PROJECTS[project]
    found = collections.Counter()
    for run in document["runs"]:
        artifacts = run.get("artifacts", [])
        for result in run.get("results", []):
            location = result["locations"][0]["physicalLocation"]
            uri = location["artifactLocation"].get("uri") or artifacts[location["artifactLocation"]["index"]]["location"]["uri"]
            if not uri.startswith(prefix):
                raise ValueError(f"{project}: alert outside the project source tree: {uri}")
            found[(project, uri[len(prefix):], location["region"]["startLine"], result["ruleId"], result["level"])] += 1
    return found


def read_triage(path):
    rows = list(csv.DictReader(path.open(newline="")))
    return rows, collections.Counter((r["project"], r["file"], int(r["line"]), r["rule"], r["level"]) for r in rows)


def seconds(profiles):
    values = sorted(p["seconds"] for p in profiles if isinstance(p.get("seconds"), (int, float)))
    if not values:
        return None
    return {"sum": round(sum(values), 1), "median": round(statistics.median(values), 2),
            "p95": round(values[min(len(values) - 1, int(0.95 * len(values)))], 2), "max": round(values[-1], 1)}


def summarize(scan, triage_path, out, previous_path=None, root=ROOT, previous_version=None):
    scan, out, root = Path(scan), Path(out), Path(root).resolve()
    rows, triaged = read_triage(Path(triage_path))
    found, coverage, timings, version = collections.Counter(), {}, [], None
    for project in PROJECTS:
        document = relativize(json.loads((scan / project / "out.sarif").read_text()), root)
        cov = json.loads((scan / project / "coverage.json").read_text())
        found += alerts(document, project)
        drivers = {run["tool"]["driver"].get("version") for run in document["runs"] if run.get("results")}
        version = version or next((v.split()[0] for v in drivers if v), None)
        (out / f"{project}.sarif").write_text(json.dumps(document, indent=2) + "\n")
        exhausted = collections.Counter(relative(entry["file"], root) for entry in cov["budgetExhausted"] if "function" in entry)
        for profile in cov["profiles"]:
            file = relative(profile["file"], root)
            timings.append({"project": project, "file": file, "seconds": round(profile["seconds"], 3),
                            "peakRssKiB": profile.get("peakRssKiB"), "functionsAnalyzed": profile.get("functionsAnalyzed"),
                            "functionsOverBudget": exhausted.get(file, 0)})
        names = sorted({function(name) for name in cov["unmodeledCudaCalls"]})
        coverage[project] = {
            "translationUnits": cov["tus"]["total"], "parsed": cov["tus"]["parsed"],
            "failed": sorted(relative(entry["file"], root) for entry in cov["tus"]["failed"]),
            "functionsAnalyzed": cov["visitCounts"].get("functionsAnalyzed"),
            "functionsOverBudget": sum(exhausted.values()),
            "wallClockTimeouts": sum("function" not in entry for entry in cov["budgetExhausted"]),
            "coverageGaps": len(cov["coverageUnknown"]),
            "unmodeledCudaFunctions": names,
            "wallSeconds": round(cov.get("orchestration", {}).get("wallSeconds", 0), 1),
            "jobs": cov.get("orchestration", {}).get("jobs"),
            "tuSeconds": seconds(cov["profiles"]),
        }
    if found != triaged:
        missing, extra = sorted(found - triaged), sorted(triaged - found)
        raise ValueError(f"triage does not match the scan: untriaged {missing[:5]}, not in scan {extra[:5]}")
    with (out / "timings.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(timings[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(timings)
    summary = {"version": version, "budget": json.loads((scan / "faiss" / "coverage.json").read_text())["budget"],
               "alerts": {p: sum(n for k, n in found.items() if k[0] == p) for p in PROJECTS},
               "alertsByCategory": dict(sorted(collections.Counter(r["category"] for r in rows).items())),
               "alertsByLevel": dict(sorted(collections.Counter(r["level"] for r in rows).items())),
               "coverage": coverage}
    if previous_path:
        previous = list(csv.DictReader(Path(previous_path).open(newline="")))
        projects = [r.get("project") or r["target"] for r in previous]
        categories = [r.get("category") or r["judgment"] for r in previous]
        summary["previous"] = {"version": previous_version,
                               "alerts": {p: projects.count(p) for p in PROJECTS},
                               "alertsByCategory": dict(sorted(collections.Counter(categories).items()))}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scan", type=Path, required=True)
    parser.add_argument("--triage", type=Path, required=True)
    parser.add_argument("--previous-triage", type=Path)
    parser.add_argument("--previous-version", help="version label of the previous triage")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    try:
        summary = summarize(args.scan, args.triage, args.out, args.previous_triage, previous_version=args.previous_version)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(2, f"summarize_scan: {error}\n")
    print(json.dumps({key: summary[key] for key in ("version", "alerts", "alertsByCategory")}))


if __name__ == "__main__":
    main()
