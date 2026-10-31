"""Compare predeclared mutant oracle sites without inventing coverage or rates."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

from .metrics import _coverage_files, _diagnostics, normalized_file


def compare(manifest, sarif, coverage, mutant_root, native_logs):
    root = Path(mutant_root).resolve()
    diagnostics, ignored = _diagnostics(sarif, root)
    analyzed, _ = _coverage_files(coverage, root)

    def telemetry_problems(value):
        problems = []
        for field in ("budgetExhausted", "unmodeledCudaCalls", "coverageUnknown"):
            if field not in value:
                problems.append("missing telemetry: " + field)
        for field in ("analyzed", "failed", "translationUnknown"):
            if field not in value.get("tus", {}):
                problems.append("missing telemetry: tus." + field)
        counts = value.get("unmodeledCudaCalls", {})
        if not isinstance(counts, dict) or any(type(count) is not int or count < 0 for count in counts.values()):
            raise ValueError("unmodeled CUDA call counts must be nonnegative integers")
        if "complete" in value and type(value["complete"]) is not bool:
            raise ValueError("coverage complete flag must be a boolean")
        return problems

    aggregate_problems = telemetry_problems(coverage)
    if coverage.get("complete") is False:
        aggregate_problems.append("aggregate coverage incomplete")
    per_tu = {}
    for path in Path(native_logs).glob("*.coverage.json"):
        native = json.loads(path.read_text())
        _coverage_files(native, root)
        telemetry_problems(native)
        for source in native.get("tus", {}).get("analyzed", []):
            source = normalized_file(source, root)
            if source in per_tu and per_tu[source] != native:
                raise ValueError("conflicting per-TU coverage; use an isolated log directory")
            per_tu[source] = native

    def complete(path):
        reasons = list(aggregate_problems)
        if path not in analyzed:
            reasons.append("TU not analyzed")
        for entries, description in ((coverage["tus"].get("failed", []), "analysis failed"),
                                     (coverage["tus"].get("translationUnknown", []), "translation unknown"),
                                     (coverage.get("budgetExhausted", []), "budget exhausted"),
                                     (coverage.get("coverageUnknown", []), "coverage unknown")):
            for entry in entries:
                filename = entry if isinstance(entry, str) else entry.get("file")
                if filename and normalized_file(filename, root) == path:
                    reasons.append(description)
        native = per_tu.get(path)
        if native is None:
            reasons.append("missing per-TU telemetry")
        else:
            reasons.extend(telemetry_problems(native))
            if native.get("complete") is False:
                reasons.append("native coverage incomplete")
            for field in ("failed", "translationUnknown", "skipped"):
                if native.get("tus", {}).get(field):
                    reasons.append("native TU " + field)
            if any(native.get("unmodeledCudaCalls", {}).values()):
                reasons.append("unmodeled CUDA calls")
            if native.get("coverageUnknown"):
                reasons.append("native coverage unknown")
            if native.get("budgetExhausted"):
                reasons.append("native budget exhausted")
        return not reasons, sorted(set(reasons)), (native or {}).get("unmodeledCudaCalls", {})

    rows = []
    for mutation in manifest["mutants"]:
        path = normalized_file(mutation["file"], root)
        baseline = normalized_file(mutation["baseline"], root)
        if sha256(Path(path).read_bytes()).hexdigest() != mutation["mutantSha256"]:
            raise ValueError("generated mutant changed after oracle definition: " + mutation["id"])
        if sha256(Path(baseline).read_bytes()).hexdigest() != mutation["sourceSha256"]:
            raise ValueError("baseline no longer matches the upstream source hash")
        covered, reasons, unmodeled = complete(path)
        baseline_covered, baseline_reasons, baseline_unmodeled = complete(baseline)
        observed = [result for result in diagnostics if result["file"] == path]
        baseline_observed = [result for result in diagnostics if result["file"] == baseline]
        site_results = []
        for site in mutation["oracle"]["sites"]:
            matching = [result for result in observed if result["ruleId"] == site["ruleId"]
                        and (site["expect"] == "none" or result["line"] == site["line"])]
            if site["expect"] == "none":
                status = "unexpected-finding" if matching else "verified-no-finding" if covered else "unverified-no-finding"
            elif any(result["level"] == site["level"] for result in matching):
                status = "observed-at-required-site"
            elif matching:
                status = "severity-mismatch"
            else:
                status = "missed-required-site" if covered else "unverified-miss"
            site_results.append({**site, "status": status, "matchingFindings": matching})
        rows.append({"id": mutation["id"], "category": mutation["category"],
                     "coverageComplete": covered, "coverageLimitations": reasons,
                     "unmodeledCudaCalls": unmodeled, "sites": site_results,
                     "allMutantFindings": observed, "baselineFindings": baseline_observed,
                     "baselineCoverageComplete": baseline_covered, "baselineCoverageLimitations": baseline_reasons,
                     "baselineUnmodeledCudaCalls": baseline_unmodeled,
                     "oracleLimitations": mutation["oracle"].get("coverageLimitations", [])})
    return {"schemaVersion": 1, "kind": "whole-source-mutation-comparison",
            "notAPrecisionBenchmark": True, "ignoredInformationalOrOtherRuleDiagnostics": ignored,
            "budget": coverage.get("budget"), "native": coverage.get("native"), "mutants": rows,
            "coverageComplete": all(row["coverageComplete"] and row["baselineCoverageComplete"] for row in rows)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--sarif", required=True)
    parser.add_argument("--coverage", required=True)
    parser.add_argument("--native-logs", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    try:
        manifest_path = Path(args.manifest)
        result = compare(json.loads(manifest_path.read_text()), json.loads(Path(args.sarif).read_text()),
                         json.loads(Path(args.coverage).read_text()), manifest_path.parent, args.native_logs)
        Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({"coverageComplete": result["coverageComplete"], "mutants": [
            {"id": row["id"], "coverageComplete": row["coverageComplete"],
             "sites": [site["status"] for site in row["sites"]]} for row in result["mutants"]]}, indent=2))
        return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f"mutation-report: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
