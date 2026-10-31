#!/usr/bin/env python3
"""Join retained cuda-lifetime and baseline measurements by original command identity."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics


MODES = ("syntax", "analyze", "tidy")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def command_hash(entry):
    # Identical to runner.py and baseline.py; do not normalize before hashing.
    return hashlib.sha256(json.dumps(entry, sort_keys=True).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def source_path(entry, database):
    return str((Path(database).parent / entry["directory"] / entry["file"]).resolve())


def original_indices(original, scoped):
    """Recover original indices only when the retained JSON proves identity.

    Same complete ordered database supports duplicates. A filtered/reordered
    database requires unique hashes; ambiguous duplicate commands stay unknown.
    """
    if original == scoped:
        return [(i, None) for i in range(len(scoped))]
    by_hash = defaultdict(list)
    for i, entry in enumerate(original):
        by_hash[command_hash(entry)].append(i)
    answer = []
    for entry in scoped:
        candidates = by_hash[command_hash(entry)]
        answer.append((candidates[0], None) if len(candidates) == 1 else
                      (None, "ambiguous-original-index" if candidates else "command-not-in-original-database"))
    return answer


def metric_summary(values):
    values = [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)
              and math.isfinite(v) and v >= 0]
    if not values:
        return {"count": 0, "sum": None, "median": None, "p95": None, "max": None}
    ordered = sorted(values)
    return {"count": len(values), "sum": sum(values), "median": statistics.median(values),
            "p95": ordered[math.ceil(.95 * len(values)) - 1], "max": max(values)}


def valid_seconds(record):
    value = record.get("seconds")
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def native_parsed(coverage, source):
    tus = coverage.get("tus", {})
    return (type(tus.get("parsed")) is int and tus["parsed"] == 1 and
            tus.get("failed") == [] and tus.get("analyzed") == [source])


def coverage_fields(coverage):
    return {key: coverage.get(key) for key in
            ("tus", "budgetExhausted", "coverageUnknown", "unmodeledCudaCalls", "astScan",
             "modeledCallVisits", "kernelLaunches", "assumptions", "maxNodes")}


def compact_baseline(record):
    result = {k: v for k, v in record.items() if k != "diagnosticLines"}
    result["diagnosticLineCount"] = len(record.get("diagnosticLines", []))
    return result


def compare(rows, mode):
    matched = [r for r in rows if r["comparisons"][mode]["eligible"]]
    ratios = [r["cudaLifetime"]["seconds"] / r["baselines"][mode]["seconds"] for r in matched]
    checker = metric_summary([r["cudaLifetime"]["seconds"] for r in matched])
    base = metric_summary([r["baselines"][mode]["seconds"] for r in matched])
    # RSS is a per-process high-water mark, never an additive quantity.
    memory = {}
    for label, values in (("cudaLifetime", [r["cudaLifetime"].get("peakRssKiB") for r in matched]),
                          ("baseline", [r["baselines"][mode].get("peakRssKiB") for r in matched])):
        memory[label] = {k: v for k, v in metric_summary(values).items() if k != "sum"}
    return {"matchedSuccessfulCommands": len(matched), "cudaLifetimeSeconds": checker, "baselineSeconds": base,
            "cudaLifetimeOverBaselineWallRatio": {k: v for k, v in metric_summary(ratios).items() if k != "sum"},
            "ratioOfSummedWallSeconds": checker["sum"] / base["sum"] if base["sum"] else None,
            "peakRssKiB": memory,
            "excludedReasons": dict(Counter(reason for row in rows
                                              for reason in row["comparisons"][mode]["reasons"]))}


def summarize_target(cuda_lifetime_root, baseline_root, target, manifest):
    root = Path(cuda_lifetime_root) / target
    database = root / "input/compile_commands.json"
    original_database = root / "input/original-compile_commands.json"
    original = read(original_database)
    entries = read(database)
    mapping = original_indices(original, entries)
    cov_path = root / "coverage.json"
    cov = read(cov_path) if cov_path.exists() else {}
    metadata = next((t for t in manifest.get("targets", []) if t["target"] == target), {})
    profiles = defaultdict(list)
    for profile in cov.get("profiles", []):
        profiles[profile.get("index")].append(profile)
    expected_db = metadata.get("inputSha256", {}).get("compile_commands.json")
    input_verified = expected_db is not None and expected_db == sha256(database)
    original_verified = metadata.get("inputSha256", {}).get("original-compile_commands.json") == sha256(original_database)
    run_verified = (bool(manifest.get("completedAt")) and
                    all(manifest.get(k) is True for k in ("snapshotUnchanged", "pythonUnchanged",
                                                         "observedSnapshotUnchanged", "observedPythonUnchanged")) and
                    metadata.get("inputChanges") == [] and metadata.get("translationUnitChanges") == [])
    binary_verified = (cov.get("native", {}).get("sha256") == manifest.get("nativeSha256") and
                       manifest.get("observedNativeUnchanged") is True and
                       not any(item.get("reason") == "native-binary-changed-during-run"
                               for item in cov.get("coverageUnknown", [])))
    baseline_maps, baseline_metadata, unmatched = {}, {}, {}
    for mode in MODES:
        report_path = Path(baseline_root) / target / mode / "report.json"
        if not report_path.exists():
            baseline_maps[mode] = {}; baseline_metadata[mode] = {"status": "missing", "path": str(report_path)}
            unmatched[mode] = []
            continue
        report = read(report_path)
        baseline_original_path = report_path.parent / "original-compile_commands.json"
        baseline_original = read(baseline_original_path)
        baseline_mapping = original_indices(original, baseline_original)
        index_map, rejected = defaultdict(list), []
        for record in report.get("records", []):
            index = record.get("compilationCommandIndex")
            if type(index) is not int or not 0 <= index < len(baseline_original):
                rejected.append({"reason": "invalid-record-index", "record": compact_baseline(record)})
                continue
            orig_index, error = baseline_mapping[index]
            expected = command_hash(baseline_original[index])
            if error or record.get("inputCommandSha256") != expected:
                rejected.append({"reason": error or "record-command-hash-mismatch", "record": compact_baseline(record)})
                continue
            index_map[(orig_index, expected)].append(compact_baseline(record))
        baseline_maps[mode] = index_map
        baseline_metadata[mode] = {"status": "available", "path": str(report_path), "sha256": sha256(report_path),
                                   **{k: report.get(k) for k in ("summary", "jobs", "toolVersion", "executableSha256",
                                                                  "compdbUnchanged", "executableUnchanged", "warningCount")},
                                   "retainedDatabaseVerified": sha256(baseline_original_path) == report.get("compdbSha256")}
        unmatched[mode] = rejected
    rows, used = [], {mode: set() for mode in MODES}
    for index, entry in enumerate(entries):
        source = source_path(entry, database)
        digest = command_hash(entry)
        orig_index, identity_error = mapping[index]
        row = {"file": source, "scopeIndex": index, "originalIndex": orig_index,
               "inputCommandSha256": digest, "identityError": identity_error, "baselines": {}, "comparisons": {}}
        candidates = profiles[index]
        profile = candidates[0] if len(candidates) == 1 else {}
        log_key = f"{index:04d}-" + hashlib.sha256(source.encode()).hexdigest()[:10]
        native_path = root / "out.logs" / (log_key + ".coverage.json")
        native = read(native_path) if native_path.exists() else {}
        errors = []
        if identity_error: errors.append(identity_error)
        if not run_verified: errors.append("cuda-lifetime-run-integrity-unverified")
        if not input_verified or not original_verified: errors.append("cuda-lifetime-input-integrity-unverified")
        if not binary_verified: errors.append("cuda-lifetime-binary-integrity-unverified")
        if len(candidates) != 1: errors.append("missing-or-duplicate-cuda-lifetime-profile")
        elif profile.get("inputCommandSha256") != digest or profile.get("file") != source:
            errors.append("cuda-lifetime-profile-identity-mismatch")
        parsed = native_parsed(native, source)
        if not parsed: errors.append("cuda-lifetime-not-proven-parsed")
        if not valid_seconds(profile): errors.append("cuda-lifetime-invalid-time")
        expected_source = metadata.get("translationUnitSha256", {}).get(source)
        source_verified = (expected_source is not None and Path(source).is_file()
                           and sha256(source) == expected_source
                           and source not in metadata.get("translationUnitChanges", []))
        if not source_verified: errors.append("cuda-lifetime-source-integrity-unverified")
        failures = [f for f in cov.get("tus", {}).get("failed", []) if f.get("file") == source]
        row["cudaLifetime"] = {**profile, "parsed": parsed, "status": "parsed" if parsed else "failed-or-unavailable",
                       "nativeCoverage": str(native_path) if native_path.exists() else None,
                       "coverage": coverage_fields(native), "fileLevelFailures": failures,
                       "sourceSha256": expected_source, "sourceUnchangedAtSummary": source_verified}
        for mode in MODES:
            reasons = list(errors)
            key = (orig_index, digest)
            matches = baseline_maps[mode].get(key, [])
            record = matches[0] if len(matches) == 1 else None
            row["baselines"][mode] = record
            if len(matches) != 1:
                reasons.append("missing-or-duplicate-baseline-command")
            else:
                used[mode].add(key)
                if record.get("status") != "passed" or record.get("returncode") != 0:
                    reasons.append("baseline-" + record.get("status", "unknown"))
                if not valid_seconds(record): reasons.append("baseline-invalid-time")
                if not all(record.get(k) is True for k in ("sourceUnchanged", "inputDatabaseUnchanged")):
                    reasons.append("baseline-record-integrity-unverified")
                bm = baseline_metadata[mode]
                if not all(bm.get(k) is True for k in ("compdbUnchanged", "executableUnchanged", "retainedDatabaseVerified")):
                    reasons.append("baseline-run-integrity-unverified")
                if not record.get("sourceSha256") or record["sourceSha256"] != row["cudaLifetime"]["sourceSha256"]:
                    reasons.append("source-content-mismatch-or-unverified")
            row["comparisons"][mode] = {"eligible": not reasons, "reasons": reasons}
        rows.append(row)
    for mode in MODES:
        for key, records in baseline_maps[mode].items():
            if key not in used[mode]:
                unmatched[mode].extend({"reason": "no-unique-matching-cuda-lifetime-command", "record": r} for r in records)
    return {"target": target, "status": "completed" if cov else "cuda-lifetime-report-pending",
            "runIntegrityVerified": run_verified,
            "input": {"path": str(database), "sha256": sha256(database), "verified": input_verified,
                      "originalPath": str(original_database), "originalVerified": original_verified},
            "cudaLifetimeSummary": cov.get("summary"), "cudaLifetimeBudget": cov.get("budget"),
            "cudaLifetimeCoverage": {k: cov.get(k) for k in ("tus", "budgetExhausted", "coverageUnknown", "unmodeledCudaCalls")},
            "baselines": baseline_metadata, "comparisons": {mode: compare(rows, mode) for mode in MODES},
            "allThreeBaselinesAndCudaLifetimeSuccessful": sum(all(r["comparisons"][m]["eligible"] for m in MODES) for r in rows),
            "rows": rows, "unmatchedBaselineRecords": unmatched}


def build_report(cuda_lifetime_root, baseline_root, expected_native_sha256=None):
    cuda_lifetime_root, baseline_root = Path(cuda_lifetime_root).resolve(), Path(baseline_root).resolve()
    manifest = read(cuda_lifetime_root / "manifest.json")
    native = cuda_lifetime_root / "cuda-lifetime-native"
    manifest["observedNativeUnchanged"] = native.is_file() and sha256(native) == manifest.get("nativeSha256")
    snapshots = manifest.get("snapshotSha256", {})
    manifest["observedSnapshotUnchanged"] = bool(snapshots) and all(
        (cuda_lifetime_root / name).is_file() and sha256(cuda_lifetime_root / name) == expected for name, expected in snapshots.items())
    python = manifest.get("python", {})
    python_path = Path(python.get("executable", ""))
    manifest["observedPythonUnchanged"] = python_path.is_file() and sha256(python_path) == python.get("sha256")
    return {"schemaVersion": 1, "createdAt": datetime.now(timezone.utc).isoformat(),
            "cudaLifetimeManifest": {"path": str(cuda_lifetime_root / "manifest.json"), "sha256": sha256(cuda_lifetime_root / "manifest.json"),
                            "nativeSha256": manifest.get("nativeSha256"), "jobs": manifest.get("jobs"),
                            "completedAt": manifest.get("completedAt"),
                            **{k: manifest.get(k) for k in ("snapshotUnchanged", "pythonUnchanged", "observedSnapshotUnchanged",
                                                            "observedPythonUnchanged", "observedNativeUnchanged")}},
            "expectedNativeSha256": expected_native_sha256,
            "expectedCandidateMatch": (manifest.get("nativeSha256") == expected_native_sha256
                                        if expected_native_sha256 else None),
            "method": "Exact retained command SHA256 plus recovered original database index; per-TU native parsed evidence; source hashes equal; pairwise successful intersection.",
            "limitations": ["Non-randomized runs under concurrent external workloads; wall ratios are exploratory, not precise speed claims.",
                            "Per-TU wall time includes driver/process startup; RSS is each process high-water mark, not aggregate memory.",
                            "Parsed does not mean exhaustive: budget exhaustion, unknown coverage and unmodeled calls remain in each row.",
                            "Baseline warningCount includes header diagnostics and is not a count of CL1–CL4 defects.",
                            "SDKs, transitive headers and dynamic libraries were external to input snapshots."],
            "targets": [summarize_target(cuda_lifetime_root, baseline_root, t["target"], manifest)
                        for t in manifest.get("targets", [])]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda-lifetime", type=Path, required=True)
    parser.add_argument("--baselines", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-native-sha256")
    args = parser.parse_args()
    report = build_report(args.cuda_lifetime, args.baselines, args.expected_native_sha256)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({t["target"]: {m: t["comparisons"][m]["matchedSuccessfulCommands"] for m in MODES}
                      for t in report["targets"]}))
    return 0 if report["expectedCandidateMatch"] is not False and all(
        t["status"] == "completed" and t["runIntegrityVerified"] for t in report["targets"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
