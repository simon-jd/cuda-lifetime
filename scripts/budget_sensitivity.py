#!/usr/bin/env python3
"""Predeclare a seed/project subset, then measure one candidate at four budgets."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

if __package__:
    from .profile_report import command_hash, coverage_fields, native_parsed, original_indices, read, sha256, source_path
else:
    from profile_report import command_hash, coverage_fields, native_parsed, original_indices, read, sha256, source_path

ROOT = Path(__file__).resolve().parents[1]
BUDGETS = (25000, 100000, 225000, 450000)


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def now():
    return datetime.now(timezone.utc).isoformat()


def reject_holdout(path):
    if any("holdout" in part.lower() for part in Path(path).resolve().parts):
        raise ValueError("budget sensitivity must not read or run holdout inputs")


def select_seed(database, labels_path):
    reject_holdout(database); reject_holdout(labels_path)
    labels = read(labels_path)
    if labels.get("split") != "seed":
        raise ValueError("only explicitly marked seed labels are allowed")
    entries = read(database)
    by_source = {}
    for index, entry in enumerate(entries):
        source = source_path(entry, database)
        by_source.setdefault(source, []).append((index, entry))
    chosen = []
    for rule in ("CL1", "CL2", "CL3", "CL4"):
        for expect in ("warning", "none"):
            case = next((case for case in labels["cases"] if case["ruleId"] == rule and case["expect"] == expect), None)
            if not case:
                raise ValueError(f"missing seed stratum {rule}/{expect}")
            source = str((ROOT / case["file"]).resolve())
            reject_holdout(source)
            matches = by_source.get(source, [])
            if len(matches) != 1:
                raise ValueError(f"seed command not unique: {source}")
            index, entry = matches[0]
            chosen.append({"group": "seed", "selection": {"ruleId": rule, "expect": expect, "caseId": case["id"]},
                           "database": str(Path(database).resolve()), "originalIndex": index,
                           "scopeIndex": index, "entry": entry, "file": source})
    return chosen


def select_target(cuda_lifetime_root, target):
    root = Path(cuda_lifetime_root) / target
    database = root / "input/compile_commands.json"
    entries = read(database)
    original = read(root / "input/original-compile_commands.json")
    mapping = original_indices(original, entries)
    examined = []
    for index, entry in enumerate(entries):
        import hashlib
        source = source_path(entry, database)
        reject_holdout(source)
        key = f"{index:04d}-" + hashlib.sha256(source.encode()).hexdigest()[:10]
        evidence = root / "out.logs" / (key + ".coverage.json")
        if not evidence.is_file():
            continue
        coverage = read(evidence)
        original_index, error = mapping[index]
        if not error and native_parsed(coverage, source) and coverage.get("modeledCallVisits", 0) > 0:
            return {"group": target, "selection": {"rule": "first retained parsed command in database order with modeledCallVisits > 0",
                                                     "evidence": str(evidence.resolve()), "evidenceSha256": sha256(evidence),
                                                     "selectionNativeSha256": read(Path(cuda_lifetime_root) / "manifest.json")["nativeSha256"]},
                    "database": str(database.resolve()), "originalIndex": original_index,
                    "scopeIndex": index, "entry": entry, "file": source}, None
        examined.append(index)
    return None, {"target": target, "reason": "no retained unambiguously indexed parsed TU with modeledCallVisits > 0",
                  "examinedCoverageIndices": examined}


def create_plan(out, native, seed_compdb, labels, cuda_lifetime_root, models, expected_sha, timeout=180):
    out = Path(out).resolve()
    reject_holdout(seed_compdb); reject_holdout(labels); reject_holdout(cuda_lifetime_root)
    if Path(out).exists() and any(out.iterdir()):
        raise ValueError("use an empty output directory to preserve prior evidence")
    if sha256(native) != expected_sha:
        raise ValueError("candidate native SHA256 does not equal the requested immutable candidate")
    selected = select_seed(Path(seed_compdb).resolve(), Path(labels).resolve())
    skipped = []
    for target in ("cuda-samples", "faiss", "rmm"):
        entry, reason = select_target(cuda_lifetime_root, target)
        if entry:
            selected.append(entry)
        else:
            skipped.append(reason)
    out.mkdir(parents=True, exist_ok=True)
    copied_native = out / "cuda-lifetime-native"
    shutil.copy2(native, copied_native)
    shutil.copytree(models, out / "models", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy2(__file__, out / "budget_sensitivity.py")
    shutil.copy2(Path(__file__).with_name("profile_report.py"), out / "profile_report.py")
    inputs = out / "inputs"; inputs.mkdir()
    input_records = []
    for number, selected_entry in enumerate(selected):
        record = dict(selected_entry)
        source = record["file"]
        record["id"] = f"{number:02d}-{record['group']}-{Path(source).stem}"
        record["sourceSha256"] = sha256(source)
        record["inputCommandSha256"] = command_hash(record["entry"])
        record["sourceDatabaseSha256"] = sha256(record["database"])
        single = inputs / record["id"]; single.mkdir()
        entry = {**record["entry"], "file": source,
                 "directory": str((Path(record["database"]).parent / record["entry"]["directory"]).resolve())}
        path = single / "compile_commands.json"
        write(path, [entry])
        record["isolatedDatabase"] = str(path)
        record["isolatedDatabaseSha256"] = sha256(path)
        input_records.append(record)
    runs = []
    for budget in BUDGETS:
        for record in input_records:
            run_dir = out / "runs" / str(budget) / record["id"]
            command = [str(copied_native), "-p", str(Path(record["isolatedDatabase"]).parent), record["file"],
                       "--sarif", str(run_dir / "out.sarif"), "--coverage", str(run_dir / "coverage.json"),
                       "--model", str(out / "models"), "--max-nodes", str(budget), "--include-contract"]
            runs.append({"id": record["id"], "maxNodes": budget, "command": command,
                         "workingDirectory": str(ROOT), "outputDirectory": str(run_dir)})
    manifest = {"schemaVersion": 1, "createdAt": now(), "state": "predeclared",
                "candidate": {"source": str(Path(native).resolve()), "path": str(copied_native), "sha256": expected_sha},
                "budgets": list(BUDGETS), "timeoutSeconds": timeout, "jobs": 1,
                "selectionRule": "Seed label order: first warning and none case for CL1–CL4. Each project: first already-parsed command with modeledCallVisits > 0 in retained database order. No outcome-driven replacement.",
                "selectionLabels": {"path": str(Path(labels).resolve()), "sha256": sha256(labels)},
                "selectionCudaLifetimeManifest": {"path": str((Path(cuda_lifetime_root) / "manifest.json").resolve()),
                                         "sha256AtSelection": sha256(Path(cuda_lifetime_root) / "manifest.json")},
                "inputs": input_records, "skippedTargets": skipped, "runs": runs,
                "snapshotSha256": {str(p.relative_to(out)): sha256(p) for folder in (out / "models", inputs)
                                   for p in sorted(folder.rglob("*")) if p.is_file()},
                "environment": {k: os.environ[k] for k in ("PATH", "LD_LIBRARY_PATH", "CUDA_HOME", "CUDA_PATH", "CPATH", "CPLUS_INCLUDE_PATH", "C_INCLUDE_PATH") if k in os.environ},
                "limitations": ["Small deterministic development subset, not holdout or project-wide accuracy evidence.",
                                "One trial per point, fixed ascending budget order, external concurrent load; timings are exploratory, not precise speed comparisons.",
                                "Source files are hashed; transitive headers, SDKs and dynamic libraries remain external.",
                                "Parsed success does not imply complete path exploration; all coverage gaps are retained."],
                "notRuntimeEvidence": True}
    manifest["snapshotSha256"].update({name: sha256(out / name) for name in ("budget_sensitivity.py", "profile_report.py")})
    write(out / "manifest.json", manifest)
    return manifest


def integrity_errors(manifest, out):
    failures = []
    paths = {manifest["candidate"]["path"]: manifest["candidate"]["sha256"]}
    paths.update({str(out / name): digest for name, digest in manifest["snapshotSha256"].items()})
    paths.update({record["file"]: record["sourceSha256"] for record in manifest["inputs"]})
    for path, expected in paths.items():
        if not Path(path).is_file() or sha256(path) != expected:
            failures.append(path)
    return failures


def run_manifest(path):
    path = Path(path).resolve(); out = path.parent
    manifest_hash = sha256(path); manifest = read(path)
    if (out / "report.json").exists() or (out / "runs").exists():
        raise ValueError("refusing to overwrite or resume a started experiment; preserve it and declare a new run")
    if manifest.get("budgets") != list(BUDGETS):
        raise ValueError("unexpected predeclared budgets")
    for entry in manifest["inputs"]:
        reject_holdout(entry["file"])
    errors = integrity_errors(manifest, out)
    if errors:
        raise ValueError("pre-run input integrity mismatch: " + ", ".join(errors))
    inputs = {item["id"]: item for item in manifest["inputs"]}
    report = {"schemaVersion": 1, "startedAt": now(), "manifest": str(path), "manifestSha256": manifest_hash,
              "nativeSha256": manifest["candidate"]["sha256"], "limitations": manifest["limitations"],
              "records": [], "complete": False}
    write(out / "report.json", report)
    for run in manifest["runs"]:
        output = Path(run["outputDirectory"]); output.mkdir(parents=True)
        entry = inputs[run["id"]]
        record = {**run, "file": entry["file"], "originalIndex": entry["originalIndex"],
                  "inputCommandSha256": entry["inputCommandSha256"], "sourceSha256": entry["sourceSha256"],
                  "startedAt": now(), "loadAverageBefore": list(os.getloadavg()), "log": str(output / "stderr.log")}
        rss = output / "rss.txt"
        command = ["/usr/bin/time", "-f", "%M", "-o", str(rss), "--", *run["command"]] if Path("/usr/bin/time").is_file() else run["command"]
        record["measurementCommand"] = command
        started = time.monotonic()
        try:
            with (output / "stderr.log").open("w") as log:
                proc = subprocess.Popen(command, cwd=run["workingDirectory"], stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    record["returncode"] = proc.wait(timeout=manifest["timeoutSeconds"])
                    record["status"] = "exited" if record["returncode"] == 0 else "failed"
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL); proc.wait()
                    record.update(status="timeout", returncode=proc.returncode)
        except OSError as exc:
            record.update(status="unavailable", error=str(exc))
        record.update(seconds=time.monotonic() - started, loadAverageAfter=list(os.getloadavg()), finishedAt=now())
        values = [line for line in rss.read_text().splitlines() if line.isdecimal()] if rss.exists() else []
        record["peakRssKiB"] = int(values[-1]) if values else None
        cov_path = output / "coverage.json"
        cov = {}
        if cov_path.is_file():
            try:
                cov = read(cov_path)
                record["coverage"] = coverage_fields(cov)
                record["nativeSeconds"] = cov.get("seconds")
                record["nativePeakRssKiB"] = cov.get("peakRssKiB")
            except (ValueError, AttributeError) as error:
                record["coverageError"] = str(error)
        record["parsed"] = record["status"] == "exited" and native_parsed(cov, entry["file"])
        if record["status"] == "exited":
            record["status"] = "parsed" if record["parsed"] else "invalid-or-failed-coverage"
        sarif = output / "out.sarif"
        record["diagnostics"] = None
        if sarif.is_file():
            try:
                record["diagnostics"] = sum(len(r.get("results", [])) for r in read(sarif).get("runs", []))
            except (ValueError, AttributeError):
                record["sarifError"] = "invalid-json-or-shape"
        record["sourceUnchanged"] = sha256(entry["file"]) == entry["sourceSha256"]
        record["isolatedDatabaseUnchanged"] = sha256(entry["isolatedDatabase"]) == entry["isolatedDatabaseSha256"]
        write(output / "record.json", record)
        report["records"].append(record)
        write(out / "report.json", report)
        print(f"{run['maxNodes']} {run['id']}: {record['status']} {record['seconds']:.3f}s", flush=True)
    report.update(finishedAt=now(), integrityErrors=integrity_errors(manifest, out), manifestUnchanged=sha256(path) == manifest_hash)
    report["complete"] = len(report["records"]) == len(manifest["runs"]) and not report["integrityErrors"] and report["manifestUnchanged"]
    report["summary"] = {str(budget): {"statuses": dict(Counter(r["status"] for r in report["records"] if r["maxNodes"] == budget)),
                                      "budgetExhausted": sum(len((r.get("coverage") or {}).get("budgetExhausted") or []) for r in report["records"] if r["maxNodes"] == budget),
                                      "coverageUnknown": sum(len((r.get("coverage") or {}).get("coverageUnknown") or []) for r in report["records"] if r["maxNodes"] == budget)}
                         for budget in BUDGETS}
    write(out / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    plan = sub.add_parser("plan")
    for name in ("out", "native", "seed-compdb", "labels", "cuda-lifetime", "models"):
        plan.add_argument("--" + name, type=Path, required=True)
    plan.add_argument("--expected-native-sha256", required=True)
    plan.add_argument("--timeout", type=float, default=180)
    run = sub.add_parser("run")
    run.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.action == "plan":
            if args.timeout <= 0:
                parser.error("timeout must be positive")
            report = create_plan(args.out, args.native, args.seed_compdb, args.labels, args.cuda_lifetime, args.models,
                                 args.expected_native_sha256, args.timeout)
            print(json.dumps({"inputs": len(report["inputs"]), "runs": len(report["runs"]), "skippedTargets": report["skippedTargets"]}))
            return 0
        report = run_manifest(args.manifest)
        return 0 if report["complete"] and all(r["status"] == "parsed" for r in report["records"]) else 2
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
