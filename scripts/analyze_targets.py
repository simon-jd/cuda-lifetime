#!/usr/bin/env python3
"""Analyze fixed target scopes using a copied native, Python driver and models."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
TARGETS = ("cuda-samples", "faiss", "rmm")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", nargs="+", choices=TARGETS, default=list(TARGETS))
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--toolchain-kind", choices=["bootstrap", "source-assertions", "container-assertions"], required=True)
    parser.add_argument("--max-nodes", type=int, default=225000)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--jobs", type=int, default=3, help="simultaneous target runs; each target uses one TU process")
    args = parser.parse_args()
    if min(args.max_nodes, args.timeout, args.jobs) <= 0:
        parser.error("budgets must be positive")
    if len(args.targets) != len(set(args.targets)):
        parser.error("targets must not repeat")
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = args.out / "manifest.json"
    if any(args.out.iterdir()):
        parser.error("use an empty output directory to preserve the existing run")
    native = args.out / "cuda-lifetime-native"
    shutil.copy2(args.native.resolve(), native)
    shutil.copytree(ROOT / "models", args.out / "models")
    driver = args.out / "driver"
    driver.mkdir()
    for package in ("tool", "translate", "eval"):
        shutil.copytree(ROOT / package, driver / package,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copy2(ROOT / "cuda-lifetime", driver / "cuda-lifetime")
    # The native executable can precede current source edits. Preserve the actual
    # bytes available now without pretending this is a binary build attestation.
    source_snapshot = args.out / "source-snapshot"
    source_snapshot.mkdir()
    rule_paths = [*sorted((ROOT / "tool").glob("*.cpp")),
                  *sorted((ROOT / "tool").glob("*.h")), *sorted((ROOT / "checker").glob("*")),
                  ROOT / "CMakeLists.txt", ROOT / "scripts/analyze_targets.py", ROOT / "environment.lock"]
    for path in rule_paths:
        if path.is_file():
            destination = source_snapshot / path.relative_to(ROOT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
    driver_environment = {**os.environ, "PYTHONPATH": str(driver), "PYTHONNOUSERSITE": "1"}
    try:
        version = subprocess.run([str(native), "--version"], capture_output=True, text=True, timeout=30)
        native_version = {"returncode": version.returncode, "stdout": version.stdout, "stderr": version.stderr}
    except (OSError, subprocess.TimeoutExpired) as error:
        native_version = {"error": str(error)}
    python_path = Path(sys.executable).resolve()
    record = {"createdAt": datetime.now(timezone.utc).isoformat(), "toolchainKind": args.toolchain_kind,
              "nativeSha256": sha256(native),
              "nativeSourcePath": str(args.native.resolve()), "nativeVersion": native_version,
              "python": {"executable": str(python_path), "version": sys.version, "sha256": sha256(python_path)},
              "driverDirectory": str(driver),
              "driverEnvironment": {"PYTHONPATH": str(driver), "PYTHONNOUSERSITE": "1"},
              "compilerEnvironment": {key: os.environ[key] for key in
                                      ("PATH", "LD_LIBRARY_PATH", "CUDA_PATH", "CUDA_HOME", "CPATH",
                                       "CPLUS_INCLUDE_PATH", "C_INCLUDE_PATH", "GCC_EXEC_PREFIX",
                                       "COMPILER_PATH", "LIBRARY_PATH") if key in os.environ},
              "snapshotSha256": {str(path.relative_to(args.out)): sha256(path)
                                 for folder in (driver, source_snapshot, args.out / "models")
                                 for path in sorted(folder.rglob("*")) if path.is_file()},
              "sourceSnapshotRelation": "working-tree source at invocation; not proof of the native binary's build source",
              "externalInputs": "TU files and input databases are hashed; SDKs, shared libraries, transitive headers and generated includes remain external and are not snapshotted",
              "scope": "all configured compilation commands; faiss restricted to faiss/gpu/ including generated CUDA TUs",
              "jobs": args.jobs,
              "notRuntimeEvidence": True, "targets": []}
    write(manifest, record)
    for target in args.targets:
        source = ROOT / "targets" / target / "clang"
        database_bytes = (source / "compile_commands.json").read_bytes()
        translation_bytes = (source / "translation-report.json").read_bytes()
        entries = json.loads(database_bytes)
        translation = json.loads(translation_bytes)
        selected = lambda item: target != "faiss" or "/faiss/gpu/" in item.get("file", "")
        inputs = args.out / target / "input"
        inputs.mkdir(parents=True)
        (inputs / "original-compile_commands.json").write_bytes(database_bytes)
        (inputs / "original-translation-report.json").write_bytes(translation_bytes)
        chosen = [entry for entry in entries if selected(entry)]
        for entry in chosen:
            entry["directory"] = str((source / entry["directory"]).resolve())
        translation["entries"] = [entry for entry in translation["entries"] if selected(entry)]
        translation["summary"] = {"total": len(translation["entries"]),
                                  "translated": len(chosen),
                                  "translationUnknown": sum(entry["status"] == "translation-unknown"
                                                            for entry in translation["entries"])}
        write(inputs / "compile_commands.json", chosen)
        write(inputs / "translation-report.json", translation)
        output = inputs.parent
        command = [sys.executable, "-m", "tool.cli", "analyze", "--compdb", str(inputs),
                   "--native", str(native), "--model", str(args.out / "models"),
                   "--sarif", str(output / "out.sarif"), "--coverage", str(output / "coverage.json"),
                   "--max-nodes", str(args.max_nodes), "--timeout", str(args.timeout), "--include-contract"]
        record["targets"].append({"target": target, "commandsInOriginalDatabase": len(entries),
                                  "commandsInScope": len(chosen), "command": command,
                                  "workingDirectory": str(driver),
                                  "sourceInputDirectory": str(source),
                                  "inputSha256": {path.name: sha256(path) for path in sorted(inputs.glob("*.json"))},
                                  "translationUnitSha256": {
                                      str((Path(entry["directory"]) / entry["file"]).resolve()):
                                      sha256((Path(entry["directory"]) / entry["file"]).resolve())
                                      if (Path(entry["directory"]) / entry["file"]).is_file() else None
                                      for entry in chosen}})
    write(manifest, record)

    def analyze(target):
        with (args.out / target["target"] / "analyze.log").open("w") as log:
            process = subprocess.run(target["command"], cwd=driver, env=driver_environment,
                                     stdout=log, stderr=subprocess.STDOUT)
        target["returncode"] = process.returncode
        target["inputChanges"] = [name for name, expected in target["inputSha256"].items()
                                  if not (args.out / target["target"] / "input" / name).is_file()
                                  or sha256(args.out / target["target"] / "input" / name) != expected]
        target["translationUnitChanges"] = [path for path, expected in target["translationUnitSha256"].items()
                                            if not Path(path).is_file() or sha256(Path(path)) != expected]
        print(f"{target['target']}: exit {process.returncode}", flush=True)
        return target

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        record["targets"] = list(pool.map(analyze, record["targets"]))
    record["completedAt"] = datetime.now(timezone.utc).isoformat()
    record["pythonUnchanged"] = python_path.is_file() and sha256(python_path) == record["python"]["sha256"]
    record["snapshotUnchanged"] = sha256(native) == record["nativeSha256"] and all(
        (args.out / path).is_file() and sha256(args.out / path) == expected
        for path, expected in record["snapshotSha256"].items())
    write(manifest, record)
    return 0 if record["snapshotUnchanged"] and record["pythonUnchanged"] and all(
        target["returncode"] == 0 and not target["translationUnitChanges"] and not target["inputChanges"]
        for target in record["targets"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
