#!/usr/bin/env python3
"""One recorded attempt on already frozen labels; this command never freezes them."""

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval import EvaluationError, evaluate, verify_frozen  # noqa: E402
from eval.metrics import normalized_file  # noqa: E402
from tool.runner import analyze  # noqa: E402


def digest(path):
    value = sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def verify_dependencies(manifest_path):
    manifest = json.loads(Path(manifest_path).read_text())
    if not isinstance(manifest, dict) or manifest.get("schemaVersion") != 1:
        raise EvaluationError("invalid dependency manifest version")
    entries = manifest.get("files")
    if not isinstance(entries, list) or not entries:
        raise EvaluationError("dependency manifest must list actual files")
    files = {}
    for entry in entries:
        if (not isinstance(entry, dict) or not isinstance(entry.get("file"), str)
                or not Path(entry["file"]).is_absolute()
                or not isinstance(entry.get("sha256"), str)
                or not re.fullmatch(r"[0-9a-f]{64}", entry["sha256"])):
            raise EvaluationError("dependency entry requires absolute file and SHA256")
        path = str(Path(entry["file"]).resolve())
        if path in files:
            raise EvaluationError("duplicate dependency file: " + path)
        if digest(path) != entry["sha256"]:
            raise EvaluationError("dependency changed: " + path)
        files[path] = entry["sha256"]
    return files


def required_rule_files(root, native):
    """Require the actual executable and all code that can change these results."""
    root = Path(root)
    files = {Path(native).resolve(), root / "CMakeLists.txt", root / "environment.lock", root / "docs/semantics.md",
             Path(__file__).resolve()}
    for directory, patterns in (("checker", ("*.cpp", "*.h")), ("tool", ("*.cpp", "*.h", "*.py")),
                                ("translate", ("*.py",)), ("models", ("*.json",)),
                                ("eval", ("*.py",)), ("corpus/include", ("*.h",))):
        for pattern in patterns:
            files.update((root / directory).rglob(pattern))
    return {str(path.resolve()) for path in files}


def write_once(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def run_frozen(labels_path, freeze_path, root, native, output, max_nodes=225000, timeout=180,
               required_files=None, dependencies_path=None):
    root, native, output = Path(root).resolve(), Path(native).resolve(), Path(output).resolve()
    labels_path, freeze_path = Path(labels_path).resolve(), Path(freeze_path).resolve()
    manifest = verify_frozen(labels_path, freeze_path, root)
    labels = json.loads(labels_path.read_text())
    if labels.get("split") != "holdout":
        raise EvaluationError("this harness requires a holdout-labelled corpus")
    if dependencies_path is None:
        raise EvaluationError("holdout execution requires a dependency manifest")
    dependencies_path = Path(dependencies_path).resolve()
    required = required_rule_files(root, native) if required_files is None else set(required_files) | {str(native)}
    required.add(str(dependencies_path))
    frozen = {normalized_file(entry["file"], root) for entry in manifest["rules"]}
    if not required <= frozen:
        raise EvaluationError("freeze omits required rule/executable files: " + ", ".join(sorted(required - frozen)))
    if not native.is_file():
        raise EvaluationError("frozen native executable is missing")
    dependencies = verify_dependencies(dependencies_path)
    source_files = sorted({normalized_file(case["file"], root) for case in labels["cases"]})
    if any(Path(source).suffix != ".cpp" for source in source_files):
        raise EvaluationError("this holdout protocol accepts .cpp fixtures with the frozen CUDA declarations")
    if output.exists():
        raise EvaluationError("holdout output directory must not exist")
    freeze_hash, native_hash = digest(freeze_path), digest(native)
    started = {"schemaVersion": 1, "kind": "first-frozen-holdout-attempt",
               "startedAt": datetime.now(timezone.utc).isoformat(), "freeze": str(freeze_path),
               "freezeSha256": freeze_hash, "nativeSha256": native_hash, "output": str(output),
               "budget": {"maxNodes": max_nodes, "timeoutSeconds": timeout},
               "labelsSha256": digest(labels_path), "dependenciesManifest": str(dependencies_path),
               "dependenciesManifestSha256": digest(dependencies_path),
               "dependencyFiles": dependencies}
    # The marker belongs to the freeze, not the chosen output directory. A crash
    # or incomplete analysis consumes the first attempt and must remain visible.
    marker = freeze_path.with_name(freeze_path.name + ".first-run.json")
    try:
        write_once(marker, started)
    except FileExistsError as exc:
        raise EvaluationError("the frozen corpus already has a first-run attempt") from exc
    output.mkdir(parents=True, exist_ok=False)
    commands = [{"directory": str(root), "file": source,
                 "arguments": ["clang++", "-std=c++17", "-I" + str(root / "corpus/include"), "-c", source]}
                for source in source_files]
    write_once(output / "compile_commands.json", commands)
    failures, metrics = [], None
    try:
        coverage = analyze(output / "compile_commands.json", output / "out.sarif", output / "coverage.json",
                           native=str(native), model=str(root / "models"), max_nodes=max_nodes,
                           timeout=timeout, include_contract=True)
        metrics = evaluate(labels, json.loads((output / "out.sarif").read_text()), coverage, root)
    except Exception as exc:
        failures.append(type(exc).__name__ + ": " + str(exc))
    try:
        verify_frozen(labels_path, freeze_path, root)
        if digest(freeze_path) != freeze_hash:
            failures.append("freeze manifest changed during the run")
    except (OSError, ValueError) as exc:
        failures.append("post-run freeze verification failed: " + str(exc))
    try:
        verified = verify_dependencies(dependencies_path)
        if verified != dependencies or digest(dependencies_path) != started["dependenciesManifestSha256"]:
            failures.append("dependency manifest changed during the run")
    except (OSError, ValueError) as exc:
        failures.append("post-run dependency verification failed: " + str(exc))
    if metrics is not None:
        metrics["freezeVerifiedBeforeAndAfter"] = not failures
        metrics["passed"] = metrics["passed"] and not failures
        write_once(output / "metrics.json", metrics)
    evidence = {**started, "finishedAt": datetime.now(timezone.utc).isoformat(),
                "attemptFailures": failures, "metricsRecorded": metrics is not None,
                "passed": metrics is not None and metrics["passed"]}
    write_once(output / "first-result.json", evidence)
    return evidence


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--freeze", required=True)
    parser.add_argument("--dependencies", required=True,
                        help="frozen manifest of actual external library/header file hashes")
    parser.add_argument("--out", required=True)
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--max-nodes", type=int, default=225000)
    parser.add_argument("--timeout", type=float, default=180)
    args = parser.parse_args(argv)
    try:
        evidence = run_frozen(args.labels, args.freeze, args.root, args.native, args.out,
                              args.max_nodes, args.timeout, dependencies_path=args.dependencies)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"holdout: {exc}\n")
    print(json.dumps(evidence, indent=2))
    return 0 if evidence["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
