#!/usr/bin/env python3
"""Replay frozen source bytes with a rebuilt environment; never a first attempt."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ORIGINAL_ENVIRONMENT = frozenset({"build-assertions/cuda-lifetime-native",
    "results/holdout-dependencies.json", "results/environment/source-toolchain.json"})
SOURCE_SCOPES = {"checker": ("*.cpp", "*.h"), "tool": ("*.cpp", "*.h", "*.py"),
    "translate": ("*.py",), "eval": ("*.py",), "models": ("*.json",), "corpus/include": ("*.h",)}


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def local_file(value, root):
    if not isinstance(value, str) or not value:
        raise ValueError("freeze file must be a nonempty relative path")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts or str(relative) != value:
        raise ValueError("nonlocal or noncanonical freeze artifact: " + value)
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("freeze artifact escapes source root: " + value)
    return path


def verify_sources(freeze, root):
    root = Path(root).resolve()
    if (not isinstance(freeze, dict) or freeze.get("schemaVersion") != 1
            or freeze.get("split") != "holdout" or not freeze.get("frozenAt")
            or not isinstance(freeze.get("sources"), list) or not freeze["sources"]
            or not isinstance(freeze.get("rules"), list) or not freeze["rules"]):
        raise ValueError("invalid holdout freeze manifest")
    verified, external, seen = {}, [], set()
    for section, entries in (("labels", [freeze.get("labels")]),
                             ("sources", freeze["sources"]), ("rules", freeze["rules"])):
        for entry in entries:
            if (not isinstance(entry, dict) or not isinstance(entry.get("sha256"), str)
                    or not re.fullmatch(r"[0-9a-f]{64}", entry["sha256"])):
                raise ValueError("invalid frozen artifact hash")
            name = entry.get("file")
            path = local_file(name, root)
            if name in seen:
                raise ValueError("duplicate frozen artifact: " + name)
            seen.add(name)
            if name in ORIGINAL_ENVIRONMENT:
                if section != "rules":
                    raise ValueError("environment replacement is allowed only for rule records")
                external.append(entry)
                continue
            if digest(path) != entry["sha256"]:
                raise ValueError("frozen source changed: " + name)
            verified[name] = entry["sha256"]
    if {entry["file"] for entry in external} != ORIGINAL_ENVIRONMENT:
        raise ValueError("freeze must identify exactly the three original environment records")
    # Extra models or declaration headers change inputs despite unchanged hashes.
    for directory, patterns in SOURCE_SCOPES.items():
        current = {str(path.relative_to(root)) for pattern in patterns
                   for path in (root / directory).rglob(pattern) if path.is_file()}
        expected = {name for name in verified if name.startswith(directory + "/")
                    and any(Path(name).match(pattern) for pattern in patterns)}
        if current != expected:
            raise ValueError("source input inventory changed in " + directory)
    return verified, external


def header_files(root):
    """Follow header-directory symlinks while detecting loops and dangling links."""
    if not root.is_dir():
        raise ValueError("missing dependency root: " + str(root))
    files, visited, pending = set(), set(), [root]
    while pending:
        directory = pending.pop().resolve(strict=True)
        if directory in visited:
            continue
        visited.add(directory)
        for entry in directory.iterdir():
            target = entry.resolve(strict=True)
            if target.is_dir():
                pending.append(target)
            elif target.is_file():
                files.add(target)
            else:
                raise ValueError("unsupported dependency entry: " + str(entry))
    if not files:
        raise ValueError("empty dependency root: " + str(root))
    return files


def dependency_files(native, roots):
    native = Path(native).resolve(strict=True)
    if not roots:
        raise ValueError("at least one explicit header dependency root is required")
    result = subprocess.run(["ldd", str(native)], capture_output=True, text=True, check=True)
    if re.search(r"=>\s+not found\b", result.stdout):
        raise ValueError("unresolved shared library in ldd output: " + result.stdout.strip())
    paths = {native}
    for name in re.findall(r"(?:=>\s+)?(/[^\s()]+)", result.stdout):
        path = Path(name).resolve(strict=True)
        if not path.is_file():
            raise ValueError("shared library is not a file: " + name)
        paths.add(path)
    for root in roots:
        paths.update(header_files(Path(root)))
    return {str(path): digest(path) for path in sorted(paths)}, result.stdout


def run_analysis(*args, **kwargs):
    # Import the analyzer protocol only after its frozen source bytes verify.
    from tool.runner import analyze
    return analyze(*args, **kwargs)


def run_replay(native, freeze_path, output, dependency_roots, expected_freeze_sha256, root=ROOT):
    root, native = Path(root).resolve(), Path(native).resolve()
    freeze_path, output = Path(freeze_path).resolve(), Path(output).resolve()
    roots = [Path(path).resolve() for path in dependency_roots]
    if not re.fullmatch(r"[0-9a-f]{64}", expected_freeze_sha256):
        raise ValueError("expected freeze SHA256 must be 64 lowercase hexadecimal digits")
    if any(output.is_relative_to(path) for path in (root / "corpus", root / "results/holdout-first")):
        raise ValueError("replay output may not be placed in the corpus or original first-run directory")
    output.mkdir(parents=True, exist_ok=False)
    record = {"schemaVersion": 1, "kind": "rebuilt-environment-replay", "notFirstAttempt": True,
              "notBitIdenticalReproduction": True, "startedAt": datetime.now(timezone.utc).isoformat(),
              "freeze": str(freeze_path), "expectedFreezeSha256": expected_freeze_sha256,
              "replayProtocolSha256": digest(Path(__file__)), "pythonVersion": sys.version,
              "native": str(native), "dependencyRoots": [str(path) for path in roots],
              "dependencyScope": "native/shared libraries and explicitly listed header roots; not a full OS snapshot",
              "budget": {"maxNodes": 225000, "timeoutSeconds": 180}}
    write(output / "started.json", record)
    failures, metrics = [], None
    analysis_complete = False
    verified = dependencies = freeze = None
    stage = "preflight"
    try:
        record["freezeSha256"] = digest(freeze_path)
        if record["freezeSha256"] != expected_freeze_sha256:
            raise ValueError("freeze does not match the pinned first-run freeze SHA256")
        freeze = json.loads(freeze_path.read_text())
        verified, original_environment = verify_sources(freeze, root)
        record.update(verifiedSourceHashes=verified, originalEnvironmentRecords=original_environment)
        dependencies, ldd_output = dependency_files(native, roots)
        record.update(dependencyFiles=dependencies, ldd=ldd_output, nativeSha256=digest(native))
        labels = json.loads(local_file(freeze["labels"]["file"], root).read_text())
        if not isinstance(labels, dict) or labels.get("split") != "holdout" or not labels.get("cases"):
            raise ValueError("replay requires the frozen holdout labels")
        labelled = {case["file"] for case in labels["cases"]}
        if labelled != {entry["file"] for entry in freeze["sources"]}:
            raise ValueError("frozen and labelled source sets differ")
        for source in labelled:
            if local_file(source, root).suffix != ".cpp":
                raise ValueError("this replay protocol requires .cpp declaration-based fixtures")
        commands = [{"directory": str(root), "file": str(root / source),
                     "arguments": ["clang++", "-std=c++17", "-I" + str(root / "corpus/include"),
                                   "-c", str(root / source)]} for source in sorted(labelled)]
        write(output / "compile_commands.json", commands)
        stage = "analysis"
        coverage = run_analysis(output / "compile_commands.json", output / "out.sarif",
                                output / "coverage.json", native=str(native), model=str(root / "models"),
                                max_nodes=225000, timeout=180, include_contract=True)
        analysis_complete = coverage.get("complete") is True
        record["coverageSummary"] = coverage.get("summary")
        stage = "evaluation"
        from eval import evaluate
        metrics = evaluate(labels, json.loads((output / "out.sarif").read_text()), coverage, root)
    except Exception as exc:
        failures.append({"stage": stage, "type": type(exc).__name__, "message": str(exc)})

    sources_intact = dependencies_intact = False
    if verified is not None:
        try:
            after_sources, _ = verify_sources(freeze, root)
            sources_intact = verified == after_sources and digest(freeze_path) == expected_freeze_sha256
            if not sources_intact:
                raise ValueError("frozen inputs or freeze manifest changed during replay")
        except Exception as exc:
            failures.append({"stage": "post-source-verification", "type": type(exc).__name__, "message": str(exc)})
    if dependencies is not None:
        try:
            after_dependencies, after_ldd = dependency_files(native, roots)
            dependencies_intact = dependencies == after_dependencies
            record["afterDependencyFiles"] = after_dependencies
            record["afterLdd"] = after_ldd
            if not dependencies_intact:
                raise ValueError("rebuilt environment dependencies changed during replay")
        except Exception as exc:
            failures.append({"stage": "post-dependency-verification", "type": type(exc).__name__, "message": str(exc)})
    intact = sources_intact and dependencies_intact
    try:
        if digest(Path(__file__)) != record["replayProtocolSha256"]:
            raise ValueError("replay script changed during execution")
    except Exception as exc:
        failures.append({"stage": "post-protocol-verification", "type": type(exc).__name__, "message": str(exc)})
    metrics_recorded = False
    if metrics is not None:
        metrics.update(replayInputsUnchanged=intact, notFirstAttempt=True)
        metrics["passed"] = metrics["passed"] and intact and not failures
        try:
            write(output / "metrics.json", metrics)
            metrics_recorded = True
        except Exception as exc:
            failures.append({"stage": "metrics-persistence", "type": type(exc).__name__, "message": str(exc)})
    record.update(finishedAt=datetime.now(timezone.utc).isoformat(), inputsUnchanged=intact,
                  failures=failures, metricsRecorded=metrics_recorded,
                  analysisComplete=analysis_complete,
                  replaySucceeded=metrics_recorded and analysis_complete and intact and not failures,
                  passed=metrics_recorded and metrics["passed"] and not failures)
    write(output / "replay-result.json", record)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, default=ROOT / "build/cuda-lifetime-native")
    parser.add_argument("--freeze", type=Path, default=ROOT / "corpus/holdout/freeze.json")
    parser.add_argument("--expected-freeze-sha256", required=True,
                        help="SHA256 pinned outside the replay environment to the original first-run freeze")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dependency-root", type=Path, action="append", required=True)
    args = parser.parse_args(argv)
    try:
        result = run_replay(args.native, args.freeze, args.out, args.dependency_root, args.expected_freeze_sha256)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"replay: {exc}\n")
    print(json.dumps({key: result[key] for key in ("kind", "inputsUnchanged", "replaySucceeded", "passed", "failures")}, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
