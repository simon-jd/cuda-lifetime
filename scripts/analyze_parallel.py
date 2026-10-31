#!/usr/bin/env python3
"""Isolated phase orchestration over the unchanged per-TU analysis CLI.

This schedules compilation-command occurrences, not unique source paths. Its
wall times include concurrent scheduling and are not independent performance
measurements. Real project runs require the phase's separate input/profile gates.
"""
import argparse
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
COUNTERS = ("modeledCallVisits", "pointerEscapes", "pendingMemberCallUses",
            "pendingArgumentUses", "pendingPointerEscapes")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8") as out:
        json.dump(value, out, indent=2, ensure_ascii=False)
        out.write("\n")
    os.replace(out.name, path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def entry_sha(entry):
    # Same JSON encoding as the existing runner's inputCommandSha256.
    return hashlib.sha256(json.dumps(entry, sort_keys=True).encode()).hexdigest()


def identity(path, ignore_cache=False):
    path = Path(path)
    if not path.exists():
        return {"missing": True}
    if path.is_file():
        return {"sha256": sha(path), "bytes": path.stat().st_size}
    files = {}
    for child in sorted(path.rglob("*")):
        if ignore_cache and ("__pycache__" in child.parts or child.suffix == ".pyc"):
            continue
        if child.is_symlink():
            raise ValueError(f"snapshot inputs must not contain symlinks: {child}")
        if child.is_file():
            files[str(child.relative_to(path))] = {"sha256": sha(child), "bytes": child.stat().st_size}
    return {"files": files}


def now():
    return datetime.now(timezone.utc).isoformat()


def empty_coverage(max_nodes, timeout):
    return {"schemaVersion": 1, "tus": {"total": 0, "parsed": 0, "analyzed": [],
            "failed": [], "translationUnknown": []}, "budgetExhausted": [],
            "unmodeledCudaCalls": {}, "coverageUnknown": [], "profiles": [],
            "visitCounts": {name: 0 for name in COUNTERS},
            "assumptions": ["single host thread", "single translation unit", "bounded CSA exploration"],
            "budget": {"maxNodes": max_nodes, "timeoutSeconds": timeout},
            "complete": False, "suppressedContractDiagnostics": 0}


def finish_coverage(cov, runs):
    cov["complete"] = not any((cov["tus"]["failed"], cov["tus"]["translationUnknown"],
                               cov["budgetExhausted"], cov["coverageUnknown"], cov["unmodeledCudaCalls"]))
    cov["summary"] = {"parsed": cov["tus"]["parsed"], "failed": len(cov["tus"]["failed"]),
        "translationUnknown": len(cov["tus"]["translationUnknown"]),
        "budgetExhausted": len(cov["budgetExhausted"]),
        "unmodeledCudaCalls": sum(cov["unmodeledCudaCalls"].values()),
        "coverageUnknown": len(cov["coverageUnknown"]),
        "diagnostics": sum(len(run["results"]) for run in runs), "complete": cov["complete"]}
    if not runs:
        runs.append({"tool": {"driver": {"name": "cuda-lifetime", "rules": []}}, "results": []})
    runs[0].setdefault("properties", {})["coverage"] = copy.deepcopy(cov["summary"])
    return {"version": "2.1.0", "$schema": "https://json.schemastore.org/sarif-2.1.0.json", "runs": runs}


class Children:
    def __init__(self):
        self.lock = threading.Lock()
        self.active = set()
        self.cancelled = False

    def start(self, command, **kwargs):
        with self.lock:
            if self.cancelled:
                raise RuntimeError("orchestration cancelled before worker start")
            process = subprocess.Popen(command, **kwargs)
            self.active.add(process)
            return process

    def forget(self, process):
        with self.lock:
            self.active.discard(process)

    def cancel(self):
        with self.lock:
            self.cancelled = True
            for process in self.active:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass


def invoke(command, directory, environment, log_path, timeout, children=None):
    """An outer guard also kills the CLI's process group if its own guard fails."""
    started = time.monotonic()
    with Path(log_path).open("w") as log:
        children = children or Children()
        process = children.start(command, cwd=directory, env=environment,
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait(timeout=timeout)
        except BaseException as exc:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            exc.returncode = process.returncode
            raise
        finally:
            children.forget(process)
    return {"returncode": code, "seconds": time.monotonic() - started}


def normalize_report(raw, original_directory):
    """Keep invalid reports invalid; relocate only well-typed directory fields."""
    try:
        document = json.loads(raw)
    except (ValueError, UnicodeError):
        return raw
    if not isinstance(document, dict) or not isinstance(document.get("entries"), list):
        return raw
    for entry in document["entries"]:
        if isinstance(entry, dict):
            directory = entry.get("directory", str(original_directory))
            if isinstance(directory, str) and "\0" not in directory:
                entry["directory"] = str((original_directory / directory).resolve())
    return (json.dumps(document, indent=2) + "\n").encode()


def preceding_assumptions(raw, original_directory):
    """Recover runner's local prefix when a later malformed row aborts parsing.

    runner does not export translationAssumptions in that failure case, but still
    attaches the successfully read prefix to diagnostics. Keep the report error;
    this projection does not turn a malformed report into valid coverage.
    """
    assumptions = {}
    try:
        report = json.loads(raw)
        entries = report["entries"]
        if not isinstance(entries, list) or not all(isinstance(row, dict) for row in entries):
            return assumptions
        if any("directory" in row and (not isinstance(row["directory"], str) or "\0" in row["directory"]) for row in entries):
            return assumptions
        # runner canonicalizes every unknown path before collecting assumptions.
        for row in entries:
            if row.get("status") == "translation-unknown" and isinstance(row.get("file"), str):
                (original_directory / row.get("directory", str(original_directory)) / row["file"]).resolve()
        for row in entries:
            if isinstance(row.get("file"), str):
                name = str((original_directory / row.get("directory", str(original_directory)) / row["file"]).resolve())
                values = row.get("assumptions", [])
                if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
                    break
                assumptions.setdefault(name, []).extend(values)
    except (ValueError, TypeError, KeyError, OSError):
        pass
    return assumptions


def normalized_entry(entry, directory):
    child = copy.deepcopy(entry)
    if isinstance(child, dict) and isinstance(child.get("directory"), str) and "\0" not in child["directory"]:
        child["directory"] = str((directory / child["directory"]).resolve())
    return child


def source_name(entry, directory, index):
    if isinstance(entry, dict) and isinstance(entry.get("file"), str) and entry["file"] and "\0" not in entry["file"]:
        work = entry.get("directory")
        if isinstance(work, str) and "\0" not in work:
            return str((directory / work / entry["file"]).resolve())
    return f"entry:{index}"


def validate_output(cov, document, total, expected_source=None):
    if not isinstance(cov, dict) or not isinstance(cov.get("tus"), dict):
        raise ValueError("worker coverage must contain tus")
    tus = cov["tus"]
    if type(tus.get("total")) is not int or tus["total"] != total:
        raise ValueError("worker coverage total does not match assigned scope")
    if type(tus.get("parsed")) is not int or not 0 <= tus["parsed"] <= total:
        raise ValueError("invalid worker parsed count")
    for name in ("analyzed", "failed", "translationUnknown"):
        if not isinstance(tus.get(name), list):
            raise ValueError("invalid worker tus." + name)
    if tus["parsed"] != len(tus["analyzed"]):
        raise ValueError("worker parsed/analyzed counts disagree")
    if expected_source is not None and any(name != expected_source for name in tus["analyzed"]):
        raise ValueError("worker coverage identifies a different source")
    for name in ("budgetExhausted", "coverageUnknown", "profiles", "assumptions"):
        if not isinstance(cov.get(name), list):
            raise ValueError("invalid worker " + name)
    if not isinstance(cov.get("budget"), dict):
        raise ValueError("invalid worker budget")
    if any(not isinstance(profile, dict) or type(profile.get("index")) is not int for profile in cov["profiles"]):
        raise ValueError("invalid worker profile")
    for name in ("unmodeledCudaCalls", "visitCounts"):
        if not isinstance(cov.get(name), dict) or any(type(n) is not int or n < 0 for n in cov[name].values()):
            raise ValueError("invalid worker " + name)
    if type(cov.get("suppressedContractDiagnostics")) is not int or cov["suppressedContractDiagnostics"] < 0:
        raise ValueError("invalid worker suppression count")
    if not isinstance(document, dict) or document.get("version") != "2.1.0" or not isinstance(document.get("runs"), list) or not document["runs"]:
        raise ValueError("invalid worker SARIF")
    for run in document["runs"]:
        if not isinstance(run, dict) or not isinstance(run.get("results"), list) or not run.get("tool", {}).get("driver", {}).get("name"):
            raise ValueError("invalid worker SARIF run")
    check = copy.deepcopy(cov)
    finish_coverage(check, copy.deepcopy(document["runs"]))
    if cov.get("complete") is not check["complete"] or cov.get("summary") != check["summary"]:
        raise ValueError("worker coverage summary is inconsistent")


def execute_task(task, environment, driver, outer_timeout, children=None):
    record = copy.deepcopy(task)
    path = Path(record["directory"])
    record.update(state="running", startedAt=now())
    write_json(path / "task.json", record)
    try:
        record.update(invoke(record["command"], driver, environment, path / "worker.log", outer_timeout, children))
        cov = json.loads((path / "coverage.json").read_text())
        document = json.loads((path / "out.sarif").read_text())
        validate_output(cov, document, 1, record["file"])
        if cov["tus"]["translationUnknown"]:
            raise ValueError("translationUnknown must only be counted by the parent")
        if record["returncode"] != (0 if cov["complete"] else 2):
            raise ValueError("worker return code disagrees with its coverage")
        record["state"] = "completed" if cov["complete"] else "completed-incomplete"
        record["coverageSummary"] = cov["summary"]
        record["outputSha256"] = {name: sha(path / name) for name in ("coverage.json", "out.sarif", "worker.log")}
    except subprocess.TimeoutExpired as exc:
        record.update(state="exception", returncode=getattr(exc, "returncode", None),
                      failure={"reason": "worker-wall-clock-timeout", "timeoutSeconds": outer_timeout})
    except Exception as exc:
        record.update(state="exception", failure={"reason": "worker-output-or-execution-failed", "detail": str(exc), "exceptionType": type(exc).__name__})
    if record["state"] == "exception":
        with (path / "worker.log").open("a") as log:
            log.write("\n[orchestrator] " + json.dumps(record["failure"]) + "\n")
    record["completedAt"] = now()
    write_json(path / "task.json", record)
    return record


def aggregate(base, tasks):
    """Reduce in original command order; never flatten run-local SARIF arrays."""
    cov = copy.deepcopy(base)
    cov["tus"]["total"] += len(tasks)
    runs = []
    for task in sorted(tasks, key=lambda task: task["index"]):
        index = task["index"]
        if task["state"] not in ("completed", "completed-incomplete"):
            failure = {"file": task["file"], "index": index,
                       **task.get("failure", {"reason": "task-not-completed"}),
                       "task": str(Path(task["directory"]) / "task.json")}
            cov["tus"]["failed"].append(failure)
            if failure["reason"] == "worker-wall-clock-timeout":
                cov["budgetExhausted"].append({"file": task["file"], "index": index, "reason": "orchestration-wall-clock-timeout"})
            continue
        path = Path(task["directory"])
        try:
            if any(not (path / name).is_file() or sha(path / name) != expected for name, expected in task.get("outputSha256", {}).items()):
                raise ValueError("worker output changed after task completion")
            child = json.loads((path / "coverage.json").read_text())
            document = json.loads((path / "out.sarif").read_text())
            validate_output(child, document, 1, task["file"])
            if child["budget"] != cov["budget"] or child["assumptions"] != cov["assumptions"]:
                raise ValueError("worker analysis policy differs from parent")
            child_runs = copy.deepcopy(document["runs"])
            child_runs[0].get("properties", {}).pop("coverage", None)
            # runner.finish emits this placeholder only when it received no native runs.
            placeholder = [{"tool": {"driver": {"name": "cuda-lifetime", "rules": []}},
                            "results": [], "properties": {}}]
            if child_runs != placeholder:
                runs.extend(child_runs)
            cov["tus"]["parsed"] += child["tus"]["parsed"]
            cov["tus"]["analyzed"].extend(child["tus"]["analyzed"])
            for failure in copy.deepcopy(child["tus"]["failed"]):
                if isinstance(failure, dict) and failure.get("file") == "entry:0":
                    failure["file"] = f"entry:{index}"
                cov["tus"]["failed"].append(failure)
            for field in ("budgetExhausted", "coverageUnknown"):
                cov[field].extend(child[field])
            for field in ("unmodeledCudaCalls", "visitCounts"):
                for key, value in child[field].items():
                    cov[field][key] = cov[field].get(key, 0) + value
            cov["suppressedContractDiagnostics"] += child["suppressedContractDiagnostics"]
            for profile in copy.deepcopy(child["profiles"]):
                profile.update(index=index, inputCommandSha256=task["entrySha256"])
                cov["profiles"].append(profile)
        except Exception as exc:
            cov["tus"]["failed"].append({"file": task["file"], "index": index,
                "reason": "invalid-completed-task-output", "detail": str(exc)})
    return cov, runs


def analyze_parallel(compdb, out, native, model, jobs=2, timeout=180, max_nodes=225000,
                     include_contract=True, driver_root=ROOT):
    if type(jobs) is not int or jobs <= 0 or type(max_nodes) is not int or max_nodes <= 0 or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("jobs, max-nodes and timeout must be positive and finite")
    out = Path(out).resolve()
    compdb = Path(compdb).resolve()
    if compdb.is_dir():
        compdb /= "compile_commands.json"
    report = compdb.parent / "translation-report.json"
    driver_source = (Path(driver_root).resolve() / "tool").resolve()
    model = Path(model).resolve()
    for input_tree in (model, driver_source):
        if out.is_relative_to(input_tree) or input_tree.is_relative_to(out):
            raise ValueError(f"output directory overlaps copied input tree: {input_tree}")
    found = shutil.which(str(native))
    original_native = Path(found if found else native).resolve()
    observed = {"compdb": compdb, "translationReport": report, "native": original_native,
                "models": model, "driver": driver_source, "orchestrator": Path(__file__).resolve(),
                "python": Path(sys.executable).resolve()}
    # All overlap checks precede the first filesystem mutation.
    out.mkdir(parents=True, exist_ok=False)
    manifest = {"schemaVersion": 1, "phase": "preparing", "createdAt": now(),
                "scope": "every compilation-command occurrence in the supplied database; no success filtering",
                "jobs": jobs, "budget": {"maxNodes": max_nodes, "timeoutSeconds": timeout},
                "outerWorkerTimeoutSeconds": timeout + 60, "includeContract": include_contract,
                "timingMeaning": "concurrent scheduled wall time; not independent performance measurement",
                "originalPaths": {key: str(path) for key, path in observed.items()},
                "driverCacheExclusions": ["__pycache__", "*.pyc"],
                "tasks": [], "before": {}, "after": {}}
    write_json(out / "manifest.json", manifest)
    base = empty_coverage(max_nodes, timeout)
    tasks = []
    entries = []
    started = time.monotonic()
    interrupted = False
    termination = getattr(signal.getsignal(signal.SIGTERM), "_parallel_termination_state", None)
    try:
        inputs = out / "inputs"
        inputs.mkdir()
        try:
            raw = compdb.read_bytes()
            (inputs / "original-compile_commands.json").write_bytes(raw)
            entries = json.loads(raw)
            if not isinstance(entries, list):
                raise ValueError("compile database must be a JSON array")
            invalid_database = False
        except (OSError, ValueError, UnicodeError) as exc:
            raw = b"null"
            entries = []
            invalid_database = True
            manifest["inputError"] = str(exc)
        for index, entry in enumerate(entries):
            path = out / "tasks" / f"{index:06d}-{entry_sha(entry)[:12]}"
            path.mkdir(parents=True)
            task = {"index": index, "entry": entry, "entrySha256": entry_sha(entry),
                    "file": source_name(entry, compdb.parent, index), "directory": str(path),
                    "state": "pending", "command": None, "returncode": None,
                    "log": str(path / "worker.log")}
            tasks.append(task)
            write_json(path / "task.json", task)
            write_json(path / "original-entry.json", entry)
            (path / "worker.log").write_text("[orchestrator] Task pending; no worker invocation yet.\n")
        report_bytes = report.read_bytes() if report.exists() else None
        original_unknowns = None
        if report_bytes is not None:
            (inputs / "original-translation-report.json").write_bytes(report_bytes)
            try:
                preliminary = json.loads(report_bytes)
                if (not invalid_database and isinstance(preliminary, dict) and isinstance(preliminary.get("entries"), list)
                        and all(isinstance(row, dict) and ("directory" not in row or
                            isinstance(row["directory"], str) and "\0" not in row["directory"]) for row in preliminary["entries"])):
                    original_unknowns = [copy.deepcopy(row) for row in preliminary["entries"] if row.get("status") == "translation-unknown"]
                    # Match runner's public coverage: only file is canonicalized;
                    # the original directory field stays as supplied in the report.
                    for row in original_unknowns:
                        if isinstance(row.get("file"), str):
                            row["file"] = str((compdb.parent / row.get("directory", str(compdb.parent)) / row["file"]).resolve())
                    base["tus"]["translationUnknown"] = original_unknowns
                    base["tus"]["total"] = len(base["tus"]["translationUnknown"])
            except (ValueError, UnicodeError):
                pass
        manifest["tasks"] = tasks
        # All assigned occurrences are visible as incomplete before any execution.
        pending_cov, pending_runs = aggregate(base, tasks)
        pending_cov["coverageUnknown"].append({"file": str(compdb), "reason": "orchestration-in-progress"})
        write_json(out / "out.sarif", finish_coverage(pending_cov, pending_runs))
        write_json(out / "coverage.json", pending_cov)
        manifest["before"] = {key: identity(path, key == "driver") for key, path in observed.items()}
        for key, filename in (("compdb", "original-compile_commands.json"), ("translationReport", "original-translation-report.json")):
            if identity(inputs / filename) != manifest["before"][key]:
                raise ValueError(f"{key} changed while its original bytes were captured")
        driver = out / "driver"
        shutil.copytree(driver_source, driver / "tool", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        if not (driver / "tool/cli.py").is_file() or not (driver / "tool/runner.py").is_file():
            raise ValueError("driver must provide tool.cli and tool.runner")
        shutil.copytree(model, out / "models")
        frozen_native = out / "native"
        if original_native.is_file():
            shutil.copy2(original_native, frozen_native)
        else:
            frozen_native = original_native
        snapshot = {"driver": driver / "tool", "models": out / "models", "native": frozen_native}
        manifest["snapshotPaths"] = {key: str(path) for key, path in snapshot.items()}
        manifest["snapshotBefore"] = {key: identity(path, key == "driver") for key, path in snapshot.items()}
        if any(manifest["snapshotBefore"][key] != manifest["before"][key] for key in snapshot):
            raise ValueError("native/models/driver changed while being snapshotted")
        environment = {**os.environ, "PYTHONPATH": str(driver), "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1"}
        manifest["driverEnvironment"] = {key: environment[key] for key in ("PYTHONPATH", "PYTHONNOUSERSITE", "PYTHONDONTWRITEBYTECODE")}
        manifest["compilerEnvironment"] = {key: os.environ[key] for key in ("PATH", "LD_LIBRARY_PATH", "CUDA_PATH", "CUDA_HOME", "CPATH", "CPLUS_INCLUDE_PATH", "C_INCLUDE_PATH", "GCC_EXEC_PREFIX", "COMPILER_PATH", "LIBRARY_PATH") if key in os.environ}

        def command(path):
            cmd = [sys.executable, "-m", "tool.cli", "analyze", "--compdb", str(path),
                   "--sarif", str(path / "out.sarif"), "--coverage", str(path / "coverage.json"),
                   "--native", str(frozen_native), "--model", str(out / "models"),
                   "--max-nodes", str(max_nodes), "--timeout", str(timeout)]
            return cmd + (["--include-contract"] if include_contract else [])

        # The unchanged CLI processes report errors/unknowns and global per-file
        # assumptions exactly once, with zero TU/native invocations here.
        metadata = out / "metadata"
        metadata.mkdir()
        (metadata / "compile_commands.json").write_bytes(raw if invalid_database else b"[]\n")
        if report_bytes is not None:
            (metadata / "translation-report.json").write_bytes(normalize_report(report_bytes, compdb.parent))
        metadata_command = command(metadata)
        write_json(metadata / "command.json", metadata_command)
        metadata_result = invoke(metadata_command, driver, environment, metadata / "worker.log", timeout + 60)
        write_json(metadata / "execution.json", metadata_result)
        base = json.loads((metadata / "coverage.json").read_text())
        validate_output(base, json.loads((metadata / "out.sarif").read_text()), base["tus"]["total"])
        if metadata_result["returncode"] != (0 if base["complete"] else 2):
            raise ValueError("metadata CLI return code disagrees with its coverage")
        if original_unknowns is not None:
            if len(original_unknowns) != len(base["tus"]["translationUnknown"]):
                raise ValueError("metadata translationUnknown count differs from original report")
            base["tus"]["translationUnknown"] = original_unknowns
        projection_assumptions = base.get("translationAssumptions")
        if projection_assumptions is None:
            projection_assumptions = preceding_assumptions(report_bytes, compdb.parent) if report_bytes is not None else {}
        projection = {"entries": [{"file": file, "directory": str(compdb.parent),
                                   "status": "translated", "assumptions": assumptions}
                                  for file, assumptions in projection_assumptions.items()],
                      "purpose": "global per-file assumptions projection only; original unknowns counted by parent"}
        for task in tasks:
            path = Path(task["directory"])
            write_json(path / "compile_commands.json", [normalized_entry(task["entry"], compdb.parent)])
            write_json(path / "translation-report.json", projection)
            task.update(command=command(path), inputSha256={name: sha(path / name) for name in ("compile_commands.json", "translation-report.json")})
            write_json(path / "command.json", task["command"])
            write_json(path / "task.json", task)
        manifest.update(phase="running", tasks=tasks)
        write_json(out / "manifest.json", manifest)
        pending_cov, pending_runs = aggregate(base, tasks)
        pending_cov["coverageUnknown"].append({"file": str(compdb), "reason": "orchestration-in-progress"})
        write_json(out / "out.sarif", finish_coverage(pending_cov, pending_runs))
        write_json(out / "coverage.json", pending_cov)
        children = Children()
        pool = ThreadPoolExecutor(max_workers=jobs)
        futures = {}
        try:
            futures = {pool.submit(execute_task, task, environment, driver, timeout + 60, children): task["index"] for task in tasks}
            for future in as_completed(futures):
                index = futures[future]
                try:
                    tasks[index] = future.result()
                except Exception as exc:
                    tasks[index].update(state="exception", failure={"reason": "task-exception", "detail": str(exc)}, completedAt=now())
                    write_json(Path(tasks[index]["directory"]) / "task.json", tasks[index])
                    with Path(tasks[index]["log"]).open("a") as log:
                        log.write("\n[orchestrator] " + json.dumps(tasks[index]["failure"]) + "\n")
                manifest["tasks"] = tasks
                write_json(out / "manifest.json", manifest)
        except BaseException:
            children.cancel()
            for future in futures:
                future.cancel()
            raise
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
    except KeyboardInterrupt as exc:
        interrupted = True
        base["coverageUnknown"].append({"file": str(compdb), "reason": "orchestration-interrupted"})
        if str(exc) in ("SIGTERM", "SIGINT"):
            manifest["terminationSignal"] = str(exc)
    except Exception as exc:
        base["tus"]["failed"].append({"file": str(compdb), "reason": "orchestration-setup-or-execution-failed", "detail": str(exc), "exceptionType": type(exc).__name__})
        manifest["failure"] = {"detail": str(exc), "exceptionType": type(exc).__name__}
    finally:
        if termination is not None:
            # Workers are stopped/joined. Record termination during potentially
            # long integrity hashing instead of throwing past final evidence.
            termination["defer"] = True
    for task in tasks:
        if task["state"] in ("pending", "running"):
            stored = json.loads((Path(task["directory"]) / "task.json").read_text())
            if stored["state"] in ("completed", "completed-incomplete", "exception"):
                task.update(stored)
        if task["state"] in ("pending", "running"):
            task.update(state="not-completed", completedAt=now())
            write_json(Path(task["directory"]) / "task.json", task)
    cov, runs = aggregate(base, tasks)
    for category, paths, before_key, after_key in (("original", observed, "before", "after"),
            ("snapshot", {key: Path(value) for key, value in manifest.get("snapshotPaths", {}).items()}, "snapshotBefore", "snapshotAfter")):
        after = {}
        for key, path in paths.items():
            try:
                after[key] = identity(path, key == "driver")
            except Exception as exc:
                after[key] = {"identityError": str(exc)}
            if after[key] != manifest.get(before_key, {}).get(key):
                cov["coverageUnknown"].append({"file": str(path), "reason": "analysis-input-changed-or-unverifiable", "category": category, "input": key})
        manifest[after_key] = after
    for task in tasks:
        for name, expected in task.get("inputSha256", {}).items():
            path = Path(task["directory"]) / name
            if not path.is_file() or sha(path) != expected:
                cov["coverageUnknown"].append({"file": str(path), "reason": "task-input-changed-during-run", "index": task["index"]})
    cov["orchestration"] = {"jobs": jobs, "wallSeconds": time.monotonic() - started,
                            "timingMeaning": manifest["timingMeaning"], "manifest": str(out / "manifest.json")}
    while True:
        recorded_term = bool(termination and termination["received"])
        if recorded_term:
            interrupted = True
            manifest["terminationSignal"] = termination["signal"]
            marker = {"file": str(compdb), "reason": "orchestration-interrupted"}
            if marker not in cov["coverageUnknown"]:
                cov["coverageUnknown"].append(marker)
        write_json(out / "out.sarif", finish_coverage(cov, runs))
        write_json(out / "coverage.json", cov)
        manifest.update(phase="interrupted" if interrupted else "completed", completedAt=now(), tasks=tasks,
                        summary=cov["summary"], wallSeconds=cov["orchestration"]["wallSeconds"])
        write_json(out / "manifest.json", manifest)
        # A first termination signal during serialization gets one extra write;
        # repeated signals do not keep the finalization transaction alive.
        if recorded_term or not (termination and termination["received"]):
            break
    return cov


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compdb", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--native", required=True, type=Path)
    parser.add_argument("--model", default=ROOT / "models", type=Path)
    parser.add_argument("--driver", default=ROOT, type=Path, help="root containing the unchanged tool package to snapshot")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--max-nodes", type=int, default=225000)
    parser.add_argument("--errors-only", action="store_true", help="hide warning-level diagnostics")
    parser.add_argument("--include-contract", action="store_true", help=argparse.SUPPRESS)  # the default
    args = parser.parse_args(argv)
    previous_handlers = {}
    if threading.current_thread() is threading.main_thread():
        previous_handlers = {signum: signal.getsignal(signum)
                             for signum in (signal.SIGTERM, signal.SIGINT)}
        termination = {"received": False, "defer": False, "signal": None}

        def terminate(signum, frame):
            if termination["received"]:
                return
            termination["received"] = True
            termination["signal"] = signal.Signals(signum).name
            if not termination["defer"]:
                raise KeyboardInterrupt(termination["signal"])

        terminate._parallel_termination_state = termination
        for signum in previous_handlers:
            signal.signal(signum, terminate)
    try:
        cov = analyze_parallel(args.compdb, args.out, args.native, args.model, args.jobs,
                               args.timeout, args.max_nodes, not args.errors_only, args.driver)
    except (ValueError, OSError) as exc:
        print(f"parallel analysis: {exc}", file=sys.stderr)
        return 2
    finally:
        for signum, previous_handler in previous_handlers.items():
            signal.signal(signum, previous_handler)
    print(json.dumps(cov["summary"]))
    return 0 if cov["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
