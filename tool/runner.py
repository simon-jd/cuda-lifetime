"""Per-TU isolation, native SARIF preservation, and explicit coverage failures."""
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import tempfile
import time

RULES = {
    "CL1": "Host access before asynchronous copy completes",
    "CL2": "Host buffer lifetime ends before asynchronous copy completes",
    "CL3": "Stream-ordered allocation lifetime violation",
    "CL4": "Event capture does not cover the producer",
}
META = re.compile(r"\b(CL[1-4]) \[(error|warning|note); memory=([^;\]]+); semantic=([^\]]+)\]:?\s*")
VISIT_COUNTERS = ("modeledCallVisits", "pointerEscapes", "pendingMemberCallUses",
                  "pendingArgumentUses", "pendingPointerEscapes")


def default_stream_mode(args):
    mode = "legacy"
    macro = False
    for i, arg in enumerate(args):
        if arg.startswith("-fgpu-default-stream="):
            mode = arg.split("=", 1)[1]
        value = args[i + 1] if arg in ("-D", "-U") and i + 1 < len(args) else arg[2:]
        if arg.startswith("-D") and value.split("=", 1)[0] == "CUDA_API_PER_THREAD_DEFAULT_STREAM":
            macro = True
        if arg.startswith("-U") and value == "CUDA_API_PER_THREAD_DEFAULT_STREAM":
            macro = False
    return "per-thread" if macro else mode


def validate_native_coverage(cov, source):
    if not isinstance(cov, dict) or not isinstance(cov.get("tus"), dict):
        raise ValueError("native coverage must include tus object")
    tus = cov["tus"]
    if type(tus.get("parsed")) is not int or tus["parsed"] not in (0, 1):
        raise ValueError("native coverage parsed count must be 0 or 1")
    for field in ("failed", "analyzed"):
        if not isinstance(tus.get(field), list):
            raise ValueError("native coverage must include tus." + field)
    if tus["parsed"] != len(tus["analyzed"]):
        raise ValueError("native coverage parsed/analyzed counts disagree")
    if any(not isinstance(path, str) or Path(path).resolve() != Path(source).resolve() for path in tus["analyzed"]):
        raise ValueError("native coverage identifies a different translation unit")
    if not tus["failed"] and tus["parsed"] != 1:
        raise ValueError("native coverage has no evidence of TU analysis")
    for field in ("budgetExhausted", "coverageUnknown", "assumptions"):
        if field in cov and not isinstance(cov[field], list):
            raise ValueError("invalid native coverage " + field)
    calls = cov.get("unmodeledCudaCalls", {})
    if not isinstance(calls, dict) or any(type(n) is not int or n < 0 for n in calls.values()):
        raise ValueError("invalid native unmodeled call counts")
    for field in VISIT_COUNTERS:
        if field in cov and (type(cov[field]) is not int or cov[field] < 0):
            raise ValueError("invalid native visit counter " + field)


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False, encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")
    os.replace(f.name, path)


def enrich_sarif(document, default_stream, include_contract):
    """Keep run-local artifact indices and CSA codeFlows intact."""
    if not isinstance(document, dict) or document.get("version") != "2.1.0" or not isinstance(document.get("runs"), list) or not document["runs"]:
        raise ValueError("native analyzer returned invalid SARIF")
    runs = copy.deepcopy(document["runs"])
    suppressed = 0
    for run in runs:
        if not isinstance(run, dict) or not isinstance(run.get("results"), list):
            raise ValueError("native SARIF run must contain results")
        driver = run.get("tool", {}).get("driver", {})
        if not isinstance(driver.get("name"), str) or not driver["name"]:
            raise ValueError("native SARIF run must identify its producer")
        rules = run.setdefault("tool", {}).setdefault("driver", {}).setdefault("rules", [])
        if not isinstance(rules, list):
            raise ValueError("native SARIF rules must be an array")
        indices = {}
        for rule, name in RULES.items():
            indices[rule] = len(rules)
            rules.append({"id": rule, "name": name, "shortDescription": {"text": name}})
        kept = []
        for result in run.get("results", []):
            message = result.get("message", {}).get("text", "")
            match = META.search(message)
            if match:
                rule, level, memory, semantic = match.groups()
                result["ruleId"] = rule
                result["ruleIndex"] = indices[rule]
                result["level"] = level
                props = result.setdefault("properties", {})
                props.update(memoryKind=memory, defaultStream=default_stream,
                             semanticId=semantic, assumptions=props.get("assumptions", []))
                result["message"]["text"] = message[:match.start()] + message[match.end():]
                if level == "warning" and not include_contract:
                    suppressed += 1
                    continue
            kept.append(result)
        run["results"] = kept
    return runs, suppressed


def analyze(compdb, sarif, coverage_path=None, native="build/cuda-lifetime-native",
            model="models", max_nodes=225000, timeout=180, include_contract=True):
    if max_nodes <= 0 or timeout <= 0:
        raise ValueError("max-nodes and timeout must be positive")
    compdb = Path(compdb).resolve()
    if compdb.is_dir():
        compdb /= "compile_commands.json"
    sarif = Path(sarif).resolve()
    coverage_path = Path(coverage_path).resolve() if coverage_path else sarif.with_name("coverage.json")
    coverage = {"schemaVersion": 1, "tus": {"total": 0, "parsed": 0, "analyzed": [],
                "failed": [], "translationUnknown": []}, "budgetExhausted": [],
                "unmodeledCudaCalls": {}, "coverageUnknown": [], "profiles": [],
                "visitCounts": {field: 0 for field in VISIT_COUNTERS},
                "assumptions": ["single host thread", "single translation unit", "bounded CSA exploration"],
                "budget": {"maxNodes": max_nodes, "timeoutSeconds": timeout},
                "complete": False, "suppressedContractDiagnostics": 0}
    output = {"version": "2.1.0", "$schema": "https://json.schemastore.org/sarif-2.1.0.json", "runs": []}
    logs = sarif.parent / (sarif.stem + ".logs")
    logs.mkdir(parents=True, exist_ok=True)
    translation_assumptions = {}
    exe = None
    native_hash = None

    def fail(file, reason, **extra):
        coverage["tus"]["failed"].append({"file": file, "reason": reason, **extra})

    def finish():
        if exe and native_hash:
            try:
                changed = hashlib.sha256(Path(exe).read_bytes()).hexdigest() != native_hash
            except OSError:
                changed = True
            if changed:
                coverage["coverageUnknown"].extend({"file": name, "reason": "native-binary-changed-during-run"} for name in coverage["tus"]["analyzed"])
        coverage["complete"] = not any((coverage["tus"]["failed"], coverage["tus"]["translationUnknown"],
                                        coverage["budgetExhausted"], coverage["coverageUnknown"],
                                        coverage["unmodeledCudaCalls"]))
        coverage["summary"] = {"parsed": coverage["tus"]["parsed"],
            "failed": len(coverage["tus"]["failed"]), "translationUnknown": len(coverage["tus"]["translationUnknown"]),
            "budgetExhausted": len(coverage["budgetExhausted"]),
            "unmodeledCudaCalls": sum(coverage["unmodeledCudaCalls"].values()),
            "coverageUnknown": len(coverage["coverageUnknown"]),
            "diagnostics": sum(len(r.get("results", [])) for r in output["runs"]),
            "complete": coverage["complete"]}
        if not output["runs"]:
            output["runs"] = [{"tool": {"driver": {"name": "cuda-lifetime", "rules": []}}, "results": []}]
        output["runs"][0].setdefault("properties", {})["coverage"] = coverage["summary"]
        write_json(sarif, output)
        write_json(coverage_path, coverage)
        return coverage

    try:
        entries = json.loads(compdb.read_text())
        if not isinstance(entries, list):
            raise ValueError("compile database must be a JSON array")
    except (OSError, ValueError) as exc:
        fail(str(compdb), "invalid-compilation-database", detail=str(exc))
        return finish()
    translation = compdb.parent / "translation-report.json"
    if translation.exists():
        try:
            report = json.loads(translation.read_text())
            if not isinstance(report, dict) or not isinstance(report.get("entries"), list) or not all(isinstance(e, dict) for e in report["entries"]):
                raise ValueError("translation report must contain an entries array of objects")
            if any("directory" in e and (not isinstance(e["directory"], str) or "\0" in e["directory"]) for e in report["entries"]):
                raise ValueError("translation directory must be a path string")
            coverage["tus"]["translationUnknown"] = [e for e in report.get("entries", []) if e.get("status") == "translation-unknown"]
            for item in coverage["tus"]["translationUnknown"]:
                if isinstance(item.get("file"), str):
                    directory = Path(item.get("directory", compdb.parent))
                    if not directory.is_absolute():
                        directory = compdb.parent / directory
                    item["file"] = str((directory / item["file"]).resolve())
            for entry in report["entries"]:
                if isinstance(entry.get("file"), str):
                    directory = Path(entry.get("directory", compdb.parent))
                    if not directory.is_absolute():
                        directory = compdb.parent / directory
                    name = str((directory / entry["file"]).resolve())
                    assumptions = entry.get("assumptions", [])
                    if not isinstance(assumptions, list) or not all(isinstance(a, str) for a in assumptions):
                        raise ValueError("translation assumptions must be a list of strings")
                    translation_assumptions.setdefault(name, []).extend(assumptions)
            coverage["translationAssumptions"] = translation_assumptions
        except (OSError, ValueError, TypeError) as exc:
            fail(str(translation), "invalid-translation-report", detail=str(exc))
    coverage["tus"]["total"] = len(entries) + len(coverage["tus"]["translationUnknown"])
    exe = shutil.which(native)
    if not exe and Path(native).is_file():
        exe = str(Path(native).resolve())
    if exe:
        native_hash = hashlib.sha256(Path(exe).read_bytes()).hexdigest()
        coverage["native"] = {"path": exe, "sha256": native_hash}
    for number, entry in enumerate(entries):
        if (not isinstance(entry, dict) or not isinstance(entry.get("file"), str) or not entry.get("file")
                or not isinstance(entry.get("directory"), str) or "\0" in entry["file"] or "\0" in entry["directory"]):
            fail(f"entry:{number}", "invalid-compilation-command")
            continue
        directory = Path(entry["directory"])
        if not directory.is_absolute():
            directory = compdb.parent / directory
        file = str((directory / entry["file"]).resolve())
        if not exe:
            fail(file, "native-analyzer-unavailable", executable=native)
            continue
        try:
            args = entry.get("arguments")
            if args is None:
                args = shlex.split(entry["command"])
            if not isinstance(args, list) or not args or not all(isinstance(x, str) for x in args):
                raise ValueError("arguments must be a nonempty string array")
        except (KeyError, TypeError, ValueError) as exc:
            fail(file, "invalid-compilation-command", detail=str(exc))
            continue
        stream = default_stream_mode(args)
        key = f"{number:04d}-" + hashlib.sha256(file.encode()).hexdigest()[:10]
        started = time.monotonic()
        peak_rss = None
        with tempfile.TemporaryDirectory(prefix="cuda-lifetime-tu-") as td:
            td = Path(td)
            write_json(td / "compile_commands.json", [{**entry, "file": file, "directory": str(directory.resolve()), "arguments": args}])
            native_sarif, native_coverage = td / "out.sarif", td / "coverage.json"
            cmd = [exe, "-p", str(td), file, "--sarif", str(native_sarif), "--coverage", str(native_coverage),
                   "--model", str(Path(model).resolve()), "--max-nodes", str(max_nodes)]
            if include_contract:
                cmd.append("--include-contract")
            write_json(logs / f"{key}.command.json", cmd)
            try:
                with (logs / f"{key}.log").open("w") as log:
                    process = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=timeout, check=False)
                if process.returncode:
                    fail(file, "native-analysis-failed", returncode=process.returncode, log=str(logs / f"{key}.log"))
                else:
                    cov = json.loads(native_coverage.read_text())
                    validate_native_coverage(cov, file)
                    peak_rss = cov.get("peakRssKiB")
                    native_mode = cov.get("defaultStreamMode", stream)
                    runs, suppressed = enrich_sarif(json.loads(native_sarif.read_text()), native_mode, include_contract)
                    assumptions = sorted(set(cov.get("assumptions", []) + translation_assumptions.get(file, [])))
                    for run in runs:
                        run.setdefault("properties", {})["assumptions"] = assumptions
                        for result in run.get("results", []):
                            props = result.setdefault("properties", {})
                            props["assumptions"] = sorted(set(props.get("assumptions", []) + assumptions))
                    output["runs"].extend(runs)
                    coverage["suppressedContractDiagnostics"] += suppressed
                    if cov.get("tus", {}).get("failed"):
                        fail(file, "native-reported-failure", detail=cov["tus"]["failed"])
                    else:
                        coverage["tus"]["parsed"] += 1
                        coverage["tus"]["analyzed"].append(file)
                    for field in ("budgetExhausted", "coverageUnknown"):
                        coverage[field].extend(cov.get(field, []))
                    for field in ("budgetExhausted", "unmodeledCudaCalls"):
                        if field not in cov:
                            coverage["coverageUnknown"].append({"file": file, "reason": "missing-native-" + field})
                    for name, count in cov.get("unmodeledCudaCalls", {}).items():
                        coverage["unmodeledCudaCalls"][name] = coverage["unmodeledCudaCalls"].get(name, 0) + count
                    for field in VISIT_COUNTERS:
                        coverage["visitCounts"][field] += cov.get(field, 0)
                    write_json(logs / f"{key}.coverage.json", cov)
                    shutil.copyfile(native_sarif, logs / f"{key}.sarif")
            except subprocess.TimeoutExpired:
                fail(file, "analysis-timeout", timeoutSeconds=timeout)
                coverage["budgetExhausted"].append({"file": file, "reason": "wall-clock-timeout"})
            except (OSError, ValueError, TypeError, AttributeError) as exc:
                fail(file, "invalid-native-output", detail=str(exc))
        coverage["profiles"].append({"file": file, "index": number,
                                     "inputCommandSha256": hashlib.sha256(json.dumps(entry, sort_keys=True).encode()).hexdigest(),
                                     "seconds": time.monotonic() - started, "peakRssKiB": peak_rss})
    return finish()
