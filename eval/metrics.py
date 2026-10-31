"""Evaluate hand-labelled CUDA lifetime cases against SARIF 2.1.0 results."""

from collections import Counter
from pathlib import Path
import posixpath
import re
from urllib.parse import unquote, urlsplit

RULES = ("CL1", "CL2", "CL3", "CL4")


class EvaluationError(ValueError):
    """The supplied artifacts cannot support a trustworthy evaluation."""


def normalized_file(value, root="."):
    """Canonical local path, accepting percent-encoded SARIF file URIs."""
    if not isinstance(value, str) or not value:
        raise EvaluationError("file must be a nonempty path or file URI")
    if re.match(r"^[a-zA-Z]:[\\/]", value):
        path = value.replace("\\", "/")
        return path[0].lower() + posixpath.normpath(path[1:])
    uri = urlsplit(value)
    if uri.scheme and uri.scheme != "file":
        raise EvaluationError(f"unsupported artifact URI scheme: {uri.scheme}")
    if uri.netloc not in ("", "localhost"):
        raise EvaluationError("non-local file URI is not supported")
    path = unquote(uri.path).replace("\\", "/")
    if re.match(r"^/[a-zA-Z]:/", path):
        return path[1].lower() + posixpath.normpath(path[2:])
    return str((Path(root) / path).resolve())


def _cases(labels, root):
    if not isinstance(labels, dict) or labels.get("schemaVersion") != 1:
        raise EvaluationError("labels must have schemaVersion 1")
    cases = labels.get("cases")
    if not isinstance(cases, list) or not cases:
        raise EvaluationError("labels.cases must be a nonempty list")
    ids, positive_keys, negative_keys, clean = set(), set(), set(), []
    for case in cases:
        if not isinstance(case, dict):
            raise EvaluationError("each label must be an object")
        if not isinstance(case.get("id"), str) or not case["id"] or case["id"] in ids:
            raise EvaluationError("case IDs must be nonempty and unique")
        ids.add(case["id"])
        if case.get("ruleId") not in RULES or case.get("expect") not in ("warning", "none"):
            raise EvaluationError(f"invalid rule or expectation: {case['id']}")
        line = case.get("line")
        if type(line) is not int or line < 1:
            raise EvaluationError(f"invalid source line: {case['id']}")
        if (not isinstance(case.get("semanticIds"), list) or not case["semanticIds"]
                or any(not isinstance(value, str) or not value for value in case["semanticIds"])):
            raise EvaluationError(f"missing semantic evidence: {case['id']}")
        if case.get("level") not in (None, "error", "warning"):
            raise EvaluationError(f"invalid severity: {case['id']}")
        row = dict(case, file=normalized_file(case.get("file"), root))
        key = (row["ruleId"], row["file"], line)
        if row["expect"] == "warning":
            if key in positive_keys:
                raise EvaluationError("duplicate positive label location")
            positive_keys.add(key)
        else:
            negative_key = (row["ruleId"], row["file"])
            if negative_key in negative_keys:
                raise EvaluationError("duplicate negative label for file/rule")
            negative_keys.add(negative_key)
        clean.append(row)
    for row in clean:
        if row["expect"] == "none" and any(
            rule == row["ruleId"] and path == row["file"] for rule, path, _ in positive_keys
        ):
            raise EvaluationError("negative file/rule label conflicts with a positive label")
    return clean


def _artifact_file(location, run, root, seen=None):
    location = dict(location)
    if "uri" not in location and "index" in location:
        try:
            index = location["index"]
            if type(index) is not int or index < 0:
                raise IndexError
            location = {**run["artifacts"][index]["location"], **location}
        except (IndexError, KeyError, TypeError) as exc:
            raise EvaluationError("invalid SARIF artifact index") from exc
    value = location.get("uri")
    base_id = location.get("uriBaseId")
    if base_id:
        seen = set() if seen is None else seen
        if base_id in seen:
            raise EvaluationError("cyclic SARIF URI base")
        seen.add(base_id)
        try:
            base = _artifact_file(run["originalUriBaseIds"][base_id], run, root, seen)
        except (KeyError, TypeError) as exc:
            raise EvaluationError("unknown SARIF URI base") from exc
        return normalized_file(value, base)
    return normalized_file(value, root)


def _diagnostics(sarif, root):
    if not isinstance(sarif, dict) or sarif.get("version") != "2.1.0":
        raise EvaluationError("expected SARIF version 2.1.0")
    if not isinstance(sarif.get("runs"), list) or not sarif["runs"]:
        raise EvaluationError("SARIF runs must be a nonempty list")
    diagnostics, ignored = [], 0
    for run in sarif["runs"]:
        if not isinstance(run, dict) or not isinstance(run.get("results"), list):
            raise EvaluationError("malformed SARIF run")
        try:
            producer = run["tool"]["driver"]["name"]
            if not isinstance(producer, str) or not producer:
                raise ValueError
        except (KeyError, TypeError, ValueError) as exc:
            raise EvaluationError("SARIF run must identify its producer") from exc
        for result in run.get("results", []):
            if not isinstance(result, dict):
                raise EvaluationError("malformed SARIF result")
            rule = result.get("ruleId")
            if rule is None and "ruleIndex" in result:
                try:
                    index = result["ruleIndex"]
                    if type(index) is not int or index < 0:
                        raise IndexError
                    rule = run["tool"]["driver"]["rules"][index]["id"]
                except (KeyError, IndexError, TypeError) as exc:
                    raise EvaluationError("invalid SARIF rule index") from exc
            level = result.get("level", "warning")
            if rule not in RULES or level in ("note", "none"):
                ignored += 1
                continue
            if level not in ("error", "warning"):
                raise EvaluationError("invalid SARIF result level")
            try:
                physical = result["locations"][0]["physicalLocation"]
                line = physical["region"]["startLine"]
                if type(line) is not int or line < 1:
                    raise ValueError
                path = _artifact_file(physical["artifactLocation"], run, root)
            except (KeyError, IndexError, TypeError, ValueError) as exc:
                raise EvaluationError(f"{rule} result lacks a valid primary source location") from exc
            diagnostics.append({"ruleId": rule, "file": path, "line": line, "level": level})
    return diagnostics, ignored


def _coverage_files(coverage, root):
    if not isinstance(coverage, dict) or not isinstance(coverage.get("tus"), dict):
        raise EvaluationError("coverage must contain a tus object")
    tus = coverage.get("tus", {})

    def files(entries):
        if not isinstance(entries, list):
            raise EvaluationError("coverage file entries must be a list")
        result = set()
        for entry in entries:
            value = entry if isinstance(entry, str) else entry.get("file") if isinstance(entry, dict) else None
            if value is None:
                raise EvaluationError("coverage entry is missing its file")
            result.add(normalized_file(value, root))
        return result

    analyzed = files(tus.get("analyzed", tus.get("parsedFiles", [])))
    incomplete = set()
    for field in ("failed", "translationUnknown", "skipped"):
        incomplete |= files(tus.get(field, []))
    incomplete |= files(coverage.get("budgetExhausted", []))
    incomplete |= files(coverage.get("coverageUnknown", []))
    return analyzed - incomplete, incomplete


def evaluate(labels, sarif, coverage, root="."):
    """Match positive labels one-to-one by rule, canonical file and exact line.

    Repeated diagnostics count as false positives. False negatives retain missed
    positives in un-analyzed TUs. Negative cases only become true negatives when
    coverage explicitly certifies their TU as analyzed without failure/budget loss.
    """
    cases = _cases(labels, root)
    diagnostics, ignored = _diagnostics(sarif, root)
    analyzed, incomplete = _coverage_files(coverage, root)
    if "complete" in coverage and type(coverage["complete"]) is not bool:
        raise EvaluationError("coverage complete flag must be a boolean")
    missing_telemetry = [field for field in ("budgetExhausted", "unmodeledCudaCalls") if field not in coverage]
    missing_telemetry.extend("tus." + field for field in ("failed", "translationUnknown") if field not in coverage["tus"])
    if missing_telemetry:
        incomplete |= analyzed
        analyzed = set()
    unmodeled = coverage.get("unmodeledCudaCalls", {})
    if not isinstance(unmodeled, dict) or any(type(count) is not int or count < 0 for count in unmodeled.values()):
        raise EvaluationError("unmodeled CUDA call counts must be nonnegative integers")
    coverage_incomplete = bool(missing_telemetry) or coverage.get("complete") is False or any(unmodeled.values())
    # Aggregate unknown-call counts do not identify which TU/path was affected.
    # A global incomplete claim cannot certify an otherwise silent negative.
    if coverage_incomplete:
        incomplete |= analyzed
        analyzed = set()
    labelled_files = {case["file"] for case in cases}
    outside = [d for d in diagnostics if d["file"] not in labelled_files]
    diagnostics = [d for d in diagnostics if d["file"] in labelled_files]
    available = {}
    for index, diagnostic in enumerate(diagnostics):
        key = (diagnostic["ruleId"], diagnostic["file"], diagnostic["line"])
        available.setdefault(key, []).append(index)
    matched, misses, severity_mismatches, uncovered = set(), [], [], []
    metrics = {rule: {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "uncoveredNegative": 0} for rule in RULES}
    case_results = []
    for case in cases:
        rule, path = case["ruleId"], case["file"]
        covered = path in analyzed
        if not covered:
            uncovered.append(case["id"])
        if case["expect"] == "warning":
            candidates = available.get((rule, path, case["line"]), [])
            if candidates:
                # Severity is validated separately; it cannot change location matching.
                index = candidates.pop(0)
                matched.add(index)
                metrics[rule]["tp"] += 1
                status = "tp"
                if case.get("level") and case["level"] != diagnostics[index]["level"]:
                    severity_mismatches.append({"id": case["id"], "expected": case["level"], "actual": diagnostics[index]["level"]})
            else:
                metrics[rule]["fn"] += 1
                misses.append(case["id"])
                status = "fn"
        elif any(d["ruleId"] == rule and d["file"] == path for d in diagnostics):
            status = "fp"
        elif covered:
            metrics[rule]["tn"] += 1
            status = "tn"
        else:
            metrics[rule]["uncoveredNegative"] += 1
            status = "uncovered"
        case_results.append({"id": case["id"], "status": status, "covered": covered})
    false_positives = [d for index, d in enumerate(diagnostics) if index not in matched]
    for diagnostic in false_positives:
        metrics[diagnostic["ruleId"]]["fp"] += 1
    for values in metrics.values():
        values["precision"] = values["tp"] / (values["tp"] + values["fp"]) if values["tp"] + values["fp"] else None
        values["recall"] = values["tp"] / (values["tp"] + values["fn"]) if values["tp"] + values["fn"] else None
    totals = {key: sum(values[key] for values in metrics.values()) for key in ("tp", "fp", "fn", "tn", "uncoveredNegative")}
    totals["precision"] = totals["tp"] / (totals["tp"] + totals["fp"]) if totals["tp"] + totals["fp"] else None
    totals["recall"] = totals["tp"] / (totals["tp"] + totals["fn"]) if totals["tp"] + totals["fn"] else None
    counts = Counter((d["ruleId"], d["file"], d["line"]) for d in diagnostics)
    return {
        "schemaVersion": 1, "split": labels.get("split", "unspecified"),
        "rules": metrics, "totals": totals, "cases": case_results,
        "falsePositives": false_positives, "falseNegatives": misses,
        "severityMismatches": severity_mismatches,
        "duplicateDiagnostics": sum(max(0, count - 1) for count in counts.values()),
        "coverage": {"labelledFiles": len(labelled_files), "analyzedLabelledFiles": len(labelled_files & analyzed), "uncoveredCases": uncovered, "incompleteFiles": sorted(incomplete),
                     "inputComplete": coverage.get("complete"), "unmodeledCudaCalls": unmodeled,
                     "coverageUnknown": coverage.get("coverageUnknown", []), "missingTelemetry": missing_telemetry},
        "ignoredDiagnostics": ignored, "outsideCorpusDiagnostics": len(outside),
        "passed": not (false_positives or misses or severity_mismatches or uncovered or coverage_incomplete),
    }
