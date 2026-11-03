#!/usr/bin/env python3
"""Run the checker on the buggy and fixed revision of each labelled real bug.

Reads corpus/real-bugs/cases.json (labels) and commands.json (how to compile
each case's translation unit), checks out both revisions from local clones of
the projects, analyzes the translation unit at each, and writes one record
per case: whether it parsed, its diagnostics, whether one is on the labelled
lines (found), and whether the fix silences it.

    python3 scripts/run_real_bugs.py --clones .deps/recall --native build/cuda-lifetime-native \\
      --cuda-path /usr/local/cuda-13.2 --gcc-toolchain /usr --out results/real-bugs/results.json

--clones holds one git clone per project, named after the repository; the
revisions are checked out as worktrees under --clones/trees. A case may name
dependencies (for example RMM for cuDF), checked out at their last commit
before the case revision and substituted as {name} in its flags.

A case may also name header patches (commands.json "headerPatches": a
project header copied with textual fixes, one "pattern"/"replacement" or a
list of "edits", into an overlay directory searched first, at "output" if
given), generated headers (a directory under the generated/ directory next
to the commands file, searched
first), and helper translation units ("summarize"): each is summarized with
cuda-lifetime-native --summarize at the same revision, in order, and the
resulting models are loaded when the case's TU is analyzed.

--tool csa or --tool clang-tidy runs a baseline instead of cuda-lifetime on
the same translation units with the same flags (no summaries): Clang's stock
static analyzer (clang++ --analyze, SARIF output) or clang-tidy with the
clang-analyzer-* and bugprone-* checks (headers included). A baseline finds a case when any of
its diagnostics, whatever the check, is on a site line, as for cuda-lifetime.
--tool syntax only checks that each translation unit compiles (clang++
-fsyntax-only), for writing a case's flags without running an analyzer.
"""
import argparse
import json
from pathlib import Path
import re
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
CUDA = ["-x", "cuda", "--cuda-host-only", "--cuda-path={cuda}", "--offload-arch=sm_89"]


def worktree(clones, repository, revision):
    clone = clones / repository.rstrip("/").split("/")[-1]
    tree = clones / "trees" / f"{clone.name}-{revision[:10]}"
    if not tree.exists():
        subprocess.run(["git", "-C", str(clone), "worktree", "add", "-q", "--detach", str(tree.resolve()), revision],
                       check=True, capture_output=True, text=True)
    return tree.resolve()


def dependency(clones, spec, clone_of, revision, tree):
    """A dependency's worktree at its last commit before the case revision's commit date.

    The commit is taken from the clone's HEAD, or from origin/<branch> when
    the spec names a "branchFile" in the case tree (RAPIDS_BRANCH: a release
    branch builds against the dependencies' same release branch). A spec's
    "commit" pins the dependency instead.
    """
    clone = clones / spec["repository"].rstrip("/").split("/")[-1]
    if spec.get("commit"):
        return worktree(clones, spec["repository"], spec["commit"]), spec["commit"]
    date = subprocess.run(["git", "-C", str(clone_of), "show", "-s", "--format=%cI", revision],
                          check=True, capture_output=True, text=True).stdout.strip()
    ref = "origin/" + (tree / spec["branchFile"]).read_text().strip() if spec.get("branchFile") else "HEAD"
    commit = subprocess.run(["git", "-C", str(clone), "rev-list", "-1", "--before=" + date, ref],
                            check=True, capture_output=True, text=True).stdout.strip()
    return worktree(clones, spec["repository"], commit), commit


def expand(values, names):
    return [value.format(**names) for value in values]


def overlay(temp, tree, patches, names=None):
    """Copy each patched header into an overlay directory; return -I flags for them.

    A patch with "in" takes its header from that dependency's tree (a key of names).
    """
    flags = []
    for index, patch in enumerate(patches):
        root = Path(temp) / f"overlay{index}"
        # "output" names the copy when it differs from the header's own path
        # (a header the build generates from a template, such as nccl.h.in).
        target = root / patch.get("output", Path(patch["header"]).relative_to(patch["includeRoot"]))
        target.parent.mkdir(parents=True, exist_ok=True)
        source = Path(names[patch["in"]]) if patch.get("in") else tree
        text = (source / patch["header"]).read_text()
        edits = patch.get("edits", [{"pattern": patch.get("pattern"), "replacement": patch.get("replacement")}])
        for edit in edits:
            text, count = re.subn(edit["pattern"], edit["replacement"], text, flags=re.M)
            if not count:
                raise SystemExit(f"header patch {patch['header']} matched nothing in {source}")
        target.write_text(text)
        flags.append(f"-I{root}")
    return flags


def summarize(native, helpers, names, temp, prefix, timeout):
    """Summarize helper TUs in order into one model directory; return it and a record per helper."""
    models = Path(temp) / "models"
    models.mkdir()
    records = []
    for index, helper in enumerate(helpers):
        source = Path(helper["file"].format(**names))
        record = {"file": helper["file"]}
        if not source.exists():
            records.append({**record, "status": "absent at this revision"})
            continue
        out = Path(temp) / f"summary{index}.json"
        language = CUDA if helper["language"] == "cuda" else ["-x", helper["language"]]
        done = subprocess.run([native, str(source), "--summarize", str(out), "--model", str(models), "--",
                               *expand(language, names), *prefix], capture_output=True, text=True, timeout=timeout)
        if done.returncode or not out.exists():
            errors = [line for line in done.stderr.splitlines() if "error:" in line]
            records.append({**record, "status": "not parsed", "errors": errors[:3]})
            continue
        generated = json.loads(out.read_text())
        (models / out.name).write_text(out.read_text())
        records.append({**record, "status": "summarized", "models": sorted(m["function"] for m in generated["models"])})
    return models, records


DIAGNOSTIC = re.compile(r"^(.+?):(\d+):\d+: (warning|error|note): (.*?)(?: \[([^\]]+)\])?$")


def sarif_results(sarif):
    """Rule, level, message, primary location and code-flow locations of each SARIF result."""
    results = []
    for run in json.loads(sarif.read_text())["runs"]:
        artifacts = run.get("artifacts", [])
        def uri(location):
            artifact = location["physicalLocation"]["artifactLocation"]
            return artifact.get("uri") or artifacts[artifact["index"]]["location"]["uri"]
        for result in run["results"]:
            flow = [{"uri": uri(item["location"]), "line": item["location"]["physicalLocation"]["region"]["startLine"]}
                    for f in result.get("codeFlows", []) for thread in f["threadFlows"] for item in thread["locations"]]
            results.append({"ruleId": result["ruleId"], "level": result["level"], "message": result["message"]["text"],
                            "uri": uri(result["locations"][0]),
                            "line": result["locations"][0]["physicalLocation"]["region"]["startLine"], "flow": flow})
    return results


def text_results(output):
    """Warnings in compiler-style text output, each with the notes that follow it as its flow."""
    results = []
    for line in output.splitlines():
        match = DIAGNOSTIC.match(line)
        if not match:
            continue
        path, number, level, message, check = match.groups()
        if level == "warning":
            results.append({"ruleId": check or "", "level": level, "message": message,
                            "uri": path, "line": int(number), "flow": []})
        elif level == "note" and results:
            results[-1]["flow"].append({"uri": path, "line": int(number)})
    return results


ERROR = re.compile(r"^(?:\S.*?:\d+:\d+: )?(?:fatal )?error: ")
ASSERTION = re.compile(r"Assertion `.*' failed")
# Overrides the projects' own .clang-tidy files (some turn warnings into errors).
TIDY_CONFIG = '{"Checks": "-*,clang-analyzer-*,bugprone-*", "WarningsAsErrors": ""}'


def compiler_errors(output):
    """Compiler error lines, not source excerpts that happen to contain "error:"."""
    return [line for line in output.splitlines() if ERROR.match(line) or ASSERTION.search(line)]


def baseline(tool, executable, source, flags, timeout):
    """Run Clang's stock analyzer or clang-tidy on one TU; the record has the same fields as analyze's."""
    with tempfile.TemporaryDirectory() as temp:
        sarif = Path(temp) / "out.sarif"
        command = ([executable, "-fsyntax-only", *flags, str(source)] if tool == "syntax" else
                   [executable, "--analyze", "--analyzer-output", "sarif", "-o", str(sarif), *flags, str(source)]
                   if tool == "csa" else
                   [executable, str(source), "--config=" + TIDY_CONFIG, "--header-filter=.*", "--quiet", "--", *flags])
        try:
            done = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return {"parsed": False, "status": "timeout", "errors": [f"no result within {timeout} s"]}
        output = done.stdout + done.stderr
        errors = compiler_errors(output)
        parsed = done.returncode == 0 and not errors and (tool != "csa" or sarif.exists())
        record = {"parsed": parsed, "returncode": done.returncode, "errorCount": len(errors), "errors": errors[:3]}
        if done.returncode < 0:
            record["status"] = f"terminated by signal {-done.returncode}"
        if parsed:
            record["diagnostics"] = sarif_results(sarif) if tool == "csa" else text_results(done.stdout + done.stderr)
        return record


def analyze(native, source, flags, timeout, models=None):
    with tempfile.TemporaryDirectory() as temp:
        sarif, coverage = Path(temp) / "out.sarif", Path(temp) / "coverage.json"
        try:
            done = subprocess.run([native, str(source), "--sarif", str(sarif), "--coverage", str(coverage),
                                   *(["--model", str(models)] if models else []), "--", *flags],
                                  capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return {"parsed": False, "status": "timeout", "errors": [f"no result within {timeout} s"]}
        errors = [line for line in done.stderr.splitlines() if "error:" in line or "Assertion" in line]
        cov = json.loads(coverage.read_text()) if coverage.exists() else {}
        parsed = done.returncode == 0 and cov.get("tus", {}).get("parsed") == 1
        record = {"parsed": parsed, "returncode": done.returncode, "errorCount": len(errors), "errors": errors[:3]}
        if done.returncode < 0:
            record["status"] = f"terminated by signal {-done.returncode}"
        if not parsed:
            return record
        results = sarif_results(sarif)
        record.update(diagnostics=results, functionsAnalyzed=cov.get("functionsAnalyzed"),
                      functionsOverBudget=len(cov.get("budgetExhausted", [])),
                      overBudgetFunctions=sorted(entry.get("function", "") for entry in cov.get("budgetExhausted", [])),
                      coverageGaps=sorted({entry["reason"] for entry in cov.get("coverageUnknown", [])}),
                      unmodeledCudaCalls=sorted(cov.get("unmodeledCudaCalls", {})), seconds=cov.get("seconds"),
                      assumptions=[a for a in cov.get("assumptions", []) if a.startswith("user model: ")])
        return record


def on_lines(diagnostic, path, lines):
    places = [(diagnostic["uri"], diagnostic["line"])] + [(item["uri"], item["line"]) for item in diagnostic["flow"]]
    return any(uri.endswith(path) and line in lines for uri, line in places)


def relative(record, tree, label):
    text = json.dumps(record).replace("file://" + str(tree) + "/", label + "/").replace(str(tree) + "/", label + "/")
    return json.loads(text.replace(str(ROOT) + "/", ""))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cases", type=Path, default=ROOT / "corpus/real-bugs/cases.json")
    parser.add_argument("--commands", type=Path, default=ROOT / "corpus/real-bugs/commands.json")
    parser.add_argument("--clones", type=Path, required=True)
    parser.add_argument("--native", help="cuda-lifetime-native, required with --tool cuda-lifetime")
    parser.add_argument("--tool", choices=("cuda-lifetime", "csa", "clang-tidy", "syntax"), default="cuda-lifetime")
    parser.add_argument("--clang", default="clang++", help="the clang++ for --tool csa and --tool syntax")
    parser.add_argument("--tidy", default="clang-tidy", help="the clang-tidy for --tool clang-tidy")
    parser.add_argument("--cuda-path", default="/usr/local/cuda-13.2")
    parser.add_argument("--gcc-toolchain", default="/usr")
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--only", action="append", default=[], help="analyze only this case (repeatable)")
    args = parser.parse_args()
    if args.tool == "cuda-lifetime" and not args.native:
        parser.error("--native is required with --tool cuda-lifetime")
    executable = {"csa": args.clang, "syntax": args.clang, "clang-tidy": args.tidy}.get(args.tool, args.native)
    version = subprocess.run([executable, "--version"], capture_output=True, text=True).stdout.strip()
    cases = [case for case in json.loads(args.cases.read_text())["cases"] if not args.only or case["id"] in args.only]
    commands = json.loads(args.commands.read_text())
    records = []
    for case in cases:
        command = commands["cases"][case["id"]]
        source = command.get("translationUnit", case["file"])
        row = {"id": case["id"], "inRuleScope": case["inRuleScope"], "rule": case["rule"]}
        for name, revision in (("buggy", case["parentCommit"]), ("fixed", case["fixCommit"])):
            tree = worktree(args.clones, case["repository"], revision)
            names = {"tree": str(tree), "root": str(ROOT), "cuda": args.cuda_path, "gcc": args.gcc_toolchain}
            pinned = {}
            for key, spec in command.get("dependencies", {}).items():
                dep_tree, pinned[key] = dependency(args.clones, spec, args.clones / case["repository"].split("/")[-1], revision, tree)
                names[key] = str(dep_tree)
            with tempfile.TemporaryDirectory() as temp:
                prefix = overlay(temp, tree, [commands["headerPatches"][name] for name in command.get("headerPatches", [])], names)
                prefix += [f"-I{args.commands.resolve().parent / 'generated' / name}" for name in command.get("generatedHeaders", [])]
                prefix += expand([*commands["common"], *command["flags"]], names)
                models, helpers = summarize(args.native, command.get("summarize", []), names, temp, prefix, args.timeout) \
                    if command.get("summarize") and args.tool == "cuda-lifetime" else (None, [])
                flags = [*expand(CUDA if command["language"] == "cuda" else ["-x", command["language"]], names), *prefix]
                record = analyze(args.native, tree / source, flags, args.timeout, models) if args.tool == "cuda-lifetime" \
                    else baseline(args.tool, executable, tree / source, flags, args.timeout)
            if helpers:
                record["summaries"] = helpers
            if pinned:
                record["dependencies"] = pinned
            for key, path in names.items():
                if key not in ("tree", "root", "cuda", "gcc"):
                    record = relative(record, Path(path), f"{key}@{pinned[key][:10]}")
            row[name] = relative(record, tree, f"{case['id']}@{revision[:10]}")
            print(case["id"], name, "parsed" if record["parsed"] else "not parsed",
                  len(record.get("diagnostics", [])), "diagnostics", flush=True)
        if args.tool != "cuda-lifetime":
            # Baselines warn throughout the headers; keep the count and the
            # diagnostics that touch the labelled file, which is all matching uses.
            for name in ("buggy", "fixed"):
                if "diagnostics" in row[name]:
                    every = row[name]["diagnostics"]
                    row[name]["diagnosticCount"] = len(every)
                    row[name]["diagnostics"] = [d for d in every if d["uri"].endswith(case["file"]) or
                                                any(f["uri"].endswith(case["file"]) for f in d["flow"])]
        buggy, fixed = row["buggy"], row["fixed"]
        if not buggy["parsed"]:
            row["result"] = "not-analyzable"
        else:
            matched = [d for d in buggy["diagnostics"] if on_lines(d, case["file"], case["siteLines"])]
            row["result"] = "found" if matched else "missed"
            if matched:
                same = {(d["ruleId"], d["message"]) for d in matched}
                row["fixSilences"] = fixed["parsed"] and not any(
                    d["uri"].endswith(case["file"]) and (d["ruleId"], d["message"]) in same for d in fixed["diagnostics"])
        records.append(row)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"schemaVersion": 1, "tool": args.tool, "toolVersion": version,
                                    "cases": records}, indent=2) + "\n")


if __name__ == "__main__":
    main()
