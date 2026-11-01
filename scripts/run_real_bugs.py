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
"""
import argparse
import json
from pathlib import Path
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


def dependency(clones, spec, clone_of, revision):
    """A dependency's worktree at its last commit before the case revision's commit date."""
    date = subprocess.run(["git", "-C", str(clone_of), "show", "-s", "--format=%cI", revision],
                          check=True, capture_output=True, text=True).stdout.strip()
    clone = clones / spec["repository"].rstrip("/").split("/")[-1]
    commit = subprocess.run(["git", "-C", str(clone), "rev-list", "-1", "--before=" + date, "HEAD"],
                            check=True, capture_output=True, text=True).stdout.strip()
    return worktree(clones, spec["repository"], commit), commit


def expand(values, names):
    return [value.format(**names) for value in values]


def analyze(native, source, flags, timeout):
    with tempfile.TemporaryDirectory() as temp:
        sarif, coverage = Path(temp) / "out.sarif", Path(temp) / "coverage.json"
        try:
            done = subprocess.run([native, str(source), "--sarif", str(sarif), "--coverage", str(coverage), "--", *flags],
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
        record.update(diagnostics=results, functionsAnalyzed=cov.get("functionsAnalyzed"),
                      functionsOverBudget=len(cov.get("budgetExhausted", [])),
                      overBudgetFunctions=sorted(entry.get("function", "") for entry in cov.get("budgetExhausted", [])),
                      coverageGaps=sorted({entry["reason"] for entry in cov.get("coverageUnknown", [])}),
                      unmodeledCudaCalls=sorted(cov.get("unmodeledCudaCalls", {})), seconds=cov.get("seconds"))
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
    parser.add_argument("--native", required=True)
    parser.add_argument("--cuda-path", default="/usr/local/cuda-13.2")
    parser.add_argument("--gcc-toolchain", default="/usr")
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cases = json.loads(args.cases.read_text())["cases"]
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
                dep_tree, pinned[key] = dependency(args.clones, spec, args.clones / case["repository"].split("/")[-1], revision)
                names[key] = str(dep_tree)
            flags = expand([*(CUDA if command["language"] == "cuda" else ["-x", command["language"]]),
                            *commands["common"], *command["flags"]], names)
            record = analyze(args.native, tree / source, flags, args.timeout)
            if pinned:
                record["dependencies"] = pinned
            for key, path in names.items():
                if key not in ("tree", "root", "cuda", "gcc"):
                    record = relative(record, Path(path), f"{key}@{pinned[key][:10]}")
            row[name] = relative(record, tree, f"{case['id']}@{revision[:10]}")
            print(case["id"], name, "parsed" if record["parsed"] else "not parsed",
                  len(record.get("diagnostics", [])), "diagnostics", flush=True)
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
    args.out.write_text(json.dumps({"schemaVersion": 1, "cases": records}, indent=2) + "\n")


if __name__ == "__main__":
    main()
