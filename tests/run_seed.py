#!/usr/bin/env python3
"""Run actual CSA on human-labelled seeds; never substitute a fixture analyzer."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval import evaluate  # noqa: E402
from tool.runner import analyze, write_json  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--labels", default=str(ROOT / "corpus/seed/labels.json"))
    parser.add_argument("--max-nodes", type=int, default=225000)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--cuda-path", help="real toolkit root, required for labelled .cu files")
    parser.add_argument("--default-stream", choices=("legacy", "per-thread"), default="legacy")
    args = parser.parse_args(argv)
    native, output, labels_path = Path(args.native).resolve(), Path(args.out).resolve(), Path(args.labels).resolve()
    labels = json.loads(labels_path.read_text())
    if labels.get("split") == "holdout":
        parser.error("use the frozen evaluation workflow for held-out data")
    label_hash = sha256(labels_path.read_bytes()).hexdigest()
    native_hash = sha256(native.read_bytes()).hexdigest() if native.is_file() else None
    output.mkdir(parents=True, exist_ok=True)
    sources = sorted({str((ROOT / case["file"]).resolve()) for case in labels["cases"]})
    commands = []
    for source in sources:
        if Path(source).suffix == ".cu":
            if not args.cuda_path:
                parser.error("labelled .cu files require --cuda-path; SDK declarations are not stubbed")
            flags = ["-x", "cuda", "--cuda-host-only", "--cuda-path=" + str(Path(args.cuda_path).resolve()),
                     "--offload-arch=sm_89", "-Xclang", "-fgpu-default-stream=" + args.default_stream]
        else:
            flags = ["-I" + str(ROOT / "corpus/include")]
        commands.append({"directory": str(ROOT), "file": source,
                         "arguments": ["clang++", "-std=c++17", *flags, "-c", source]})
    write_json(output / "compile_commands.json", commands)
    coverage = analyze(output / "compile_commands.json", output / "out.sarif", output / "coverage.json",
                       native=str(native), model=str(ROOT / "models"), max_nodes=args.max_nodes,
                       timeout=args.timeout, include_contract=True)
    sarif = json.loads((output / "out.sarif").read_text())
    metrics = evaluate(labels, sarif, coverage, ROOT)
    results = [result for run in sarif["runs"] for result in run.get("results", [])
               if result.get("ruleId") in ("CL1", "CL2", "CL3", "CL4")]
    integration_failures = []
    for index, result in enumerate(results):
        flows = result.get("codeFlows", [])
        if not any(flow.get("threadFlows") for flow in flows):
            integration_failures.append(f"diagnostic {index} has no CSA threadFlows")
        properties = result.get("properties", {})
        for field in ("memoryKind", "defaultStream", "semanticId", "assumptions"):
            if field not in properties:
                integration_failures.append(f"diagnostic {index} is missing {field}")
    if sha256(labels_path.read_bytes()).hexdigest() != label_hash:
        integration_failures.append("labels changed during the run")
    if native.is_file() and sha256(native.read_bytes()).hexdigest() != native_hash:
        integration_failures.append("native executable changed during the run; results may mix builds")
    metrics["coverageSummary"] = coverage["summary"]
    metrics["integrationFailures"] = integration_failures
    metrics["passed"] = metrics["passed"] and not integration_failures and coverage["complete"]
    write_json(output / "metrics.json", metrics)
    evidence = {"kind": "actual-native-seed-run" if labels.get("split") == "seed" else "actual-native-development-run", "notHeldOut": True,
                "native": str(native), "nativeSha256": native_hash,
                "labelsSha256": label_hash, "sourceSha256": {source: sha256(Path(source).read_bytes()).hexdigest() for source in sources},
                "coverage": coverage["summary"], "totals": metrics["totals"], "passed": metrics["passed"]}
    write_json(output / "evidence.json", evidence)
    print(json.dumps(evidence, indent=2))
    return 0 if metrics["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
