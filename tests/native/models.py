#!/usr/bin/env python3
"""Native model-loader check: invalid wrapper models are rejected, a valid one is disclosed.

Usage: models.py NATIVE. Each variant of one synthetic sync_device model is
written to its own model directory and loaded while analyzing model_repro.cpp.
"""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile

SOURCE = Path(__file__).resolve().parent / "model_repro.cpp"
BASE = {"schemaVersion": 1, "models": [{
    "function": "project_wrapper", "effect": "sync_device", "args": {}, "aliases": [],
    "doc": {"url": "https://docs.nvidia.com/cuda/archive/13.2.2/cuda-runtime-api/group__CUDART__DEVICE.html",
            "anchor": "cudaDeviceSynchronize", "cudaVersion": "13.2.2"}}]}


def variant(change):
    model = copy.deepcopy(BASE)
    change(model, model["models"][0])
    return model


INVALID = {
    "bad-schema": variant(lambda m, _: m.update(schemaVersion="1")),
    "empty-function": variant(lambda _, e: e.update(function="")),
    "unsupported-effect": variant(lambda _, e: e.update(effect="silently_guess")),
    "alias-not-array": variant(lambda _, e: e.update(aliases="bad")),
    "empty-alias": variant(lambda _, e: e.update(aliases=[""])),
    "bad-assumption": variant(lambda _, e: e.update(userAssumption="true")),
    "missing-documentation": variant(lambda _, e: e.pop("doc")),
    "empty-documentation-url": variant(lambda _, e: e["doc"].update(url="")),
    "empty-documentation-anchor": variant(lambda _, e: e["doc"].update(anchor="")),
    "noninteger-argument": variant(lambda _, e: e.update(args={"stream": "0"})),
    "missing-required-argument": variant(lambda _, e: e.update(effect="malloc_async", args={"ptr": 0})),
}


def run(native, model, temp):
    folder = Path(tempfile.mkdtemp(dir=temp))
    (folder / "models").mkdir()
    (folder / "models" / "models.json").write_text(json.dumps(model, indent=2) + "\n")
    process = subprocess.run([native, str(SOURCE), "--model", str(folder / "models"),
                              "--sarif", str(folder / "out.sarif"), "--coverage", str(folder / "coverage.json"),
                              "--", "-std=c++17"], capture_output=True, text=True)
    return process, folder / "coverage.json"


def main(native):
    with tempfile.TemporaryDirectory() as temp:
        accepted = [name for name, model in INVALID.items() if run(native, model, temp)[0].returncode == 0]
        assert not accepted, ("invalid models were accepted", accepted)
        process, coverage = run(native, BASE, temp)
        assert process.returncode == 0, process.stderr
        assert "user model: project_wrapper" in json.loads(coverage.read_text())["assumptions"]
    print(f"Model loader rejected {len(INVALID)} invalid variants and disclosed the valid model")


if __name__ == "__main__":
    main(sys.argv[1])
