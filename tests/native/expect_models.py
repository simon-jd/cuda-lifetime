#!/usr/bin/env python3
"""Check a generated model file: expect_models.py FILE "name key=index ..." ...

Each expectation names one generated model and all its arguments and
constants, "[]" marking an array argument whose first element is the
operand; the file must contain exactly the expected models."""
import json
import sys

models = json.load(open(sys.argv[1]))["models"]
actual = sorted(" ".join([m["function"], *sorted(f"{k}={v}" + ("[]" if k in m.get("elements", []) else "")
                                                for k, v in {**m["args"], **m.get("constants", {})}.items())])
                for m in models)
expected = sorted(" ".join([e.split()[0], *sorted(e.split()[1:])]) for e in sys.argv[2:])
if actual != expected:
    sys.exit(f"models {actual} != expected {expected}")
