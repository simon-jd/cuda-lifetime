#!/usr/bin/env python3
"""Independent actual-native tests for the conservative pre-analysis AST scan."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    native, output = Path(args.native).resolve(), Path(args.out).resolve()
    output.mkdir(parents=True, exist_ok=True)
    declarations = 'extern "C" int cudaDeviceSynchronize();\n'
    fixtures = {
        "pure_host": ("", "int answer() { return 42; }\n", True),
        "unused_header_cuda": (declarations + "inline void unused() { cudaDeviceSynchronize(); }\n", "int answer() { return 42; }\n", True),
        "direct_header_wrapper": (declarations + "inline void wrapper() { cudaDeviceSynchronize(); }\n", "void entry() { wrapper(); }\n", False),
        "header_constructor": (declarations + "struct Buffer { Buffer() { cudaDeviceSynchronize(); } };\n", "void entry() { Buffer local; }\n", False),
        "header_destructor": (declarations + "struct Buffer { ~Buffer() { cudaDeviceSynchronize(); } };\n", "void entry() { Buffer local; }\n", False),
        "temporary_destructor": (declarations + "struct Buffer { ~Buffer() { cudaDeviceSynchronize(); } };\ninline Buffer make() { return {}; }\n", "void entry() { make(); }\n", False),
        "delete_destructor": (declarations + "struct Buffer { ~Buffer() { cudaDeviceSynchronize(); } };\n", "void entry(Buffer *value) { delete value; }\n", False),
        "indirect_call": ("", "void entry(void (*callback)()) { callback(); }\n", False),
        "virtual_call": ("struct Base { virtual void run(); };\n", "void entry(Base &value) { value.run(); }\n", False),
        "global_initializer": (declarations + "inline int initialize() { cudaDeviceSynchronize(); return 0; }\n", "int value = initialize();\n", False),
        "global_destructor": (declarations + "struct Buffer { ~Buffer() { cudaDeviceSynchronize(); } };\n", "Buffer global;\n", False),
        "member_constructor": (declarations + "struct Inner { Inner() { cudaDeviceSynchronize(); } };\nstruct Outer { Inner inner; Outer() : inner() {} };\n", "void entry() { Outer local; }\n", False),
        "implicit_member_destructor": (declarations + "struct Inner { ~Inner() { cudaDeviceSynchronize(); } };\nstruct Outer { Inner inner; };\n", "void entry() { Outer local; }\n", False),
    }
    start_hash = sha256(native.read_bytes()).hexdigest()
    results = []
    for name, (header, source, expected_skip) in fixtures.items():
        directory = output / name
        directory.mkdir(exist_ok=True)
        (directory / "fixture.h").write_text(header)
        path = directory / "case.cpp"
        path.write_text('#include "fixture.h"\n' + source)
        command = [str(native), str(path), "--sarif", str(directory / "out.sarif"),
                   "--coverage", str(directory / "coverage.json"), "--", "-std=c++17"]
        (directory / "command.json").write_text(json.dumps(command, indent=2) + "\n")
        process = subprocess.run(command, capture_output=True, text=True, timeout=60)
        (directory / "stderr.log").write_text(process.stderr)
        coverage = json.loads((directory / "coverage.json").read_text())
        actual_skip = coverage.get("astScan", {}).get("skippedNoCudaCalls")
        results.append({"scenario": name, "expectedSkip": expected_skip, "actualSkip": actual_skip,
                        "returncode": process.returncode, "passed": process.returncode == 0 and actual_skip is expected_skip,
                        "modeledCallVisits": coverage.get("modeledCallVisits")})
    unchanged = sha256(native.read_bytes()).hexdigest() == start_hash
    report = {"kind": "actual-native-ast-scan", "nativeSha256": start_hash,
              "binaryUnchanged": unchanged, "results": results,
              "passed": unchanged and all(result["passed"] for result in results)}
    (output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
