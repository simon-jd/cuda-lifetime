# lit configuration for the annotated checker fixtures in this directory.
#
#   llvm-lit -sv tests/native --param native=build/cuda-lifetime-native
#
# Each test runs the native analyzer on itself and checks the result with
# verify.py against its "// CLn level" annotations.
import os
import sys

import lit.formats

config.name = "cuda-lifetime"
config.test_format = lit.formats.ShTest(execute_external=False)
config.suffixes = [".cpp"]
config.test_source_root = os.path.dirname(os.path.abspath(__file__))
# Fixtures checked by run.py, prescan.py and models.py instead.
config.excludes = ["prescan", "cases.cpp", "model_repro.cpp"]

native = lit_config.params.get("native")
if not native:
    lit_config.fatal("pass --param native=PATH/cuda-lifetime-native")
native = os.path.abspath(native)
config.test_exec_root = lit_config.params.get("exec_root", os.path.join(os.path.dirname(native), "lit-tests"))
verify = os.path.join(config.test_source_root, "verify.py")
config.substitutions.append(("%native", f'"{native}"'))
config.substitutions.append(("%verify", f'"{sys.executable}" "{verify}"'))
config.substitutions.append(("%python", f'"{sys.executable}"'))
