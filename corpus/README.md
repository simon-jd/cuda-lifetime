# Hand-labelled analysis fixtures

`seed/` contains 20 independently labelled cases, ten positive/negative pairs.
Labels refer to S1–S10 in `docs/semantics.md`; they come from the specified CUDA
semantics, not from observed checker output. Source comments identify the exact
primary diagnostic line. `expected-warning` is for Clang `-verify`; diagnostic
severity in SARIF is checked separately against each label's `level`.

`include/cuda_runtime.h` supplies declarations for CPU-only analyzer development.
It neither replaces the CUDA SDK nor demonstrates support for a toolkit version.
The fixtures need no GPU and are deliberately ordinary C++ to isolate analyzer
semantics from CUDA parsing and kernel launch compatibility.

`dev/kernel_cross_stream_{bad,good}.cu` is a separate development pair that uses
the real toolkit headers. Its labels are in `dev/kernel-labels.json`. Both legacy
and per-thread default stream modes can be tested with the same fixed labels:

```sh
python3 tests/run_seed.py --native build/cuda-lifetime-native \
  --labels corpus/dev/kernel-labels.json --cuda-path /usr/local/cuda-13.2 \
  --default-stream per-thread --out results/kernel-per-thread
```

The runner retains actual native SARIF, coverage, per-TU command/log files,
metrics, and executable/label/source hashes. These development fixtures are not
held-out data, GPU execution, or evidence of runtime behavior.

Run the analyzer with contract diagnostics enabled when evaluating the entire
seed corpus: pageable CL1 and `cudaFreeHost` CL2 intentionally expect warning,
not error. Missing or budget-limited translation units remain uncovered, and
their negative cases cannot be counted as true negatives.

`holdout/` holds 48 translation units and 49 labels written before the checker
ran on them, with authorship hashes (`authorship.json`) and the freeze manifest
(`freeze.json`) of the first run; [docs/evaluation.md](../docs/evaluation.md)
reports that run.
