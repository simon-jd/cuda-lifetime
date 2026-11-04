# Held-out fixed bugs for version 0.6.0

`cases.json` lists 6 stream-ordering and lifetime bugs that public CUDA
projects fixed in host code. Version 0.6.0 was developed on the 13 cases in
[`../real-bugs/`](../real-bugs/README.md) (development set v1) and the 10 in
[`../real-bugs-heldout/`](../real-bugs-heldout/README.md), which were held out
from 0.5.0 and drive 0.6.0's changes (development set v2). These six are held
out from 0.6.0.

**How they were found.** The fixes come from projects whose histories were not
searched for either earlier set (cuDF, RMM, llama.cpp, faiss, cuda-samples,
nccl-tests, CUDALibrarySamples, darknet, Caffe, tiny-cuda-nn, TensorRT, NCCL,
cuVS, cuML, RAFT, cuGraph, Warp, AMReX, DALI, gpe, cail,
high-level-gpu-programming, AMGX, CuPy, Ginkgo, KvikIO, OpenMM, ONNX Runtime
and CCCL). cuCollections was prepared as a scan target earlier
(`targets/cuCollections`), but its history was not searched. The histories of
cuOpt, rapidsmpf, cuSpatial, cuCIM, XGBoost, LightGBM, CTranslate2, MatX,
ucxx, cugraph-gnn, WholeGraph, Kokkos, GROMACS, HOOMD-blue, Open3D, DGL, QUDA,
instant-ngp, cuCollections, spark-rapids-jni, TensorRT-LLM, legate,
nvImageCodec, stdexec, CUDA-Q, FasterTransformer, HugeCTR, MinkowskiEngine,
Umpire, CHAI, hypre, Horovod, CUTLASS, TVM and a few smaller projects were
searched with `git log --grep` for messages about synchronization, races,
lifetimes, dangling or out-of-scope buffers, pinned or pageable staging and
event records or waits, and, in the full clones, with `git log -G` for added
stream, event and device synchronization, `cudaFreeAsync`, `cudaEventRecord`
and `cudaStreamWaitEvent` calls. Each match's diff was read. The selection
rule is the 0.5.0 held-out set's: host-code fixes that add a stream, event or
device synchronization, or move a free, read or destruction (for CL4, the
producer relative to the event capture) to the right side of one. Races
inside kernels, new code, unmerged branches, device-side ordering between
streams and Python-only fixes were dropped.

Some in-scope bugs were left out before labelling. Three are real CL2 bugs
whose fix removes the copy or gives its source static storage instead of
synchronizing (cuOpt `6cce8f76` and `195c8e8a`, GROMACS `dd53a5457`); GROMACS
`fd997bc1e` fixes a wait on an event never recorded by removing the wait.
rapidsmpf's test fixes `97383ad`, `5c08242` and `ed7e621` (CL2) were left out
because every one of those tests includes `<mpi.h>`, which is not installed
here, and cuSpatial `d43f5064` (2021) because it needs cuDF and RMM 0.19 and
Thrust 1.10-era headers (not tried, the reason the 0.5.0 set gave for two
cases).

The six kept: two CL1 (a host read of a pending D2H copy's destination), two
CL2 (a host source destroyed with its copy pending) and two CL4 (an event
wait that does not cover the producer). cuOpt and spark-rapids-jni go through
RAPIDS wrappers (`set_element_async`, `make_device_uvector_async`),
cuCollections and instant-ngp call `cudaMemcpyAsync`, and the two GROMACS
cases go through its GPU state and nonbonded modules, whose event records and
waits are in other translation units.

| Case | Project, fix | Rule |
|---|---|---|
| `cuco-dynamic-map-success-count` | cuCollections `71bcfa43` | CL1 |
| `cuopt-route-view-staging` | cuOpt `c7c94d2e` | CL2 |
| `spark-rapids-shuffle-offset-info` | spark-rapids-jni `013b0671` | CL2 |
| `ngp-exposure-gradient-read` | instant-ngp `a96f4544` | CL1 |
| `gromacs-shift-vector-event` | GROMACS `4a676c0a` | CL4 |
| `gromacs-nonlocal-x-wait` | GROMACS `422429f5` | CL4 |

**Protocol.** Each label (rule, file, lines, a one-sentence description) was
written from the fix commit's message and diff, and this file and
`cases.json` were committed before the checker, or any analyzer, ran on
either revision of any case. While the cases were chosen, only the compiler
(`scripts/run_real_bugs.py --tool syntax`) was run on them, to tell which
translation units can parse here. The compile commands are committed
afterwards in `commands.json`. The matching rule is that of both earlier
sets: a case is found when the buggy revision has a CL1–CL4 diagnostic on
one of its `siteLines` in the labelled file, and the fix silences it when the
fixed revision has none there. Labels are not changed after the first run,
which is run once, with the 0.6.0 build.

## Results of version 0.6.0

After the labels and then the compile commands were committed,
`scripts/run_real_bugs.py --cases corpus/real-bugs-heldout-0.6.0/cases.json
--commands corpus/real-bugs-heldout-0.6.0/commands.json` ran once with the
0.6.0 build and produced `results/real-bugs-heldout-0.6.0/results.json`.
Four of the six translation units parse. The checker reports **2 of the 6**
on a labelled line of the buggy revision, and in both the fixed revision has
no report on the labelled lines.

| Case | Rule | Result | Why |
|---|---|---|---|
| `cuco-dynamic-map-success-count` | CL1 | found; the fix removes it | CL1 reads of `h_num_successes` and `h_submap_num_successes` on lines 178, 179, 226 and 227 after the D2H copies; the fixed revision has no report in the file. The buggy revision also has CL2 reports on lines 149 and 219 (the same stack counters at the functions' ends), not triaged |
| `cuopt-route-view-staging` | CL2 | found; the fix removes it | `route_view`'s storage ends with the `set_element_async` copy from it (line 299) on the path; the report is placed where the inlined `set_route_views` returns to its callers (lines 68 and 239). The fixed revision keeps other CL2 and CL1 reports in the file, all with copies through `device_scalar` at unlabelled sites (not triaged), which is why the runner's coarse `fixSilences` (no report with the same rule and message anywhere in the file) is false |
| `spark-rapids-shuffle-offset-info` | CL2 | not analyzable | the file itself writes `operator+ __host__ __device__(int)`, which Clang rejects, and cuDF's `type_dispatcher.hpp` reports 24 deferred host/device errors |
| `ngp-exposure-gradient-read` | CL1 | not analyzable | 18 calls from host code to `constexpr __device__` functions, which nvcc accepts with `--expt-relaxed-constexpr` and Clang does not |
| `gromacs-shift-vector-event` | CL4 | missed | the event record and the wait are in the nonbonded module's translation units; `sim_util.cpp` sees only calls into them, and no diagnostic is reported in either revision |
| `gromacs-nonlocal-x-wait` | CL4 | missed | the 2019 `sim_util.cpp` reaches no CUDA call or model in its own translation unit, so no function is analyzed |

Two of the four analyzable cases are found; the two misses are event
dependencies that live in other translation units, which a per-TU checker
does not see.
