# Runtime probes

These programs observe CUDA runtime behavior on the machine they run on. An
observation describes one GPU and driver; it never changes a static rule or
label.

`completion.py` builds `completion_contract.cu` and measures when
`cudaMemcpyAsync` D2H returns for pinned and pageable host memory, with a
kernel that spins for about 20 ms queued ahead of it:

```sh
python3 probes/completion.py --ccbin /usr/bin/g++-13 --out results/probes/completion.json
```

On an RTX 4090 with driver 595.99 the pinned copy returns in 0.007 ms, before
the kernel finishes; the pageable copy returns after 18.4 ms, when the kernel
and the copy are done. This is why the faiss code reported as a portability
risk in [docs/evaluation.md](../docs/evaluation.md) works today.

`stale_read.cu` is the pattern cuda-lifetime reports as a CL1 error, run on
the GPU: with a 20 ms kernel queued ahead, it reads the pinned destination of a
`cudaMemcpyAsync` D2H before synchronizing the stream, then again after.
`stale_read.py` builds it, runs it for pinned and for pageable host memory, and
with `--native` also records what the checker reports for the same source:

```sh
flock /var/tmp/rtx4090.lock python3 probes/stale_read.py --ccbin g++-13 \
  --native build/cuda-lifetime-native --cuda-path /usr/local/cuda-13.2 \
  --out results/probes/stale-read.json
```

On the RTX 4090 with driver 595.99, all 20 pinned runs read the sentinel -1
before synchronization and 42 after it; all 20 pageable runs read 42 both
times. The checker reports the early read as CL1 error (pinned path) and CL1
warning (pageable path).

`sanitizer.py` runs every compute-sanitizer tool on each scenario of
`stale_read.cu` and `lifetime.cu`, and the checker on both sources; see
[docs/evaluation.md](../docs/evaluation.md#what-compute-sanitizer-sees):

```sh
flock /var/tmp/rtx4090.lock python3 probes/sanitizer.py --ccbin g++-13 \
  --native build/cuda-lifetime-native --out results/probes/sanitizer.json
```

`lifetime.cu` and `run.py` are deliberately unsafe microprograms for pinned and
pageable D2H, ordinary `cudaFree`, `cudaFreeHost`, and cross-stream
stream-ordered allocation use with and without an event edge, run directly and
under compute-sanitizer:

```sh
nvcc -ccbin /usr/bin/g++-13 -std=c++17 -arch=sm_89 probes/lifetime.cu -o build/lifetime-probe
python3 probes/run.py
```
