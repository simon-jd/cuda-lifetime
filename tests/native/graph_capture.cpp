// RUN: %native %s --sarif %t.sarif --coverage %t.json --include-contract -- -std=c++17
// RUN: %verify %s %t.sarif %t.json
#include "cuda_runtime_api.h"
// Work enqueued during stream capture is recorded into a graph, not run.
void captured_copy_has_not_run(void *device, cudaStream_t stream) {
  void *host;
  cudaMallocHost(&host, 16);
  cudaStreamBeginCapture(stream, 0);
  cudaMemcpyAsync(device, host, 16, 1, stream);
  cudaGraph_t graph;
  cudaStreamEndCapture(stream, &graph);
  *static_cast<int *>(host) = 1;
}
void forked_capture(void *device, cudaStream_t stream) {
  cudaStream_t side;
  cudaStreamCreateWithFlags(&side, 1);
  cudaEvent_t fork, join;
  cudaEventCreate(&fork);
  cudaEventCreate(&join);
  void *host;
  cudaMallocHost(&host, 16);
  cudaStreamBeginCapture(stream, 0);
  cudaEventRecord(fork, stream);
  cudaStreamWaitEvent(side, fork, 0);
  cudaMemcpyAsync(device, host, 16, 1, side);
  cudaEventRecord(join, side);
  cudaStreamWaitEvent(stream, join, 0);
  cudaGraph_t graph;
  cudaStreamEndCapture(stream, &graph);
  *static_cast<int *>(host) = 1;
}
int work_after_capture_runs(void *device, cudaStream_t stream) {
  void *host;
  cudaMallocHost(&host, 16);
  cudaStreamBeginCapture(stream, 0);
  cudaGraph_t graph;
  cudaStreamEndCapture(stream, &graph);
  cudaMemcpyAsync(host, device, 16, 2, stream);
  return *static_cast<int *>(host); // CL1 error
}
