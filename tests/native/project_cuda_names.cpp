// RUN: %native %s --sarif %t.sarif --coverage %t.json --include-contract -- -std=c++17
// RUN: %verify %s %t.sarif %t.json --unmodeled cudaGraphCreate
#include "cuda_runtime_api.h"
// Project functions whose names start like CUDA API calls are not CUDA calls,
// whether or not CSA inlines them; only the stub-declared cudaGraphCreate is an
// unmodeled CUDA API call here.
int cudaGraphsManual(int value);
float cuGet(int value) { return static_cast<float>(value); }
int uses_project_names(cudaStream_t stream, void *device) {
  int *host;
  cudaMallocHost((void **)&host, sizeof(int));
  cudaMemcpyAsync(host, device, sizeof(int), 2, stream);
  int sum = cudaGraphsManual(1) + static_cast<int>(cuGet(2));
  cudaStreamSynchronize(stream);
  cudaGraph_t graph;
  cudaGraphCreate(&graph, 0);
  return sum + *host;
}
