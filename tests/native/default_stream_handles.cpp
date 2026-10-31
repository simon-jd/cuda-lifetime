// RUN: %native %s --sarif %t.sarif --coverage %t.json --include-contract -- -std=c++17
// RUN: %verify %s %t.sarif %t.json --default-stream legacy
// RUN: %native %s --sarif %t.pt.sarif --coverage %t.pt.json --include-contract -- -std=c++17 -DCUDA_API_PER_THREAD_DEFAULT_STREAM=1
// RUN: %verify %s %t.pt.sarif %t.pt.json --default-stream per-thread
#include "cuda_runtime_api.h"
#ifndef CUDA_API_PER_THREAD_DEFAULT_STREAM
#define EXPLICIT_DEFAULT ((cudaStream_t)1)
#else
#define EXPLICIT_DEFAULT ((cudaStream_t)2)
#endif
int explicit_default_synchronizes_null(void *device) {
  int *host;
  cudaMallocHost((void **)&host, 4);
  cudaMemcpyAsync(host, device, 4, 2, nullptr);
  cudaStreamSynchronize(EXPLICIT_DEFAULT);
  return *host; // no CL1: null names this explicit default stream
}
void explicit_default_orders_null(void *other_device) {
  int *device;
  cudaMallocAsync((void **)&device, 4, nullptr);
  cudaMemcpyAsync(device, other_device, 4, 3, EXPLICIT_DEFAULT); // no CL3
  cudaDeviceSynchronize();
}
