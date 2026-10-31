// RUN: %native %s --sarif %t.sarif --coverage %t.json --include-contract -- -std=c++17
// RUN: %verify %s %t.sarif %t.json
#include "cuda_runtime_api.h"
void default_and_nonblocking_are_distinct(void *source) {
  cudaStream_t stream;
  cudaStreamCreateWithFlags(&stream, 1);
  int *device;
  cudaMallocAsync((void **)&device, 4, nullptr);
  cudaMemcpyAsync(device, source, 4, 3, stream); // CL3 error: a new nonblocking stream differs from the default stream
  cudaDeviceSynchronize();
}
