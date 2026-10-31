// RUN: %native %s --sarif %t.sarif --coverage %t.json --include-contract -- -std=c++17
// RUN: %verify %s %t.sarif %t.json
#include "cuda_runtime_api.h"
int elapsed_success(void *device, cudaStream_t stream) {
  void *host;
  cudaMallocHost(&host, 4);
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, stream);
  cudaMemcpyAsync(host, device, 4, 2, stream);
  cudaEventRecord(stop, stream);
  float ms;
  if (cudaEventElapsedTime(&ms, start, stop) != 0) return 0;
  return *static_cast<int *>(host);
}
int elapsed_failure(void *device, cudaStream_t stream) {
  void *host;
  cudaMallocHost(&host, 4);
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, stream);
  cudaMemcpyAsync(host, device, 4, 2, stream);
  cudaEventRecord(stop, stream);
  float ms;
  if (cudaEventElapsedTime(&ms, start, stop) != 0)
    return *static_cast<int *>(host); // CL1 error
  return 0;
}
int elapsed_ignored(void *device, cudaStream_t stream) {
  void *host;
  cudaMallocHost(&host, 4);
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, stream);
  cudaMemcpyAsync(host, device, 4, 2, stream);
  cudaEventRecord(stop, stream);
  float ms;
  cudaEventElapsedTime(&ms, start, stop);
  return *static_cast<int *>(host); // CL1 error
}
int elapsed_stale_capture(void *device, cudaStream_t stream) {
  void *host;
  cudaMallocHost(&host, 4);
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, stream);
  cudaEventRecord(stop, stream);
  cudaMemcpyAsync(host, device, 4, 2, stream);
  float ms;
  if (cudaEventElapsedTime(&ms, start, stop) != 0) return 0;
  return *static_cast<int *>(host); // CL1 error
}
int elapsed_two_streams(void *device, cudaStream_t one, cudaStream_t two) {
  if (one == two) return 0;
  void *first, *second;
  cudaMallocHost(&first, 4);
  cudaMallocHost(&second, 4);
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaMemcpyAsync(first, device, 4, 2, one);
  cudaMemcpyAsync(second, device, 4, 2, two);
  cudaEventRecord(start, one);
  cudaEventRecord(stop, two);
  float ms;
  if (cudaEventElapsedTime(&ms, start, stop) != 0) return 0;
  return *static_cast<int *>(first) + *static_cast<int *>(second);
}
int elapsed_empty_start(void *device, cudaStream_t stream) {
  void *host;
  cudaMallocHost(&host, 4);
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaMemcpyAsync(host, device, 4, 2, stream);
  cudaEventRecord(stop, stream);
  float ms;
  cudaEventElapsedTime(&ms, start, stop);
  return *static_cast<int *>(host); // CL1 error
}
int elapsed_disabled_timing(void *device, cudaStream_t stream) {
  void *host;
  cudaMallocHost(&host, 4);
  cudaEvent_t start, stop;
  cudaEventCreateWithFlags(&start, 2);
  cudaEventCreate(&stop);
  cudaEventRecord(start, stream);
  cudaMemcpyAsync(host, device, 4, 2, stream);
  cudaEventRecord(stop, stream);
  float ms;
  cudaEventElapsedTime(&ms, start, stop);
  return *static_cast<int *>(host); // CL1 error
}
