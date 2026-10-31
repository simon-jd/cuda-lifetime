#include "cuda_runtime_api.h"
// Synchronous cudaMemcpy and cudaMemset (legacy default stream).
void blocking_d2h_completes_default_stream(void *device) {
  void *allocated;
  cudaMalloc(&allocated, 16);
  cudaMemcpyAsync(device, allocated, 16, 3, nullptr);
  float host[4];
  cudaMemcpy(host, device, 16, 2);
  cudaFree(allocated);
}
void device_copy_does_not_synchronize(void *device) {
  void *allocated;
  cudaMalloc(&allocated, 16);
  cudaMemcpyAsync(device, allocated, 16, 3, nullptr);
  cudaMemcpy(device, device, 16, 3);
  cudaFree(allocated); // CL3 note
}
int blocking_copy_orders_blocking_streams(void *device) {
  cudaStream_t blocking;
  cudaStreamCreateWithFlags(&blocking, 0);
  void *host;
  cudaMallocHost(&host, 16);
  cudaMemcpyAsync(host, device, 16, 2, blocking);
  float value[4];
  cudaMemcpy(value, device, 16, 2);
  return *static_cast<int *>(host);
}
void blocking_copy_races_nonblocking_stream(void *device) {
  cudaStream_t other;
  cudaStreamCreateWithFlags(&other, 1);
  void *host;
  cudaMallocHost(&host, 16);
  cudaMemcpyAsync(host, device, 16, 2, other);
  cudaMemcpy(host, device, 16, 2); // CL1 error
  cudaStreamSynchronize(other);
}
int async_memset_of_pinned_memory(cudaStream_t stream) {
  void *host;
  cudaMallocHost(&host, 16);
  cudaMemsetAsync(host, 0, 16, stream);
  return *static_cast<int *>(host); // CL1 error
}
void memset_of_device_memory_is_asynchronous(void *device) {
  void *allocated;
  cudaMalloc(&allocated, 16);
  cudaMemset(allocated, 0, 16);
  cudaFree(allocated); // CL3 note
}
