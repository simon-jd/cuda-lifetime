#include "cuda_runtime_api.h"
#define LEGACY ((cudaStream_t)1)
#define PER_THREAD ((cudaStream_t)2)
#ifdef CUDA_API_PER_THREAD_DEFAULT_STREAM
#define CURRENT_DEFAULT PER_THREAD
#else
#define CURRENT_DEFAULT LEGACY
#endif

void per_thread_to_legacy(void *device) {
  void *allocated;
  cudaMallocAsync(&allocated, 4, PER_THREAD);
  cudaMemcpyAsync(device, allocated, 4, 3, LEGACY);
}
void legacy_to_per_thread(void *device) {
  void *allocated;
  cudaMallocAsync(&allocated, 4, LEGACY);
  cudaMemcpyAsync(device, allocated, 4, 3, PER_THREAD);
}
void zero_to_matching_explicit(void *device) {
  void *allocated;
  cudaMallocAsync(&allocated, 4, nullptr);
  cudaMemcpyAsync(device, allocated, 4, 3, CURRENT_DEFAULT);
}
void matching_explicit_to_zero(void *device) {
  void *allocated;
  cudaMallocAsync(&allocated, 4, CURRENT_DEFAULT);
  cudaMemcpyAsync(device, allocated, 4, 3, nullptr);
}
void per_thread_to_blocking(void *device) {
  cudaStream_t stream;
  cudaStreamCreateWithFlags(&stream, 0);
  void *allocated;
  cudaMallocAsync(&allocated, 4, PER_THREAD);
  cudaMemcpyAsync(device, allocated, 4, 3, stream); // CL3 error
}
void blocking_to_per_thread(void *device) {
  cudaStream_t stream;
  cudaStreamCreateWithFlags(&stream, 0);
  void *allocated;
  cudaMallocAsync(&allocated, 4, stream);
  cudaMemcpyAsync(device, allocated, 4, 3, PER_THREAD); // CL3 error
}
void per_thread_to_nonblocking(void *device) {
  cudaStream_t stream;
  cudaStreamCreateWithFlags(&stream, 1);
  void *allocated;
  cudaMallocAsync(&allocated, 4, PER_THREAD);
  cudaMemcpyAsync(device, allocated, 4, 3, stream); // CL3 error
}
void legacy_to_nonblocking(void *device) {
  cudaStream_t stream;
  cudaStreamCreateWithFlags(&stream, 1);
  void *allocated;
  cudaMallocAsync(&allocated, 4, LEGACY);
  cudaMemcpyAsync(device, allocated, 4, 3, stream); // CL3 error
}
int device_order_does_not_complete_host(void *device, void *other_device) {
  void *host;
  cudaMallocHost(&host, 4);
  cudaMemcpyAsync(host, device, 4, 2, PER_THREAD);
  cudaMemcpyAsync(other_device, device, 4, 3, LEGACY);
  return *static_cast<int *>(host); // CL1 error
}
int synchronized_legacy_completes_preceding_per_thread(void *device, void *other_device) {
  void *host;
  cudaMallocHost(&host, 4);
  cudaMemcpyAsync(host, device, 4, 2, PER_THREAD);
  cudaMemcpyAsync(other_device, device, 4, 3, LEGACY);
  cudaStreamSynchronize(LEGACY);
  return *static_cast<int *>(host);
}
