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
