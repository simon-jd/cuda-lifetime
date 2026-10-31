#include "cuda_runtime_api.h"
int offset_copy(cudaStream_t stream, void *device) {
  int *host;
  cudaMallocHost((void **)&host, 16);
  cudaMemcpyAsync(host + 1, device, 8, 2, stream);
  int result = host[2]; // CL1 error: inside bytes [4,12)
  cudaDeviceSynchronize();
  return result;
}
int after_copy_range(cudaStream_t stream, void *device) {
  int *host;
  cudaMallocHost((void **)&host, 16);
  host[3] = 42;
  cudaMemcpyAsync(host, device, 4, 2, stream);
  int result = host[3]; // no CL1: outside bytes [0,4)
  cudaDeviceSynchronize();
  return result;
}
