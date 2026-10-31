#include "../native/runtime.h"
int offset_copy(cudaStream_t stream, void *device) {
  int *host;
  cudaMallocHost((void **)&host, 16);
  cudaMemcpyAsync(host + 1, device, 8, 2, stream);
  int result = host[2]; // review: inside bytes [4,12), CL1 error
  cudaDeviceSynchronize();
  return result;
}
int after_copy_range(cudaStream_t stream, void *device) {
  int *host;
  cudaMallocHost((void **)&host, 16);
  host[3] = 42;
  cudaMemcpyAsync(host, device, 4, 2, stream);
  int result = host[3]; // review: outside bytes [0,4), no CL1
  cudaDeviceSynchronize();
  return result;
}
