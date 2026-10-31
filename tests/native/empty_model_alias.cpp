#include "cuda_runtime_api.h"
int indirect_call_is_not_sync(void (*hook)(), cudaStream_t stream, void *device) {
  int *host;
  cudaMallocHost((void **)&host, 4);
  cudaMemcpyAsync(host, device, 4, 2, stream);
  hook();
  return *host; // CL1 error: an unrelated indirect callback establishes no completion
}
