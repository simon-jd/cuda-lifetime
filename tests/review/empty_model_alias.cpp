#include "../native/runtime.h"
int indirect_call_is_not_sync(void (*hook)(), cudaStream_t stream, void *device) {
  int *host;
  cudaMallocHost((void **)&host, 4);
  cudaMemcpyAsync(host, device, 4, 2, stream);
  hook();
  return *host; // review: CL1, an unrelated indirect callback establishes no completion
}
