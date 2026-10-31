#include "../native/runtime.h"
extern void review_wrapper();
int wrapper_without_assumption(cudaStream_t stream, void *device) {
  int *host;
  cudaMallocHost((void **)&host, sizeof(int));
  cudaMemcpyAsync(host, device, sizeof(int), 2, stream);
  review_wrapper();
  return *host;
}
