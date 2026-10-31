#include "runtime.h"
extern void project_wrapper();
int wrapper_without_assumption(cudaStream_t stream, void *device) {
  int *host;
  cudaMallocHost((void **)&host, sizeof(int));
  cudaMemcpyAsync(host, device, sizeof(int), 2, stream);
  project_wrapper();
  return *host;
}
