#include "cuda_runtime.h"

void cl2_stack_return_good(void *device, cudaStream_t producer, cudaStream_t consumer) {
  int host[4] = {0};
  cudaHostRegister(host, sizeof(host), cudaHostRegisterDefault);
  cudaMemcpyAsync(device, host, sizeof(host), cudaMemcpyHostToDevice, producer);
  cudaStreamSynchronize(producer);
  return; // label-site
}
