#include "cuda_runtime.h"

void cl2_stack_return_bad(void *device, cudaStream_t producer, cudaStream_t consumer) {
  int host[4] = {0};
  cudaHostRegister(host, sizeof(host), cudaHostRegisterDefault);
  cudaMemcpyAsync(device, host, sizeof(host), cudaMemcpyHostToDevice, producer);
  return; // expected-warning{{CL2}}
}
