#include "cuda_runtime.h"

void case_bad(void *device, cudaStream_t stream) {
  int values[6] = {};
  cudaMemcpyAsync(values, device, sizeof(values), cudaMemcpyDeviceToHost, stream);
  return; // expected-warning{{CL2}}
}
