#include "cuda_runtime.h"

struct Pair { int first[4]; int second[4]; };
void case_bad(void *device, cudaStream_t stream) {
  Pair data = {};
  cudaHostRegister(data.first, sizeof(data.first), cudaHostRegisterDefault);
  cudaMemcpyAsync(device, data.first, sizeof(data.first), cudaMemcpyHostToDevice, stream);
  return; // expected-warning{{CL2}}
}
