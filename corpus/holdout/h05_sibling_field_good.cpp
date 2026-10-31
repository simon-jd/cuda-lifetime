#include "cuda_runtime.h"

struct Pair { int first[4]; int second[4]; };
void case_good(void *device, cudaStream_t stream) {
  Pair data = {};
  cudaHostRegister(data.first, sizeof(data.first), cudaHostRegisterDefault);
  cudaMemcpyAsync(device, data.first, sizeof(data.first), cudaMemcpyHostToDevice, stream);
  data.second[1] = 8; // label-site
  cudaStreamSynchronize(stream);
}
