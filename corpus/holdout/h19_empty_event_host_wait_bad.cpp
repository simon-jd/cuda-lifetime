#include "cuda_runtime.h"

void case_bad(cudaStream_t stream) {
  cudaEvent_t ready;
  cudaEventCreate(&ready);
  cudaEventSynchronize(ready); // expected-warning{{CL4}}
}
