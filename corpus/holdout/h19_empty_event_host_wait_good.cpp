#include "cuda_runtime.h"

void case_good(cudaStream_t stream) {
  cudaEvent_t ready;
  cudaEventCreate(&ready);
  cudaEventRecord(ready, stream);
  cudaEventSynchronize(ready); // label-site
}
