#include "cuda_runtime.h"

void cl4_empty_event_good(void *device, cudaStream_t producer, cudaStream_t consumer) {
  cudaEvent_t ready;
  cudaEventCreate(&ready);
  cudaEventRecord(ready, producer);
  cudaStreamWaitEvent(consumer, ready); // label-site
}
