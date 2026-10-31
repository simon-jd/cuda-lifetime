#include "cuda_runtime.h"

void cl4_empty_event_bad(void *device, cudaStream_t producer, cudaStream_t consumer) {
  cudaEvent_t ready;
  cudaEventCreate(&ready);
  cudaStreamWaitEvent(consumer, ready); // expected-warning{{CL4}}
}
