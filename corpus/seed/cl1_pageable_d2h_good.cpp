#include "cuda_runtime.h"

void cl1_pageable_d2h_good(void *device, cudaStream_t producer, cudaStream_t consumer) {
  int *host = (int *)malloc(sizeof(int));
  cudaMemcpyAsync(host, device, sizeof(int), cudaMemcpyDeviceToHost, producer);
  cudaStreamSynchronize(producer);
  int value = host[0]; // label-site
}
