#include "cuda_runtime.h"

void cl2_free_host_good(void *device, cudaStream_t producer, cudaStream_t consumer) {
  int *host;
  cudaMallocHost((void **)&host, sizeof(int));
  cudaMemcpyAsync(device, host, sizeof(int), cudaMemcpyHostToDevice, producer);
  cudaStreamSynchronize(producer);
  cudaFreeHost(host); // label-site
}
