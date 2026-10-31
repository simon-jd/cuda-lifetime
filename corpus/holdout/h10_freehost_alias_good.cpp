#include "cuda_runtime.h"

void case_good(void *device, cudaStream_t stream) {
  int *host;
  cudaMallocHost((void **)&host, 8 * sizeof(int));
  int *alias = host;
  cudaMemcpyAsync(device, host, 8 * sizeof(int), cudaMemcpyHostToDevice, stream);
  cudaStreamSynchronize(stream);
  cudaFreeHost(alias); // label-site
}
