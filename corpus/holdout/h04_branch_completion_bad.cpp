#include "cuda_runtime.h"

void case_bad(void *device, cudaStream_t stream, bool finish) {
  int *host;
  cudaMallocHost((void **)&host, 8 * sizeof(int));
  cudaMemcpyAsync(host, device, sizeof(int), cudaMemcpyDeviceToHost, stream);
  if (finish) cudaStreamSynchronize(stream);
  int value = host[0]; // expected-warning{{CL1}}
}
