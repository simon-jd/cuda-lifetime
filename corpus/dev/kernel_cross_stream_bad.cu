#include <cuda_runtime.h>

__global__ void consume_bad(int *pointer) { pointer[0] = 42; }

void kernel_cross_stream_bad(cudaStream_t producer, cudaStream_t consumer) {
  if (producer == consumer) return;
  int *device;
  cudaMallocAsync((void **)&device, sizeof(int), producer);
  consume_bad<<<1, 1, 0, consumer>>>(device); // expected-warning{{CL3}}
}
