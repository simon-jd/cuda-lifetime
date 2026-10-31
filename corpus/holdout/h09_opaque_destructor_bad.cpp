#include "cuda_runtime.h"

struct Buffer { int values[4]; ~Buffer(); };
void case_bad(void *device, cudaStream_t stream) {
  {
    Buffer buffer;
    cudaHostRegister(buffer.values, sizeof(buffer.values), cudaHostRegisterDefault);
    cudaMemcpyAsync(device, buffer.values, sizeof(buffer.values), cudaMemcpyHostToDevice, stream);
  } // expected-warning{{CL2}}
  cudaStreamSynchronize(stream);
}
