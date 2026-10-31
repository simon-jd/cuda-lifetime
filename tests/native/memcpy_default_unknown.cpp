#include "cuda_runtime_api.h"
// Neither operand's memory kind is known, so no direction can be modeled: no
// diagnostic, and coverage records the gap instead of treating the copy as safe.
int default_unknown(void *destination, const void *source, cudaStream_t stream) {
  cudaMemcpyAsync(destination, source, sizeof(int), 4, stream);
  return *static_cast<int *>(destination); // no CL1: direction unknown
}
int default_unknown_sync(void *destination, const void *source) {
  return cudaMemcpy(destination, source, sizeof(int), 4); // no diagnostic: direction unknown
}
