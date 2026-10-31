#include "cuda_runtime_api.h"
extern void current_state();
// A failed synchronization cannot establish host completion.
int failed_sync_branch(cudaStream_t s, void *device) {
  int *host;
  if (cudaMallocHost((void **)&host, sizeof(int))) return 0;
  if (cudaMemcpyAsync(host, device, sizeof(int), 2, s)) return 0;
  if (cudaStreamSynchronize(s)) return *host; // CL1 error: the failure branch has no completion
  return 0;
}
// Registration of one field must not certify another field as pinned.
int partial_registration(cudaStream_t s, void *device) {
  struct Buffer { int pinned_part; int other_part; } buffer;
  cudaHostRegister(&buffer.pinned_part, sizeof(int), 0);
  cudaMemcpyAsync(&buffer.other_part, device, sizeof(int), 2, s);
  int result = buffer.other_part; // CL1 warning: only the other field is registered
  cudaDeviceSynchronize();
  return result;
}
// A distinct field must not be treated as the pending target.
int disjoint_fields(cudaStream_t s, void *device) {
  struct Buffer { int pending; int unrelated; } buffer;
  buffer.unrelated = 42;
  cudaHostRegister(&buffer, sizeof(buffer), 0);
  cudaMemcpyAsync(&buffer.pending, device, sizeof(int), 2, s);
  int result = buffer.unrelated; // no CL1: distinct object range
  cudaDeviceSynchronize();
  return result;
}
// Ordinary C++ names beginning in cu are not CUDA Driver API calls.
void non_cuda_name() { current_state(); }
