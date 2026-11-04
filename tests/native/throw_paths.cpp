// RUN: %native %s --sarif %t.sarif --coverage %t.json -- -std=c++17
// RUN: %verify %s %t.sarif %t.json
// RUN: %python -c "import json, sys; c = json.load(open(sys.argv[1])); assert c['pathsEndedBeforeThrow'] > 0, c" %t.json
#include "cuda_runtime_api.h"
// CSA ends a path at a throw. A path that can only end in one (an error
// branch that formats a message, here in a loop, and throws) is ended where
// that becomes certain, so the branch is not explored; the paths that
// continue are checked as before.
extern void append(char *message, int digit);
[[noreturn]] extern void fail(const char *message);
inline void check(int status) {
  if (status != 0) {
    char message[64];
    for (int i = 0; i < status; ++i)
      append(message, i);
    throw status;
  }
}
float checked_copy(void *device, cudaStream_t stream) {
  float *host;
  cudaMallocHost((void **)&host, 4);
  check(cudaMemcpyAsync(host, device, 4, 2, stream));
  float v = host[0]; // CL1 error
  check(cudaStreamSynchronize(stream));
  return v;
}
// A check that returns before its throw (RAFT's check_cuda_error) is not
// an error path: the paths through it continue.
inline void check_returns_first(int status) {
  if (status == 0)
    return;
  throw status;
}
float checked_like_raft(void *device, cudaStream_t stream) {
  float *host;
  cudaMallocHost((void **)&host, 4);
  check_returns_first(cudaMemcpyAsync(host, device, 4, 2, stream));
  float v = host[0]; // CL1 error
  check_returns_first(cudaStreamSynchronize(stream));
  return v;
}
// Accesses on a branch that must throw are not checked: CSA would stop at
// the throw, and the checker stops at the branch.
extern void log_value(float);
float read_on_error_branch(void *device, cudaStream_t stream, bool bad) {
  float *host;
  cudaMallocHost((void **)&host, 4);
  cudaMemcpyAsync(host, device, 4, 2, stream);
  if (bad) {
    log_value(host[0]);
    fail("bad input");
  }
  float v = host[0]; // CL1 error
  cudaStreamSynchronize(stream);
  return v;
}
// A branch that can still return is explored.
float read_on_recoverable_branch(void *device, cudaStream_t stream, bool retry) {
  float *host;
  cudaMallocHost((void **)&host, 4);
  cudaMemcpyAsync(host, device, 4, 2, stream);
  if (retry) {
    if (host[0] > 0) // CL1 error
      throw 1;
  }
  cudaStreamSynchronize(stream);
  return host[0];
}
