// RUN: %native %s --sarif %t.sarif --coverage %t.json -- -std=c++17
// RUN: %verify %s %t.sarif %t.json
// RUN: %python -c "import json, sys; c = json.load(open(sys.argv[1])); assert c['libraryCallsNotInlined'] > 0, c" %t.json
#include "cuda_runtime_api.h"
#include <memory>
#include <mutex>
#include <sstream>
// shared_ptr's reference counts, stream output and locking are evaluated
// without inlining when no pending buffer reaches them; the checker's view of
// the copies is unchanged.
struct config {
  int rows;
};
std::mutex lock;
float with_shared_state(std::shared_ptr<config> shared, void *device, cudaStream_t stream) {
  float *host;
  cudaMallocHost((void **)&host, 4);
  cudaMemcpyAsync(host, device, 4, 2, stream);
  std::shared_ptr<config> copy = shared;
  std::ostringstream message;
  message << "rows " << copy->rows;
  std::lock_guard<std::mutex> guard(lock);
  float v = host[0]; // CL1 error
  cudaStreamSynchronize(stream);
  return v;
}
float synchronized_with_shared_state(std::shared_ptr<config> shared, void *device, cudaStream_t stream) {
  float *host;
  cudaMallocHost((void **)&host, 4);
  cudaMemcpyAsync(host, device, 4, 2, stream);
  std::shared_ptr<config> copy = shared;
  cudaStreamSynchronize(stream);
  std::ostringstream message;
  message << host[0] << copy->rows;
  return host[0];
}
