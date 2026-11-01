// RUN: %native %s --sarif %t.sarif --coverage %t.json --include-contract -- -std=c++17
// RUN: %verify %s %t.sarif %t.json --no-coverage-gaps
#include "cuda_runtime_api.h"
// The body of a generic lambda is a template pattern. Its constant locals have
// value-dependent initializers, so reading them from the store made Clang's
// constant evaluator assert (cuDF's parquet reader_impl.cpp hit this).
int generic_lambda_local(void *device, cudaStream_t stream) {
  auto width = [](auto value) { const int bytes = sizeof(value); return bytes; };
  int host;
  cudaMemcpyAsync(&host, device, sizeof host, 2, stream);
  int early = host; // CL1 warning
  cudaStreamSynchronize(stream);
  return early + host + width(1);
}
template <typename T> int templated_local(void *device, cudaStream_t stream) {
  constexpr int bytes = sizeof(T);
  T host;
  cudaMemcpyAsync(&host, device, bytes, 2, stream);
  cudaStreamSynchronize(stream);
  return static_cast<int>(host) + bytes;
}
int instantiate(void *device, cudaStream_t stream) { return templated_local<int>(device, stream); }
