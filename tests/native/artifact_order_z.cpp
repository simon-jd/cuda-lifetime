// RUN: %native %s --sarif %t.sarif --coverage %t.json --include-contract -- -std=c++17
// RUN: %verify %s %t.sarif %t.json
#include "artifact_order_a.h"
int caller(void *device, cudaStream_t stream) {
  void *host = enqueue_in_header(device, stream);
  return *static_cast<int *>(host); // CL1 error
}
