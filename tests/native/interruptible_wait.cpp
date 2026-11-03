// RUN: %native %s --sarif %t.sarif --coverage %t.json -- -std=c++17
// RUN: %verify %s %t.sarif %t.json
#include "cuda_runtime_api.h"
// RAFT's interruptible wait: synchronize(stream) passes cudaStreamQuery to a
// member template that polls it in a loop until it stops returning
// cudaErrorNotReady. The loop reaches CSA's block visit limit on the
// not-ready path, so CSA replays the call to the polling template without
// inlining; its summary records that it calls its first parameter with its
// second on every path, and the function passed at the call resolves it.
enum { cudaErrorNotReady = 600 };
struct stream_view {
  cudaStream_t value;
  operator cudaStream_t() const { return value; }
};
inline void check_cuda_error(int status, const char *) {
  if (status != 0)
    throw status;
}
extern void yield_thread();
struct interruptible {
  bool cancelled = false;
  void yield_impl() {
    if (cancelled)
      throw 1;
  }
  template <typename Query, typename Object> void synchronize_impl(Query query, Object object, const char *call) {
    int result;
    while (true) {
      yield_impl();
      result = query(object);
      if (result != cudaErrorNotReady)
        break;
      yield_thread();
    }
    check_cuda_error(result, call);
  }
  static interruptible *get_token();
  static void synchronize(stream_view stream) { get_token()->synchronize_impl(cudaStreamQuery, stream, "cudaStreamQuery"); }
  static void synchronize(cudaEvent_t event) { get_token()->synchronize_impl(cudaEventQuery, event, "cudaEventQuery"); }
  // A wrapper that passes its own query parameter on.
  template <typename Query> static void wait_with(Query query, stream_view stream) {
    get_token()->synchronize_impl(query, stream, "query");
  }
};
int not_a_query(cudaStream_t);

float stream_wait(void *device, stream_view stream) {
  float *host;
  cudaMallocHost((void **)&host, 4);
  cudaMemcpyAsync(host, device, 4, 2, stream);
  interruptible::synchronize(stream);
  return host[0];
}
float event_wait(void *device, stream_view stream, cudaEvent_t done) {
  float *host;
  cudaMallocHost((void **)&host, 4);
  cudaMemcpyAsync(host, device, 4, 2, stream);
  cudaEventRecord(done, stream);
  interruptible::synchronize(done);
  return host[0];
}
float forwarded_query(void *device, stream_view stream) {
  float *host;
  cudaMallocHost((void **)&host, 4);
  cudaMemcpyAsync(host, device, 4, 2, stream);
  interruptible::wait_with(cudaStreamQuery, stream);
  return host[0];
}
float no_wait(void *device, stream_view stream) {
  float *host;
  cudaMallocHost((void **)&host, 4);
  cudaMemcpyAsync(host, device, 4, 2, stream);
  return host[0]; // CL1 error
}
// A function that is not a query synchronizes nothing.
float other_function(void *device, stream_view stream) {
  float *host;
  cudaMallocHost((void **)&host, 4);
  cudaMemcpyAsync(host, device, 4, 2, stream);
  interruptible::wait_with(not_a_query, stream);
  return host[0]; // CL1 error
}
