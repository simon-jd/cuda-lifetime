#include "runtime.h"
#include <cstdlib>
#include <error.h>

static void require_success(int status) {
  if (status) std::exit(EXIT_FAILURE);
}
static void continue_on_failure(int status) { (void)status; }

int sync_helper_good(void *device) {
  int *host;
  cudaMallocHost((void **)&host, 4);
  cudaMemcpyAsync(host, device, 4, 2, 0);
  require_success(cudaStreamSynchronize(0));
  return *host;
}

int nonfatal_helper_bad(void *device) {
  int *host;
  cudaMallocHost((void **)&host, 4);
  cudaMemcpyAsync(host, device, 4, 2, 0);
  continue_on_failure(cudaStreamSynchronize(0));
  return *host; // CL1 error
}

int elapsed_helper_good(void *device) {
  int *host;
  cudaMallocHost((void **)&host, 4);
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  cudaMemcpyAsync(host, device, 4, 2, 0);
  cudaEventRecord(stop, 0);
  float ms;
  require_success(cudaEventElapsedTime(&ms, start, stop));
  return *host;
}

int elapsed_stale_snapshot_bad(void *device) {
  int *host;
  cudaMallocHost((void **)&host, 4);
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  cudaMemcpyAsync(host, device, 4, 2, 0);
  cudaEventRecord(stop, 0);
  float ms;
  require_success(cudaEventElapsedTime(&ms, start, start));
  return *host; // CL1 error
}

void allocation_helper_good(void *host) {
  cudaStream_t stream;
  require_success(cudaStreamCreateWithFlags(&stream, 1));
  void *device;
  require_success(cudaMallocAsync(&device, 4, stream));
  require_success(cudaMemcpyAsync(device, host, 4, 1, stream));
}

void allocation_cross_stream_bad(void *host) {
  cudaStream_t first, second;
  require_success(cudaStreamCreateWithFlags(&first, 1));
  require_success(cudaStreamCreateWithFlags(&second, 1));
  void *device;
  require_success(cudaMallocAsync(&device, 4, first));
  require_success(cudaMemcpyAsync(device, host, 4, 1, second)); // CL3 error
}

void panic() {}

int returning_named_helper_bad(void *device) {
  int *host;
  cudaMallocHost((void **)&host, 4);
  cudaMemcpyAsync(host, device, 4, 2, 0);
  panic();
  return *host; // CL1 error
}

int returning_gnu_error_bad(void *device) {
  int *host;
  cudaMallocHost((void **)&host, 4);
  cudaMemcpyAsync(host, device, 4, 2, 0);
  error(0, 0, "report only");
  return *host; // CL1 error
}

// A call that is the body of an if statement discards its status, so its
// failure path ends like any other unchecked call's.
int synchronize_in_branches(void *device, cudaStream_t stream, bool finish) {
  int *host;
  cudaMallocHost((void **)&host, 4);
  cudaMemcpyAsync(host, device, 4, 2, stream);
  if (finish) cudaStreamSynchronize(stream);
  if (!finish) cudaDeviceSynchronize();
  return *host;
}
int checked_condition(void *device, cudaStream_t stream) {
  int *host;
  cudaMallocHost((void **)&host, 4);
  cudaMemcpyAsync(host, device, 4, 2, stream);
  if (cudaStreamSynchronize(stream)) return *host; // CL1 error
  return *host;
}
