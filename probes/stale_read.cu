// The pattern cuda-lifetime reports as CL1 error, run on a GPU: the host reads
// the pinned destination of a cudaMemcpyAsync D2H before the stream is
// synchronized. A kernel queued ahead of the copy spins for about 20 ms and
// then writes 42, so the copy cannot have run yet when the host reads.
//
//   stale_read pinned    reads cudaMallocHost memory (the reported pattern)
//   stale_read pageable  reads malloc memory (the copy call blocks instead)
//
// Prints one JSON line: the value read before synchronization and after it.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

static void check(cudaError_t code, const char *call) {
  if (code != cudaSuccess) {
    std::fprintf(stderr, "%s: %s\n", call, cudaGetErrorString(code));
    std::exit(1);
  }
}

__global__ void delayed_write(int *value, unsigned long long cycles) {
  const long long start = clock64();
  while (clock64() - start < static_cast<long long>(cycles)) {}
  *value = 42;
}

int main(int argc, char **argv) {
  if (argc != 2 || (std::strcmp(argv[1], "pinned") && std::strcmp(argv[1], "pageable"))) return 2;
  const bool pinned = std::strcmp(argv[1], "pinned") == 0;
  cudaStream_t stream;
  int *device;
  int *host;
  check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags");
  check(cudaMalloc((void **)&device, sizeof(int)), "cudaMalloc");
  check(cudaMemset(device, 0, sizeof(int)), "cudaMemset");
  check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
  if (pinned) {
    check(cudaMallocHost((void **)&host, sizeof(int)), "cudaMallocHost");
  } else {
    host = static_cast<int *>(std::malloc(sizeof(int)));
    if (!host) return 1;
  }
  *host = -1;  // sentinel: what the host sees if the copy has not run
  delayed_write<<<1, 1, 0, stream>>>(device, 50000000ULL);
  check(cudaGetLastError(), "kernel launch");
  check(cudaMemcpyAsync(host, device, sizeof(int), cudaMemcpyDeviceToHost, stream), "cudaMemcpyAsync");
  const int early = *host;  // read before the stream is synchronized
  check(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
  const int late = *host;
  std::printf("{\"memory\":\"%s\",\"beforeSync\":%d,\"afterSync\":%d}\n", argv[1], early, late);
  if (pinned) check(cudaFreeHost(host), "cudaFreeHost");
  else std::free(host);
  check(cudaFree(device), "cudaFree");
  check(cudaStreamDestroy(stream), "cudaStreamDestroy");
  return 0;
}
