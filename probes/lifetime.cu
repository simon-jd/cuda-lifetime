// Runtime evidence only. Run on an isolated CUDA-capable test host.
#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define CUDA(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
  std::fprintf(stderr, "%s:%d: %s: %s\n", __FILE__, __LINE__, #call, cudaGetErrorString(e)); return 2; } } while (0)

__global__ void delayed_write(int* value) {
  auto start = clock64();
  while (clock64() - start < 50000000ULL) {}
  *value = 42;
}

int main(int argc, char** argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: lifetime-probe pinned|pageable|free|free-host|ordered-bad|ordered-good\n");
    return 2;
  }
  const char* mode = argv[1];
  bool pinned = !std::strcmp(mode, "pinned"), pageable = !std::strcmp(mode, "pageable");
  bool free_host = !std::strcmp(mode, "free-host"), ordinary_free = !std::strcmp(mode, "free");
  bool ordered_bad = !std::strcmp(mode, "ordered-bad"), ordered_good = !std::strcmp(mode, "ordered-good");
  if (!(pinned || pageable || free_host || ordinary_free || ordered_bad || ordered_good)) return 2;
  int driver = 0, runtime = 0;
  cudaDeviceProp props{};
  CUDA(cudaDriverGetVersion(&driver)); CUDA(cudaRuntimeGetVersion(&runtime));
  CUDA(cudaGetDeviceProperties(&props, 0));
  std::printf("{\"gpu\":\"%s\",\"driver\":%d,\"runtime\":%d,\"mode\":\"%s\"}\n", props.name, driver, runtime, mode);
  cudaStream_t a{}, b{}; CUDA(cudaStreamCreateWithFlags(&a, cudaStreamNonBlocking));
  CUDA(cudaStreamCreateWithFlags(&b, cudaStreamNonBlocking));
  int* device = nullptr;
  if (ordered_bad || ordered_good) {
    // Delay allocation stream so the unsynchronized consumer is observable.
    int* delay; CUDA(cudaMalloc(&delay, sizeof(int)));
    delayed_write<<<1,1,0,a>>>(delay); CUDA(cudaGetLastError());
    CUDA(cudaMallocAsync(&device, sizeof(int), a));
    cudaEvent_t event{}; CUDA(cudaEventCreate(&event));
    if (ordered_good) { CUDA(cudaEventRecord(event,a)); CUDA(cudaStreamWaitEvent(b,event,0)); }
    delayed_write<<<1,1,0,b>>>(device); CUDA(cudaGetLastError());
    CUDA(cudaStreamSynchronize(b)); CUDA(cudaStreamSynchronize(a));
    CUDA(cudaFreeAsync(device, b)); CUDA(cudaStreamSynchronize(b));
    CUDA(cudaEventDestroy(event)); CUDA(cudaFree(delay));
  } else {
    CUDA(cudaMalloc(&device, sizeof(int)));
    delayed_write<<<1,1,0,a>>>(device); CUDA(cudaGetLastError());
    auto start = std::chrono::steady_clock::now();
    if (ordinary_free) {
      CUDA(cudaFree(device)); device = nullptr;
      std::printf("{\"freeCallMilliseconds\":%.6f}\n", std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count());
    } else {
      int* host = nullptr;
      if (pageable) host = static_cast<int*>(std::malloc(sizeof(int)));
      else { CUDA(cudaMallocHost(&host, sizeof(int))); }
      if (!host) return 2;
      *host = -1;
      CUDA(cudaMemcpyAsync(host, device, sizeof(int), cudaMemcpyDeviceToHost, a));
      if (free_host) {
        start = std::chrono::steady_clock::now();
        CUDA(cudaFreeHost(host)); host = nullptr;
        std::printf("{\"freeHostCallMilliseconds\":%.6f}\n", std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count());
        CUDA(cudaStreamSynchronize(a));
      } else {
        int observed = *static_cast<volatile int*>(host);
        CUDA(cudaStreamSynchronize(a));
        std::printf("{\"immediate\":%d,\"afterSynchronize\":%d}\n", observed, *host);
        if (pageable) std::free(host); else { CUDA(cudaFreeHost(host)); }
      }
    }
    if (device) { CUDA(cudaFree(device)); }
  }
  CUDA(cudaStreamDestroy(a)); CUDA(cudaStreamDestroy(b));
  return 0;
}
