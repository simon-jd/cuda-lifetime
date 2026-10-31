// Environment observation only: never turn an observed behavior into a CUDA guarantee.
#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

static void checked(cudaError_t code, const char *call) {
  if (code != cudaSuccess) {
    std::fprintf(stderr, "%s: %s (%d)\n", call, cudaGetErrorString(code), int(code));
    throw std::runtime_error(call);
  }
}
#define CUDA(call) checked((call), #call)

__global__ void delayed_write(int *value, unsigned long long cycles) {
  auto start = clock64();
  while (clock64() - start < cycles) {}
  *value = 42;
}

static const char *query(cudaEvent_t event) {
  cudaError_t code = cudaEventQuery(event);
  if (code == cudaSuccess) return "ready";
  if (code == cudaErrorNotReady) return "not-ready";
  checked(code, "cudaEventQuery");
  return "error";
}

struct Resources {
  cudaStream_t stream{};
  cudaEvent_t producer{}, transfer{};
  int *device{}, *host{};
  bool pageable{};
  bool cleanup() {
    bool ok = true;
    auto release = [&](cudaError_t c, const char *name) {
      std::printf("{\"cleanup\":\"%s\",\"code\":%d}\n", name, int(c));
      ok = ok && c == cudaSuccess;
    };
    if (stream) release(cudaStreamSynchronize(stream), "stream-synchronize");
    if (host) {
      if (pageable) std::free(host);
      else release(cudaFreeHost(host), "host-free");
      host = nullptr;
    }
    if (device) release(cudaFree(device), "device-free");
    if (producer) release(cudaEventDestroy(producer), "producer-event-destroy");
    if (transfer) release(cudaEventDestroy(transfer), "transfer-event-destroy");
    if (stream) release(cudaStreamDestroy(stream), "stream-destroy");
    release(cudaDeviceReset(), "owned-context-reset");
    std::fflush(stdout);
    return ok;
  }
};

int main(int argc, char **argv) {
  if (argc != 2) return 2;
  const std::string mode = argv[1];
  bool pageable = mode == "pageable-observe" || mode == "pageable-synchronized";
  bool free_host = mode == "free-host-unsynchronized" || mode == "free-host-synchronized";
  bool sync = mode == "pinned-synchronized" || mode == "pageable-synchronized" || mode == "free-host-synchronized";
  if (!(pageable || free_host || mode == "pinned-observe" || mode == "pinned-synchronized")) return 2;
  Resources r;
  r.pageable = pageable;
  int result = 0;
  try {
    int driver{}, runtime{};
    cudaDeviceProp properties{};
    CUDA(cudaSetDevice(0));
    CUDA(cudaDriverGetVersion(&driver)); CUDA(cudaRuntimeGetVersion(&runtime));
    CUDA(cudaGetDeviceProperties(&properties, 0));
    // Names supplied by the driver are encoded by the Python collector as raw output.
    std::fprintf(stderr, "device=%s\n", properties.name);
    std::printf("{\"mode\":\"%s\",\"driverApiVersion\":%d,\"runtimeVersion\":%d,\"computeMajor\":%d,\"computeMinor\":%d,\"allocationBytes\":4096,\"delayCycles\":50000000}\n",
                mode.c_str(), driver, runtime, properties.major, properties.minor);
    CUDA(cudaStreamCreateWithFlags(&r.stream, cudaStreamNonBlocking));
    CUDA(cudaEventCreateWithFlags(&r.producer, cudaEventDisableTiming));
    CUDA(cudaEventCreateWithFlags(&r.transfer, cudaEventDisableTiming));
    CUDA(cudaMalloc(&r.device, 4096));
    if (pageable) r.host = static_cast<int *>(std::malloc(4096));
    else CUDA(cudaMallocHost(&r.host, 4096));
    if (!r.host) throw std::runtime_error("host allocation");
    *r.host = -1;
    // Exclude first kernel/module loading from the observation, without benchmarking it.
    delayed_write<<<1,1,0,r.stream>>>(r.device, 0);
    CUDA(cudaGetLastError()); CUDA(cudaStreamSynchronize(r.stream));
    delayed_write<<<1,1,0,r.stream>>>(r.device, 50000000ULL);
    CUDA(cudaGetLastError());
    CUDA(cudaEventRecord(r.producer, r.stream));
    auto start = std::chrono::steady_clock::now();
    CUDA(cudaMemcpyAsync(r.host, r.device, sizeof(int), cudaMemcpyDeviceToHost, r.stream));
    double copy_ms = std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
    CUDA(cudaEventRecord(r.transfer, r.stream));
    std::string producer_after_copy = query(r.producer);
    if (sync) CUDA(cudaStreamSynchronize(r.stream));
    std::string before = query(r.producer), after;
    double free_ms = 0;
    if (free_host) {
      // Intentional missing-completion variant. Never dereference the released host pointer.
      start = std::chrono::steady_clock::now();
      CUDA(cudaFreeHost(r.host)); r.host = nullptr;
      free_ms = std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
      after = query(r.producer);
    } else after = query(r.transfer);
    CUDA(cudaStreamSynchronize(r.stream));
    int value = r.host ? *r.host : -999; // Only inspect a live buffer after checked completion.
    if (r.host && value != 42) result = 3;
    std::printf("{\"observation\":true,\"mode\":\"%s\",\"producerAfterCopy\":\"%s\",\"producerBeforeAction\":\"%s\",\"eventAfterAction\":\"%s\",\"copyCallMilliseconds\":%.6f,\"freeCallMilliseconds\":%.6f,\"synchronizedValue\":%d,\"explicitSyncBeforeAction\":%s}\n",
                mode.c_str(),producer_after_copy.c_str(),before.c_str(),after.c_str(),copy_ms,free_ms,value,sync?"true":"false");
  } catch (const std::exception &e) {
    std::fprintf(stderr,"probe-error: %s\n",e.what()); result=2;
  }
  bool clean = r.cleanup();
  return clean ? result : 4;
}
