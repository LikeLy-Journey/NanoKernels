#pragma once
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// ---- Error checking ----
#define CHECK_CUDA(call)                                                      \
  do {                                                                        \
    cudaError_t _e = (call);                                                  \
    if (_e != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error %s:%d: '%s'\n", __FILE__, __LINE__,         \
              cudaGetErrorString(_e));                                        \
      exit(EXIT_FAILURE);                                                     \
    }                                                                         \
  } while (0)

#define CHECK_LAST_CUDA() CHECK_CUDA(cudaGetLastError())

// ---- GPU event timer (returns milliseconds) ----
struct GpuTimer {
  cudaEvent_t start_, stop_;
  GpuTimer()  { cudaEventCreate(&start_); cudaEventCreate(&stop_); }
  ~GpuTimer() { cudaEventDestroy(start_); cudaEventDestroy(stop_); }
  void start() { cudaEventRecord(start_, 0); }
  float stop() {
    cudaEventRecord(stop_, 0);
    cudaEventSynchronize(stop_);
    float ms = 0.f;
    cudaEventElapsedTime(&ms, start_, stop_);
    return ms;
  }
};

// ---- Time a kernel lambda: warmup + averaged repeats. Returns avg ms. ----
template <typename F>
float time_kernel(F&& fn, int warmup = 5, int repeats = 50) {
  for (int i = 0; i < warmup; ++i) fn();
  CHECK_CUDA(cudaDeviceSynchronize());
  GpuTimer t;
  t.start();
  for (int i = 0; i < repeats; ++i) fn();
  float ms = t.stop();
  return ms / repeats;
}

// ---- Effective bandwidth in GB/s given total bytes moved ----
inline double bandwidth_gbps(double bytes, double ms) {
  return bytes / (ms * 1e-3) / 1e9;
}

// ---- ceil division ----
inline int cdiv(int a, int b) { return (a + b - 1) / b; }
