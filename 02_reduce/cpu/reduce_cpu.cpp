// CPU baseline for reduce-sum: out = sum(in[0..n)).
// Uses OpenMP reduction; double accumulator for a stable reference value.
// Compile: g++ -O3 -fopenmp -march=native reduce_cpu.cpp -o reduce_cpu
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>
#include <cmath>
#ifdef _OPENMP
#include <omp.h>
#endif

static double reduce_cpu(const float* in, size_t n) {
  double sum = 0.0;
#pragma omp parallel for schedule(static) reduction(+ : sum)
  for (size_t i = 0; i < n; ++i) sum += (double)in[i];
  return sum;
}

int main(int argc, char** argv) {
  size_t n = (argc > 1) ? (size_t)atoll(argv[1]) : (1u << 24);  // 16M
  std::vector<float> in(n);
  // values in [0,1): keep the exact sum bounded and comparable across versions
  for (size_t i = 0; i < n; ++i) in[i] = (float)((i % 1000) * 0.001);

  double ref = reduce_cpu(in.data(), n);  // warmup + reference

  const int iters = 20;
  auto t0 = std::chrono::high_resolution_clock::now();
  volatile double sink = 0.0;
  for (int it = 0; it < iters; ++it) sink += reduce_cpu(in.data(), n);
  auto t1 = std::chrono::high_resolution_clock::now();
  (void)sink;
  double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;

  double bytes = (double)n * sizeof(float);  // read all elements once
  double gbps = bytes / (ms * 1e-3) / 1e9;
#ifdef _OPENMP
  int threads = omp_get_max_threads();
#else
  int threads = 1;
#endif
  printf("[CPU ] n=%zu threads=%d  %.4f ms  %.1f GB/s  sum=%.6f\n",
         n, threads, ms, gbps, ref);
  return 0;
}
