// CPU baseline for vector add: y = a*x + y (SAXPY-style) and plain add.
// Compile: g++ -O3 -fopenmp -march=native vector_add_cpu.cpp -o vadd_cpu
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>
#include <cmath>
#ifdef _OPENMP
#include <omp.h>
#endif

static void vadd_cpu(const float* a, const float* b, float* c, int n) {
#pragma omp parallel for schedule(static)
  for (int i = 0; i < n; ++i) c[i] = a[i] + b[i];
}

int main(int argc, char** argv) {
  int n = (argc > 1) ? atoi(argv[1]) : (1 << 24);  // 16M elements default
  std::vector<float> a(n), b(n), c(n);
  for (int i = 0; i < n; ++i) { a[i] = 1.0f * i; b[i] = 2.0f * i; }

  // warmup
  vadd_cpu(a.data(), b.data(), c.data(), n);

  const int iters = 20;
  auto t0 = std::chrono::high_resolution_clock::now();
  for (int it = 0; it < iters; ++it) vadd_cpu(a.data(), b.data(), c.data(), n);
  auto t1 = std::chrono::high_resolution_clock::now();
  double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;

  // verify
  double max_err = 0.0;
  for (int i = 0; i < n; ++i) max_err = std::fmax(max_err, std::fabs(c[i] - (a[i] + b[i])));

  double bytes = 3.0 * n * sizeof(float);  // 2 read + 1 write
  double gbps = bytes / (ms * 1e-3) / 1e9;
#ifdef _OPENMP
  int threads = omp_get_max_threads();
#else
  int threads = 1;
#endif
  printf("[CPU ] n=%d threads=%d  %.4f ms  %.1f GB/s  max_err=%.3e\n",
         n, threads, ms, gbps, max_err);
  return 0;
}
