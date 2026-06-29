// CPU baseline for matrix transpose: out[j*rows + i] = in[i*cols + j]
// Row-major MxN -> NxM.
// Compile: g++ -O3 -fopenmp -march=native transpose_cpu.cpp -o transpose_cpu
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>
#include <cmath>
#ifdef _OPENMP
#include <omp.h>
#endif

// Naive transpose: read row-wise from in, scattered write to out.
static void transpose_cpu(const float* in, float* out, int rows, int cols) {
#pragma omp parallel for schedule(static)
  for (int i = 0; i < rows; ++i)
    for (int j = 0; j < cols; ++j)
      out[(size_t)j * rows + i] = in[(size_t)i * cols + j];
}

int main(int argc, char** argv) {
  int rows = (argc > 1) ? atoi(argv[1]) : 4096;
  int cols = (argc > 2) ? atoi(argv[2]) : 4096;
  size_t n = (size_t)rows * cols;
  std::vector<float> in(n), out(n);
  for (size_t i = 0; i < n; ++i) in[i] = (float)(i % 1000);

  // warmup
  transpose_cpu(in.data(), out.data(), rows, cols);

  const int iters = 20;
  auto t0 = std::chrono::high_resolution_clock::now();
  for (int it = 0; it < iters; ++it) transpose_cpu(in.data(), out.data(), rows, cols);
  auto t1 = std::chrono::high_resolution_clock::now();
  double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;

  // verify a few elements
  double max_err = 0.0;
  for (int i = 0; i < rows; i += 97)
    for (int j = 0; j < cols; j += 89) {
      double e = std::fabs(out[(size_t)j * rows + i] - in[(size_t)i * cols + j]);
      max_err = std::fmax(max_err, e);
    }

  double bytes = 2.0 * n * sizeof(float);  // 1 read + 1 write
  double gbps = bytes / (ms * 1e-3) / 1e9;
#ifdef _OPENMP
  int threads = omp_get_max_threads();
#else
  int threads = 1;
#endif
  printf("[CPU ] %dx%d threads=%d  %.4f ms  %.1f GB/s  max_err=%.3e\n",
         rows, cols, threads, ms, gbps, max_err);
  return 0;
}
