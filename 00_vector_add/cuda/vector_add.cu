// CUDA vector add: y = a + b
// Two versions:
//   v0_naive   : one thread per element, scalar loads
//   v1_float4  : vectorized float4 loads/stores (4 elems/thread), grid-stride
// Compile: nvcc -O3 -arch=sm_80 vector_add.cu -o vadd_cuda -I../../common
#include <cstdio>
#include <cstdlib>
#include "cuda_utils.cuh"

// ---------- v0: naive, one thread per element ----------
__global__ void vadd_naive(const float* __restrict__ a,
                           const float* __restrict__ b,
                           float* __restrict__ c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}

// ---------- v1: float4 vectorized + grid-stride loop ----------
__global__ void vadd_float4(const float4* __restrict__ a,
                            const float4* __restrict__ b,
                            float4* __restrict__ c, int n4) {
  int stride = blockDim.x * gridDim.x;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += stride) {
    float4 av = a[i];
    float4 bv = b[i];
    float4 cv;
    cv.x = av.x + bv.x;
    cv.y = av.y + bv.y;
    cv.z = av.z + bv.z;
    cv.w = av.w + bv.w;
    c[i] = cv;
  }
}

int main(int argc, char** argv) {
  int n = (argc > 1) ? atoi(argv[1]) : (1 << 24);  // 16M
  size_t bytes = (size_t)n * sizeof(float);

  float *ha = (float*)malloc(bytes), *hb = (float*)malloc(bytes), *hc = (float*)malloc(bytes);
  for (int i = 0; i < n; ++i) { ha[i] = 1.0f * i; hb[i] = 2.0f * i; }

  float *da, *db, *dc;
  CHECK_CUDA(cudaMalloc(&da, bytes));
  CHECK_CUDA(cudaMalloc(&db, bytes));
  CHECK_CUDA(cudaMalloc(&dc, bytes));
  CHECK_CUDA(cudaMemcpy(da, ha, bytes, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(db, hb, bytes, cudaMemcpyHostToDevice));

  const int block = 256;
  double total_bytes = 3.0 * n * sizeof(float);  // 2 read + 1 write

  // --- v0 naive ---
  {
    int grid = cdiv(n, block);
    auto run = [&]{ vadd_naive<<<grid, block>>>(da, db, dc, n); };
    run(); CHECK_LAST_CUDA();
    float ms = time_kernel(run);
    printf("[v0 naive  ] %.4f ms  %.1f GB/s\n", ms, bandwidth_gbps(total_bytes, ms));
  }

  // --- v1 float4 ---
  {
    int n4 = n / 4;  // assume n divisible by 4
    int grid = cdiv(n4, block);
    grid = grid > 1024 ? 1024 : grid;  // cap for grid-stride
    auto run = [&]{
      vadd_float4<<<grid, block>>>((const float4*)da, (const float4*)db, (float4*)dc, n4);
    };
    run(); CHECK_LAST_CUDA();
    float ms = time_kernel(run);
    printf("[v1 float4 ] %.4f ms  %.1f GB/s\n", ms, bandwidth_gbps(total_bytes, ms));
  }

  // verify v1 result
  CHECK_CUDA(cudaMemcpy(hc, dc, bytes, cudaMemcpyDeviceToHost));
  double max_err = 0.0;
  for (int i = 0; i < n; ++i) {
    double e = fabs((double)hc[i] - (double)(ha[i] + hb[i]));
    if (e > max_err) max_err = e;
  }
  printf("[verify    ] max_err=%.3e\n", max_err);

  cudaFree(da); cudaFree(db); cudaFree(dc);
  free(ha); free(hb); free(hc);
  return 0;
}
