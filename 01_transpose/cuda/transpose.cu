// CUDA matrix transpose: out[NxM] = in[MxN]^T  (row-major)
// Three versions to teach the classic shared-memory / bank-conflict lesson:
//   v0_naive       : direct index swap -> coalesced READ, strided (non-coalesced) WRITE
//   v1_tiled       : 32x32 shared-memory tile -> coalesced read AND write,
//                    but the tile column access causes 32-way bank conflicts
//   v2_tiled_pad   : same tile padded to [32][33] -> bank conflicts eliminated
// Plus a copy kernel as the effective-bandwidth ceiling (no transpose, pure mem).
// Compile: nvcc -O3 -arch=sm_80 transpose.cu -o transpose_cuda -I../common
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include "cuda_utils.cuh"

#define TILE 32
#define BLOCK_ROWS 8   // each thread handles TILE/BLOCK_ROWS = 4 rows

// ---------- v0: naive ----------
// Read in[row,col] coalesced; write out[col,row] is column-strided -> uncoalesced.
__global__ void transpose_naive(const float* __restrict__ in,
                                float* __restrict__ out, int rows, int cols) {
  int x = blockIdx.x * TILE + threadIdx.x;  // col in input
  int y = blockIdx.y * TILE + threadIdx.y;  // row in input
  for (int j = 0; j < TILE; j += BLOCK_ROWS) {
    if (x < cols && (y + j) < rows)
      out[(size_t)x * rows + (y + j)] = in[(size_t)(y + j) * cols + x];
  }
}

// ---------- v1: shared-memory tiled (bank conflicts on tile[ty][tx] read) ----------
__global__ void transpose_tiled(const float* __restrict__ in,
                                float* __restrict__ out, int rows, int cols) {
  __shared__ float tile[TILE][TILE];
  int x = blockIdx.x * TILE + threadIdx.x;
  int y = blockIdx.y * TILE + threadIdx.y;
  // coalesced read from in into shared tile
  for (int j = 0; j < TILE; j += BLOCK_ROWS)
    if (x < cols && (y + j) < rows)
      tile[threadIdx.y + j][threadIdx.x] = in[(size_t)(y + j) * cols + x];
  __syncthreads();
  // transposed block coordinates
  x = blockIdx.y * TILE + threadIdx.x;  // col in output
  y = blockIdx.x * TILE + threadIdx.y;  // row in output
  // coalesced write; tile[threadIdx.x][...] reads a column -> 32-way bank conflict
  for (int j = 0; j < TILE; j += BLOCK_ROWS)
    if (x < rows && (y + j) < cols)
      out[(size_t)(y + j) * rows + x] = tile[threadIdx.x][threadIdx.y + j];
}

// ---------- v2: shared-memory tiled + padding (bank-conflict free) ----------
__global__ void transpose_tiled_pad(const float* __restrict__ in,
                                    float* __restrict__ out, int rows, int cols) {
  __shared__ float tile[TILE][TILE + 1];  // +1 padding shifts banks, removes conflict
  int x = blockIdx.x * TILE + threadIdx.x;
  int y = blockIdx.y * TILE + threadIdx.y;
  for (int j = 0; j < TILE; j += BLOCK_ROWS)
    if (x < cols && (y + j) < rows)
      tile[threadIdx.y + j][threadIdx.x] = in[(size_t)(y + j) * cols + x];
  __syncthreads();
  x = blockIdx.y * TILE + threadIdx.x;
  y = blockIdx.x * TILE + threadIdx.y;
  for (int j = 0; j < TILE; j += BLOCK_ROWS)
    if (x < rows && (y + j) < cols)
      out[(size_t)(y + j) * rows + x] = tile[threadIdx.x][threadIdx.y + j];
}

// ---------- copy: bandwidth ceiling (no transpose) ----------
__global__ void copy_kernel(const float* __restrict__ in,
                            float* __restrict__ out, int rows, int cols) {
  int x = blockIdx.x * TILE + threadIdx.x;
  int y = blockIdx.y * TILE + threadIdx.y;
  for (int j = 0; j < TILE; j += BLOCK_ROWS)
    if (x < cols && (y + j) < rows) {
      size_t idx = (size_t)(y + j) * cols + x;
      out[idx] = in[idx];
    }
}

static bool verify(const float* h_in, const float* h_out, int rows, int cols) {
  double max_err = 0.0;
  for (int i = 0; i < rows; i += 97)
    for (int j = 0; j < cols; j += 89) {
      double e = fabs((double)h_out[(size_t)j * rows + i] - (double)h_in[(size_t)i * cols + j]);
      if (e > max_err) max_err = e;
    }
  printf("    verify max_err=%.3e\n", max_err);
  return max_err == 0.0;
}

int main(int argc, char** argv) {
  int rows = (argc > 1) ? atoi(argv[1]) : 4096;
  int cols = (argc > 2) ? atoi(argv[2]) : 4096;
  size_t n = (size_t)rows * cols;
  size_t bytes = n * sizeof(float);

  float *h_in = (float*)malloc(bytes), *h_out = (float*)malloc(bytes);
  for (size_t i = 0; i < n; ++i) h_in[i] = (float)(i % 1000);

  float *d_in, *d_out;
  CHECK_CUDA(cudaMalloc(&d_in, bytes));
  CHECK_CUDA(cudaMalloc(&d_out, bytes));
  CHECK_CUDA(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

  dim3 block(TILE, BLOCK_ROWS);
  dim3 grid(cdiv(cols, TILE), cdiv(rows, TILE));
  double total_bytes = 2.0 * n * sizeof(float);  // 1 read + 1 write

  printf("transpose %dx%d  (move %.1f MB/iter)\n", rows, cols, total_bytes / 1e6);

  // copy ceiling
  {
    auto run = [&]{ copy_kernel<<<grid, block>>>(d_in, d_out, rows, cols); };
    run(); CHECK_LAST_CUDA();
    float ms = time_kernel(run);
    printf("[copy       ] %.4f ms  %.1f GB/s  (ceiling)\n", ms, bandwidth_gbps(total_bytes, ms));
  }
  // v0 naive
  {
    auto run = [&]{ transpose_naive<<<grid, block>>>(d_in, d_out, rows, cols); };
    run(); CHECK_LAST_CUDA();
    float ms = time_kernel(run);
    CHECK_CUDA(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
    printf("[v0 naive   ] %.4f ms  %.1f GB/s\n", ms, bandwidth_gbps(total_bytes, ms));
    verify(h_in, h_out, rows, cols);
  }
  // v1 tiled (bank conflicts)
  {
    CHECK_CUDA(cudaMemset(d_out, 0, bytes));
    auto run = [&]{ transpose_tiled<<<grid, block>>>(d_in, d_out, rows, cols); };
    run(); CHECK_LAST_CUDA();
    float ms = time_kernel(run);
    CHECK_CUDA(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
    printf("[v1 tiled   ] %.4f ms  %.1f GB/s\n", ms, bandwidth_gbps(total_bytes, ms));
    verify(h_in, h_out, rows, cols);
  }
  // v2 tiled + padding (no bank conflicts)
  {
    CHECK_CUDA(cudaMemset(d_out, 0, bytes));
    auto run = [&]{ transpose_tiled_pad<<<grid, block>>>(d_in, d_out, rows, cols); };
    run(); CHECK_LAST_CUDA();
    float ms = time_kernel(run);
    CHECK_CUDA(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
    printf("[v2 tiled+pad] %.4f ms  %.1f GB/s\n", ms, bandwidth_gbps(total_bytes, ms));
    verify(h_in, h_out, rows, cols);
  }

  cudaFree(d_in); cudaFree(d_out);
  free(h_in); free(h_out);
  return 0;
}
