// CUDA reduce-sum: the classic Mark Harris 7-version optimization ladder.
// Each block reduces its chunk to one partial in d_partials[blockIdx.x];
// the host sums the (small) partials array for verification. Only the main
// kernel is timed. Bandwidth = n * 4B (every input element read once).
//
//   v0 interleaved + modulo : warp divergence + shared bank conflicts
//   v1 interleaved strided   : no divergence, still bank conflicts
//   v2 sequential addressing : bank-conflict free
//   v3 first add on load     : half the blocks, 1 add fused into the load
//   v4 unroll last warp      : drop __syncthreads in the warp-synchronous tail
//   v5 warp shuffle          : __shfl_down_sync, no shared for the final warp
//   v6 grid-stride + shuffle : each thread grabs many elements, then v5
//
// Compile: nvcc -O3 -arch=sm_80 reduce.cu -o reduce_cuda -I../common
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include "cuda_utils.cuh"

#define BLOCK 256

// ---------- v0: interleaved addressing with modulo (divergent) ----------
__global__ void reduce_v0(const float* __restrict__ in, float* __restrict__ out, int n) {
  __shared__ float sdata[BLOCK];
  int tid = threadIdx.x;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  sdata[tid] = (i < n) ? in[i] : 0.0f;
  __syncthreads();
  // stride doubles; active threads selected by modulo -> heavy warp divergence
  for (int s = 1; s < blockDim.x; s *= 2) {
    if (tid % (2 * s) == 0) sdata[tid] += sdata[tid + s];
    __syncthreads();
  }
  if (tid == 0) out[blockIdx.x] = sdata[0];
}

// ---------- v1: interleaved, strided index (no divergence, bank conflicts) ----------
__global__ void reduce_v1(const float* __restrict__ in, float* __restrict__ out, int n) {
  __shared__ float sdata[BLOCK];
  int tid = threadIdx.x;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  sdata[tid] = (i < n) ? in[i] : 0.0f;
  __syncthreads();
  for (int s = 1; s < blockDim.x; s *= 2) {
    int index = 2 * s * tid;          // contiguous active threads
    if (index < blockDim.x) sdata[index] += sdata[index + s];
    __syncthreads();
  }
  if (tid == 0) out[blockIdx.x] = sdata[0];
}

// ---------- v2: sequential addressing (bank-conflict free) ----------
__global__ void reduce_v2(const float* __restrict__ in, float* __restrict__ out, int n) {
  __shared__ float sdata[BLOCK];
  int tid = threadIdx.x;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  sdata[tid] = (i < n) ? in[i] : 0.0f;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) sdata[tid] += sdata[tid + s];
    __syncthreads();
  }
  if (tid == 0) out[blockIdx.x] = sdata[0];
}

// ---------- v3: first add during load (grid handles 2*BLOCK per block) ----------
__global__ void reduce_v3(const float* __restrict__ in, float* __restrict__ out, int n) {
  __shared__ float sdata[BLOCK];
  int tid = threadIdx.x;
  int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
  float v = (i < n) ? in[i] : 0.0f;
  if (i + blockDim.x < n) v += in[i + blockDim.x];   // one add fused into load
  sdata[tid] = v;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) sdata[tid] += sdata[tid + s];
    __syncthreads();
  }
  if (tid == 0) out[blockIdx.x] = sdata[0];
}

// ---------- v4: unroll the last warp (no __syncthreads once s<=32) ----------
__device__ void warpReduce_v4(volatile float* sdata, int tid) {
  sdata[tid] += sdata[tid + 32];
  sdata[tid] += sdata[tid + 16];
  sdata[tid] += sdata[tid + 8];
  sdata[tid] += sdata[tid + 4];
  sdata[tid] += sdata[tid + 2];
  sdata[tid] += sdata[tid + 1];
}
__global__ void reduce_v4(const float* __restrict__ in, float* __restrict__ out, int n) {
  __shared__ float sdata[BLOCK];
  int tid = threadIdx.x;
  int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
  float v = (i < n) ? in[i] : 0.0f;
  if (i + blockDim.x < n) v += in[i + blockDim.x];
  sdata[tid] = v;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 32; s >>= 1) {
    if (tid < s) sdata[tid] += sdata[tid + s];
    __syncthreads();
  }
  if (tid < 32) warpReduce_v4(sdata, tid);
  if (tid == 0) out[blockIdx.x] = sdata[0];
}

// ---------- v5: warp shuffle (__shfl_down_sync), shared only across warps ----------
__inline__ __device__ float warpReduceSum(float val) {
  for (int offset = 16; offset > 0; offset >>= 1)
    val += __shfl_down_sync(0xffffffff, val, offset);
  return val;
}
__global__ void reduce_v5(const float* __restrict__ in, float* __restrict__ out, int n) {
  __shared__ float warpsum[BLOCK / 32];
  int tid = threadIdx.x;
  int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
  float v = (i < n) ? in[i] : 0.0f;
  if (i + blockDim.x < n) v += in[i + blockDim.x];
  v = warpReduceSum(v);                       // reduce within each warp
  int lane = tid & 31, wid = tid >> 5;
  if (lane == 0) warpsum[wid] = v;            // one value per warp -> shared
  __syncthreads();
  if (wid == 0) {                             // first warp reduces the warp sums
    v = (tid < blockDim.x / 32) ? warpsum[lane] : 0.0f;
    v = warpReduceSum(v);
    if (lane == 0) out[blockIdx.x] = v;
  }
}

// ---------- v6: grid-stride multi-element load + warp shuffle ----------
__global__ void reduce_v6(const float* __restrict__ in, float* __restrict__ out, int n) {
  __shared__ float warpsum[BLOCK / 32];
  int tid = threadIdx.x;
  // each thread accumulates a grid-stride slice, decoupling work from grid size
  float v = 0.0f;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + tid;
       i < (size_t)n; i += (size_t)blockDim.x * gridDim.x)
    v += in[i];
  v = warpReduceSum(v);
  int lane = tid & 31, wid = tid >> 5;
  if (lane == 0) warpsum[wid] = v;
  __syncthreads();
  if (wid == 0) {
    v = (tid < blockDim.x / 32) ? warpsum[lane] : 0.0f;
    v = warpReduceSum(v);
    if (lane == 0) out[blockIdx.x] = v;
  }
}

static double sum_partials(const float* d_partials, int blocks) {
  float* h = (float*)malloc(blocks * sizeof(float));
  CHECK_CUDA(cudaMemcpy(h, d_partials, blocks * sizeof(float), cudaMemcpyDeviceToHost));
  double s = 0.0;
  for (int i = 0; i < blocks; ++i) s += (double)h[i];
  free(h);
  return s;
}

int main(int argc, char** argv) {
  int n = (argc > 1) ? atoi(argv[1]) : (1 << 24);   // 16M
  size_t bytes = (size_t)n * sizeof(float);

  float* h_in = (float*)malloc(bytes);
  double ref = 0.0;
  for (int i = 0; i < n; ++i) { h_in[i] = (float)((i % 1000) * 0.001); ref += h_in[i]; }

  float *d_in, *d_partials;
  CHECK_CUDA(cudaMalloc(&d_in, bytes));
  CHECK_CUDA(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

  double rbytes = (double)n * sizeof(float);  // every element read once
  printf("reduce-sum n=%d  (read %.1f MB)  ref=%.6f\n", n, rbytes / 1e6, ref);

  auto check = [&](const char* tag, double got, float ms, int blocks) {
    double rel = fabs(got - ref) / fabs(ref);
    printf("[%-22s] %.4f ms  %8.1f GB/s  sum=%.4f  rel_err=%.2e %s\n",
           tag, ms, bandwidth_gbps(rbytes, ms), got, rel,
           rel < 1e-3 ? "OK" : "FAIL");
  };

  // ---- shared-memory ladder v0..v4: one element (v0..v2) or 2 (v3..v4) per thread ----
  {
    int blocks = cdiv(n, BLOCK);
    CHECK_CUDA(cudaMalloc(&d_partials, blocks * sizeof(float)));
    auto run0 = [&]{ reduce_v0<<<blocks, BLOCK>>>(d_in, d_partials, n); };
    run0(); CHECK_LAST_CUDA(); float t0 = time_kernel(run0);
    check("v0 modulo divergent", sum_partials(d_partials, blocks), t0, blocks);

    auto run1 = [&]{ reduce_v1<<<blocks, BLOCK>>>(d_in, d_partials, n); };
    run1(); float t1 = time_kernel(run1);
    check("v1 strided index", sum_partials(d_partials, blocks), t1, blocks);

    auto run2 = [&]{ reduce_v2<<<blocks, BLOCK>>>(d_in, d_partials, n); };
    run2(); float t2 = time_kernel(run2);
    check("v2 sequential addr", sum_partials(d_partials, blocks), t2, blocks);
    cudaFree(d_partials);
  }
  {
    int blocks = cdiv(n, BLOCK * 2);  // v3,v4 handle 2*BLOCK per block
    CHECK_CUDA(cudaMalloc(&d_partials, blocks * sizeof(float)));
    auto run3 = [&]{ reduce_v3<<<blocks, BLOCK>>>(d_in, d_partials, n); };
    run3(); float t3 = time_kernel(run3);
    check("v3 first-add-on-load", sum_partials(d_partials, blocks), t3, blocks);

    auto run4 = [&]{ reduce_v4<<<blocks, BLOCK>>>(d_in, d_partials, n); };
    run4(); float t4 = time_kernel(run4);
    check("v4 unroll last warp", sum_partials(d_partials, blocks), t4, blocks);

    auto run5 = [&]{ reduce_v5<<<blocks, BLOCK>>>(d_in, d_partials, n); };
    run5(); float t5 = time_kernel(run5);
    check("v5 warp shuffle", sum_partials(d_partials, blocks), t5, blocks);
    cudaFree(d_partials);
  }
  {
    // v6: cap grid (grid-stride), e.g. enough blocks to saturate 108 SMs
    int blocks = 1024;
    CHECK_CUDA(cudaMalloc(&d_partials, blocks * sizeof(float)));
    auto run6 = [&]{ reduce_v6<<<blocks, BLOCK>>>(d_in, d_partials, n); };
    run6(); float t6 = time_kernel(run6);
    check("v6 grid-stride+shuffle", sum_partials(d_partials, blocks), t6, blocks);
    cudaFree(d_partials);
  }

  cudaFree(d_in);
  free(h_in);
  return 0;
}
