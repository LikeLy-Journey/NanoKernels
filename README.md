# NanoKernels

学习高性能 GPU 算子开发的练习仓库。每个算子从最朴素版本逐步优化,覆盖三套技术栈
(原始 CUDA / Triton / CuTe),并与 CPU 并行版本、torch 原生算子做性能对比,
全程结合 Nsight (nsys/ncu) 做微架构分析。

## 五段式分析法

| 阶段 | 实现 | 目的 |
|---|---|---|
| ① CPU baseline | 朴素 C++ / OpenMP | 正确性基准 + Roofline 下界 |
| ② CUDA naive | 一线程一元素 | 暴露访存/occupancy 问题 |
| ③ CUDA optimized | tiling / shared / cp.async / 向量化 | 逼近硬件极限 |
| ④ Triton | block-level 编程 | 手写 vs 编译器调优 |
| ⑤ CuTe / CUTLASS | layout/tensor 抽象 | 生产级抽象 |
| 基准 | torch 原生 | cuBLAS/cuDNN 天花板 |

## 学习路线(Level)

- **Level 0** 访存绑定:00_vector_add, transpose, elementwise
- **Level 1** Reduction:reduce, softmax, layernorm
- **Level 2** GEMM(核心):sgemm, hgemm_tensorcore, gemm_fused
- **Level 3** FlashAttention
- **Level 4** 量化/卷积等特色算子

## 工作流

```bash
# 1. 本地编写
# 2. 同步到 remote A100 节点
scripts/sync.sh
# 3. 在 remote 上选空闲 GPU 运行
cd 00_vector_add && scripts/../scripts/run_on_idle_gpu.sh make run
# 4. profile
ncu --set full -k "regex:vadd" -o vadd_report ./cuda/vadd_cuda
```

## 运行环境

- GPU: 8 × A100-SXM4-80GB (SM80)
- CUDA 12.9 / torch 2.8 / Triton 3.3.0
- remote 同步目录: /mnt/bn/jianglielin-yg/codes/NanoKernels_v0.1

## 目录约定

```
NN_<algo>/
├── cpu/            # OpenMP baseline
├── cuda/           # naive + optimized versions
├── triton/         # Triton 版
├── cute/           # CuTe 版(GEMM 及以后)
├── Makefile        # build + run
└── profile_notes.md
```
