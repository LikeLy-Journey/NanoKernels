# 01_transpose — profile notes

> 五段式分析法的第二个算子。Matrix transpose(4096×4096 fp32)仍是访存绑定,
> 但与 vector_add 不同:它的难点在**写出端的访存模式**——naive 版按列写出,
> 完全非合并(uncoalesced),带宽掉到峰值的 ~12%。这是 shared memory tiling +
> bank conflict 的经典教学算子。
> 数据移动量 = 2 × n × 4 字节(1 读 + 1 写),n = 4096×4096 = 16.78M,共 134.2 MB/iter。

## 各版本结果(4096×4096, fp32, GPU 4)

| 版本 | 时间 (ms) | 带宽 (GB/s) | % of copy 上限 | 备注 |
|---|---|---|---|---|
| CPU (OpenMP, 128 线程) | 20.47 | 6.6 | — | 非合并写 + cache 抖动,比 GPU naive 还慢 |
| CUDA copy(上限) | 0.0860 | 1560 | 100% | 纯拷贝,无转置,作为有效带宽天花板 |
| CUDA v0 naive | 0.5562 | 241 | 15.5% | 读合并 / **写按列非合并** → 32× 访存放大 |
| CUDA v1 tiled | 0.1339 | 1002 | 64.3% | 32×32 shared tile,读写都合并,但**列读 bank conflict** |
| CUDA v2 tiled+pad | 0.0869 | 1545 | 99.0% | tile 补到 [32][33],消除 bank conflict,**追平 copy** |
| Triton (BLOCK 32×32) | 0.0855 | 1570 | ~100% | tl.trans + block ptr,编译器自动消冲突 |
| torch (ref) | 0.2235 | 600 | 38.5% | `.t().contiguous()`,通用实现未做 tile 优化 |

> 时间口径:CUDA event 计时;nsys `cuda_gpu_kern_sum` 交叉验证完全吻合
> (naive 554µs / tiled 134µs / tiled+pad 86µs / copy 86µs)。正确性 max_err = 0.000e+00。

## 关键结论

1. **naive 的瓶颈是写出端非合并,不是读入端。** 读 `in[row,col]` 连续线程读连续地址(合并),
   但写 `out[col,row]` 是按列跨行写,相邻线程的写地址相隔 `rows` 个元素 → 每个 32B/128B
   事务只用上 4B,有效带宽掉到 241 GB/s(copy 上限的 ~15%)。这正是 transpose 区别于
   vector_add 的核心:**同样 memory-bound,但访存模式决定一切**。

2. **shared memory tiling 把"非合并写"换成"合并写 + 片上转置"(v0→v1, +4.2×)。** 先把一个
   32×32 块合并读入 shared,`__syncthreads` 后再合并写出,转置发生在片上。带宽从 241→1002 GB/s。
   但还没到顶——因为读 shared 时 `tile[threadIdx.x][...]` 是按列访问。

3. **bank conflict 是 v1 的最后一道坎,padding 一招解决(v1→v2, +1.54×)。** shared memory 有
   32 个 bank,`tile[32][32]` 时同一 warp 的 32 个线程读同一列恰好落在同一个 bank → 32-way
   conflict,串行 32 次。把 tile 补成 `[32][33]`,列元素错开到不同 bank,冲突消失,带宽
   1002→1545 GB/s,**追平纯 copy 上限(99%)**。代价仅 32×4=128B 额外 shared/block。

4. **手写 CUDA v2 与 Triton 几乎平手,且都远超 torch。** Triton 用 `tl.trans` + block ptr
   达到 1570 GB/s,与手写 v2 持平;torch 的 `.t().contiguous()` 是通用 path、未做 tile/padding,
   只有 600 GB/s。说明对 transpose 这类有明确访存模式的算子,**针对性优化能大幅超过通用库**。

5. **CPU 只有 6.6 GB/s,比 vector_add(14.6)更慢。** 因为非合并写在 CPU 上同样致命——
   按列写出击穿 cache line,DDR 有效带宽进一步恶化。

## nsys profile(可用,不受 DCGM 影响)

```bash
CUDA_VISIBLE_DEVICES=4 nsys profile --stats=true -o transpose_nsys --force-overwrite true \
    ./cuda/transpose_cuda 4096 4096
# 看 cuda_gpu_kern_sum:naive 554µs / tiled 134µs / tiled+pad 86µs / copy 86µs
```

实测 kernel 占比(nsys):naive 64.4% / tiled 15.6% / tiled+pad 10.0% / copy 10.0%,
直观印证 naive 是绝对热点,padding 后与 copy 同量级。

## ncu 分析要点(本节点暂不可用,DCGM 占用计数器)

```bash
# bank conflict 是本算子的核心指标,ncu 可用时重点看:
ncu --section MemoryWorkloadAnalysis --section Occupancy \
    -k "regex:transpose" ./cuda/transpose_cuda 4096 4096
```

关注指标(待独占节点 / DCGM 释放后补):
- `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum` —— shared load bank conflict 数,
  应见 v1 高、v2 ≈ 0(本算子最关键的对比指标)
- `l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum` / `..._st.avg.pct_of_peak` —— 写出端
  global 事务数与合并率,naive 应远高于 tiled
- `dram__throughput.avg.pct_of_peak_sustained_elapsed` —— v2 应 >95%
- `sm__warps_active.avg.pct_of_peak_sustained_active` —— occupancy

> **环境限制:** 与 00_vector_add 相同,本 A100 节点性能计数器被容器外宿主机层 DCGM 独占,
> ncu 报 `driver resource was unavailable ... DCGM`(code 9)。已验证容器内无 dcgmi /
> nv-hostengine 可暂停、`RmProfilingAdminOnly=0`(非权限)、sudo 提权无效——锁在宿主机层。
> 故改用 nsys(tracing)+ 实测带宽 + copy 上限三方对比定位瓶颈,结论链条完整:
> 非合并写(241)→ tiling(1002)→ 消 bank conflict(1545 ≈ copy 1560)。ncu 的
> bank-conflict 计数只是对第 3 步的定量佐证,不影响优化路径与结论。

## 下一步

- transpose 收尾:v2 已追平 copy 上限(99%),Triton 持平,优化路径完整闭环。
- 进入 Level 0 收官算子:**copy/memset 带宽基准**(确立 A100 实际 HBM 上限),
  或直接进 **Level 1 — reduce sum**(7 版本经典优化,引入 warp shuffle 与 block 级归约)。
