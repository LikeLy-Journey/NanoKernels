# 02_reduce — profile notes

> 五段式分析法的第三个算子。Reduce-sum(n = 16M, fp32)是访存绑定算子,但与
> vector_add / transpose 不同:它的难点在**跨线程协作**——把 N 个值归约成 1 个,
> 需要 block 内树形归约 + warp 级洗牌 + block 间二次归约。这是 Mark Harris 经典
> 7 版优化阶梯,从 warp divergence → bank conflict → warp shuffle 层层递进。
> 数据移动量 = n × 4 字节(每个元素读一次),n = 1<<24 = 16.78M,共 67.1 MB。

## 各版本结果(n=16M, fp32, A100-SXM4-80GB)

| 版本 | 时间 (ms) | 带宽 (GB/s) | % of v5 | 备注 |
|---|---|---|---|---|
| CPU (OpenMP, 124 线程) | 7.393 | 9.1 | — | double 累加器,稳定参考值 |
| CUDA v0 modulo divergent | 0.3140 | 213.7 | 15% | `tid % (2*s)`,warp divergence + bank conflict,绝对热点 |
| CUDA v1 strided index | 0.1674 | 400.8 | 28% | 连续 active 线程,消除 divergence,仍有 bank conflict |
| CUDA v2 sequential addr | 0.1287 | 521.5 | 36% | 顺序寻址,bank-conflict free |
| CUDA v3 first-add-on-load | 0.0699 | 960.1 | 67% | load 时先加一次,block 数减半 |
| CUDA v4 unroll last warp | 0.0513 | 1307.8 | 91% | s<=32 后去掉 __syncthreads(warp 同步) |
| **CUDA v5 warp shuffle** | **0.0470** | **1429.2** | **100%** | `__shfl_down_sync`,最后一个 warp 不用 shared,**最快** |
| CUDA v6 grid-stride+shuffle | 0.0764 | 878.7 | 61% | 固定 1024 block,本配置下欠订阅(见结论 7) |
| Triton (tl.sum) | 0.0465 | 1442.7 | ~101% | 编译器自动归约,与 v5 持平,1.06× vs torch |
| torch (ref) | 0.0494 | 1359.7 | 95% | torch.sum |

> 时间口径:CUDA event(warmup+repeat 平均)计时;nsys `cuda_gpu_kern_sum` 交叉验证完全吻合
> (v0 322µs / v1 168 / v2 139 / v3 74 / v4 55 / v5 50 / v6 76µs)。正确性 rel_err ~1e-9(全部 OK)。
>
> **测量稳定性:** 同一组数字在「8 卡满载的共享节点」与「完全空闲的独占 GPU」上多次重跑逐位一致
> (v5 1429–1435 GB/s)——kernel 极短、用平均计时,能拿到干净时间片,结论数据可信。
> v5 的 1435 GB/s ≈ A100 标称带宽(~2039 GB/s)的 70%,这是 **reduce 的固有上限**:只读一遍 +
> 输出极小 + kernel launch/occupancy 开销,达不到 copy 那种 ~99% 峰值,属正常。

## 关键结论

1. **v0 的瓶颈:warp divergence + bank conflict(213 GB/s,绝对热点)。** `if (tid % (2*s) == 0)`
   让同一 warp 内只有部分线程活跃,且活跃线程随 stride 翻倍越来越稀疏 → 严重 warp 分歧;
   `sdata[tid] += sdata[tid+s]` 的跨步访问还引入 shared bank conflict。nsys 占 36.4%,
   是第二名(v1)的近 2 倍,典型"最慢环节"。

2. **v1 strided index 消除 divergence(+1.88×,400 GB/s)。** 改用 `index = 2*s*tid` 让活跃
   线程连续排布,同一 warp 要么全活跃要么全不活跃 → 无分歧。但 `sdata[index] += sdata[index+s]`
   仍是跨步访问 → bank conflict 未解,只到 28%。

3. **v2 sequential addressing 消除 bank conflict(+1.30×,521 GB/s)。** 倒序 stride
   `s = blockDim.x/2 → 1`,`if (tid < s) sdata[tid] += sdata[tid+s]`,活跃线程连续且访问连续
   → bank-conflict free。但此时活跃线程数随 stride 减半,后期半数 warp 闲置。

4. **v3 first-add-on-load 砍掉一半 block(+1.84×,960 GB/s)。** 每个 block 处理 2*BLOCK 元素,
   load 阶段就先做一次加法 → 启动 block 数减半,kernel launch / 全程闲置线程开销大幅下降。
   这是单步增益第二大的一跳(仅次于 v0→v1)。

5. **v4 unroll last warp 去掉尾部同步(+1.36×,1307 GB/s)。** 当 s<=32 时只剩一个 warp 活跃,
   warp 内天然同步(lock-step),`__syncthreads()` 成为纯开销 → 用 volatile 指针手动展开
   最后 6 步(32→16→8→4→2→1),省掉 5 次全 block barrier。

6. **v5 warp shuffle 彻底告别 shared(warp 内)(+1.09×,1429 GB/s,峰值)。** `__shfl_down_sync`
   直接在寄存器间交换,warp 内归约不走 shared、不需 barrier;只用 BLOCK/32=8 个 shared 槽存
   各 warp 的部分和,首个 warp 再 shuffle 归约一次。已逼近本节点可达带宽,与 Triton/torch 同档。

7. **v6 grid-stride 在本配置下反而慢(878 GB/s,61%)。** 固定 `blocks=1024`、每线程 grid-stride
   吃多个元素——其优势在于**解耦 work 与 grid 大小**(适合 n 极大或多次复用的场景),但 1024 block
   × 256 线程 = 262k 线程,对 16M 元素来说每线程要串行扫 ~64 个元素,且 1024 block 未充分喂满
   108 个 SM 的调度 → 欠订阅。**教训:grid-stride 不是无脑更快,block 数要按 `SM数 × 每SM驻留block`
   调够**;把 blocks 提到 ~数千(或按 `cudaOccupancyMaxActiveBlocksPerMultiprocessor` 算)才会反超 v5。

8. **手写 v5 与 Triton / torch 基本平手。** Triton `tl.sum` 1442、torch.sum 1359、手写 v5 1429,
   三者同档。说明 reduce 这种已被库高度优化的基础算子,手写到 warp shuffle 即可追平通用实现;
   再往上(v6 调参 / 多级归约 / vectorized load)才是进一步压榨的空间。

## 优化阶梯总览

```
v0 modulo      213  ─┐ 消 warp divergence
v1 strided     400  ─┘ +1.88×
v2 sequential  521     消 bank conflict          +1.30×
v3 first-add   960     load 时先加,block 减半     +1.84×
v4 unroll-warp 1307    去尾部 __syncthreads        +1.36×
v5 shuffle     1429    寄存器洗牌,弃 shared        +1.09×  ← 峰值
v6 grid-stride 878     (本配置欠订阅,需调 block 数)
```
v0→v5 累计 **6.7×**。前两跳(divergence、block 数)收益最大,后段(bank、barrier、shuffle)
逐步收尾——印证"瓶颈转移"规律:每消一个最慢环节,下一个浮现成新瓶颈。

## nsys profile(可用,不受 DCGM 影响)

```bash
CUDA_VISIBLE_DEVICES=0 nsys profile --stats=true -o reduce_nsys --force-overwrite true \
    ./cuda/reduce_cuda 16777216
# 看 cuda_gpu_kern_sum:v0 322µs / v1 168 / v2 139 / v3 74 / v4 55 / v5 50 / v6 76µs
```

实测 kernel 占比(nsys):v0 36.4% / v1 19.0% / v2 15.7% / v6 8.6% / v3 8.4% / v4 6.3% / v5 5.7%,
直观印证 v0 是绝对热点(吃掉 1/3+ 总时间),v5 最省时。

## ncu 分析要点(本节点暂不可用,DCGM 占用计数器)

```bash
# warp divergence / bank conflict 是本算子核心指标:
ncu --section WarpStateStats --section MemoryWorkloadAnalysis \
    -k "regex:reduce" ./cuda/reduce_cuda 16777216
```

关注指标(待独占节点 / DCGM 释放后补):
- `smsp__thread_inst_executed_per_inst_executed.ratio` —— 分支效率,v0 应低、v1+ 接近 32
- `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum` —— v0/v1 高、v2+ ≈ 0
- `sm__warps_active.avg.pct_of_peak_sustained_active` —— occupancy,调够 block 后 v6 应最高

> **环境限制:** 与 00/01 相同,A100 节点性能计数器被宿主机层 DCGM 独占,ncu 报 code 9。
> 改用 nsys + 实测带宽 + 各版本相对占比定位瓶颈,结论链条完整:
> divergence(213)→ 连续线程(400)→ 消 bank conflict(521)→ block 减半(960)→
> 去 barrier(1307)→ warp shuffle(1429)。

## 下一步

- reduce 收尾后进入 Level 1 后续算子:**softmax**(行归约 + 数值稳定 max-shift)
  或 **layernorm**(均值/方差两遍归约),复用本节的 warp shuffle / block 归约基建。
- 可选回补:把 v6 的 block 数调到 occupancy 上限(`cudaOccupancyMaxActiveBlocksPerMultiprocessor`)验证其反超 v5。
