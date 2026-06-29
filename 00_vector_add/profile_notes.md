# 00_vector_add — profile notes

> 五段式分析法的第一个算子。Vector add 是纯访存绑定(memory-bound)算子,
> 算术强度极低(每 12 字节只有 1 次 FLOP, AI ≈ 0.083 FLOP/Byte),性能上限 = HBM 带宽。
> A100-SXM4-80GB 理论带宽约 2039 GB/s,实测可达约 1900+ GB/s。

## 各版本结果(n = 16M, fp32, GPU 4)

| 版本 | 时间 (ms) | 带宽 (GB/s) | % of 2039 peak | 备注 |
|---|---|---|---|---|
| CPU (OpenMP, 128 线程) | 13.83 | 14.6 | — | DDR 带宽上限,比 GPU 慢 ~112× |
| CUDA v0 naive | 0.1234 | 1631 | 80.0% | 一线程一元素 |
| CUDA v1 float4 | 0.1185 | 1699 | 83.3% | 向量化 + grid-stride |
| Triton (BLOCK=1024) | 0.1170 | 1720 | 84.4% | block-level,编译器调优 |
| torch (ref) | 0.1165 | 1729 | 84.8% | ATen element-wise(天花板) |

> 时间口径:CUDA/Triton 为 CUDA event 计时;另用 nsys `cuda_gpu_kern_sum` 交叉验证
> (naive 123.4 µs / float4 118.5 µs,完全吻合)。正确性 max_err = 0.000e+00。

## 关键结论

1. **彻底 memory-bound,已逼近 HBM 上限。** 四个 GPU 版本带宽都落在 1631–1729 GB/s,
   即峰值的 80%–85%。这正是 memory-bound 算子的健康区间——剩下的 ~15% 缺口来自
   kernel 启动/收尾 (tail effect)、ECC 开销与 HBM 刷新,在如此短(~0.12 ms)的 kernel 上无法消除。
2. **float4 向量化收益有限(+4%)。** naive 已经是完美合并访存(连续线程读连续地址),
   float4 的增益主要来自更少的访存指令数与更高的 in-flight 请求数(MLP),而非合并质量。
   在 memory-bound 且 naive 已打满带宽的场景,向量化的天花板就在这。
3. **手写 CUDA 与 torch/Triton 几乎平手。** 说明对于这种平凡算子,编译器/库不存在魔法,
   带宽就是物理上限;真正拉开差距的是后面 compute-bound 的 GEMM/Attention。
4. **CPU 比 GPU 慢约 112×。** 直接量化了 HBM(~2 TB/s)对 DDR(~15 GB/s 有效)的代际差距,
   也印证 Roofline 下界。

## 如何用 nsys / ncu 定位 naive 算子的问题(方法论)

> 本节回答一个核心问题:**拿到一个 kernel,怎么用工具判断它"卡在哪、还有没有救"?**
> 对 memory-bound 算子,诊断流程是固定的三步漏斗:先用 nsys 看"是不是这个 kernel 慢"
> → 再用 ncu 看"它在等什么(stall 原因)" → 最后看"带宽/合并率到没到顶"。

### 第 1 步:nsys —— 定位热点、量化耗时(tracing,不占计数器)

nsys 是 timeline tracing 工具,**不抓硬件计数器**,因此在被 DCGM 占用的共享节点上仍可用。
它回答"时间花在哪",但不回答"为什么慢"。

```bash
CUDA_VISIBLE_DEVICES=4 nsys profile --stats=true -o vadd_nsys --force-overwrite true \
    ./cuda/vadd_cuda 16777216
```

看 `cuda_gpu_kern_sum` 段(按 kernel 聚合的 GPU 时间):

```
 Time(%)  Total Time   Instances   Avg (ns)   Name
   51.0     123,400         1       123,400   vadd_naive(...)
   49.0     118,500         1       118,500   vadd_float4(...)
```

**从 nsys 能读出的 naive 问题信号:**
- **Avg 耗时 123.4 µs** → 反推有效带宽 = 3×16M×4B / 123.4µs ≈ 1631 GB/s。先把"实测带宽 ÷ 理论峰值"
  算出来(80%),这一步就能判断"是否还有优化空间"。memory-bound 算子 >85% 即接近触顶。
- 配合 `cuda_api_sum` 看 `cudaLaunchKernel` 与 H2D/D2H memcpy 的占比:若 kernel 本身只有 0.12 ms
  而 memcpy 占了几 ms,说明瓶颈在数据搬运而非 kernel——这是 nsys 最容易暴露的"假优化"陷阱。

> nsys 的局限:它只告诉你"naive 花了 123µs",但**说不清这 123µs 是被访存延迟、合并不佳、
> 还是 occupancy 不足拖住的**。要回答这个,必须上 ncu。

### 第 2 步:ncu —— 拆解 naive 到底"在等什么"(硬件计数器)

ncu(Nsight Compute)逐 kernel 重放并采集硬件计数器,是唯一能给出"stall 原因 / 合并率 /
DRAM 利用率"的工具。判断一个访存算子健康与否,重点看四类指标:

```bash
# 完整 section(慢,但信息全)
ncu --set full -k "regex:vadd" -o vadd_report ./cuda/vadd_cuda 16777216
# 快速迭代:只看访存与占用
ncu --section MemoryWorkloadAnalysis --section Occupancy --section WarpStateStats \
    -k "regex:vadd" ./cuda/vadd_cuda 16777216
```

| 诊断维度 | ncu 指标 | naive 的预期表现 | 怎么解读 |
|---|---|---|---|
| **是不是 memory-bound** | `sm__throughput` vs `dram__throughput` 的 SOL% | DRAM SOL ≫ Compute SOL | DRAM 那条接近 100%、compute 很低 → 实锤 memory-bound,优化方向只能是"减少/优化访存" |
| **带宽到顶没有** | `dram__throughput.avg.pct_of_peak_sustained_elapsed` | naive ~80% | <85% 说明还有空间;float4 后应往上抬 |
| **访存合不合并** | `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` ÷ 请求数 = 每请求 sector 数 | 连续访问本就合并,每 128B 事务 4 线程满载 | 一个 32-thread warp 的 load 若是 4 个 128B 事务即完美合并;vector_add 这里 naive 已是满分,所以"合并"不是它的问题 |
| **在等什么(stall)** | Warp State 里的 stall 分布 | 以 `stall_long_scoreboard` 为主 | long scoreboard = 等 global memory 返回。占比最高 → 典型 memory-bound,印证"瓶颈是 HBM 延迟/带宽,不是算力" |
| **并发够不够** | `sm__warps_active.avg.pct_of_peak_sustained_active`(occupancy) | 应较高 | occupancy 低 → in-flight 访存请求不足,无法用并发掩盖延迟(MLP 不够),这正是 float4/grid-stride 要改善的点 |

**ncu 给 naive 的"诊断书":** `dram__throughput` ≈ 80% + stall 几乎全是 `long_scoreboard`
→ 结论是"已经基本打满带宽,瓶颈是物理 HBM 延迟,剩余 20% 靠提高访存并行度(MLP)小幅压榨"。
这恰好解释了为什么 float4 只能 +4%——naive 并没有"合并不佳"这种可大幅修复的硬伤,
它的问题只是"每线程发起的 in-flight 请求太少",而这正是向量化能改善的部分。

> **对比 transpose 看价值:** vector_add 的 naive 已经合并良好,ncu 主要用于"确认健康";
> 而 transpose 的 naive 写出端非合并,ncu 的 `..._st.sum`(global store 事务数)会比 tiled 高数倍——
> 那才是 ncu 真正"抓出 bug"的场景。同一套指标,在不同算子上一个用于"证明触顶"、
> 一个用于"定位 32× 访存放大"。

### 第 3 步:闭环验证

nsys 量化前后耗时 → ncu 解释原因 → 改完再跑一遍 nsys 看耗时是否下降、ncu 看
`dram__throughput`/stall 分布是否改善。三步形成闭环,而不是"改完跑个 wall-clock 就完事"。

> **环境限制:** 本 A100 节点由 DCGM 持续采集性能计数器,ncu 抓取硬件计数器时报
> `ERROR: a driver resource was unavailable ... DCGM`(code 9)。无 `dcgmi` 可暂停,且停掉
> host engine 会影响共享节点其他用户,故本算子改用 nsys(基于 tracing,不占计数器)
> + 实测带宽推算占峰值比例,结论一致(80–85%)。上表 ncu 指标为"独占节点 / DCGM 释放后"
> 的标准诊断清单,留作复现参考。

## 为什么要做向量化(float4)与 grid-stride —— 原理

> 这两个是 memory-bound kernel 最常用的一对手法,经常一起出现,但解决的是**两个不同的问题**:
> 向量化解决"每条访存指令搬的字节太少 / 指令开销占比高",grid-stride 解决"线程网格如何复用、
> 如何与数据规模解耦"。下面分别讲清原理。

### A. 向量化(float4):用更宽的访存指令,提高 MLP、摊薄指令开销

**朴素版每线程做的事:** `c[i] = a[i] + b[i]`,即 1 次 32-bit load a + 1 次 32-bit load b
+ 1 次 32-bit store c。GPU 的访存事务以 **sector(32B)/ cache line(128B)** 为单位,
一个 32-thread 的 warp 连续访问 32 个 float = 128B,恰好一个事务——所以 vector_add 的 naive
**合并质量已经满分**,向量化并不是来修合并的。

**float4 改成每线程处理 4 个连续元素:** 把指针 reinterpret 成 `float4*`,一条
`LD.E.128` 指令一次性读 16B(4 个 float)。收益有三处:

1. **更少的访存指令数(指令开销摊薄)。** 同样搬 16M 元素,naive 发 16M 条 load.32,
   float4 只发 4M 条 load.128。地址计算、边界判断、指令发射的固定开销除以 4,指令流水更短。
2. **更高的访存并行度 MLP(Memory-Level Parallelism)。** 这是 memory-bound 场景的关键。
   GPU 靠"同时让大量访存请求 in-flight"来掩盖单次 HBM 几百 cycle 的延迟。每线程一条
   128-bit 请求,等价于一次性把 4 个元素的数据需求压进访存系统,**单位时间在途字节更多**,
   更容易把 HBM 带宽喂满。naive 的瓶颈(那剩下的 20%)正是 in-flight 请求不足,float4 改善的就是这个。
3. **更高的有效带宽/指令比。** 在 LSU(load/store unit)发射能力有限时,宽指令让每条指令搬运更多有用字节。

**为什么只 +4%?** 因为 naive 已经打到 80% 峰值、合并也满分,向量化只能从"指令开销 + MLP"
这点边角压榨,天花板就在带宽物理上限。**向量化的大收益场景是 naive 访存未合并 / 指令受限的 kernel**,
此处属于"锦上添花"而非"雪中送炭"——这本身就是一个有价值的结论:对已触顶的算子别指望向量化翻倍。

> 代价/前提:指针需 16B 对齐,元素数需被 4 整除(否则要处理尾部 remainder);寄存器压力略升。
> vector_add 里 `n = 16M` 天然满足,故直接 `n4 = n/4`。

### B. grid-stride loop:让 kernel 与数据规模解耦,且复用常驻线程

**朴素版的网格策略:** "一线程一元素",`grid = cdiv(n, block)`。n=16M 就要起 65536 个 block。问题有二:

1. **网格大小随数据线性膨胀,且受上限约束。** 数据再大,block 数可能撞到硬件 grid 维度上限;
   而且海量 block 的调度/退场本身有开销(tail effect——最后一批 block 收尾时 SM 已空转)。
2. **block 数与 GPU 实际并行宽度无关。** A100 只有 108 个 SM,一次能驻留的 block 有限;
   起 65536 个 block,绝大多数在排队,调度器要反复换上换下。

**grid-stride 改法:** 起一个**固定大小、刚好填满 GPU 的网格**(这里 cap 到 1024 个 block),
每个线程用步长 `stride = blockDim.x * gridDim.x` 的循环,跨着把整个数组处理完:

```cpp
int stride = blockDim.x * gridDim.x;
for (int i = blockIdx.x*blockDim.x + threadIdx.x; i < n4; i += stride) { ... }
```

原理与收益:

1. **kernel 与数据规模解耦(可伸缩性)。** 同一份 kernel,n 从 1M 到 1B 都能正确跑,
   不必担心 grid 维度溢出——线程不够就多转几圈循环。这是工程上最实在的好处。
2. **线程常驻、复用,摊薄启动/退场开销。** 固定数量的 block 一次性铺满 SM 并长期驻留,
   循环体内连续处理多个元素,避免"起海量 block → 各跑一下就退场"的调度抖动和 tail effect。
3. **天然制造连续访存 + 流水。** 相邻迭代 `i += stride`,同一 warp 内仍是连续地址(合并保持),
   且循环让每个线程持续有访存在途,进一步抬高 MLP——与 float4 协同把带宽喂满。
4. **更好的负载均衡。** 元素数不是 block 整数倍时,grid-stride 自动把余量摊到已有线程上,
   不需要额外的尾部处理 kernel。

> **二者的协同关系:** float4 让"每条指令/每个线程一次搬更多字节",grid-stride 让"固定的常驻线程
> 高效复用、循环喂出连续访存流"。一个优化"访存指令的宽度与并行度",一个优化"线程网格的形状与复用"。
> 在 vector_add 这种已触顶的算子上二者合计 +4%;但它们是后续所有 memory-bound kernel
> (reduce / softmax / layernorm)的标准底座,价值在于**形成可复用的写法范式**,而非这 4%。

## 下一步

- v1 已基本触顶,vector_add 收尾。带宽利用率 80–85% 即视为达标。
- 进入 Level 0 下一个算子:**transpose**(会暴露 naive 版的非合并写 / bank conflict,
  ncu/nsys 对比收益明显),或 **elementwise** 融合(fusion 减少访存往返)。


# 附录:性能分析答疑(FAQ)

> 本节汇总围绕 vector_add 五段式分析延伸出的高频性能问题,作为前面方法论小节的补充注脚。

## A. nsys 命令逐参数详解

```bash
CUDA_VISIBLE_DEVICES=4 nsys profile --stats=true -o vadd_nsys --force-overwrite true \
    ./cuda/vadd_cuda 16777216
```

| 部分 | 含义 |
|---|---|
| `CUDA_VISIBLE_DEVICES=4` | 环境变量(非 nsys 参数),把进程可见 GPU 限定为物理 4 号卡;程序内部仍视其为 device 0。避免干扰共享节点其他用户 |
| `nsys profile` | 启动一次 timeline tracing 采集 |
| `--stats=true` | 采集结束后自动解析并在终端打印汇总表(`cuda_gpu_kern_sum` / `cuda_api_sum` 等),省去手动 `nsys stats` |
| `-o vadd_nsys` | 输出报告前缀,生成 `vadd_nsys.nsys-rep`(GUI 可开)与 `.sqlite` |
| `--force-overwrite true` | 同名报告存在则覆盖,避免中断 |
| `./cuda/vadd_cuda` | 被 profile 的可执行程序 |
| `16777216` | 传给程序的参数,即 n = 16M = 2²⁴ 个元素 |

## B. 带宽数字怎么来的 & 理论峰值

**有效带宽 = 搬运字节数 ÷ 耗时:**
- 搬运字节数 = `3 × n × 4B`:vector add 是 `c=a+b`,读 a + 读 b + 写 c = 3 个数组;n=16M;fp32 每元素 4B。合计 = 3×16777216×4 ≈ 192 MiB = 201,326,592 B。
- 耗时 123.4 µs:取自 nsys `cuda_gpu_kern_sum` 表中 naive kernel 的 Avg 列(123,400 ns)。
- 有效带宽 = 201,326,592 B ÷ 123.4e-6 s ≈ **1631 GB/s**(十进制 GB,÷10⁹)。

**理论峰值 = 2039 GB/s(A100-SXM4-80GB):**
- 由 HBM2e 规格算出:等效时钟 ≈1593 MHz × 双沿(×2)× 总线 5120-bit ÷ 8 ≈ 2.039e12 B/s。
- ⚠️ 40GB 版 A100 峰值是 1555 GB/s,只有 80GB 版才是 2039;取值要对应实际卡型。

**占比与阈值:**
- 1631 ÷ 2039 = **80%**。
- 经验判据:memory-bound 算子实测 > ~85% 峰值即视为触顶。纯访存 kernel 摸不到 100%,因有 kernel 启停(tail effect)、ECC 开销、HBM 刷新等固定损耗,在 ~0.12 ms 的短 kernel 上无法摊掉。naive 在 80% 说明还差一点,这正是 float4+grid-stride 的空间。

## C. ECC 是什么

**ECC = Error-Correcting Code(纠错码)**,显存的数据完整性保护机制。
- DRAM/HBM 里数据偶尔因噪声/射线发生位翻转;ECC 通过额外校验位,自动纠正单比特错误、检测双比特错误(SECDED)。A100 等数据中心卡默认开启。
- 代价:存校验位要占额外显存容量与带宽,每次读写多搬一点 → 有效带宽比理论峰值低几个百分点。这是 memory-bound kernel 永远摸不到 100% 峰值的损耗来源之一(与 tail effect、HBM 刷新共同构成那 ~15% 缺口)。

## D. 如何确认一台机器上 ncu 能否使用(四步确认法)

| 步骤 | 命令 / 检查 | 判读 |
|---|---|---|
| 1. 二进制+版本 | `which ncu && ncu --version` | 拿不到 → 没装或不在 PATH(常在 `/usr/local/cuda/bin`)|
| 2. 内核权限锁 | `cat /proc/driver/nvidia/params \| grep RmProfilingAdminOnly` | `0`=放开普通用户可抓;`1`=仅 root,需 `sudo` 或管理员设 `NVreg_RestrictProfilingToAdminUsers=0` |
| 3. 计数器占用嫌疑 | `ps -ef \| grep -Ei "dcgm\|nv-hostengine\|ncu\|nsys"` | DCGM/hostengine 会持续独占计数器。⚠️ 但容器内 `ps` 看不到 host 层进程,**此步只能排查嫌疑,不能定论** |
| 4. 决定性:真跑一次 | `CUDA_VISIBLE_DEVICES=0 ncu --launch-count 1 --section MemoryWorkloadAnalysis -k "regex:vadd" ./cuda/vadd_cuda 16777216` | 打印出 Memory Workload 表=可用;报 `code 9 / driver resource unavailable ... DCGM`=被独占;`ERR_NVGPUCTRPERM`=权限没放开;`No kernels profiled`=正则没匹配 |

**口诀:** 装了没 → 权限放开没 → 被占没(ps 嫌疑 + 真跑一次定论)。

**本项目实测结论:** 旧节点与新换的 8 卡全空节点,前三关均通过(`ncu` 2025.2.1.0、`RmProfilingAdminOnly: 0`、`ps` 看不到 DCGM),但第四关真实采集**两台都报 `code 9`**。说明独占来自**容器外 host 层 nv-hostengine**(容器内无 `dcgmi` 可暂停),属集群基础监控,与具体哪台节点/哪张卡空闲无关。**故本项目 ncu 不可用,改用 nsys(tracing 不占计数器)+ 实测带宽推算。**

## E. ncu 跑不起来,为什么前面还有"诊断书"?(证据链澄清)

前面那份对 naive 的"诊断书"**不是本项目 ncu 实测**,而是三类证据拼出来的:
1. **实测硬数据**:`dram__throughput ≈ 80%` 来自 nsys 耗时反推有效带宽 ÷ 理论峰值,不依赖硬件计数器,DCGM 占用下仍成立。
2. **算子原理推断**:`stall 几乎全是 long_scoreboard`、`合并已满分`、`MLP 不足` 从 vector_add 的连续访存模式推断——warp 内 32 线程读 32 个连续 float=128B 完美合并;已合并满分却停在 80% 的纯访存 kernel,瓶颈必然是等 HBM 返回。
3. **结果反证**:float4 实测 +4%(118.5 µs)反过来印证了"瓶颈是 MLP 不足,而非合并不佳"的推断。
> 严谨表述:文档中那张 ncu 指标表是**"独占节点 / DCGM 释放后"的标准诊断清单,留作复现参考**,非本次实测结果。

## F. stall 是什么

**stall = warp 在某周期想发指令却因原料没就绪而被迫空等**(数据还在搬、依赖未算完、执行单元被占、等同步)。
- GPU 设计哲学:靠海量并发掩盖延迟——一个 warp stall,调度器立刻切到另一就绪 warp;只要驻留 warp 足够多(occupancy 高),执行单元就不闲。问题不在"有没有 stall",而在"stall 时有没有别的 warp 顶上"。
- ncu 按原因分类:`long_scoreboard`(等 global/HBM,memory-bound 特征)、`short_scoreboard`(等 shared mem)、`wait`(等算术依赖)、`barrier`(等 `__syncthreads`)、`not_selected`(自己就绪但没被选中,健康信号)。
- 套回 vector_add:naive 的 stall 几乎全是 `long_scoreboard`,即大部分时间在等 HBM 搬回 a/b——memory-bound 实锤。float4/grid-stride 通过提高 in-flight 请求(MLP)填满这些空等周期。

## G. CPU 侧 OpenMP 使用与原理

本项目 CPU baseline 的并行仅靠一行 pragma:
```cpp
#pragma omp parallel for schedule(static)
for (int i = 0; i < n; ++i) c[i] = a[i] + b[i];
```
配合编译期 `-fopenmp`(没有它该行被当注释,退化单线程)与 `#include <omp.h>`。

**原理要点:**
1. **fork-join 模型**:平时单线程,遇 `parallel` 从线程池 fork 出 N 个线程(默认=逻辑核数,`OMP_NUM_THREADS` 可改),并行区末尾有隐式 barrier(join)。
2. **work-sharing**:`parallel for` 把 16M 次迭代切成 N 段分给 N 线程,前提是**各迭代独立**(此处每个 i 只读写自己,无跨迭代依赖)。
3. **`schedule(static)`**:编译期均匀连续切块,适合每次迭代等耗时的规整循环;连续切块还让各线程访问连续内存,对 cache 预取与带宽友好。(对比 `dynamic`/`guided` 适合负载不均场景。)
4. **为何 CPU 仍比 GPU 慢 ~112×**:vector_add 纯 memory-bound,CPU 天花板是 DDR 带宽(~十几 GB/s 有效),128 线程拉满也只 ~14.6 GB/s;OpenMP 的作用是"把 DDR 带宽吃满",而非提升上限。A100 的 HBM ~2 TB/s 才是代差来源。
5. **常见坑**:共享写需 `reduction(+:sum)` 防数据竞争;不同线程写同一 cache line 导致 false sharing;NUMA 机器的 first-touch 亲和性。vector_add 无共享写、`static` 连续切块,天然规避前两者。

## H. Roofline 模型

**定义:** 用一张图同时画出"算子特性"与"硬件上限",一眼看出瓶颈类型与优化天花板。Berkeley 2009 提出。

- **算术强度 AI = 计算量(FLOP) ÷ 访存量(Byte)**,算子固有属性。vector_add:1 FLOP / 12 B ≈ **0.083 FLOP/Byte**,极低。
- **两条屋顶**:带宽屋顶(斜线,性能 ≤ AI×峰值带宽)与算力屋顶(水平线,性能 ≤ 峰值 FLOP/s)。算子能达到的最高性能 = `min(AI×带宽, 峰值算力)`。
- **拐点(ridge point)左侧 = memory-bound**,右侧 = compute-bound。
- **套回 vector_add**:AI 极小,远在拐点左侧 → 必然 memory-bound,上限=AI×带宽。GPU 斜线(HBM 2039 GB/s)远高于 CPU 斜线(DDR ~十几 GB/s),两者斜率差两个量级,正是 CPU 慢 ~112× 的本质,也即"印证 Roofline 下界"的含义。
- **实战价值**:落在斜线上→减少访存/提带宽利用率(向量化、合并、融合);落在水平线下→上 Tensor Core/提并行;远低于两屋顶→occupancy 不足/stall/启动开销,有大优化空间。
