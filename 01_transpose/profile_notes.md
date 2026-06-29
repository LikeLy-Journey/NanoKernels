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

---

# 附录:transpose 优化答疑(FAQ)

> 本节汇总围绕 transpose 三版优化(naive → tiled → tiled+pad)延伸出的高频问题,作为前面方法论小节的补充注脚。

## A. TILE 是什么,为什么设成 32

**TILE 是分块(tiling)的块边长**——把 4096×4096 大矩阵切成一个个 32×32 小方块,每个 CUDA block 负责搬运并转置其中一块。`#define TILE 32`,grid 维度 `cdiv(4096,32)=128`,即 128×128 个块铺满矩阵。

为什么是 32,四条理由叠加:
1. **对齐 warp = 32 线程**:一行 tile 正好 32 个元素,一个 warp 横着读一整行 = 32 连续 float = 128B = 一个完整合并事务,不多不少。
2. **对齐 shared 的 32 个 bank**:shared 物理上正好 32 个 bank,TILE=32 让行访问天然每线程落一个 bank,列访问的冲突规律也变得干净可分析、可被 padding 精准修复。
3. **shared 容量够用**:`32×32×4B = 4KB`(padding 版 4.125KB)。A100 每 SM ≤164KB,4KB/block 可同时驻留很多 block,occupancy 不受限;TILE=64 要 16KB,occupancy 开始受限。
4. **经典甜点值**:NVIDIA 官方 transpose 博客用的就是 32×32 tile + `BLOCK_ROWS=8`(每线程处理 4 行),教科书级标准配置 [[An Efficient Matrix Transpose in CUDA C/C++]](https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/)。

> 代码里 block 是 `dim3(32,8)`=256 线程,但 tile 是 32×32,所以每线程用 `for(j=0;j<32;j+=8)` 循环处理 4 行(32/8)。

## B. 不知道 tiled 优化时,如何定位"写出端非合并"

完整诊断链(不靠背答案):
1. **先确认 memory-bound 且离上限多远**:算有效带宽 241 GB/s,对比 copy 上限 1560,只有 15%。纯访存 kernel 只跑到上限 15%,几乎一定是访存模式问题(不是算力/occupancy)。这步不需要 ncu,只要会算带宽 + 有 copy 基线。
2. **用 ncu 高层 section 让工具指方向**:`ncu --set full -k "regex:transpose_naive"`。GPU Speed of Light + Memory Workload Analysis 会给结论性提示,Rules 会直接打印 *"Uncoalesced Global Accesses ... excessive sectors"* 并指到那条 store 的行号。
3. **看 sector 利用率这个决定性指标**:`l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_st` = 每个 store 请求平均触发几个 sector。完美合并应是 4(128B=4×32B),naive 写出端接近 32 → 放大 8 倍。或对比读(`_op_ld`)与写(`_op_st`)两方向:读正常、写爆炸 → 瓶颈在写出端,不在读入端(transpose 反直觉点)。
4. **没 ncu 时(本项目即如此)的退路**:写对照实验——把 kernel 改成"只读不写"和"只写不读"分别计时,去掉按列写后速度飙升 → 反证瓶颈在写;或直接和 copy 比,唯一区别就是写地址转置了。

> 口诀:带宽占比低 → 先怀疑访存模式 → ncu 看 sectors/request 或 coalescing rule → 没 ncu 就用拆读/拆写对照反证。心法:不要盯代码猜,用"每请求 sector 数"量化说话。

## C. shared memory 是什么;转置后块坐标如何变换;瓶颈转移怎么看出

**shared memory** 是位于 SM 片上、一个 block 内所有线程共享的高速暂存区。相比 global(HBM,片外,延迟 ~400 cycle),shared 延迟仅 ~20-30 cycle,带宽高一个量级,但很小(A100 每 SM ≤164KB)、生命周期仅限一个 block、需 `__syncthreads()` 手动同步。本质是"程序员手动管理的 cache"。

为什么 transpose 需要它:global 端读能合并、写不能;而 shared 没有"合并"约束(按 bank 而非 cache line 组织)。策略变成:合并读 global→shared → 在 shared 里做转置(随便换索引都不触发 global 非合并)→ 合并写 shared→global。**把不可避免的"非合并"从昂贵的 global 端转移到便宜的 shared 端。**

坐标变换(注意阶段 2 `blockIdx.x/.y` 互换):
```cpp
// 阶段1 读入:block 负责输入的 (blockIdx.y, blockIdx.x) 块
int x = blockIdx.x*TILE + threadIdx.x;   // 输入列
int y = blockIdx.y*TILE + threadIdx.y;   // 输入行
tile[threadIdx.y+j][threadIdx.x] = in[(y+j)*cols + x];   // global 读连续 → 合并
__syncthreads();
// 阶段2 写出:blockIdx 互换!
x = blockIdx.y*TILE + threadIdx.x;       // 用 blockIdx.y 算输出列
y = blockIdx.x*TILE + threadIdx.y;       // 用 blockIdx.x 算输出行
out[(y+j)*rows + x] = tile[threadIdx.x][threadIdx.y+j];  // tile 下标也交换 [tx][ty]
```
两层转置同时发生:**块级**(输入第 `(By,Bx)` 块写到输出第 `(Bx,By)` 块,blockIdx 对调)+ **块内**(shared tile 内 `[ty][tx]` 存、`[tx][ty]` 取,行列下标互换)。关键:两次 index 交换都在 shared 或 block 编号上,global 读写各自都保持连续合并。

瓶颈转移怎么看出:
- **数字证据**:v0→v1 带宽 241→1002(4.2×)。ncu 看 `global_op_st` 的 sector/request 回到正常的 4(写出端合并问题解决),但 `l1tex__data_bank_conflicts_..._shared_op_ld` 突然冒高(原本为 0)。
- **逻辑推理**:1002 还差 copy 上限 36%,而 global 两端都已合并 → 剩余损耗只能来自新引入的环节(shared 访问)→ 顺着阶段 2 `tile[threadIdx.x][...]` 发现是按列读 shared → 定位 bank conflict。
- **方法论**:瓶颈转移是优化常态,每消除一个瓶颈,下一个就浮现成新的最慢环节。判据永远是"当前带宽 vs 理论上限的差距 + 哪个指标异常"。

## D. warp 读 shared 一整列的问题;bank 是什么;为什么相差 128B 全落一个 bank

**bank**:shared 在硬件上切成 32 个独立存储体,每个 bank 每周期能独立服务一次访问。地址→bank 按 4 字节(一个 float)交错映射:`bank = (字节地址 / 4) % 32`。一个 warp 32 线程访问 32 个连续 float 时正好分散到 32 个不同 bank,一周期并行完成(理想情况)。

为什么"相差 128B 全落一个 bank":v1 阶段 2 按列读 `tile[threadIdx.x][threadIdx.y+j]`,一个 warp 里 `threadIdx.x`=0~31(固定 ty),访问的是 `tile[0][k]…tile[31][k]`(同一列、不同行)。在 `tile[32][32]` 行主序布局里,相邻行同列元素地址相差一整行 = 32 float = 128B。代入映射:
```
线程i: tile[i][k] 偏移 = (i*32+k)*4 → bank = (i*32+k) % 32 = k   (对所有 i 都一样!)
```
因为行跨度 32 恰好是 bank 数 32 的整数倍,`%32` 把行的贡献完全消掉 → 32 个地址全映射到同一 bank k。这就是 128B 间隔的"致命对齐"。

对一个 warp 的所有线程意味什么:一个 bank 每周期只服务一个线程,32 线程都访问同一 bank → 硬件串行排队 32 次(**32-way bank conflict**)。本来 1 周期的 shared 读变成 32 周期,整个 warp 被这条最慢指令拖住 → v1 卡在 64% 上不去的根因。

padding 为何能解(v2):tile 改 `[32][33]`,行跨度变 33 float,`(33*i+k)%32 = (i+k)%32` → i 每加 1 bank 就 +1,32 线程错开落到 32 个不同 bank,冲突归零。代价仅每行多 4B。

## E. nsys kernel 占比 10% 是什么意思

`cuda_gpu_kern_sum` 汇总本次 profile 期间所有 kernel 的 GPU 执行时间,算各自占总时间百分比。本轮 4 个 kernel(copy/naive/tiled/tiled+pad)GPU 时间约 86/554/134/86 µs,总和 ≈ 860 µs:
- naive = 554/860 ≈ **64.4%** → 吃掉总时间近 2/3,绝对热点。
- tiled+pad = 86/860 ≈ **10.0%**;copy = 86/860 ≈ **10.0%**。

"占比 10%"本身只是相对数字,意义来自对比对象。关键不是"10% 大不大",而是:**tiled+pad 占比(10.0%)和 copy 占比(10.0%)完全相同**。同一次 profile、同样数据量下,占比相同 ⟺ 绝对耗时相同 ⟺ 带宽相同。copy 是"读写都合并、不做转置"的理论上限,tiled+pad 做了完整转置却用了和 copy 一样的时间 → 转置的额外开销已优化到趋近于零。

> 区分两个百分比口径:这里"占比 10%"是 nsys 里 kernel 之间的相对时间分配(找热点);表格里"99% of copy"是该 kernel 带宽 ÷ copy 上限带宽(衡量离理论极限多远)。

## F. A100 每 SM 有 164KB shared,为什么 16KB/block 就开始限 occupancy

限 occupancy 的不是"够不够放下一个 block",而是"够同时放下几个 block"。

**occupancy = SM 上实际同时驻留 warp 数 ÷ 硬件上限(A100 = 64 warp / 2048 线程)**。要拉满 occupancy,SM 必须同时驻留足够多 block,而每 block 的 shared 是从 SM 总量里切走的:
```
SM 能同时驻留 block 数 ≤ SM 总 shared ÷ 每 block shared 用量
```

| 每 block shared | 能同时驻留 block 数 |
|---|---|
| 4KB(TILE=32) | 164/4 ≈ 41 个 → shared 远非瓶颈 |
| 16KB(TILE=64) | 164/16 ≈ 10 个 → 接近其它上限 |
| 32KB | 164/32 ≈ 5 个 → shared 成主约束 |

一个 block 256 线程 = 8 warp,要喂满 64 warp 上限需 8 个 block 同时驻留。TILE=32(4KB)能放 41 个,远超需要 → shared 完全不是瓶颈,occupancy 由线程/warp 上限决定可拉满。TILE=64(16KB)只能放 10 个,虽 >8 看似够,但 occupancy 还同时受寄存器、block 数硬上限(A100 每 SM 最多 32 block)等联合约束;shared 从"绰绰有余"变成"刚好卡线",任一其它资源稍紧,shared 就变成决定性短板。这就是"开始受限"——不是归零,而是从"非约束项"变成"潜在约束项"。

> 一句话:shared 是 SM 上所有 block 共享分配的资源,block 用得越多能并存的越少,可隐藏延迟的 warp 越少。16KB 不是"放不下",而是"放不下足够多份"。

## G. Memory Throughput vs DRAM Throughput,为什么后者低 = 有效数据少

ncu GPU Speed of Light 里两个易混指标,区别在衡量内存层级的不同位置:

| 指标 | 衡量什么 | 位置 |
|---|---|---|
| Memory Throughput | 整个内存管线繁忙程度,取 L1TEX/L2/DRAM 各级利用率最大值 | 片上 L1/L2 + 片外 DRAM 综合 |
| DRAM Throughput | 只看 HBM 实际搬运字节速率 ÷ HBM 峰值 | 仅 HBM |

关键:前者统计"事务/请求的流量",后者统计"真正进出 HBM 的有用字节"。"前者高、后者低"的剪刀差正是非合并访存的信号。

为什么 DRAM 低 = 有效数据少(回到 naive 按列写):硬件最小搬运单位 32B sector,每线程只写 4B 有用数据。
- **管线层面**:每线程发起一个事务,32 线程=32 个独立事务,管线被请求塞满 → Memory Throughput 高(单元一直忙)。
- **HBM 层面**:32 个事务每个搬 32B(共 1024B 过总线),有效载荷只有 32×4=128B(1/8)→ 总线被占满但有用字节吞吐低 → DRAM 有效吞吐低。

即:Memory 高 = 内存系统很忙,DRAM 低 = 忙的大部分是浪费(搬被丢弃的 sector 字节)。两指标背离 = 事务利用率低 = 合并差。完美合并时(copy)每事务 128B 全有用,两指标同步走高、基本贴合。

## H. 如何判断算子是 memory-bound 还是 compute-bound

三层判据,从事前估算到 profile 实测:

**判据一:算术强度 AI + Roofline(事前估算)**
AI = 计算量(FLOP) ÷ 访存量(Byte),算子固有属性。机器平衡点(ridge point)= 峰值算力 ÷ 峰值带宽,A100 ≈ 19.5 TFLOP/s ÷ 2.0 TB/s ≈ 9.75 FLOP/Byte。
- AI < 平衡点 → memory-bound(拐点左侧);AI > 平衡点 → compute-bound(拐点右侧)。

| 算子 | AI | 判定 |
|---|---|---|
| vector_add | 1/12 ≈ 0.083 | ≪ 9.75 → 极度 memory-bound |
| transpose | 0/8 = 0 | 纯搬运 → memory-bound |
| 大矩阵乘 GEMM | ~N/2(随规模涨) | > 9.75 → compute-bound |

**判据二:ncu Speed of Light(实测一锤定音)**
- Compute(SM)Throughput 高(>60-70%)、Memory 低 → compute-bound。
- Memory Throughput 高、Compute 低 → memory-bound。
- 两者都高 → 平衡,接近 well-optimized;两者都低 → latency-bound(occupancy 不足/stall/启动开销),查 stall 原因。
- ncu Rules 通常直接打印结论,如 *"bound by memory bandwidth"*。

**判据三:有效带宽占比(最省事工程判据)**
不开 ncu 也能用:算有效带宽 ÷ 该卡 HBM 峰值。
- 占比高(>60-70%)→ 已吃满内存,基本 memory-bound,优化在减少访存/提合并/提 AI(融合算子)。
- 占比低 + 算力也没满 → 既非 memory- 也非 compute-bound,而是 latency-bound。

> 实战:先用 AI 粗判方向,再用 ncu SoL 或带宽占比坐实。绝大多数深度学习算子(elementwise、norm、激活、softmax)都是 memory-bound,只有大 GEMM、大卷积是 compute-bound。这也是 vector_add / transpose 入门算子全在讲合并、bank、带宽,而非讲怎么算得更快的原因。
