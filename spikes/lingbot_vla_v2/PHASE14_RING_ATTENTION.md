# Phase 14 — Ring attention 跨 iGPU/dGPU 做序列并行：内核对了，前提不成立

## Context

用户提的方案：用 **ring attention（序列并行）** 把 denoise 循环切到两张卡上跑，
提高 denoise 性能，在容器 `test-image_zy_scaler0260b2_lingbot_omni` 里验证。

本阶段**把两种切法都实现了、都在两张卡上跑通了、也都做了精度验收**，不是估。
结论是否，但有价值的不是这个否，而是**它在哪里输**——和之前六次拒绝的位置都不一样。

四条已有的实测数字在写代码之前就框住了答案，这也是本阶段落在 spike 而不落生产的原因：

| 事实 | 出处 |
|---|---|
| torch-xpu 一个进程只能枚举一个 Level-Zero platform，所以“两张卡”等于两个进程，各自一个 `ZE_AFFINITY_MASK`，没有 `.to("xpu:1")` | §K1 |
| iGPU 跑真实 MoE layer-step 3.246 ms，dGPU 0.251 ms，`k` = 12.9×；读带宽 29 vs 449 GB/s | §K2 |
| 一次 dGPU↔iGPU 往返 **0.155 ms**，而**整个** 32 专家块只要 0.251 ms；循环有 36 层 × 10 步 = **360** 个 layer-step | §M.6 |
| 切 51 行 suffix 的上限约 13%——**即使 iGPU 免费、通信为零**——因为循环的成本是专家权重字节，与行数无关 | §G5 |

ring attention 并行的是 **attention**，而这个循环的瓶颈是**流式读 MoE 专家权重**。
所以预期是大幅回退。但有两个决定性数字从来没在这个模型上量过，本阶段量它们而不是断言它们：

1. **attention 到底占 denoise layer-step 的多少**（Gate 1）。这是任何“只并行 attention”
   方案的天花板，而 §F2 分解过 layer-step，从没把 attention 单独拎出来。
2. **ring 的真实载荷下一次交换多少钱**（Gate 2/3）。§M.6 量的是 76.5 KiB 的 MoE 激活交换，
   ring 的载荷是另一个量级（408 KiB 的 query、822 KiB 的 partial output）。
   ——这一条量出来的答案和问题本身相反：电线不是主导项，**iGPU 的计算是**（§0 ③）。

探针：`phase14_ring_attention_probe.py`（四个 arm）+ `test_ring_attention.sh`。
torch 2.12.0+xpu，checkpoint `/tmp/lingbot-vla-v2-perf`（基础模型），eager fp16，
`moe=dense`，`inference_mode`。

---

## 0. 结论先放

两种切法都在两张卡上实测，eager、同进程对照，所以比值是同口径的；
213.5 ms 是编译过的单卡成绩，只作为目标引用。

| 切法 | dGPU 单卡 | dGPU + iGPU | |
|---|---:|---:|---|
| **context parallel** — prefix KV 切 143/143，query 复制，iGPU **只做 attention、一个权重都不加载** | 454.6 ms | **1124–1207 ms** | **0.38–0.40×** |
| **sequence parallel** — 51 行 suffix 切 26/25，两张卡都跑 6B 模型 | 448.6 ms | **2440–2460 ms** | **0.18×** |

区间是 load < 1.0 下重复跑的散布；单次数字在下面的表里和 `/tmp/lingbot-ring/*.json` 里。

**ring 本身没问题。** 它在 1/2/4/8 个 shard 下都与 `eager_attention` 差**一个 fp16 ulp**，
跨设备边界也一样；而序列并行的 action chunk 精度**比单卡答案更好**
（rms 1.202e-3，fp16 地板是 3.9e-3），因为两个 rank 都用 fp32 累加 softmax。
失败的是前提：**attention 只占循环的 10.8%**，而两种切法各自输给一个更大的项。

三个新结果，其中两个的价值超出这次拒绝：

> **① attention 占 denoise 循环 10.8%**（三次重复 10.7–10.9%，Gate 1）。这封顶了一个
> KV 切分能拿到的 213.5 ms 中的 **11.5 ms**——也封顶了所有**别的** attention 侧优化（fmha、flash-suffix、
> 换 SDPA 后端）到 ~23 ms，不只是并行类的。
>
> **② 把 action 行数砍一半只买到 +0.4%，不是 13%**（Gate 3）。§G5 从 MoE GEMM 的
> M 曲线拟出“至多 ~13%”；在**整个 forward** 上实测是 +0.4%，因为循环是 dispatch-bound
> （§F1），行数砍半**一个算子都没少**。§G5 的结论对，界宽了 30×。
>
> **③ 两种切法都是 iGPU 计算受限，不是通讯受限**（Gate 2/3）。这条和动手前的预期相反
> ——预期是「360 次往返压死它」。实测：
>
> | | iGPU 计算 | 电线 |
> |---|---:|---:|
> | context parallel，每 layer-step 交换 1.652 ms | **1.309 ms（79%）** | 0.460 ms（21%，部分重叠） |
> | sequence parallel，每 layer-step 阻塞 5.34–5.40 ms | **5.304 ms（98%）** | 0.107 ms（2%） |
>
> **两列都是直接量的**，不是残差也不是从带宽曲线折的：iGPU 侧在 iGPU 进程里自报
> （Gate 2/3），电线用 `--split wire` 单独量——worker 只回显一个预分配缓冲区，
> 不加载权重、不跑 kernel（Gate 5）。**把电线变成免费，两种切法都还是输。**
>
> **④ eager 下 attention 是 launch-bound；编译下不是，而生产跑的是编译**（Gate 1 / Gate 4）。
> 这是本文件第三处更正，也是最重要的一处。两条曲线，同一个循环：
>
> | prefix KV | eager | compiled |
> |---:|---:|---:|
> | 286 key | 452.1 ms | **206.3 ms** |
> | 143 key | 452.9 ms（+0.2%） | **181.7 ms（−11.9%，−24.6 ms）** |
> | 8 key | 452.1 ms（+0.3%） | 171.2 ms（−17.0%，−35 ms） |
>
> **eager 平，编译不平。** inductor 把 launch 开销融掉之后，KV 变成真成本，
> 减半真的省 **24.6 ms**。本文件前几版的「dGPU 一分不省」是在 **eager** 路径和
> 一个独立 `block_attention` 微基准上量的——两者都 launch-bound——**而生产配置是
> `compile_denoise_step=True`**。所以那个说法对 eager 成立，对出货的东西不成立。
>
> **⑤ 「给 iGPU 少分一点」：sequence 切法不行，context 切法可调，但收益 24.6 ms 对电线 158 ms**
> （Gate 4）。sequence 并行**复制权重**，iGPU 拿 1 行还是要把 27.2 GB 专家权重流完——
> 实测 **1704 ms 饱和**（1 行 1704，25 行 1948，50 行 2195），是 206 ms 的 **8.3 倍**，
> 而 27.2 GB ÷ 29 GB/s = 937 ms 的算术地板已经是 4.5 倍。**最优份额是 0 行。**
> context 切法现在有真实收益（编译循环 −24.6 ms at 50/50，上限 −35 ms），
> 但 query 408 KiB + reply 822 KiB **由输出形状决定、不随份额缩**，
> 实测最快 0.438 ms/往返 = **158 ms/循环**。**光电线就是收益的 6.4 倍**，
> iGPU 的 467 ms attention 还没算。

还有一条工程结论：**这个拓扑上 POSIX 共享内存打赢 oneCCL**，一次性 setup 传输快 15×
（Gate 5）。PHASE10 §9 在 293 KiB 上量到两者相等，那个相等**不往上延伸**。

> **第二处自我更正（与 ④ 那处并列）。** 本文件第一版把 context parallel 的
> 1.79 ms「交换」整段算成电线，
> 得出「电线是天花板的 56 倍」。**那是错的**：那一段计时把 dGPU **自己**的 attention 块
> 也圈在里面了。现在计时分成 `send` / `local` / `recv` 三段，iGPU 那边也自己报它的块耗时，
> 于是电线只占设备边界的 13%。结论的方向没变（还是否），但**归因变了**，
> 而归因才是「下一步该动哪里」的依据。

---

## 1. Gate 0 — 内核正确性：**过，而且压在 fp16 地板上**

`--arm ring-math`。一张卡、不走传输：这条不过，后面量的就不是 ring attention。

denoise 的 attention 带一个**任意 `[B, Lq, Lk]` bool 块掩码**
（`make_att_2d_masks` + `_block_query_columns`），这是不能直接用仓库里已有的
`ring_pytorch_attn.py` 的原因——那个只支持 `causal` / `joint`。探针的内核把掩码的
**列**轴和 K/V 一起切，用 online softmax 合并。形状是真实的：
`q [1, 51, 32, 128]`、`k/v [1, 337, 8, 128]`（286 缓存 prefix + 51 suffix）、
`mask [1, 51, 337]`。

| 输入 | softmax 累加 | shards | max abs | max abs / ref absmax | rel L2 |
|---|---|---:|---:|---:|---:|
| fp32 | fp32 | 1 | 2.09e-7 | 7.3e-7 | 2.5e-7 |
| fp32 | fp32 | 2 | 9.54e-7 | 3.3e-6 | 6.3e-7 |
| fp32 | fp32 | 4 / 8 | 9.84e-7 | 3.4e-6 | 8.6e-7 / 9.5e-7 |
| fp16 | **fp32** | 1 / 2 / 4 / 8 | **2.441e-4** | 8.5e-4 | 2.24e-4 |
| fp16 | fp16 | 1 | 2.441e-4 | 8.5e-4 | 4.2e-4 |
| fp16 | fp16 | 2 | 8.545e-4 | 3.0e-3 | 5.9e-4 |

**2.441e-4 正好是输出自身量级（0.287）上的一个 fp16 ulp。** 带 fp32 合并的 fp16 那几行
在**每一个** shard 数下都压在这个地板上，这是能给出的最强结论：ring 已经和
`eager_attention` 近到 fp16 张量能表示的极限。fp32 输入那几行存在的理由是这个地板
**会盖住真 bug**——把地板抬掉，代数本身保持到 3.4e-6 相对。

最后两行是「ring 有可能**不如**它替掉的整块」的唯一途径，值得留着：
**块内**用 fp16 softmax，块越小丢得越多（1 shard 是 1 ulp，2 shard 是 3.5 ulp）。
合并用 fp32 就完全消掉，而且在这里不要钱——块内的两个 matmul 仍然是 fp16。
所以后面每个 arm 都走 fp32 合并。

另外三项：

* **死 shard 路径。** 某一行在**全局**没被完全掩掉，但在某个 **shard** 里可能被完全掩掉：
  suffix 的 `att_masks` 是 `[True, True, False, …]`，state token（第 0 行）在 suffix 里只看得见自己，
  所以一个只装 action key 的 shard 对它是死的。只支持 causal 的实现永远碰不到这个，
  不加保护就是一次 0/0 重标定。只切 suffix key 可以强制触发它：
  **max abs / absmax = 1.0e-6**。
* **合并算子对着仓库里已有的那个验。**
  `vllm_omni/diffusion/attention/backends/ring/ring_utils.update_out_and_lse`
  与本文件的 `merge_out_lse` 差 **1.8e-7** 相对。探针验的是仓库**已经在发**的代数，
  不是它自己的一份私货。
* **端到端，走 Phase 7 指标**（Rules #3）：2-shard ring 放进真实十步循环，
  action chunk 相对未打补丁的答案 **rms 1.294e-2**——是 3.9e-3 fp16 地板的 3.3 倍，
  远在 `spec_tau` 0.15 之内。每次 attention 差一个 ulp，经 360 次调用和 10 步 Euler 累起来
  就是这个量级；和 FINDINGS 里 2.6% 的 bf16 跨设备差同一个数量级。

### 但 ring 内核自己的算术就已经很贵——而且贵在 kernel 数

这张表是后来补的，因为 Gate 2 的传输对照跑出来**比不切还慢**，那就必须先在一张卡上
把 ring 自己的成本定价，那里怪不到电线。同一形状，50 次迭代、10 次预热：

| | µs | vs eager |
|---|---:|---:|
| `eager_attention`，整块 337 key | **128.7–131.2** | 1.00× |
| `sdpa_attention`，整块 337 key（**融合**） | **115.4** | **0.90×** |
| ring，1 shard，`accum=input` | 309.7 | 2.36× |
| ring，1 shard，`accum=fp32` | 314.5 | 2.40× |
| ring，2 shard，`accum=input` | 683.3 | 5.21× |
| ring，**2 shard，`accum=fp32`** | **684.3** | **5.21×** |
| ring，4 shard，`accum=input` | 1385.9 | 10.56× |
| ring，4 shard，`accum=fp32` | 1386.6 | 10.57× |
| ring，8 shard，`accum=input` | 2758.1 | 21.11× |
| ring，8 shard，`accum=fp32` | 2741.8 | 20.98× |

两件事：

1. **fp32 合并是免费的。** `accum=input` 和 `accum=fp32` 差 1.5%，在噪声里。
   所以 Gate 0 上半部分那个「fp32 合并把 3.5 ulp 压回 1 ulp」的选择**不花钱**，
   没有精度/速度的取舍要做。
2. **成本 ≈ 340 µs × shard 数，和 shard 有多大无关**，1 到 8 个 shard 都线性。
   一个 shard 就已经 2.36× 整块。
   这不是算术——切成两半 FLOP 总数不变——而是 **kernel 数**：`block_attention` 要
   matmul / masked_fill / amax / isfinite / where / exp / sum / matmul / div / where / log，
   比 `eager_attention` 的 matmul / masked_fill / softmax / matmul 多一倍有余，
   而且**每个 shard 付一遍**。§F1 已经量过这个循环是 dispatch-bound（286.5 ms 请求里
   282.6 ms 是 host 提交），所以 kernel 数就是成本。

对上了另外两个数：131 µs × 360 = 47 ms，正是 Gate 1 消融出来的 48–49 ms attention；
684 µs × 360 = 246 ms，而 Gate 2 的传输对照实测比不切的循环多 **216 ms**。
**这一条本来看着是本阶段唯一一个「更好的工程真能修」的项**，但 Gate 4 把它也量掉了：
融合（`sdpa_attention`）只买到 **10%**，因为这个形状下 attention 是 launch-bound，
而一个**完美融合**的 ring 在 2 shard 下仍然是 2 次 launch（≈2 × 115 = 230 µs，
对不切的 115 µs）。所以融合把这 216 ms 降到约 **41 ms**——**仍然是加法，不是节省**。
在这个形状上 dGPU 永远不会因为切分而变快。细节在 Gate 4。

## 2. Gate 1 — 天花板：**eager 下 attention 占 10.8%，编译下 prefix KV 值 17%**

`--arm attn-share`，dGPU，eager fp16，`moe=dense`，`inference_mode`，10 次迭代，load 0.97。

两个互相独立的估计，因为在 dispatch-bound 的路径上单独任何一个都不可信。
§F1 的方法论注意事项在这里照样成立：这个后端上 profiler 报 `mm`/`einsum` 的
device time 为零，那正是当初那个被撤回的「51 ms device / 243 ms host」的来源，
所以两个估计**都不用它**。

| | | |
|---|---:|---|
| 循环，未打补丁（eager） | **450.8–453.8 ms** | 360 个 layer-step |
| attention，前后加 sync 栅栏，360 次求和 | 63.64–63.74 ms | 每次 177 µs——**上界**：它把它自己逼出来的 sync 和被它破坏的重叠都算进了 attention |
| 循环，attention 换成形状正确的全零张量 | 404.5–404.8 ms | |
| **attention，消融法** | **48.4–49.3 ms** | **占循环 10.7–10.9%**，三次重复——要引用的就是这个 |

消融是可信的那个，而且在这里特别干净，理由很具体：
`moe_implementation="dense"` 无条件跑全部 32 个专家（§F2），
所以改 attention 的输出**一个字节的权重流量都不会变**。桩改的是数值，不是工作量。

### 但 10.8% 是 eager 的数，而生产跑编译——编译下是 17%

前几版把这 10.8% **投影**到编译过的 213.5 ms 上，得出「编译循环里 attention 约 23 ms」。
`--compile-denoise-step` 现在把两边都量了，投影错了 1.5×：

| | |
|---|---:|
| 编译循环，未打补丁（实测） | **206.3–206.4 ms**（记录基线 213.5 ms） |
| 编译循环，去掉 prefix KV 的贡献（Gate 4 的 prefix 长度扫描） | 171.2–173.5 ms |
| **⇒ 286 token 的 prefix KV 值** | **≈ 33–35 ms，占编译循环 16–17%** |
| ⇒ 50/50 KV 切分的 dGPU 侧收益 | **24.6 ms**（206.3 → 181.7） |

**投影的方向我说对了（「编译后占比只会更大」），但量级差了 1.5×**：投影 23 ms，
实测 33–35 ms。

> **注意别用消融法读这个数。** `--arm attn-share --compile-denoise-step` 的消融读出
> 71.45 ms（34.7%），那是**上界且被污染**：桩返回一个形状固定的全零张量，
> 于是 inductor 把它背后**整个融合 q/k/v GEMM 和 `apply_mrope` 都死代码消除了**。
> eager 消融没有这个问题（Python 无论如何都会跑 `compute_qkv`），这正是两个占比
> 不一致的原因。要 KV 切分的天花板，用 Gate 4 的 prefix 长度扫描——它**真的**把
> KV 缩短，不需要桩。

所以本节的天花板表是（全部实测，无投影）：

| | |
|---|---:|
| 编译循环 | 206.3 ms |
| **上限，KV 切分 50/50** | **24.6 ms** |
| 上限，KV 切分的极限（prefix → 8 key） | 35 ms |
| 上限，行切分 — §G5 的曲线，已被 Gate 3 推翻（实测 +0.4%） | — |
| **实测电线**，本切法的载荷，shm（Gate 5） | **158–166 ms** |
| **电线 / 上限** | **6.4×** |

6.4× 而不是前几版写的 9.7×——两个数都变了（上限从 11.5 涨到 24.6，电线从投影的
111.6 涨到实测的 158–166），方向没变。

## 3. Gate 2 — context parallel：**两项开销，都不是电线**

`--arm ring-2p --split kv`。286 token 的 prefix KV cache 切 143/143，51 个 query 复制。
iGPU **一个权重都不加载**——它一次性收到自己那半 KV，之后永远只算 partial attention。
这是能压给 iGPU 的最小负担，也是这个 arm 存在的意义：**把传输从 §K2 的 `k` 里隔离出来**。

三行，而中间那行是关键——它是**传输对照**：同样两个 block、同样的 merge、同样的 Python，
但两块都在 dGPU 上算，没有电线也没有 iGPU。

| | ms | |
|---|---:|---|
| `denoise_actions`，整块 337 key，单卡 | **454.6** | 基准 |
| 两个 ring block 都在 dGPU 上算 *（传输对照）* | **670.8** | **+216 ms，还没碰电线** |
| dGPU + iGPU ring | **1206.8** | +536 ms 设备边界，**0.38×** |

| 传输 | setup（20.1 MiB） | dGPU + iGPU | |
|---|---:|---:|---:|
| shm | 16.9 ms | 1124–1207 ms | 0.38–0.40× |
| oneCCL | 524.0 ms | 1141.6 ms | 0.40× |

> **这个 arm 全程 eager**，两个 rank 都是。所以它的 216 ms / 536 ms 分解描述的是
> eager 路径。**编译路径的经济账不同，而且是生产配置**——放在本节末尾。

### 第一项：ring 自己多付的 216 ms，是 kernel 数（eager）

传输对照比不切的循环慢 216 ms，**一个字节都没过线**。Gate 0 末尾那张表就是它的定价：
2 shard 的 ring 要 684 µs 而整块 eager 只要 131 µs，×360 = 246 ms vs 47 ms，
差 199 ms——和这里的 216 ms 同一个数。原因不是算术而是 launch 数，
循环 dispatch-bound（§F1）。编译路径下 inductor 把这些 launch 融掉，这一项大部分消失
（Gate 1 / Gate 4）。

### 第二项：536 ms 的设备边界，87% 是 iGPU 在算

per-layer-step 的计时现在分三段（**这是对第一版的更正**，见 §0：第一版把 `local`
也圈进了「交换」，于是把计算算成了电线）：

| | ms/layer-step | 是什么 |
|---|---:|---|
| `send` | 0.127 | 交出 408 KiB 的 query，不等 ack |
| `local` | 0.382 | **dGPU 自己**那块 attention——不是交换 |
| `recv` | 1.526 | 阻塞等待：回程电线 + iGPU 那块里没被 `local` 遮住的部分 |
| **iGPU 侧自己报的块耗时** | **1.298** | 在 iGPU 进程里直接量的，min 1.078，1260 个样本 |

加上 Gate 5 单独量到的**纯电线** 0.460 ms（同样载荷，两端都不算东西），
`send + recv` 这 1.652 ms 就可以按测量对账，三项都是实测的：

```
send + recv           1.652 ms/layer-step   (实测)
  iGPU 的 attention   1.309 ms              (iGPU 进程自报)   79%
  纯电线往返          0.460 ms              (--split wire)    21%
  和                  1.769 ms  →  比 1.652 多 7%，即约 0.12 ms 的电线
                                 被 `local` 那 0.385 ms 遮掉了
```

三项不是严格可加的（有重叠），所以这里给的是**两个独立测量的相对大小**，
不是一个精确的分解。结论不依赖精度：**iGPU 是电线的 2.8 倍。**

**iGPU 用 1.298 ms 算 143 个 key，dGPU 用 0.382 ms 算 194 个——按 key 算慢 4.6×。**
这就是 §K2 的 `k` 出现在一个「我故意压到最小」的负载上。把电线变成免费
（165 ms/循环），这个 arm 从 0.38× 只走到 0.43×。

### 精度：切分是真的切开了

chunk 出来 **rms 1.291e-2**——和 Gate 0 里单进程 2-shard ring 的结果**每一位都相同**。
把一份 partial `(out, lse)` 送过设备边界再合并，数值上不加任何东西。

载荷，给下一个人参考：

| | |
|---|---:|
| query，dGPU → iGPU，`[51, 32, 128]` fp16 | 408 KiB |
| reply，iGPU → dGPU，`[out \| lse]` fp32 | 822 KiB |
| **一次往返** | **1.2 MB**（§M.6 那 76.5 KiB 的 16×） |

reply 用 fp32 曾被当成这里贵的原因；Gate 0 的定价表否掉了这个说法
（fp32 合并是免费的），而边界成本的主项是 iGPU 的计算，所以把 reply 砍成 fp16
省的是电线里的一部分，代价 2.5 个 ulp——不值。

### 编译路径下的账，这才是出货配置的账

上面 eager 的 0.38× 有两项与编译无关的噪声（ring 的 216 ms launch 税、eager 循环本身
是编译的 2.2 倍）。用 Gate 1 和 Gate 5 的实测数把编译路径的账单独算一遍，
**收益一项、成本两项，全部实测**：

| | ms | 出处 |
|---|---:|---|
| 编译循环，单卡 | 206.3 | Gate 1 |
| **收益**：50/50 KV 切分后 dGPU 侧变快 | **−24.6** | Gate 4 prefix 长度扫描（206.3 → 181.7） |
| 收益上限：把几乎全部 prefix KV 交出去 | −35 | 同上，prefix → 8 key |
| **成本 1**：电线，1.2 MB 往返 × 360，shm | **+158** | Gate 5，`--split wire` |
| **成本 2**：iGPU 的 attention 块，143 key × 360 | **+467** | Gate 2，iGPU 自报 1.298 ms |

```
最乐观：收益取上限 35 ms，iGPU 当成免费，电线取最快的 158 ms
  206.3 - 35 + 158 = 329 ms   ->  0.63×   仍然亏
光电线就是收益的 6.4 倍（158 / 24.6），iGPU 的 467 ms 还没算。
```

> **所以 context 切法的否决理由从「切的不是成本」改成「收益 24.6 ms，最便宜的电线
> 158 ms」。** 前者是我在 eager 路径上得出的，对出货配置不成立；后者两项都是实测，
> 而且**即使 iGPU 免费、ring 内核完美融合，也还是 6.4× 亏**。

## 4. Gate 3 — sequence parallel：**在电线和 iGPU 介入之前就已经输了**

`--arm ring-2p --split seq`。用户要的那个：51 行 suffix 切 26/25（state token 是第 0 行，
所以两半不等），两张卡都装 6B 模型，每个 layer-step 走一次 ring 交换对端的 suffix K/V，
每个 rank 跑自己那些行的 q/k/v、o_proj、两个 AdaRMSNorm 和 MoE。
prefix KV（42.2 MB）每请求复制一次，不是每步。

`ExpertDecoderLayer` 里除了 attention 全是按行的，所以这个切分是**精确**的——下面的实测精度确认了这点。

| | shm | oneCCL |
|---|---:|---:|
| setup，42.2 MB prefix KV | 35.5–36.2 ms | 391.6 ms |
| `denoise_actions`，全 51 行，单卡 | 448.6–454.1 ms | 454.8 ms |
| 本探针的 forward，全 51 行，无 ring *（对照）* | 507.6–514.5 ms | — |
| 本探针的 forward，**26 行**，ring 打桩 | **505.7–512.0 ms** | — |
| dGPU + iGPU，序列并行 | **2440.3–2459.6 ms** | 2644.8 ms |
| | **0.18×** | 0.17× |
| 每 layer-step ring 交换（104 KiB/hop，2 hop） | 4.93–5.00 ms | 5.42 ms |
| 阻塞在交换里 | 1936.4–1953.9 ms（**占墙钟 79%**） | — |
| action chunk vs 单卡 | **rms 1.202e-3** | rms 1.202e-3 |

### 对照行才是这一节的发现

第三、第四行是关键，而对照行是它们**能被读懂**的原因。`sp_predict_velocity` 是
`LingbotJointModel.forward` 的一份重写，所以它自带开销：跑全 51 行是 507.6 ms，
对 `denoise_actions` 的 448.6 ms，即 **+13% 的探针税**，和切分毫无关系。把这一项按住：

> **507.55 ms（51 行）→ 505.69 ms（26 行）。把 action 行数砍一半，买到 +0.4%。**
> （重复一次读 514.46 → 512.03，+0.5%。绝对值随 host 状态动，比值不动。）

§G5 从 MoE GEMM 的 M 扫描（M=51 0.251 ms、M=26 0.219 ms）拟出「至多 ~13%」。
在**整个 forward** 上实测是 0.4%，原因是 §F1 的而不是 §G5 的：循环是 dispatch-bound，
行数砍半**一个算子都没删掉**。host 两边都得提交同样 360 个 layer-step 的 kernel，
只有 kernel 内部的 device time 变小了一点，而 §F1 实测 host 是紧约束——
286.5 ms 请求里的 282.6 ms。

所以序列并行**把一半活儿送出去、把 ~100% 的成本留下**，
而且这发生在**任何通信之前、iGPU 跑任何东西之前**。§G5 的拒绝是对的，界宽了 30×。

### 然后 iGPU 再加 1910 ms，而电线只有 13 ms

这一段原本是按带宽曲线折出来的；现在 worker 自己报它那半的耗时，所以是加法而不是模型。
同一次运行（shm，load 0.26）：

| | ms | ms/layer-step |
|---|---:|---:|
| dGPU + iGPU 墙钟 | **2433.65** | 6.760 |
| − dGPU 自己那半（ring 打桩） | 511.12 | 1.420 |
| = 阻塞在交换里 | **1922.53** | 5.340 |
| **iGPU 自己那半（ring 打桩，在 iGPU 进程里量）** | **1909.57** | **5.304** |
| **纯电线，104 KiB 双向（`--split wire`）** | **38.6** | **0.107** |
| 两者之和 | 1948.2 | 5.411 |

511.12 + 1922.53 = 2433.65，墙钟正好闭合；而 iGPU 计算 + 电线 = 5.411 ms/layer-step
对实测阻塞的 5.340，多 1.3%（有重叠，也有 iGPU solo 是在更安静的 host 窗口里量的
偏差，§L §6）。两条路对得上，所以：

> **阻塞的 98% 是 iGPU 在算，2% 是电线**（5.304 对 0.107 ms/layer-step，两个数独立量）。
> iGPU 用 5.304 ms 跑它 25 行的一个
> layer-step，dGPU 用 1.420 ms 跑它 26 行的——**慢 3.7×**。这与 §K2 的 3.246 ms
> MoE 块加上 attention、norm、o_proj 一致，也与 §M §3 的 233 ms `predict_velocity`
> 一致（51 行下 6.5 ms/layer-step，25 行略少）。

这一段原先是用「阻塞 − iGPU solo」的**残差**当电线（12.96 ms），那个残差同时装着
测量偏差，比直接量到的 38.6 ms 小 3×。现在两个量都独立测过，残差法作废。

这个实现里的交换是**有序而非全双工**的——一个载荷一个 slot 就意味着 rank 0 先发、
rank 1 再回，于是 dGPU 本地那块计算**没有**和等待重叠。这是真实的实现成本，
而且值得把它界掉，因为它是这里唯一一件「更好的工程能修」的事：

```
最好情况：电线免费 + 完美重叠
  墙钟 = max(dGPU 511.1, iGPU 1909.6) = 1909.6 ms   →  0.24×
```

> **把传输变成免费、重叠变成完美，0.19× 只变成 0.24×。**

iGPU 那一半以 3.7× 成为关键路径，这就是 §K2 的 12.9× `k` 作用在一个「本身就占整体
99.6%」的半份上。**没有任何一个 share 尺寸能救**：dGPU 的成本对行数是平的，
所以每一行搬到 iGPU 都是纯加法。

### 精度结果是好消息

**rms 1.202e-3，低于 3.9e-3 的 fp16 地板**——序列并行的答案比「一份 fp16 chunk 离另一份」
还要近。它也比 Gate 2 的 context parallel（1.291e-2）好 11×，原因是结构性的：
序列切分里两个 rank 都用 fp32 累加自己那些行的 softmax，只有过电线的 K/V 是 fp16；
而 context 切分送的是一份成品 partial output（走 fp32），但 dGPU 自己那块是拿
fp16 舍入过的 query 算的。**ring attention 在这里不花精度**；
这个模型真要做生产级 SP，数值上是站得住的。

## 5. Gate 4 — 那就给 iGPU 少分一点？两条曲线，一条平一条不平

Gate 2/3 的归因是 iGPU 算力，所以下一个问题是负载均衡：**少分给 iGPU 一点。**
这只在「iGPU 的成本真的随份额缩」时成立，本节两张卡各跑一遍同一个扫描，把它量出来
（`--arm share-sweep`，两条曲线因为「少干活」在两种切法里是两件不同的事）。

### 先量融合值多少钱：10%

`sdpa_attention` 就是一个融合实现（整个带掩码 softmax 一次 launch）。同一形状：

| | µs | vs eager |
|---|---:|---:|
| `eager_attention`（4 个算子） | 128.7 | 1.00× |
| **`sdpa_attention`（融合）** | **115.4** | **0.90×** |
| ring，1 shard（11 个算子） | 301.0 | 2.34× |
| ring，2 shard | 668.9 | 5.20× |

> **融合只买到 10%。** 因为在 51 query × 337 key 这个形状上 attention 是
> **launch-bound 而不是算力-bound**——融合减少的是 launch 数，而一个完美融合的
> ring 在 2 shard 下仍然是 **2 次 launch**：约 2 × 115 = 230 µs，对不切的 115 µs。
>
> 所以「写个融合 ring kernel」把 Gate 2 那 223 ms 降到约 **41 ms**（(230−128) µs × 360），
> 但它**仍然是加法，不是节省**。在这个形状上，dGPU **永远不会因为切分而变快**。

### 行份额（sequence 切法）：两张卡都是平的，iGPU 平在 1704 ms

10 步，ring 打桩，无对端。同一份代码，`ZE_AFFINITY_MASK` 换卡：

| 给一个 rank 的 action 行 | dGPU | | iGPU | |
|---:|---:|---:|---:|---:|
| 50 / 50 | 513.78 ms | 1.00× | 2194.74 ms | 1.00× |
| 25 / 50 | 510.45 ms | 0.99× | 1947.67 ms | 0.89× |
| 12 / 50 | 511.62 ms | 1.00× | 1808.62 ms | 0.82× |
| 6 / 50 | 512.43 ms | 1.00× | 1748.27 ms | 0.80× |
| 2 / 50 | 510.45 ms | 0.99× | 1706.48 ms | 0.78× |
| **1 / 50** | **507.89 ms** | **0.99×** | **1704.02 ms** | **0.78×** |

**iGPU 拿 1 行还是要 1704 ms，而且已经饱和**（2 行→1 行只差 2.5 ms）。原因是结构性的：
**序列并行复制权重**，所以 iGPU 无论拿几行都要把 36 层全部专家权重流一遍。

| | |
|---|---:|
| 每步流过的专家权重（32 专家 × 3 矩阵 × 36 层，fp16） | **2718 MB** |
| × 10 步 | **27.2 GB** |
| 零行地板，dGPU 449 GB/s | 60.5 ms |
| 零行地板，iGPU 29 GB/s | **937.2 ms** |
| 实测 iGPU 饱和值 | **1704 ms**（= 达到其带宽的 55%） |

> **iGPU 参与 sequence 切法的最小代价是 1704 ms——编译过的单卡目标 213.5 ms 的 8.0 倍，
> 纯算术地板 937 ms 也已经是它的 4.4 倍。** 任何 kernel 层面的工作都碰不到这个数，
> 它是 27.2 GB ÷ 29 GB/s。**最优份额是 0 行。**

### 顺便回答「耗时是不是在给 51 个 token 生成 KV」——不是，而且它们不进 cache

先纠一个用词：denoise 循环里 51 个 suffix token 的 K/V **从来不缓存**。
`predict_velocity` 传 `fill_kv_cache=False`，`LingbotJointModel.forward:1168` 走的是
`torch.cat([cached_key, key_states], dim=1)`——**cache 里只有 286 个 prefix token**，
suffix 的 K/V 每层每步重算一次（360 次），用完就扔。所以没有「生成 KV cache 的过程」，
只有「每个 layer-step 重算 51 行的 q/k/v」。

它值多少？两个角度，字节是精确的，时间是实测的。

**字节，每 layer-step，fp16，由 config 精确算出：**

| | MiB | 占比 |
|---|---:|---:|
| **routed experts（MoE）** | **72.00** | **81.7%** |
| 融合 q/k/v 投影（造那 51 行的 K/V 的就是它） | 9.00 | 10.2% |
| o_proj | 6.00 | 6.8% |
| prefix KV 读 | 1.12 | 1.3% |
| （这四项） | 88.12 | |

**顺带把 MoE 的实际形状 hook 出来**，因为「51×337」是个很自然的误读——
337 只活在 attention 里（score 矩阵 `[1, 32, 51, 337]`），**它不进 MoE**。
MoE 拿到的是 attention→`o_proj`→`post_attention_layernorm` 之后的东西：

| | 形状 |
|---|---|
| `TokenMoeBlock.forward` 输入 | `[1, 51, 768]` |
| `forward_dense` 的 `hidden_flat` | **`[51, 768]`** |
| `gate_proj` / `up_proj` | `[32, 512, 768]` |
| `down_proj` | `[32, 768, 512]` |
| einsum 1/2（`td,eid->eti`）输出 | `[32, 51, 512]` |
| einsum 3（`eti,edi->etd`）输出 | `[32, 51, 768]` |
| `selected_experts` / `routing_weights` | `[51, 4]` |
| 最后 `etd,te->td` 输出 | `[51, 768]` |

即三个 batched GEMM，batch=32、M=51、K=768→N=512（两次）和 K=512→N=768（一次）。
`3 × 32 × 51 × 768 × 512 × 2 = 3.85 GFLOP`，权重 `3 × 32 × 768 × 512 × 2 = 75.5 MB`，
**算术强度 = 51 = M**，正好是 §F2 那个数——机器平衡是 207，所以 memory-bound 4×。

> 注意 `dense` 路径下 **top-4 一点都不省**：32 个专家全都在 51 个 token 上算完，
> `selected_experts` 只用来在 GEMM **之后**造那个把未选中对置零的 `[51, 32]` 权重。
> 这就是 §F2「top-4 把权重字节只减 4%」的机制。

**时间，编译路径，实测**——用行数扫描：所有随 51 行缩放的东西（融合 q/k/v、o_proj、
两个 norm、attention 的 query 侧）都骑在这条斜率上，而 MoE 的**权重字节**不骑
（§F2：51 行和 1 行都是 72 MiB/layer-step）：

| action 行 | eager | | compiled | |
|---:|---:|---:|---:|---:|
| 50 / 50 | 513.8 ms | 1.000× | **196.1 ms** | **1.000×** |
| 25 / 50 | 512.8 ms | 0.998× | 178.7 ms | 0.911× |
| 12 / 50 | 514.1 ms | 1.001× | 177.7 ms | 0.906× |
| **1 / 50** | 510.6 ms | 0.994× | **176.3 ms** | **0.899×** |

> **编译下，51 行自己的活儿最多值 19.8 ms / 196.1 ms ≈ 10%**，而且 25 行就基本饱和
> （178.7 → 176.3 只差 2.4 ms）。**主要耗时不在这里。**

（这条曲线的绝对值 196.1 ms 对 Gate 1 的 206.2 ms：`sp_predict_velocity` 是重写的
forward，编译后反而比出货那条快 5%——它省掉了双塔的 `inputs_embeds=[None, suffix]`
索引。**斜率**才是这里要的东西，绝对值不要和别处混用。）

**所以编译循环 206 ms 的时间去哪了**，三项都是实测、但来自不同代码路径，不要精确相加：

| | ms | 占 206 ms | 怎么量的 |
|---|---:|---:|---|
| 286 个 prefix token 的 attention | 33–35 | 17% | prefix 长度扫描（本节上文） |
| 随 51 行缩放的全部（含 q/k/v 投影） | ≤ 20 | ≤ 10% | 行数扫描（本小节） |
| MoE GEMM | ~90 | ~44% | §F2，0.251 ms/layer-step × 360 |
| 其余（router / 共享专家 / norm / mrope / 残差 / 提交） | ~60 | ~29% | 余项 |

> **denoise 的主要耗时是流 MoE 专家权重**：81.7% 的字节、每步 2718 MB、
> 每请求 27.2 GB，在 449 GB/s 上有 60.5 ms 的纯 DRAM 地板。
> 而这些字节**和那 51 个 token 一点关系都没有**——§F2 实测 51 个 token 的 top-4 路由
> 会选中 32 个专家里的 30.4 个，所以权重字节由**并集**决定，1 行和 51 行几乎一样多。

### key 份额（context 切法）：iGPU 这条**不平**，但省下来的钱在电线上

| 给一个 rank 的 prefix key | dGPU | | iGPU | |
|---:|---:|---:|---:|---:|
| 286 / 286 | 301.2 µs | 1.00× | 950.3 µs | 1.00× |
| 143 / 286 | 292.8 µs | 0.97× | 605.3 µs | 0.64× |
| 72 / 286 | 303.4 µs | 1.01× | 489.8 µs | 0.52× |
| 36 / 286 | 294.0 µs | 0.98× | 402.6 µs | 0.42× |
| 8 / 286 | 303.2 µs | 1.01× | 347.3 µs | 0.37× |
| **1 / 286** | **299.2 µs** | **0.99×** | **339.7 µs** | **0.36×** |

iGPU 拟合出来是 **340 µs 固定 + 2.1 µs/key**——所以这里**确实有一个可调的份额**。
但它调不出收益，两个原因叠加：

1. **dGPU 那条是平的。** 1 个 key 和 286 个 key 都是 ~300 µs。所以**把 key 送出去，
   dGPU 一分钱都不省**——它 launch-bound，不是 work-bound。
2. **载荷不随份额缩。** query 是 `[51, 32, 128]` = 408 KiB、reply 是
   `[out|lse]` = 822 KiB，**由 query/输出的形状决定，与 KV 份额无关**。
   iGPU 只拿 1 个 key，1.2 MB 的往返照旧。

所以份额一缩，瓶颈就从 iGPU 算力**搬到电线上，而电线不缩**：

```
最优份额（iGPU 只拿 1 个 key）下的每 layer-step 新增成本
  电线，1.2 MB 往返（实测，Gate 5）        0.460 ms     <- 与份额无关
  iGPU 的块，1 个 key（实测，上表）         0.341 ms
                                        ---------
  两者有重叠，取 Gate 2 实测的 send+recv 结构  ~0.80 ms  × 360 = ~288 ms
```

分母是 **0**，而且是实测的 0：上表 dGPU 那一列告诉我们，把 key 送出去它省不到钱。

> **这里要更正我自己的一个早期说法。** 我曾经把这笔账写成「288 ms 的成本对
> 21 ms 的收益，亏 14 倍」。那 21 ms 是 **Gate 1 的天花板**（`115 µs ÷ 2 × 360`），
> 它隐含「attention 的开销与 key 数成正比」——而本节 dGPU 那条平线**正是在否认这件事**。
> 拿一个被自己实测推翻的天花板当分母，是把结论算得比实际**宽容**了。
> 正确的说法没有倍数：**成本 ~288 ms，实测收益 0。**
>
> Gate 1 的天花板是**上界**——它界住任何 attention 并行方案——上界和可达收益是两回事，
> 我之前把它们混用了。

### 但上面两张 µs 级的表都是 eager 的，而 KV 切分要看**编译循环**

这是本文件第三处更正，也是最要紧的一处。上面 dGPU 那两条平线是真的，
但它们量的是 **eager 路径**和一个独立的 `block_attention` 微基准——两者都 launch-bound。
把 prefix KV **真的**缩短、量**整个循环**（`--sweep loop`，无桩、无消融、无天花板算术）：

| prefix KV | eager | | compiled | |
|---:|---:|---:|---:|---:|
| 286 / 286 | 452.1 ms | 1.000× | **206.3 ms** | **1.000×** |
| 214 / 286 | — | — | 203.9 ms | 0.988× |
| **143 / 286** | 452.9 ms | 1.002× | **181.7 ms** | **0.881×** |
| 72 / 286 | 452.2 ms | 1.003× | 176.7 ms | 0.856× |
| 8 / 286 | 452.1 ms | 1.003× | 171.2 ms | 0.830× |

> **eager 平（+0.3%），编译不平（−17.0%）。** inductor 把 launch 开销融掉之后，
> KV 就变成真成本，减半真的省 **24.6 ms**。
>
> **所以「dGPU 一分不省」只对 eager 成立，而生产配置是 `compile_denoise_step=True`。**
> 那句话在本文件前几版里是无条件写的，那是错的。

#### 为什么 1.6% 的字节能占 17% 的时间

这是「循环是 memory-bound，为什么减 KV 没用」这个问题的另一半，而它现在有答案了
（前半是上面那条：编译下**有用**）：

| | |
|---|---:|
| prefix KV，每 layer-step（286 × 8 × 128 × 2 tensor，fp16） | **1.12 MiB** |
| 专家权重，每 layer-step（§F2） | **72.0 MiB** |
| ⇒ KV 占 layer-step 字节的 | **1.6%** |
| 但 KV 占编译循环时间的 | **17%** |

把省下的时间除以省下的字节：143 key 省 0.56 MiB/layer-step 换来 68 µs/layer-step，
即这些字节的有效带宽约 **8.6 GB/s**，而 §M.6 实测整个循环跑在 **183 GB/s**
（B60 峰值 449 的 41%）。

> **KV 是这个循环里最低效的字节，效率差 21×。** 所以它只占 1.6% 的字节却占 17% 的时间。
> 「循环 memory-bound」说的是**专家权重**（98.4% 的字节）；KV 贵不是因为量大，
> 而是因为它是一次 `[286, 8, 128]` 的读去喂一个 M=51 的小 GEMM，
> 算术强度低、张量小到跑不满带宽。

### 两条 share 曲线合起来的结论

> **sequence 切法：给 iGPU 少分一点完全无效**——它的成本是**复制的权重字节**，
> 与份额无关，1 行也要 1704 ms。
> **context 切法：编译下确实有收益（−24.6 ms at 50/50，上限 −35 ms），但载荷不随份额缩**，
> query 408 KiB + reply 822 KiB 由输出形状决定，实测最快 158 ms/循环——
> **光电线就是收益的 6.4 倍。**

## 6. Gate 5 — 传输：给下一个拓扑定价的人看

**shm 在两个拓扑上都打赢 oneCCL**，一次性 setup 快 15×：

| | shm | oneCCL |
|---|---:|---:|
| setup，20.1 MiB（`--split kv`） | 16.9 ms | 524.0 ms |
| setup，42.2 MB（`--split seq`） | 35.5 ms | 391.6 ms |
| 交换（含计算，见下），1.2 MB 往返 | 1.65–1.79 ms | 1.855 ms |
| 交换（含计算），208 KiB 往返 | 4.94–5.00 ms | 5.42 ms |

PHASE10 §9 在 293 KiB 载荷上量到两者相等（0.155 / 0.166 ms），这**没有被推翻**——
它只是不往上延伸。oneCCL 注册过的 pt2pt 路径赢在**小而固定**的缓冲区，
那正是 speculative 一个 tick 发的东西；载荷一旦大到握手不再重要，
经 `/dev/shm` 的一次 host 暂存就赢。**这台机器上 ~300 KiB 以上用 shm。**

**上面那两行「交换」里都含计算，不要当成电线读**——这正是 §0 那处自我更正的内容。

纯电线现在有直接测量：`--split wire`，worker 只回显一个预分配缓冲区，
**不加载权重、不跑 kernel**，载荷就是 ring 的真实载荷（300 次往返）：

| 载荷 | shm | ×360 layer-step | oneCCL | ×360 |
|---|---:|---:|---:|---:|
| context 切法：query 408 KiB 上 + reply 822 KiB 下 | **0.438–0.460 ms** | **158–166 ms** | 0.546–0.569 ms | 197–205 ms |
| sequence 切法：ring 104 KiB 双向（2 hop） | **0.107–0.108 ms** | **38.6–38.8 ms** | 0.248–0.250 ms | 89–90 ms |

有效带宽 2.7–2.9 GB/s（1.2 MB）和 ~2.0 GB/s（208 KiB），shm；两次独立运行，
load 分别 0.45 和 0.07。**这两个数就是「把电线变成免费」能省掉的全部**——
~160 ms 和 ~39 ms，对 Gate 2/3 里 iGPU 的 467 ms 和 1910 ms。

作为交叉验证，也可以用 `phase10_ipc_probe.py` 量对称载荷（shm，50 次迭代）：

| 载荷（每向） | 纯往返 |
|---|---:|
| 76.5 KiB（§M.6 的 MoE 激活） | 0.169 ms |
| 293 KiB（`refresh_prefix_kv`） | 0.282 ms |
| 1.46 MiB（`reground_embs`） | 0.946 ms |

拟合下来是 **0.126 ms 固定开销 + 0.56 ms/MiB**，也就是 ~1.8 GB/s 有效带宽，
而不是本文件第一版写的 0.67 GB/s。**0.155 ms 确实不是传输常数**——它是 76.5 KiB 的
往返，1.46 MiB 要 5.6 倍于它。把这条曲线插值到 ring 的非对称载荷得到 ~0.46 ms，
和 `--split wire` 直接量到的 0.460 ms 一致——**但插值当初是被当成实测写进本文件的，
那是不该做的事**，所以现在留的是直接测量，曲线只当交叉验证。
**结论：传输不是这次的杀手，iGPU 的算力是。** 任何将来的方案仍然要按它自己的载荷
定价，而且要用不含计算的对照去量，别把计算圈进计时窗口。

## 7. 结论

**不变的：** Steps 表一格都没动。§G、§G2、§G4、§G5、§K、§L、§M 各自否掉过一种
把两张卡塞进模型路径的安排，这是第七次，也是第一次**内核真的写出来并验过**的。
§K 的规则原封不动——*iGPU 的活儿只要不把 EU 阵列占住就是免费的；把它排到 dGPU 的
空闲里，绝不和请求并发* ——§L 那半个 16 MiB 工作集预算也一样。
真正砍掉那十步的仍然是 Phase 10 的 speculative round，它一个 tick 一次往返，不是 720 次。

**变了的，值得带走的：**

* **§G5 比它当初写的更强。** 行切分买到 0.4% 而不是 13%，因为循环是 dispatch-bound。
  任何「靠每次 forward 少处理 suffix 行」来省时间的方案都是开局即死，
  **包括和第二张卡毫无关系的那些**。
* **attention 占循环 10.8%。** 这现在是实测数，它封顶所有 attention 侧优化
  在编译后 213.5 ms 里的 ~23 ms，不只是并行类的。
* **ring 内核是对的，留着很便宜。** 探针里的 `ring_masked_attention` 支持任意块掩码，
  这是仓库里 `ring_pytorch_attn.py` 做不到的。这一族里将来若有模型的上下文长到
  attention 真的占大头，那个内核就是起点，而 Gate 0 的 fp32-合并结论就是
  「合并必须留在 fp32」的理由——**而且是免费的**（Gate 0 的定价表）。
* **但那个内核要先融合。** 非融合的 `block_attention` 每 shard 340 µs，
  是整块 eager 的 2.6 倍，纯粹因为 kernel 数多而循环 dispatch-bound。
  任何在这个栈上落 ring 的人，**先写融合 kernel，再谈切分**。
* **瓶颈归因：iGPU 算力 ≫ 电线。** 两种切法都是。这条改了「下一步动哪里」的答案：
  之前 §G/§M.6 的框架是「per-layer collective 太多」，实测下来 collective 反而是
  小项（13% / 0.7%）。真正的墙还是 §K2 的 `k`——iGPU 的 29 GB/s。
* **负载均衡救不了它，而且理由值得记住：iGPU 的成本不随份额缩。** 序列并行复制权重，
  所以 iGPU 参与的最小代价是把 27.2 GB 流一遍 = 实测 1704 ms，8× 目标。
  **任何要求 iGPU 持有全部 36 层权重的方案都被这 1704 ms 封死**，与份额、与 kernel
  质量都无关。唯一能减少 iGPU **字节**的方向是按层切（PHASE13 §5c 已经定过价），
  不是按行或按 key 切。
* **286 token 的 prefix KV 值编译循环 33–35 ms（17%），而它只占 1.6% 的字节。**
  这是本阶段对 §F2 的补充：循环 memory-bound 说的是专家权重，而 KV 是循环里
  **最低效的字节**（8.6 GB/s 对循环平均 183 GB/s，差 21×）。想动 attention 侧的人
  应该按这 33–35 ms 定价，不是按 §F2 的字节占比。
* **eager 的曲线形状不能外推到编译路径。** eager 下 attention launch-bound
  （1 key = 286 key），编译下不是（−11.9% at 143 key）。生产是编译的。
  §8 ⑤ 记了这个坑的完整代价。
* **传输按自己的载荷定价，而且别把计算圈进计时窗口。** Gate 5 和 §8 ③。

## 8. 方法论：五个会让结论反号的坑

**① `max_rel` 在 attention 输出上是废指标。** 第一版 Gate 0 用逐元素相对误差做闸门，
读出 `max_rel = 3.4`，看着像内核全错。实际上 `max_abs = 2.441e-4`，
正好一个 fp16 ulp——张量里有 ~1e-5 量级的元素，一个 ulp 的绝对误差落在那儿
就显示成 3.4 的相对误差。闸门现在用 `max_abs_norm`（最大误差 / 参考自身的最大幅值）
和 `rel_l2`，并且**显式打印那个 ulp**，好让「压在地板上」和「有 bug」能分开。

**② 重写过的 forward 必须有同代码路径的对照。** Gate 3 的 505.7 ms（26 行）
单看比单卡 51 行的 448.6 ms 还**贵**，很容易写成「切分本身让它变慢」。
真相是 `sp_predict_velocity` 自带 +13% 的重写税：同一份代码跑全 51 行是 507.6 ms。
把对照量出来，0.4% 这个数才立得住。**没有这一行，这一节的结论会是错的。**

**③ 计时窗口里不能圈计算——这次真的把归因搞反了。** 第一版 Gate 2 的「交换」是
一个 `perf_counter` 跨 `send` → 本地 block → `recv` 的整段，于是 1.794 ms 里
只有约 0.5 ms 是电线，剩下是 dGPU **自己**的 attention 和 iGPU 的块。
按那个数写出来的结论是「电线是天花板的 56 倍，传输杀死了方案」。
修法有两条，两条都做了：驱动端把计时**拆成 `send` / `local` / `recv` 三段**，
worker 端**自己报它的块耗时**。修完的归因是反过来的——电线只占设备边界 13%。
> 一个跨越「发出去 → 自己算 → 收回来」的计时器量的是**延迟**，不是**传输**。
> 要传输就必须有一个不含计算的对照（`--split wire`）或者对端的自报。

**④ 插值来的数不能和实测混排，天花板不能当可达收益。** 修完 ③ 之后我犯了两个同类的错，
都是在**汇报**层面而不是代码层面：

* 电线成本一度是从 `phase10_ipc_probe.py` 的**对称**载荷曲线插值到 ring 的**非对称**
  载荷（408 KiB 上 / 822 KiB 下）的，却和实测数并排写在同一张表里。
  事后 `--split wire` 直接量到 0.460 ms，插值给的 ~0.46 ms 恰好对——**但那是运气，
  不是方法**，而且它和另一条残差算法（阻塞 − iGPU solo）差 2.6×，当时没有任何东西
  能判定哪条对。现在两条都作废，留直接测量。
* Gate 4 的账一度写成「288 ms 成本对 21 ms 收益，亏 14 倍」。那 21 ms 是 **Gate 1
  的天花板**，而它隐含「attention 开销与 key 数成正比」——恰恰是 Gate 4 自己的平曲线
  否掉的假设。**上界不等于可达收益**；把上界当分母会把结论算得偏宽容。
  实测的分母是 0。

> 两条都不是数值错误，是**把不同证据等级的数排在一起**。表里标清「实测 / 插值 /
> 上界」比多量三个数更重要。

**⑤ eager 的曲线形状不能外推到编译路径，而出货跑的是编译。** 这是本文件最贵的一个坑。
`--arm share-sweep` 的 dGPU key 曲线是平的（1 个 key 和 286 个都 ~299 µs），
`ring-math` 的 `block_attention` 微基准也是平的，两条都真实，于是我写下了
「attention 在这个形状上 launch-bound，所以切 KV 切的不是成本」。
把 prefix KV 真的缩短并量**编译**循环，−11.9%。

> eager 和一个 11-算子的微基准都受 launch 支配，所以它们对「工作量」不敏感；
> inductor 把 launch 融掉之后工作量就回来了。**任何「X 不是瓶颈」的结论，
> 都必须在出货用的那条路径上重量一遍**——这里是 `compile_denoise_step=True`。
> 代价：三段结论要改，而且改的是归因不是方向。

## 9. 没测的，和不能拿这次结果说的话

* **两个 `ring-2p` arm 只能是 eager，而且这不是偷懒，是结构性的。** `ring_kernel`
  里有 `transport.send` / `recv`——host 侧调用，`fullgraph=True` 的图里放不下。
  真做一个编译版的两进程 ring，需要在每次交换处**主动开图断点**，那是另一件工程，
  不是探针改几行。所以本文件**不能**说「编译后切分是 0.18×」。
  编译路径的账是用三个独立实测项算的（收益 24.6 ms / 电线 158 ms / iGPU 467 ms，
  Gate 2 末尾），不是跑出来的——而它的结论（光电线就是收益的 6.4 倍）不依赖那三项
  如何重叠。
* **`--sweep loop` 截断 prefix KV 之后输出是没有意义的**（suffix 看到的 prefix token
  变少了）。那一组只量成本，不量行为。它也不是「attention 的总成本」——
  51 个 suffix key、融合 q/k/v 和 mrope 都还在图里，所以 33–35 ms 是
  **「286 个 prefix token 值多少」**，不是 attention 的全部。
* **214 → 143 那一段占了整个降幅的大部分**（−1.2% 然后 −10.9%），非线性。
  大概是融合 attention kernel 的分块/对齐效应（总 key 数 337 / 265 / 194），
  没有追下去——本文件需要的是 50/50 那一点的值，它有了。
* **全双工传输没实现，只界掉了。** Gate 3 的「最好情况 0.23×」是用 `blocked_ms`
  当 iGPU 下界算的界，不是一次实测。它不需要实测——因为它已经是负的。
* **只量了 B=1。** §F2 说 B=4 是带宽交叉点；批量下 attention 的占比会变，
  Gate 1 的 10.8% 是 B=1 的数。
* ~~`--split seq` 只测了 26/25 一种划分。~~ **已补**：Gate 4 两张卡各扫了
  50/25/12/6/2/1 行和 286/143/72/36/8/1 个 key。最优 share 确实是 0 行，但理由不是
  原先写的「dGPU 平」，而是「**iGPU** 也平，平在 1704 ms」。
* **两条 share 曲线都是 `sp_predict_velocity`（重写的 forward）量的，带 +13% 探针税。**
  比值不受影响（分子分母同税），绝对值受。iGPU 那条的 1704 ms 因此是略高的估计，
  但它要降到 213.5 ms 以下需要 8×，探针税只有 1.13×。
* **精度只在基础 checkpoint、随机相机帧、固定 prompt 上量过。** rms 1.202e-3 是
  数值结论，不是机器人行为结论。要做任务级结论得上 RoboTwin 微调 checkpoint
  和 `open_loop_*` 那一套（§11.7 记过拿错 checkpoint 的坑）。
* **没有碰 `vllm_omni/`。** 本阶段不落生产。

---

## 10. 复现

```bash
D=test-image_zy_scaler0260b2_lingbot_omni
R=/llm/zhuyong/lingbovla/my/vllm-omni

# 全部四个 arm，~12 min。arm 之间会等 load 掉下来（Rules #2）
docker exec $D bash -lc "cd $R && bash spikes/lingbot_vla_v2/test_ring_attention.sh"

# Gate 0 单独跑，~1 min，不需要第二张卡
docker exec $D bash -lc "cd $R && ARMS=ring-math bash spikes/lingbot_vla_v2/test_ring_attention.sh"

# Gate 1 天花板，~2 min，不需要第二张卡
docker exec $D bash -lc "cd $R && ARMS=attn-share bash spikes/lingbot_vla_v2/test_ring_attention.sh"

# Gate 1/Gate 4 — 编译循环 + prefix KV 长度扫描。这是 KV 切分天花板的来源
docker exec $D bash -lc "cd $R && ZE_AFFINITY_MASK=0 PYTHONPATH=. \
    python spikes/lingbot_vla_v2/phase14_ring_attention_probe.py --arm share-sweep \
    --sweep loop --compile-denoise-step --loop-prefix 286,214,143,72,8 \
    --model /tmp/lingbot-vla-v2-perf --iters 8"

# Gate 5 — 纯电线。两端都不跑 kernel，~30 s，两种传输都量一遍
for t in shm oneccl; do
  docker exec $D bash -lc "cd $R && ZE_AFFINITY_MASK=0 PYTHONPATH=. \
      python spikes/lingbot_vla_v2/phase14_ring_attention_probe.py --arm ring-2p --split wire \
      --transport \$t --worker-cpu 10-11 --wire-iters 300 --model /tmp/lingbot-vla-v2-perf"
done

# Gate 4 — share 扫描。两张卡**分别**跑，不要并发（会互相污染，§L §6）
for card in 0 1; do
  docker exec $D bash -lc "cd $R && ZE_AFFINITY_MASK=$card PYTHONPATH=. \
      python spikes/lingbot_vla_v2/phase14_ring_attention_probe.py --arm share-sweep \
      --model /tmp/lingbot-vla-v2-perf --iters 2"
done

# Gate 2/3，两张卡。TRANSPORT 默认 shm（Gate 5），oneccl 可复现对比
docker exec $D bash -lc "cd $R && ARMS=ring-2p-kv,ring-2p-seq bash spikes/lingbot_vla_v2/test_ring_attention.sh"
docker exec $D bash -lc "cd $R && TRANSPORT=oneccl ARMS=ring-2p-kv,ring-2p-seq \
    bash spikes/lingbot_vla_v2/test_ring_attention.sh"
```

| arm | 它定什么 | 要 iGPU |
|---|---|---|
| `ring-math` | 内核和合并算子是对的 | 不要 |
| `attn-share` | 天花板：attention 占循环 10.8% | 不要 |
| `ring-2p --split kv` | context parallel；电线是天花板的 56 倍 | 要 |
| `ring-2p --split seq` | sequence parallel；行切分买到 0.4% | 要 |
| `share-sweep --sweep rows,keys` | 少分给 iGPU 有没有用（两张卡各跑一次） | 两张卡分别跑，不并发 |
| `share-sweep --sweep loop --compile-denoise-step` | **编译**循环 vs 真实 prefix KV 长度——KV 切分天花板的唯一可信来源 | 不要 |
| `ring-2p --split wire` | 纯电线：ring 的真实载荷，两端都不算东西 | 要（但不加载权重） |

`ring-2p-seq` 会**加载两次 6B 模型**，第二次进 host DRAM、29 GB/s，光这一步就 ~35 s。
每个 arm 都打印它启动时的 load average；Rules #2 要求 < 2.0，本阶段有一次跑在
1.75 上启动，那组数被丢掉重跑在 0.08（结果复现了，但规则是规则）。
`oneccl` 那条需要 `LD_LIBRARY_PATH` 和 `CCL_PLUGIN=ONECCL_IGPU`，
脚本已经设好，和 `test_spec_oneccl.sh` 同一份安装。
