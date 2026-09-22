# Phase 13 — 用 PTQ 压缩的真实模型当 iGPU 草稿：精度前提成立，速度前提不成立

## Context

用户提的方案：把 lingbot-vla-v2-6b 量化剪枝后放到 iGPU 上跑 10 步 denoise，
dGPU 正常跑 ViT+text 并验证，接受则保留、拒绝则 dGPU 重算。

**调度部分已经存在。** `spec_reground`（`PHASE10_SPECULATIVE.md` §12.9，提交 `4681a21c`）
就是这四条，落在 `spec_decode.py:538` `_spec_round`。用户方案里新的只有一件事：
**草稿从 4 M 参数的 `LingbotDraftHead` 换成压缩后的真实模型**。
这值得量，因为它一次打掉 Phase 10 的两个未决项 —— 草稿头要训练（今天接受率 0）、
以及 KV 陈旧性（探针 2 从没跑过，而 `reground` 每 tick 重新 grounding，根本不存在陈旧）。

**预算是 58 ms，不是 90 ms。** `_ground` 的 80.8 ms 里，草稿只要 `embs`
（`_spec_round` 调的是 `_refresh_draft(ground.embs, state)`，不碰 KV），
而 `embed_prefix` 22.6 ms 就产出了它；dGPU 随后还要花 `prefix_forward` 58.1 ms
建 verifier 的 KV（`spec_decode.py:440`）。落在这个窗口里的草稿是免费的
（§M §4 实测：memory-bound 的 iGPU 负载对 dGPU 只收 1.01×，不是 §K 的 1.75×）。

> **这个重叠今天还不存在。** `_spec_round` 是严格串行的：`_ground` 整个跑完才调
> `_refresh_draft`。58 ms 预算隐含一次重排（把 `_ground` 拆成 `embed_prefix` /
> `prefix_forward`，草稿在两者之间发车）。这个重排很小，但它是本文件所有预算比较的前提，
> 而且**它没做**——因为 Gate 2 先关掉了方案，做了也没用。若 iGPU 侧将来翻案，这是第一步。

按用户的选择，本阶段**只做 PTQ，不做需要重训的剪枝**，目标是量出差距。

探针：`phase13_ptq_draft_probe.py`（四个 arm）+ `phase10_ipc_probe.py` 的两个新载荷。
容器 `test-image_zy_scaler0260b2_lingbot_omni`，torch 2.12.0+xpu，checkpoint
`/tmp/lingbot-open-loop`（RoboTwin 微调）。

---

## 0. 前置：MoE 是 `bmm`，而 XPU 上的量化算子只有 2D `mm`

路由专家是一个 denoise step 字节的 69.5%，绕不开。`GroupedExperts.forward_dense`
用 einsum 收缩专家轴，而所有 WOQ 算子都是 2D GEMM。两者可以对上，且**不改变算术**：

```
gate/up   [E,I,H] -> [E*I, H]                      x @ W.T -> [T, E*I]
down      [E,H,I] -> [H, E*I]，路由权重先折进激活    H @ D.T -> [T, H]
```

因为 `H[t,(e,i)] = w[t,e]·h[e,t,i]`，所以 `H @ D.T = Σ_e w_e·h_e·down_e`，
正是那两个 einsum 算的东西。**每层仍然是 3 次 launch，和 `bmm` 一样。**

折叠本身单独验收（`--check-folding`，fp16 不量化）：整条 10 步 denoise 的按维 RMS
**2.807e-04**，低于 fp16 重构底噪 3.9e-3。先验收折叠再量化，否则折叠 bug 和量化损失
是同一个数字，分不开。

---

## 1. Gate 0 — 内核选型：只有一个能用，而且只在 iGPU 上有收益

一个 MoE layer，`M=51`，两张卡，**背靠背计时**（见下方"方法论"）：

| 内核 | iGPU ms/layer | dGPU ms/layer | iGPU GB/s | dGPU GB/s |
|---|---:|---:|---:|---:|
| fp16 `mm`（基线） | 3.256 | 0.255 | 23.2 | 296.5 |
| `aten::_weight_int8pack_mm` | **469.2** | 496.2 | 0.1 | 0.1 |
| `aten::_weight_int4pack_mm` | 8.83 | 1.13 | 2.3 | 17.8 |
| **`esimd_gemm_int4_pgrp`（Q4_0）** | **0.911** | 0.300 | 21.4 | 64.9 |

iGPU 侧三次重复：fp16 `3.236 / 3.248 / 3.286`，int4 `0.912 / 0.917 / 0.933`，
离散 ±1.5%，比值稳定在 **3.5–3.6×**。另有一次读到 fp16 6.899 / int4 3.751（**2.1–4.1× 偏慢**），
那次 `load1m=1.27` 而干净的几次是 0.59–0.81 —— 这就是 §L §6 记的
"iGPU 吞吐随 host CPU 活动摆动 2.4×"，它是真的，而且一次就能骗过你。
**iGPU 的绝对值必须重复三次并报 load。**

1. **stock PyTorch 的 WOQ 算子在两张卡上都比 fp16 慢。** int8 是标量回退（0.1 GB/s，
   慢 144×）；int4 慢 2.7×。两者都在任何 torch-xpu 里，都是陷阱，所以记下数字。
2. **能用的是容器自带的 Intel ESIMD Q4_0 内核**
   （`vllm/model_executor/layers/quantization/sym_int4.py`）。`gemm` 变体限
   `2 ≤ rows ≤ 64`，而 `M=51` 正好落在里面 —— 它是为小 batch decode 写的，
   和 denoise 循环的形状天然对上。iGPU 上 **3.60×**，且有效带宽与 fp16 持平
   （21.5 vs 23.2 GB/s），说明字节换时间近似线性。
3. **dGPU 上 int4 是负收益**（0.255 → 0.301）：它的 fp16 路径已经 295 GB/s，
   而 ESIMD 内核封顶 64.7 GB/s。**压缩草稿只可能放 iGPU** —— 这一条独立地支持了
   用户的设备切分，也是本阶段唯一支持 iGPU 的证据。

布局（照抄 `_register_linear_int4_layouts`）：输入 fp16 `[M,K]`、
`qweight` uint8 `[N,K/2]`（量化器出 int32 `[N,K/8]` 再 `.view(torch.uint8)`）、
`scales` fp16 `[N,K/128]`、输出 fp16 `[M,N]`。约束 `N % 16 == 0`、`K % 128 == 0`。

---

## 2. Gate 1 — 接受率：**过，而且余量极大**

dGPU，真实帧（`adjust_bottle_3ep_2chunks.npz`，6 个 grounding），
判据直接 import 生产的 `radius_prefix_acceptance` / `truncate_on_gripper_switch`，
`tau=0.15`，teacher 是同一份权重的 fp16 10 步：

| 模式 | 步数 | 对 teacher 按维 RMS | 接受率 | 接受前缀 | dist | 对 tau 的余量 |
|---|---:|---:|---:|---:|---:|---:|
| fp16 | 10 | 0.0000 | 100% | 12/12 | 0.0030 | 50× |
| fp16 | 4 | 0.0036 | 100% | 12/12 | 0.0035 | 43× |
| fp16 | 2 | 0.0041 | 100% | 12/12 | 0.0040 | 37× |
| int4-moe | 10 | 0.0022 | 100% | 12/12 | 0.0037 | 40× |
| int4-moe | 4 | 0.0042 | 100% | 12/12 | 0.0041 | 36× |
| int4-moe | 2 | 0.0052 | 100% | 12/12 | 0.0047 | 32× |
| int4-all | 10 | 0.0053 | 100% | 12/12 | 0.0056 | 27× |
| int4-all | 4 | 0.0079 | 100% | 12/12 | 0.0068 | 22× |
| **int4-all** | **2** | **0.0083** | **100%** | **12/12** | 0.0076 | **20×** |

`int4-moe` = 36 个 MoE 块折叠成 Q4_0；`int4-all` = 再加 252 个 Linear
（qkv 融合、o_proj、两个 AdaRMSNorm 的 gamma/beta、shared expert 的融合 gate_up）。
36 个 `shared_expert.down_proj` 跳过（`K=704` 不是 128 的倍数），路由 gate 故意不量化
（MODEL_ARCH §4 注 1：bf16 会翻转 top-4 选择，那是**选择**变了，不是精度损失）。

**读法：用户的核心假设成立。** 最狠的一格（int4-all + 2 步）按维 RMS 0.0083，
比探针 3 的 0.02 门限还好 2.4×，全部 12 步被接受，距离对 tau 有 20× 余量。
对照 4 M 草稿头今天的 RMS **0.888** 和接受率 **0**：**PTQ 草稿好 107×，而且不需要训练。**

一个次级但清楚的分解：**步数比量化贵**。int4-all 在 10 步上只加 0.0053，
而从 10 步降到 2 步（fp16）加 0.0041 —— 两者同量级，而 int4 省的字节多得多。

（合成噪声帧上同一张表是 0.0064–0.0222，也全过，但更差。噪声图像给不出结构化轨迹，
精度数字必须用真实帧，探针里 `--dataset` 缺省时会明说自己用的是噪声帧。）

---

## 3. Gate 2 — iGPU 速度：**不过，而且不是字节的问题**

eager、`B=1`、真实模型。`floor` 臂 = `int4-all` 再把专家截到 1 个：
**kernel 数完全不变、字节掉 71%**，所以它量的是与字节无关的地板。
它的输出是错的，这是故意的；只有时间有意义。

### iGPU（`ZE_AFFINITY_MASK=1`）

| 模式 | 权重 MB | 1 步 | 其中 host issue | 10 步 | 2 步 | vs 58 ms |
|---|---:|---:|---:|---:|---:|---:|
| fp16 | ~3610 | 221.5 | 46.1 | 2207.7 | 442.6 | 38.0× |
| int4-moe | 1591 | 140.3 | 42.9 | 1396.9 | 280.2 | 24.0× |
| int4-all | 953 | 118.4 | 41.1 | 1186.4 | 238.5 | 20.4× |
| **floor（E=1）** | **274** | **75.9** | **39.9** | 757.4 | 152.0 | **13.0×** |

"权重 MB" = 被换掉的那些模块的实际读字节，**加上 42 MB 的 prefix KV**
（每步 attention 都要重读，不随量化变化）。`fp16` 一行是按形状算的，因为那一行
什么都没换、探针的字节计数器是 0。

对 `int4-all` 与 `floor` 两点做线性拟合（时间 = a·字节 + 地板）：

    a    = 42.51 ms / 678.9 MB = 0.0626 ms/MB  ->  16.0 GB/s
    地板 = 75.92 - 0.0626 x 273.8 = 58.8 ms / 步

> **iGPU 上一个 denoise step 的零字节地板是 58.8 ms。整个十步预算是 58.1 ms。**
> 也就是说，把权重压到零，**一步**的纯开销就等于**十步**的全部预算。

这是比 §L "没有 reuse 可抓" 更强的说法，因为它不依赖 roofline：量化和剪枝都只攻字节，
而字节在 `int4-all` 的 118.4 ms 里只占 59.6 ms。

**地板的一半是 host issue。** 不同步计时（Python + 下发，设备工作被队列吸收）
在四个模式上是 46.1 / 42.9 / 41.1 / **39.85** ms —— 几乎不随压缩变化，
在 floor 臂上占 75.9 ms 的 **52%**。剩下约 19 ms 是设备侧的非权重工作
（337 token 的 attention、norm、router，以及 36 层小 kernel 的执行本身）。

**最乐观的上界也不够。** 假设图捕获把 host issue 全部消掉（§F3b 在 dispatch-bound
链上实测到 4.6×），并且字节压到零，剩下约 19 ms/步：10 步 = 190 ms，仍是预算的 **3.3×**。
而且 §F3b 的结论是 XPU Graph 在这个栈上**不能安全部署**（Inductor 复用缓冲导致跨 replay
污染，五 seed 门禁抓到 MAE 5.388e-01）。所以这个上界目前也拿不到。

### dGPU（`ZE_AFFINITY_MASK=0`）—— 对照，解释为什么压缩在这边也没用

| 模式 | 1 步 | 其中 host issue |
|---|---:|---:|
| fp16 | 48.80 | 49.05 |
| int4-moe | 46.08 | 45.57 |
| int4-all | 46.03 | 45.30 |
| floor（E=1） | 45.40 | 45.06 |

**eager 下 dGPU 的一步几乎全是 host issue**（issue ≈ 步时间），
把 71% 的字节拿掉只快 7%。dGPU 的杠杆是编译（已出货：21.3 ms/步 对 eager 49），
不是量化。这和 Gate 0 的 dGPU 列是同一件事的两种测法。

---

## 4. Gate 3 — 传输：42 MB 的 prefix KV

草稿跑真实的 36 层 expert tower，就要拿到 VLM 的 prefix K/V 去做跨塔 attention：
`36 层 × 286 token × 1024 × 2(K,V) × 2 B = 42.2 MB`，每 tick 一次（`reground` 每 tick 重建）。
§9 只把这条边定价到 293 KiB，这是它的 147×。往返中位数：

| 载荷 | shm | oneCCL |
|---|---:|---:|
| `reground_prefix_kv` 42.2 MB | **31.71 ms** | 51.91 ms |
| `reground_embs` 1.46 MB | 1.01 ms | 1.26 ms |

单向约为一半，即 prefix KV 下行 **~16 ms**，占 58 ms 预算的 27%，在草稿开跑之前就花掉。

**§9 "oneCCL 快 15%" 在这个尺寸上反号了**（51.9 对 31.7，慢 64%）。原因在 §9 自己
记过：只有 iGPU 那一侧需要 plugin 管理的 USM host 中转（`_prepare_send_tensor:399`）。
293 KiB 时这笔中转被延迟掩盖，42 MB 时它就是主项。**§9 的传输选型结论只对小载荷成立。**

---

## 5. 结论

| 闸门 | 结果 |
|---|---|
| Gate 0 内核 | **过**。ESIMD Q4_0，iGPU 3.60×，dGPU 负收益 |
| Gate 1 接受率 | **过，余量 20–50×**。int4-all + 2 步：RMS 0.0083、接受 100%、12/12 |
| Gate 2 iGPU 速度 | **不过**。零字节地板 58.8 ms/**步**，预算是 58.1 ms/**十步** |
| Gate 3 传输 | 不是决定项，但要计价：prefix KV 单向 ~16 ms |
| §5b 减层 | **不过**。训练自由上限 24 层（接受率 100%），iGPU 需要 8 层，差 **3×** |
| §5c 整条 tick | 最优格 L=24/2 步 = **217.8 ms**，比同样压缩留在 dGPU（109.2 ms）慢 **2.0×** |

**方案的精度前提是对的，速度前提是错的，而且错在一个 PTQ 够不着的地方。**
用户的直觉——"压缩后的真实模型是个好草稿，误差由 verify 兜底"——被 Gate 1
以 20–50× 的余量证实了，它确实解决了 Phase 10 卡住的那个问题。
但 Gate 2 说 iGPU 的瓶颈不是它读多少字节，而是它跑 36 层小 kernel 这件事本身：

    今天最好的一格（int4-all，2 步，iGPU）     238.5 ms + 16 ms 传输 = 254 ms
    去掉全部权重字节                            117.6 ms + 16 ms      = 134 ms
    再去掉全部 host issue（需要不安全的图捕获）   ~38 ms + 16 ms      =  54 ms
    预算                                                                58 ms

只有最后一行进得去，而它要求两件都还没有的东西同时成立。
**按用户选定的"只做 PTQ"范围，差距是 4.4×**（254 / 58）。
补齐这 4.4× 不能靠继续压字节——字节只剩 59.6 ms 里的一部分——只能靠减层。
**减层已在 §5b 量完**：训练自由的上限是 24 层（砍 33%，接受率仍 100%），
而 iGPU 预算要求 8 层，**两条曲线差 3× 且不相交**。所以这条路也走到头了，
再往下就是用户明确排除的蒸馏恢复。

### 反而更有价值的副产品：`num_steps` 本身

Gate 1 顺带显示 2 步对 10 步的 RMS 只有 0.0041。用现成的
`open_loop_steps_sweep.py` 对**真值**扫了一遍（同一数据集，fp16，dGPU）：

| 步数 | MAE(0-6) | MAE(all) | jerk(0-6) |
|---:|---:|---:|---:|
| 1 | 0.0139 | 0.0090 | 0.0059 |
| **2** | **0.0126** | **0.0071** | 0.0035 |
| 4 | 0.0137 | 0.0076 | 0.0032 |
| 5 | 0.0127 | 0.0071 | 0.0032 |
| 10（出货值） | 0.0137 | 0.0078 | 0.0037 |
| 20 | 0.0157 | 0.0098 | 0.0053 |

真值 jerk 0.0023；hold-state 基线 MAE 0.4199。

**MAE 在 1→20 步之间基本持平，2 步（0.0126）比出货的 10 步（0.0137）还略好，
jerk 也更平滑。** 也就是说 `num_steps=10` 在这个数据集上没有买到精度。
若成立，full round 从 `80.8 + 10×21.3 = 294 ms` 降到 `80.8 + 2×21.3 = 123 ms`，
**2.4×，不需要投机、不需要 iGPU、不需要草稿头、不需要 verifier** ——
比目前实测的任何投机臂都快（`cached` 在真实接受率 0 下是 164.7 ms，baseline 292.6 ms）。

这条**必须先扩大验证再当真**：只有一个任务的 6 个样本，且开环 MAE 不是闭环成功率。
但它是目前杠杆最大的一条，且成本远低于本文件里的任何一项。

### 建议的下一步，按杠杆排序

1. **把 `num_steps` 扫描扩到多任务多 episode**（`export_open_loop_bundle.py` 已有
   `--stride`）。如果 2–4 步站得住，先落这个，别的都先放着：单这一条就是 2.4×。
2. **再叠上 24 层**（§5b 的 greedy 集合，训练自由、接受率 100%）。
   `num_steps=2` + `L=24` + 编译 = full round 约 **109 ms，2.7×**，不需要 iGPU。
   落地前必须先按 §5b.4 补对真值的开环检查——投机里 verifier 是满层模型，
   直接出货就没有这个兜底了。
3. **如果还要投机**：草稿放 dGPU，省法用步数和层数，不要用 int4 ——
   Gate 0 和 Gate 2 的 dGPU 列都说明那边的杠杆是编译，量化是负收益。
4. **iGPU 只在以下前提翻案**：XPU Graph 变得可安全部署（§F3b 的 Inductor 缓冲问题解决），
   届时重跑 Gate 2 的 `floor` 臂即可，一条命令。在那之前 iGPU 上的
   model-path 方案不用再论证了 —— §K4 是 `k`、§L 是 reuse、§M §3 是粒度、
   Gate 2 是**与字节无关的每步地板**、§5b 是**减层的精度上限只有 24 层**，
   五个独立的理由。

---

## 5b. 减层 —— Gate 2 之后唯一的杠杆，量完了：两条曲线差 3×

Gate 2 说 iGPU 的地板是**每层**开销，而量化只攻字节，所以剩下的唯一变量是层数。
`--arm depth`，dGPU，真实帧 4 个 grounding，草稿 2 步，判据仍是生产的接受规则。

### 1. 单层敏感度：关键层在**两端**，不在深处

逐个丢掉一层、跑完整条 denoise、对满 36 层的 teacher 打分（摘录）：

| 丢掉 | rms | | 丢掉 | rms |
|---|---:|---|---|---:|
| **L0** | **0.3882** | | L20 | 0.0321 |
| L5 | 0.0032（比满层的 0.0041 还低） | | L32 | 0.0435 |
| L16 | 0.0037 | | L33 | 0.0713 |
| L21 | 0.0038 | | L34 | 0.0982 |
| L9 / L4 | 0.0041 / 0.0039 | | **L35** | **0.1841** |

由低到高的完整顺序：
`5 9 4 21 24 17 16 8 29 19 3 18 14 26 30 6 27 15 22 13 28 1 11 25 12 23 31 2 7 20 32 33 10 34 35 0`

**这和 LLM 的减层经验相反。** LLM 里冗余的是深层连续块；这里**越深越关键**
（L32→L35 单调升到 0.1841），而最关键的是 **L0**。两个原因都是结构性的：

* L0 是 action query 第一次遇到 VLM 前缀的地方。
* 输出是**回归**——最后一层直接过 `action_out_proj` 出速度场。LLM 的末层是在
  已经成形的分布上做锐化，下游有冗余可以吸收；这里没有。

### 2. 所以固定模式全军覆没，只有按实测敏感度挑有用

| 模式 | L=30 | L=24 | L=18 | L=12 | L=8 |
|---|---|---|---|---|---|
| **greedy**（保留最敏感） | **100%**, rms 0.0107 | **100%**, rms 0.0286 | 75% | 50% | 0% |
| `stride`（等距采样） | 100%, rms 0.0863 | 100%, prefix 9.2 | 25% | 0% | 0% |
| `head`（留浅层） | **0%** | 0% | 0% | 0% | 0% |
| `tail`（留深层） | **0%** | 0% | 0% | 0% | 0% |

`head` 和 `tail` 在**只丢 6 层**时就已经接受率 0 —— 因为关键层在两端，
任何连续模式都必然砍掉其中一端。`stride` 能同时保住两端所以不至于崩，但仍远逊于
按敏感度挑。**结论：减哪些层必须实测，而且要用 verifier 自己的度量去测**，
这一步很便宜（36 次 denoise，几十秒）。

> 顺带记一个我自己踩的方向错误：第一版把 `order[:count]` 当成保留集，
> 而 `order` 是**升序**敏感度，于是保留了最不敏感的、丢掉了 L0/L34/L35。
> 结果是 "greedy 崩溃、stride 获胜"，一个完全反的结论。修正后 greedy 在每个深度都赢。

### 3. 训练自由的深度上限是 **24 层（砍 33%）**，而 iGPU 需要 8 层

层数对时间是线性的，实测验证过（不是外推）：

| L | greedy 接受率 | iGPU int4-all 2 步 | + prefix KV 传输 | dGPU int4-all 2 步 |
|---:|---:|---:|---:|---:|
| 36 | 100% (rms 0.0041) | 238.5 | 254.4 | 92.1 |
| 24 | **100%** (rms 0.0286, 余量 6×) | **159.6** | **170.2** | **60.8** |
| 12 | 50% | 79.1 | 84.4 | 30.0 |
| 8 | **0%** | ~53 | ~56（**进得去**） | ~20 |

（iGPU 实测 36/24/12 层 = 118.4 / 79.8 / 40.4 ms 每步，线性外推预测 118.4 / 79.0 / 39.5，
吻合。传输按 15.86 × L/36 ms 单向。）

> **两条曲线不相交。精度要求 L ≥ 24，iGPU 预算要求 L ≤ 8，差 3×。**
> L=24 时 iGPU 要 170.2 ms 对 58.1 ms 的预算，仍然 **2.9× over**；
> 而预算进得去的 L=8 接受率是 0。

24 层这个上限（砍 33%）正好落在 LLM 减层文献报告的 20–30% 区间里 ——
也就是说这个模型在这件事上**没有额外的冗余可捡**，PTQ + 训练自由减层这条路到此为止。
要到 L=8 就必须蒸馏恢复，那是用户明确排除的范围。

### 4. 但减层在 dGPU 上是真的有用

同一个 L=24（接受率 100%、余量 6×）放到 dGPU、配合 `num_steps=2`：

    full round = _ground 80.8 + 2 x 21.3 x 24/36 = 80.8 + 28.4 = 109 ms
    对照 baseline 292.6 ms ->  2.7x，不需要 iGPU、不需要投机、不需要 verifier

注意这里 21.3 是**编译后**的每步耗时；上表 dGPU 那一列是 eager（60.8 ms），
而 dGPU 的 eager 步几乎全是 host issue（30.64 中 30.33），编译正是攻这一块的。

**这条要落地必须先补一个检查**：上表所有精度都是"对满 36 层 teacher 的距离"，
在投机里这是对的（verifier 就是满层模型）。但如果把 24 层当**产品**直接出货，
就没有 verifier 了，判据得换成对**真值**的开环误差，也就是 `open_loop_steps_sweep.py`
那张表的做法。这个检查还没做。

---

## 5c. 减层 + int4 之后，iGPU 跑完整流程到底多久

前面每一段都单独测过了，这里把它们拼成一个控制 tick。**除 verify 外每一项都是实测**；
verify 用 §12.8 的 22.8 ms（K=1、批量、编译，与草稿深度无关——验证器是 dGPU 上的满 36 层模型）。

### 1. iGPU 侧每步耗时（int4-all，eager，实测）

| L | 1 步 | 其中 host issue | 10 步 | 4 步 | 2 步 |
|---:|---:|---:|---:|---:|---:|
| 36 | 118.43 | 41.08 | 1186.4 | — | 238.5 |
| 24 | 79.30 | 28.23 | 787.7 | 314.6 | 157.0 |
| 18 | 62.81 | 20.83 | 620.8 | 248.1 | 124.3 |
| 12 | 40.85 | 14.19 | 398.9 | 160.2 | 79.4 |
| 8 | 27.55 | 9.26 | 280.4 | 106.8 | 53.4 |

### 2. 传输按层数线性，三点实测

| 载荷 | 往返 | ms/MB |
|---|---:|---:|
| `reground_prefix_kv` 42.2 MB（L=36） | 32.12 | 0.762 |
| `reground_prefix_kv_l24` 28.1 MB | 21.66 | 0.771 |
| `reground_prefix_kv_l12` 14.1 MB | 10.82 | 0.769 |

单向 ≈ **0.385 ms/MB**，即 prefix KV 下行 **0.45 · L ms**。另加 `embs` 下行 0.55 ms、
`x0` 上行 0.07 ms。

### 3. iGPU 这条道的总时长 = embs + KV + denoise + x0

| L | 10 步 | 4 步 | 2 步 | vs 58.1 ms 预算（2 步） |
|---:|---:|---:|---:|---:|
| 36 | 1203.1 | — | 255.2 | 4.4× |
| 24 | **799.2** | 326.1 | **168.5** | 2.9× |
| 18 | 629.6 | 257.0 | 133.1 | 2.3× |
| 12 | 404.9 | 166.2 | 85.4 | 1.5× |
| 8 | 284.6 | 111.0 | **57.6** | **进得去** |

### 4. 整个 tick

按 §Context 那条重排（草稿在 `embed_prefix` 之后 26.5 ms 发车，与 `prefix_forward` 并行）：

    tick = max(_ground 80.8, 26.5 + iGPU 这条道) + verify 22.8
           + (1 − 接受率) × 213    ← 拒绝就在同 tick 跑满 10 步 Euler

接受率取 §5b 的 greedy 实测值：

| L | 10 步 tick | 2 步 tick | 2 步接受率 | **2 步期望 tick** |
|---:|---:|---:|---:|---:|
| 36 | 1252.4 | 304.5 | 100% | 304.5 |
| **24** | **848.5** | **217.8** | **100%** | **217.8** |
| 18 | 678.9 | 182.4 | 75% | 235.7 |
| 12 | 454.2 | 134.7 | 50% | 241.2 |
| 8 | 333.9 | 106.9 | **0%** | 319.9 |

**直接回答：减到 24 层 + int4，iGPU 跑完整流程，10 步是 ~848 ms、2 步是 ~218 ms。**
最优格是 L=24 / 2 步的 **217.8 ms**。

### 5. 但这个最优格输给不用 iGPU 的做法

| 方案 | 每 tick | 来源 |
|---|---:|---|
| 不投机 baseline | 292.6 | §12.9 实测 |
| **iGPU 最优（L=24，2 步，int4）** | **217.8** | 本节，1.34× |
| `cached` K=1 @ 今天真实接受率 0 | 164.7 | §12.9 实测 |
| 只降步：dGPU，`num_steps=2` | 123.4 | 80.8 + 2×21.3 |
| **降步 + 减层：dGPU，2 步 + 24 层** | **109.2** | 80.8 + 2×21.3×24/36 |

**iGPU 那条最优路径比同样的压缩留在 dGPU 上慢 2.0×**（217.8 对 109.2），
而后者还不需要第二个进程、不需要 oneCCL、不需要给 worker 留一个核（§11.4 的 215 ms 陷阱）、
也不需要 verifier。原因在拼装里看得很清楚：iGPU 这条道 168.5 ms 里，
**157.0 ms 是 denoise 本身**——它比 dGPU 上同样 24 层 2 步的编译耗时（28.4 ms）慢 5.5×，
而能藏起来的窗口只有 54–58 ms。

即使给 iGPU 最乐观的假设——编译/图捕获把 host issue 全部消掉
（L=24 时 28.23 的一半以上），每步降到 51.1 ms、2 步 102.1 ms、整道 113.6 ms、
tick **162.9 ms**——仍然输给 dGPU 的 109.2 ms。**iGPU 侧没有能翻盘的配置。**

---

## 6. 方法论：两个会让结论反号的计时坑

* **短 kernel 必须背靠背计时，而且 batch 要按时间定而不是定成常数。**
  每次调用都 sync 会给 iGPU 的 int4 层加 0.37 ms（提交/排空往返），那是 int4 层的 30%、
  fp16 层的 0%，于是**谁慢就偏袒谁**。本探针第一版就这么测的，int4 读到 1.238 ms；
  改成背靠背后是 0.911 ms。但固定 batch=40 又反向踩坑：`aten` 的 int8 一次 469 ms，
  40 次就是 19 秒一个计时区间，整个 arm 跑不完。`auto_batch()` 先测一次再把区间凑到
  ~120 ms，上限 40。
* **输出缓冲复用会造成别名。** ESIMD 内核写调用方给的 buffer，缓存它是对的
  （形状固定），但一个无 bias、输入已是 fp16 的 Linear 会把**原始 buffer**
  直接交给下游——§F3b 的 XPU-graph 探针正是这样产出了看起来 bit-exact 的别名引用。
  `Q4Linear` 的无 bias 分支显式 `clone()`。

---

## 7. 没测的，和不能拿这次结果说的话

* **闭环成功率完全没有验证**，本机没有仿真环境。Gate 1 的 100% 接受率是
  "verifier 不反对"，不是 "机器人能完成任务"。
* **Gate 1 只有 6 个样本、一个任务、一个 checkpoint。** 余量是 20–50× 所以结论稳，
  但 `num_steps` 那张表的样本量同样是 6，而它的结论差距小得多，**必须扩样**。
* **`gripper_prev=None`**，所以夹爪守卫在本次全程没有触发过。生产里它跨 tick 携带。
  余量大到不太可能翻盘，但这一格确实比生产宽松。
* **§5b 的 greedy 顺序是在同一批 4 个 grounding 上选出来又在上面评的**，
  这是在评测集上做选择，L=24 那个上限偏乐观。bundle 只有 6 个样本，做不了
  像样的留出集；扩样时顺序要在留出集上重定。
* **§5b 只扫了单层敏感度，没做迭代式重排序。** 每丢一层后重算全部剩余层的敏感度
  （逐层贪心而不是一次排序）通常还能再多丢几层，代价是 O(L²) 次 denoise，
  按本次每格几秒算约 20 分钟，没跑。
* **Gate 2 全部是 eager。** 编译的 iGPU 数字没测；地板里的 40 ms host issue
  正是编译该攻的那部分，但 §F3b 已记录图捕获在此栈不安全，而 Inductor 在
  `torch.ops.custom_esimd_kernels_vllm` 上会不会 graph-break 也没试。
* **int4 的量化误差只在这个 checkpoint 上量过。** Q4_0 是对称、group=128、无校准集的
  最朴素 PTQ；换任务或换 checkpoint 要重跑 Gate 1，那是一条命令。
* **没有碰 `vllm_omni/`。** 本阶段不落生产。

---

## 8. 复现

```bash
D=test-image_zy_scaler0260b2_lingbot_omni
M=/tmp/lingbot-open-loop
DS=/llm/zhuyong/lingbovla/datasets/open_loop/adjust_bottle_3ep_2chunks.npz
X="docker exec -w /llm/zhuyong/lingbovla/my/vllm-omni -e PYTHONPATH=."

# Gate 0 — 内核选型，两张卡，~1 min，不需要 checkpoint
$X -e ZE_AFFINITY_MASK=1 $D python spikes/lingbot_vla_v2/phase13_ptq_draft_probe.py \
    --arm kernel --include-stock
$X -e ZE_AFFINITY_MASK=0 $D python spikes/lingbot_vla_v2/phase13_ptq_draft_probe.py \
    --arm kernel --include-stock

# Gate 1 — 接受率，dGPU，含折叠验收
$X -e ZE_AFFINITY_MASK=0 $D python spikes/lingbot_vla_v2/phase13_ptq_draft_probe.py \
    --arm accept --model $M --dataset $DS --check-folding \
    --quant fp16,int4-moe,int4-all --num-steps 10,4,2 --observations 6 \
    --json-out /tmp/phase13_accept.json

# Gate 2 — 速度曲线 + 零字节地板。两张卡分别跑，不要并发（会互相污染）
for card in 1 0; do
  $X -e ZE_AFFINITY_MASK=$card $D python spikes/lingbot_vla_v2/phase13_ptq_draft_probe.py \
      --arm step --model $M --quant fp16,int4-moe,int4-all --num-steps 10,2 \
      --launch-floor --iters 6 --warmup 2 --json-out /tmp/phase13_step_$card.json
done

# Gate 3 — 42 MB prefix KV
$X $D python spikes/lingbot_vla_v2/phase10_ipc_probe.py \
    --payloads reground_prefix_kv reground_embs --transport shm --iters 50 --warmup 10
I=/llm/zhuyong/libraries.performance.communication.oneccl-v2/build/_install
$X -e LD_LIBRARY_PATH=$I/lib:$I/opt/mpi/lib -e CCL_PLUGIN=ONECCL_IGPU $D \
    python spikes/lingbot_vla_v2/phase10_ipc_probe.py \
    --payloads reground_prefix_kv reground_embs --transport oneccl --iters 50 --warmup 10

# §5b — 减层：单层敏感度扫描 + 四种保留模式 × 六个深度
$X -e ZE_AFFINITY_MASK=0 $D python spikes/lingbot_vla_v2/phase13_ptq_draft_probe.py \
    --arm depth --model $M --dataset $DS --quant fp16 --num-steps 2 --observations 4 \
    --depths 30,24,18,12,8,4 --patterns stride,head,tail --json-out /tmp/phase13_depth.json

# §5c.1 — 减层后 iGPU 每步耗时（也验证对层数线性，别只信外推）
for L in 24 18 12 8; do
  $X -e ZE_AFFINITY_MASK=1 $D python spikes/lingbot_vla_v2/phase13_ptq_draft_probe.py \
      --arm step --model $M --quant int4-all --num-steps 10,4,2 --keep-layers $L --iters 5 --warmup 2
done

# §5c.2 — 传输随层数缩放（三点，确认线性）
$X $D python spikes/lingbot_vla_v2/phase10_ipc_probe.py --transport shm --iters 50 --warmup 10 \
    --payloads reground_prefix_kv reground_prefix_kv_l24 reground_prefix_kv_l12

# 副产品 — num_steps 对真值
$X -e ZE_AFFINITY_MASK=0 $D python spikes/lingbot_vla_v2/open_loop_steps_sweep.py \
    --model $M --dataset $DS --dtype float16 --steps 1 2 4 5 10 20
```

`--model` 必须指向 RoboTwin 微调 checkpoint；`/tmp/lingbot-vla-v2-perf` 是基础模型
（§11.7 记过这个坑）。跑之前确认 `load < 2.0` 且没有别的容器占着 GPU：
§L §6 实测 iGPU 吞吐随 host CPU 活动摆动 2.4×。
