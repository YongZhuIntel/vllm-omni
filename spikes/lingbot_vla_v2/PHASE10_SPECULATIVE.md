# Phase 10 — 用 FLASH 投机推理改造 LingBot-VLA-2.0-6B 的 denoise（iGPU 草稿 / dGPU 真实模型）

## Context

现状（`PHASE9_FLASHRT.md` 执行日志，P1+P4 已落地）：单请求 model path **301.2 ms**，
其中 `embed_prefix` 23.7 ms、`prefix_fill` 57.1 ms、**denoise loop 214.7 ms（10 步 × 21.3 ms）**。
Phase 9 剩下的单设备算子级优化（P2/P3/P5/P6/P7）加起来最多把 model path 推到 ~220 ms —— 仍然是
**10 次完整的 36 层 MoE 前向**。算子优化已经接近天花板。

`~/zhuyong/realtime-vla-flash`（Dexmal RealtimeVLA-FLASH）攻击的是另一个维度：
**不优化每一步，而是把 10 步变成 1 步**。它是第一个面向 diffusion-VLA 的投机推理框架，
机制与我们的 Phase 9 完全正交，因此是唯一还能带来数量级收益的方向。

用户要求：草稿模型跑在 iGPU、真实模型跑在 dGPU，**iGPU 作为硬性要求**来设计。

> **另一条路，已测并关闭**：ParaDiGMS（Picard 并行采样）同样消 10 步顺序依赖，
> 但它验的是**路径**而不是**终点**——把 10 个中间点一起并行求值再迭代收敛。
> `PHASE8_LATENCY_PARITY.md` §M 实测：一轮 sweep 97.5 ms，盈亏平衡 2.19 轮，
> 实际需要 **9 轮**收敛（877 ms，**0.24x**），且没有任何轮数能赢。
> 双卡分摊也关闭：iGPU 跑**一个**点 233 ms，已超过 dGPU 跑**全部十个**的 153 ms。
> 本节的投机轮在 K=2 下是 30.6 ms，比 ParaDiGMS 最便宜的一轮还便宜 3.2x ——
> 因为在 flow matching 里中间态 `x_t` 是脚手架而非输出，验它是在为不执行的东西付钱。

---

## 1. FLASH 的机制（已读代码，不是读论文）

`src/openpi/models_pytorch/spec_pi0_pytorch.py:799` `_sample_actions_impl` 每个控制 tick：

| 阶段 | 做什么 | 代码 |
|---|---|---|
| encoder | ViT + 语言 embedding → `prefix_embs` | `_encoder_stage_impl:610` |
| VLM prefill | **投机轮跳过**，复用上一次 full round 的 KV cache | `_get_cached_past_key_values:651` |
| draft | 1 层 Gemma over `prefix_embs + state token + M 个 action query` → **直接回归出整条 chunk** `x0_draft`，没有去噪循环 | `draft.py:104` |
| verify | 取 K 个近终点时刻 `t_list=(0.10, 0.05)`，构造 `x_t = t·noise + (1−t)·x0_draft`，**跑一次** `denoise_step`（batch B·K）得 `v_t`，还原 `x0_hat = x_t − t·v_t` | `_action_stage_impl:662` |
| accept | 逐步比较 `‖x0_hat[k] − x0_draft‖₂/√d ≤ τ`，取所有 K 上的**最长公共前缀**；尾部用 `mean_k(x0_hat)` 缝合 | `_compute_radius_prefix_acceptance:126`、`_stitch_radius_prefix_output:160` |
| fallback | `accepted_prefix_len == 0` 或 gripper 翻转 → 下一轮强制 full round | `_should_schedule_full_fallback:181` |

**关键：flow-matching 约定完全一致。** LingBot `sample_actions:1589` 的 docstring 写明
`t=1 是噪声、t=0 是动作`，`dt = −1/num_steps`；FLASH 的 `x_t = t·noise+(1−t)·x0` 与
`x0_hat = x_t − t·v_t` 逐字成立。**verify 的数学可以直接移植，不需要重新推导。**

同样重要：**投机轮里 `prefix_embs` 只喂给 draft**，verify 用的是缓存的 KV。
这决定了下面的设备切分。

## 2. 本机已测定的硬约束（`PHASE8_LATENCY_PARITY.md` §K，不要重新论证）

| 事实 | 数字 | 后果 |
|---|---|---|
| `torch.xpu.device_count()` | **1**（两张卡是不同 L0 platform，20.1.0 vs 30.0.4） | 没有 in-process `.to("xpu:1")`，**必须双进程**，每进程一个 `ONEAPI_DEVICE_SELECTOR` |
| iGPU 相对 dGPU 的 `k` | MoE GEMM 12.9×、读带宽 15.4×（29 vs 449 GB/s） | 任何"宽"的 draft 在 iGPU 上都不可承受 |
| iGPU 持续占满 EU 时 dGPU 请求的代价 | **1.74–1.76×** | 不能让 draft 与 verify 并发 |
| iGPU 短核 / 低占空比 | 128² 核 +2.5 ms；10 ms/320 ms（3% duty）**+0.2 ms** | **有预算**，draft 必须落在这个预算里 |

§K 给出的规则原文：*"iGPU work is free when it does not keep the EU array busy …
Schedule iGPU work into the dGPU's idle time; never run it concurrently with the request."*

**投机推理天然满足这条规则**：`encoder → draft → verify` 是串行的，draft 执行时 dGPU 无事可做，
不存在并发。iGPU 的问题不是干扰，而是**它自己的延迟**。所以 draft 的设计目标是
"dGPU 上 ≤ 0.2 ms 量级的工作量"，而不是照搬 FLASH 的全宽 1 层 Gemma
（在 hidden=2560、337 token 上约 1.59 ms dGPU → iGPU ~20 ms，会吃掉全部收益）。

## 3. 目标架构

```
┌── dGPU 进程 (ONEAPI_DEVICE_SELECTOR=level_zero:0) ───────────────┐
│  full round  : embed_prefix → prefix_fill → 10 步 denoise        │  ~301 ms
│                └─ 顺带算 draft prefix 投影 [286,2560]→[286,512]  │
│                   把 293 KB 送给 iGPU                            │
│  spec round  : verify — K 次 predict_velocity（复用缓存 KV）      │  21.3 ms × K
└──────────────────────────────────────────────────────────────────┘
            ▲ x0_draft [1,50,55] fp32 = 11 KB   │ state [1,55] = 220 B
┌── iGPU 进程 (ONEAPI_DEVICE_SELECTOR=level_zero:1) ───────────────┐
│  常驻 draft prefix K/V（286×512），full round 时刷新              │
│  每 tick : 51 个 query × 1 个窄 decoder layer → x0_draft          │  目标 ≤ 5 ms
└──────────────────────────────────────────────────────────────────┘
```

**Draft 架构（推荐，不是照抄 FLASH）**——`LingbotDraftHead`，约 4 M 参数：

- `prefix_proj`: `Linear(2560, 512)`，**在 dGPU 的 full round 里算**，只传 286×512 fp16 = 293 KB
  （而不是 1.4 MiB 的 `prefix_embs`）；iGPU 侧缓存其 K/V。
- `state_proj`: `Linear(55, 512)`；`action_queries`: `Embedding(50, 512)`。
- 一层 decoder：51 个 query 自注意力 + 对 286 个缓存 prefix slot 的交叉注意力，`d=512`,
  `intermediate=1024`, GQA。
- `action_out`: `Linear(512, 55)`。
- 可选 `history_proj`：最近 6 步已执行动作（FLASH 的 draft 丢弃了 `last_actions`，
  但它的 prefix 是每 tick 新鲜的；我们的 prefix 是陈旧的，历史信息值得加回来，用 flag 控制）。

每 tick FLOPs ≈ 0.3 GFLOP、权重 8 MB —— 在 iGPU 29 GB/s 下是**毫秒级的短促 burst**，
落在 §K 的免费区间内。若精度不够，回退方案是 FLASH 的全宽 1 层（dGPU 1.59 ms / iGPU ~20 ms），
成本已知。

**预期收益**（K=2 顺序执行）：spec round model path ≈ 5（draft+传输）+ 42.6（2×verify）≈ **50 ms**；
按 `periodic_full_every_n=4` 摊销，平均 `(301 + 4×50)/5 ≈ 100 ms`，对比今天的 301 ms 是 **~3×**；
若 staleness 探针允许 `n=9`，平均 **~75 ms**，**~4×**。K=1 再减半 verify。

---

## 4. 工作计划

### Phase 10.0 — 四个闸门探针（**先做，不写任何模型代码**）

这四个探针决定后面所有工作是否值得做，都只用现有 checkpoint 和现有 bundle。
沿用 `spikes/lingbot_vla_v2/` 的 phaseN 命名与 "先写步骤、后写结果" 的执行日志惯例。

1. **`phase10_steps_sweep_finetune.py`** — 在 **RoboTwin 微调 checkpoint** 上扫 `num_steps ∈ {1,2,3,4,6,10}`，
   对 `adjust_bottle_3ep_2chunks.npz` 记 MAE / jerk。
   `PHASE5_PERF.md:748` 的 "5→100 步 MAE 不变" 是在**基础模型**上测的，而基础模型的速度场本身是错的
   （mae 0.615），那个结论对微调模型无效。
   **退出条件：如果 num_steps=2 就达到 10 步的精度，直接改 `num_steps` 即可，本计划的收益大部分蒸发，
   需重新定范围。**（`open_loop_steps_sweep.py` 已有骨架可复用。）
2. **`phase10_kv_staleness_probe.py`** — 用第 t 帧的 KV cache + 第 t+n 帧的 state 跑完整 10 步，
   对比第 t+n 帧自己的 KV，扫 `n ∈ {1,2,4,8,16}`（数据集 15 fps）。
   **这是整个方案最大的精度风险**：投机轮全部建立在"视觉上下文可以陈旧"之上。
   输出直接决定 `periodic_full_every_n_draft_rounds` 的可行取值。
3. **`phase10_verify_oracle_probe.py`** — 把 teacher 自己 10 步的输出当作"完美 draft"喂进 FLASH 的
   verify 数学，扫 `τ ∈ [0.05, 0.5]`、`K ∈ {1,2,3}`、`t_list`，画 accepted_prefix_len 分布。
   这给出**接受率的上界**：完美草稿都接受不了的 τ/K 组合，真草稿更不可能。
   同时确认 `x0_hat = x_t − t·v_t` 在我们的 fp16 路径上数值稳定。
4. **`phase10_igpu_draft_cost_probe.py`** — 在 `ONEAPI_DEVICE_SELECTOR=level_zero:1` 下，用随机权重
   计时候选 draft 架构（窄版 512 / FLASH 全宽 2560 两个规格），同时在 dGPU 侧按真实占空比
   跑 `run_openvino_comparison.sh --no-prepare` 量 §K 的干扰税。
   **退出条件：窄版 draft 在 iGPU 上 > 15 ms，或 dGPU 请求被拖慢 > 5%。**

### Phase 10.1 — Draft 数据与训练

数据只有 `datasets/lerobot/adjust_bottle_demo_clean`（50 episode / 7153 帧 / 15 fps / state 14 维），
**单任务**。按 FLASH 的两段式做，训练成本极低：

5. **`phase10_draft_cache.py`**（对标 `realtime-vla-flash/scripts/spec/enc_cache.py`）——
   遍历 lerobot 数据集，对每帧跑一次 teacher，落盘 safetensors shard：
   `prefix_proj` 输入所需的 `prefix_embs`、`prefix_pad_masks`、`state`、
   以及 target。**target 用 `teacher_zero_noise`**（teacher 自己 10 步的 x0），
   不是数据集 GT —— 草稿要模仿的是被验证的那个模型，不是真值。
   7153 帧 × ~0.3 s ≈ 35 分钟一次，缓存 `prefix_embs` fp16 约 10 GB。
6. **`LingbotDraftHead`** + **`phase10_draft_train.py`**（对标 `spec_draft_train.py`）——
   在缓存上训窄 head。沿用 FLASH 的 step-weighted Huber + `sampled_prefix` 加权
   （`_loss_step_weights:155`，前 `max_exec_steps` 步权重高、尾部 0.1），
   按 episode 切 train/val。只训 4 M 参数，dGPU 上分钟级。
   **`out_dim` 用 55（padded），loss 只在 RobotSpec 的真实 14 维上算**；
   gripper 维从 `RobotSpec` 推导（左 6 / 右 13），**不要照抄 FLASH 硬编码的 index 6**。

### Phase 10.2 — 单设备投机运行时（正确性基线 / A-B 对照臂）

7. **`phase10_spec_runtime.py`** — 在 spike 里先把整条投机路径跑通，draft 与 verify 都在 dGPU。
   复用 `transformer.embed_prefix` / `prefix_forward` / `predict_velocity`，新增：
   - `SpecSession`：跨 tick 状态（`past_key_values`、`prefix_pad_masks`、`prefix_position_ids`、
     draft prefix K/V、`action_chunk_cache` + ptr、`last_actions`、`pending_full_fallback`、
     `draft_rounds_since_full`）。
   - `verify_step()`：移植 `_compute_radius_prefix_acceptance` / `_stitch_radius_prefix_output` /
     `_truncate_accepted_prefix_on_gripper_switch`，**改成按 RobotSpec 取 gripper 维**。
   - K 的执行方式：**默认顺序跑 K 次 `predict_velocity`**。FLASH 的 `expand_past_key_values`
     会把 42.2 MB 的 prefix KV 物化 K 份；K=2 时顺序 2×21.3 ms 与 batch 版差距有限，
     batch 版留作后续优化并单独计时。
   - 这一步就能出**端到端延迟数字**和**开环精度**，且不依赖双进程机制。

### Phase 10.3 — 双进程：draft 上 iGPU（用户的硬性要求）

8. **`phase10_draft_worker.py`** — iGPU 侧常驻进程，`ONEAPI_DEVICE_SELECTOR=level_zero:1`
   （§K 已验证此时 iGPU 以 `xpu:0` 出现，80 EU / 56.40 GiB 共享内存）。协议走
   **Unix domain socket + host 端 numpy**，不碰 `oneccl_igpu_communicator`
   —— §G 记录了那个插件的窄限制（仅 XPU tensor、单 dtype 打包、多次顺序传输会 hang），
   而我们的载荷只有 11 KB / 293 KB，host 拷贝完全够用。
   两条消息：`REFRESH(prefix_kv[286,512] fp16)` 和 `DRAFT(state, last_actions) → x0_draft`。
9. **`phase10_two_process_latency.py`** — 三臂对比：draft-on-dGPU / draft-on-iGPU / 无投机基线。
   必须同时报告**每 tick 延迟**与 **staleness**（观测年龄），因为 §G4 的教训是
   "period 变好而 staleness 变坏"是机器人场景里的假胜利。
10. **iGPU 独有的收益，单独测**：full round 刷新（~301 ms）期间，让 iGPU 继续用**陈旧 prefix**
    出草稿来维持控制 tick，而不是让控制环硬停 301 ms。这是单设备方案**做不到**的事，
    也是 iGPU 在这个架构里真正的价值主张。注意此时 draft 与 dGPU 请求并发，
    要按 §K 的 1.75× 规则如实计量（301 ms 可能变 ~520 ms，但 tick 不中断）。

### Phase 10.4 — 上行到 `vllm_omni/`（闸门全过之后）

11. `config.py`：`spec_decode: bool`、`spec_tau_radius`、`spec_t_list`、`spec_k`、
    `spec_max_exec_steps`、`spec_periodic_full_every_n`、`spec_draft_device: {"local","igpu"}`、
    `spec_draft_path`，沿用 P1/P4 建立的 flag 惯例（默认值旁边写上测量数字）。
12. `pipeline_lingbot_vla_v2.py`：目前 `forward()` 是**完全无状态**的（每个请求走完整
    `sample_actions`）。`extra_args` 里 **`session_id` 与 `reset` 已经由
    `entrypoints/openpi/serving.py:153` 铺好**，直接用它们索引 `SpecSession`；
    `reset=True` 清会话。这是现成的接缝，不需要新 hook。
13. `modeling_lingbot_vla_v2.py`：`LingbotDraftHead` 类 + `load_weights` 里可选加载 draft 权重
    （draft 是独立 checkpoint，不进主 checkpoint 的 strict 检查）。
14. 测试进 `tests/diffusion/models/lingbot_vla_v2/`：tiny-config 上的 verify 数学单测
    （接受/拒绝/缝合/gripper 截断）、session 生命周期、`spec_decode=False` 时**逐位等同**今天的路径。

---

## 5. 关键文件

**读（参考实现）**：`~/zhuyong/realtime-vla-flash/src/openpi/models_pytorch/spec_pi0_pytorch.py`（投机主循环）、
`.../draft.py`（草稿头形状）、`scripts/spec/enc_cache.py`（缓存构建）、
`scripts/spec/spec_draft_train.py`（训练与 loss 加权）。

**改**：`vllm_omni/diffusion/models/lingbot_vla_v2/{config.py,modeling_lingbot_vla_v2.py,pipeline_lingbot_vla_v2.py}`。

**新增**：`spikes/lingbot_vla_v2/phase10_*.py` 与 `PHASE10_SPECULATIVE.md`（执行日志，
沿用 Phase 9 的"每项先写步骤、落地后写回实测值"格式）。

**复用**：`examples/offline_inference/lingbot_vla_v2/open_loop_eval.py`（开环打分）、
`spikes/lingbot_vla_v2/phase7_numeric_parity.py`（五种子数值闸门）、
`spikes/lingbot_vla_v2/phase5_latency.py`（分阶段计时）。

## 6. 验证

按 Phase 8 Rule 3，但**必须如实声明其局限**：

- **数值闸门**：`phase7_numeric_parity.py` ≥5 种子。`spec_decode=False` 必须与今天**逐位一致**。
  投机路径本身不做 bit-exact 要求（它按设计就是近似）。
- **开环精度**：`run_open_loop_eval.sh` 在 6-chunk bundle 上，外加**在 50 个 episode 上新增一个
  replay 谐波**（推进 chunk 指针、每 `max_exec_steps` 步重规划），报告 MAE、jerk、
  accepted_prefix_len 分布、full-fallback 触发率。
- **延迟**：`phase5_latency.py` 风格的交替臂 × 3 次重复（P1/P4 用的方法，防止漂移伪装成效果），
  同时报 per-tick 延迟与 staleness。
- **诚实声明**：`~/zhuyong/RoboTwin` 是空的，**本机没有闭环仿真**。FLASH 的半径接受是行为启发式，
  开环 MAE 无法验证它。所有结论都要写明"闭环成功率未经验证"。
  训练数据只有 `adjust_bottle` 单任务 50 个 episode，草稿头会过拟合到这一个任务 ——
  这是机制演示，不是通用策略。

## 7. 风险与退出条件（按先后顺序）

| # | 风险 | 何时知道 | 退出动作 |
|---|---|---|---|
| 1 | 微调模型本来就只需 2 步 | 10.0 探针 1 | 直接降 `num_steps`，重新定范围 |
| 2 | KV cache 不能陈旧 | 10.0 探针 2 | 投机轮必须重跑 `prefix_fill`（+57 ms），收益从 ~6× 降到 ~2.5×，仍值得做 |
| 3 | 完美草稿也接受不了 | 10.0 探针 3 | 方案作废，止损在探针阶段 |
| 4 | 窄 draft 在 iGPU 上太慢/干扰太大 | 10.0 探针 4 | 先落 10.2 单设备版；iGPU 版按 10.3-10 的"刷新期不停机"价值重新论证 |
| 5 | 单任务数据训不出可用草稿 | 10.1 | 机制仍可用 `teacher_zero_noise` 演示，但明确标注为 demo |

---

## 8. 执行日志

沿用 Phase 9 §7 的格式：**每项先写下步骤，落地后把实测值写回来**。
未测量的数字一律标注为估算，不得当作结论引用。

环境：容器 `test-image_zy_scaler0260b2_lingbot_omni`（`intel/llm-scaler-vllm:0.26.0-b2`），
`torch 2.12.0+xpu`、`vllm 0.26.1.dev0`。宿主 `/home/user/zhuyong/` 挂载为 `/llm/zhuyong`，
所以本目录所有文档里的 `/llm/zhuyong/...` 都是容器内路径。
`torch.xpu.device_count()` 在此容器中同样是 **1** —— §K 的双进程结论在这套 runtime 上依然成立。

### 前置改动 — `export_open_loop_bundle.py --stride` — **已落地**

探针 2（KV staleness）和 10.1 的训练缓存都需要**连续帧**，而导出脚本的采样写死为
`range(start, end, args.horizon)`，步长等于 horizon=50 —— 已提交的
`adjust_bottle_3ep_2chunks.npz` 每个 episode 只有 2 个样本、间隔 50 帧，
正好是这两项工作唯一不能用的形式。

改动：新增 `--stride`，默认 `None → args.horizon`，**默认行为与今天逐位相同**；
`stride` 写进 manifest 使 bundle 自描述。导出稠密 bundle（需要上游 lingbot-vla-v2 源码 + LeRobot）：

```bash
python examples/offline_inference/lingbot_vla_v2/export_open_loop_bundle.py \
    --lingbot-root /llm/zhuyong/lingbovla/frameworks.robotics.embodied-intelligence.lingbot-vla-v2/lingbot-vla-v2 \
    --data-path /llm/zhuyong/lingbovla/datasets/lerobot/adjust_bottle_demo_clean \
    --output /llm/zhuyong/lingbovla/datasets/open_loop/adjust_bottle_dense.npz \
    --episodes 0 1 2 --stride 1 --max-chunks-per-episode 200
```

### 共享件 — `phase10_spec_common.py` — **已落地并验证**

FLASH verify 代数的移植，`phase10_port_exactness.py` 是它的验收探针。
移植有意保留两处**不同**，其余要求逐位一致。

**验收结果，`PYTHONPATH=. python spikes/lingbot_vla_v2/phase10_port_exactness.py`：**

| 检查 | 结果 |
|---|---|
| 200 组随机数据 × 3 个函数 vs FLASH 原实现（配置回 LIBERO 的 7-DoF / pose 0..5 / gripper 6） | **ALL EQUAL**（`torch.equal`，不是容差） |
| `x0_hat = x_t − t·v_t` 恒等式，fp32 | `4.768e-07` |
| 同上，**fp16（实际部署 dtype）** | **`3.906e-03`** |
| 多 gripper 泛化：任一夹爪翻转都要截断 | OK |

**新发现，值得记下来：fp16 下 verify 的重构噪声是 `3.9e-3`，比 fp32 差 4 个数量级，
而且它与 `tau` 同量纲。** 原因是 `x_t − t·v_t` 中 `x_t` 是 O(1) 而 `t·v` 在 t=0.05 时是
O(0.05)，减法直接继承 `x_t` 的 fp16 量化误差（相对 ~1e-3）。
**后果：`tau` 低于 ~0.01 在 fp16 下没有意义**，它量的是舍入噪声而不是草稿误差。
这条在跑 tau 扫描之前就该知道，否则探针 3 的表格底部几行会被误读。

**gripper 维的实测，这是不照抄 FLASH 的理由：**

```
[ 0:14) arm.position       max_dim=14  action=yes
[14:28) end.position       max_dim=14  action=no
[28:30) effector.position  max_dim=2   action=yes
30 packed slots, then zero-padded to max_action_dim (processor.py:591)

pose_dims    (12): [0..11]
gripper_dims  (2): [28, 29]
```

**FLASH 硬编码的 action index 6 在这里是左臂的一个关节，不是夹爪。** 照抄会让
gripper 保护安静地监视错误的维度 —— 不报错、不崩，只是永远不触发（或者乱触发）。
真实夹爪在模型空间的 **28 和 29**，既不是 6 也不是 13（`arm.position` 真实宽 12 但槽位宽 14，
所以 12、13 是 padding）。`gripper_dims()` 从 `RobotSpec` 推导，这个数字由它算出来并已核对。

### 探针 1 — 微调模型的 `num_steps` 扫描 — **无需新代码，未运行**

`open_loop_steps_sweep.py` 已完全覆盖这项（MAE + jerk + hold-state 基线），只需换 checkpoint 和 dtype：

```bash
PYTHONPATH=. python spikes/lingbot_vla_v2/open_loop_steps_sweep.py \
    --model /tmp/lingbot-open-loop --dtype float16 \
    --dataset /llm/zhuyong/lingbovla/datasets/open_loop/adjust_bottle_3ep_2chunks.npz \
    --steps 1 2 3 4 6 10
```

必须用 **RoboTwin 微调** checkpoint，不是 6B 基础模型：
`PHASE5_PERF.md:748` 那条 "5→100 步 MAE 不变" 是在基础模型上测的，而基础模型的速度场本身就是错的
（mae 0.615），**那个结论对微调模型无效**，这正是必须重测的原因。

顺带踩到 Phase 9 §7 记的那个坑：**`/tmp/lingbot-vla-v2-perf` 指向的是基础模型**
（`readlink -f` 确认指到 `models/lingbot-vla-v2-6b/`），用它跑精度会得到 mae 0.6 的假结论。
本次重新 prepare 了 `/tmp/lingbot-open-loop` → `lingbot-vla-v2-6b-robotwin/checkpoints/global_step_50000/hf_ckpt`，
并 `readlink -f` 核对过。**引用任何绝对精度数字前先核对链接目标。**

#### 结果 — **闸门触发：`num_steps ≥ 2` 与 10 步无法区分**

三个 seed（1234 / 7 / 99），每个 6 个 chunk，fp16 eager。
noise 由 `make_noise(seed, sample_index)` 生成、**不依赖 num_steps**，所以这是配对比较。

`MAE(all)`，对数据集真值：

| steps | seed 1234 | seed 7 | seed 99 | 均值 |
|---|---:|---:|---:|---:|
| 1 | 0.0090 | 0.0092 | 0.0091 | **0.0091** |
| 2 | 0.0071 | 0.0072 | 0.0060 | **0.0068** |
| 3 | 0.0071 | 0.0069 | 0.0055 | **0.0065** |
| 4 | 0.0076 | 0.0081 | 0.0054 | 0.0070 |
| 6 | 0.0081 | 0.0083 | 0.0058 | 0.0074 |
| 10 | 0.0078 | 0.0083 | 0.0050 | **0.0070** |
| 20 | 0.0098 | — | — | — |

`jerk(0-6)`（平滑度，真值 0.0023）：1 步 **0.0059**；2 步起一律 **0.0032–0.0038**，与 10 步持平。
`hold-state` 基线 MAE 0.4199 —— 0.007 是真在工作，不是退化解。

**读法，以及它的边界：**

* **1 步确实不够**，三个 seed 一致（0.0091 vs ≥2 步的 0.0068），jerk 也高出 60%。
  所以这个积分循环不是摆设。
* **2 步和 3 步已经到顶**。`num_steps` 在 ≥2 之后的差异（0.0065–0.0074）**小于 seed 之间的差异**
  （10 步一列：0.0050–0.0083）。诚实的表述是"≥2 步之后 `num_steps` 的影响淹没在噪声里"，
  **不是** "3 步比 10 步好" —— 后者这个表里读不出来。
* 20 步反而更差，与 `PHASE5_PERF.md:780` 记的 bf16/fp16 timestep 累积漂移方向一致。
* 3 步比 4 步略好，而 `-1/num_steps` 只在 `num_steps ∈ {1,2,4,…}` 时可精确表示 ——
  所以驱动这个结果的**不是**可表示性。

**局限，必须一起引用：** 6 个 chunk、单任务、开环、对数据集真值。
这足以触发闸门（"降 `num_steps` 更划算"），**不足以直接改默认值**——
那需要更宽的 bundle，并且闭环行为在本机无法验证。

#### 后果：Phase 10 的收益需要重算

若 `num_steps=3` 成立，denoise 从 10×21.3 = 214.7 ms 降到 **3×21.3 = 64 ms**，
model path **301.2 → ~150 ms**，**一个配置项换 2×**，不需要草稿头、不需要训练、不需要 iGPU、
不需要任何接受率启发式。

以 ~150 ms 为新基线重算投机推理：

| | 10 步基线（原估算） | **3 步基线（新）** |
|---|---:|---:|
| full round | 301 ms | ~150 ms |
| spec round（draft 5 + K=2 verify 42.6） | ~50 ms | ~50 ms |
| 摊销 @ `n=4` | 100 ms（**3.0×**） | 70 ms（**2.1×**） |
| 摊销 @ `n=4`, K=1 | 76 ms（4.0×） | 51 ms（2.9×） |

更关键的是**构成变了**：3 步之后 denoise 只剩 64 ms，而 `embed_prefix + prefix_fill` 是 80.8 ms ——
**请求里最大的一块不再是去噪循环，而是前缀**。而投机推理攻击前缀靠的是
**KV cache 跨 tick 复用**，也就是探针 2，**它不需要草稿头、不需要训练、不需要 iGPU**。

草稿头 + verify 真正买到的，只是 "3 步（64 ms）" 与 "1 次 verify（21 ms）" 之间的 ~43 ms，
代价是：训一个草稿、一套无法闭环验证的接受启发式、一套双进程 iGPU 运行时。
在 10 步基线下这笔账是划算的；在 3 步基线下不是。

**建议的重新排序**（风险 #1 的退出动作，按计划执行）：

1. 先在更宽的 bundle 上确认 `num_steps=2..3`，然后改默认值。**2× 免费。**
2. 再跑探针 2（KV staleness）。它是现在最大的单项杠杆（80.8 ms），且**不依赖草稿头**。
   如果 KV 可以陈旧 n 帧，光靠"缓存前缀 + 3 步去噪"就能到 ~70 ms，**总计 4×，仍然不需要草稿头**。
3. 草稿头 + iGPU（10.1/10.3）降级为第 3 优先级，在 1 和 2 的实测结果之上重新定值。

### 探针 2 — `phase10_kv_staleness_probe.py` — **已落地，未运行**

同一个 anchor 帧填一次 prefix KV，对 `n ∈ {1,2,4,8,16,32}` 用第 `t+n` 帧的 state 跑完整 10 步，
与第 `t+n` 帧自己的 prefix 对照。两臂共用同一份 noise（按目标帧索引），
所以唯一的差别就是 KV 来自哪一帧。

脚本会**拒绝**步长不为 1 的 bundle 并打印重新导出的命令 —— 用 strided bundle 跑出来的
"staleness" 单位会是 50 帧而不是 1 帧，是那种能安静地得出错误结论的失败。

**状态：探针已落地，本轮不跑**（2026-09-14 决定：本次任务目标是 iGPU 投机推理，
KV staleness 归档待跑）。它仍然是最大的单项杠杆（`embed_prefix + prefix_fill` = 80.7 ms 实测），
而且**不依赖草稿头** —— 谁先捡起这条线，先跑它。
前置条件：需要 `--stride 1` 的稠密 bundle，导出脚本要上游 lingbot-vla-v2 + LeRobot 环境，
不在 `test-image_zy_scaler0260b2_lingbot_omni` 容器里。

结果：_未跑_

### 探针 3 — `phase10_verify_oracle_probe.py` — **已落地，未运行**

把 teacher 自己 10 步的输出当作**完美草稿**喂进 verify，隔离出
"单步近终点估计 vs 十步答案" 这一项误差。这是接受率的**上界**。

然后扫合成草稿误差 `x0_draft = x0_teacher + eps·N(0,1)`（只加在真实动作维上）。
因为接受距离就是按维 RMS，`eps` 与 `tau` 同量纲，输出表可直接读作
**"按维 RMS 误差 eps 的草稿，在阈值 tau 下能被接受 n 步"** —— 这就是 10.1 训练要达到的精度目标，
而且是在训练**之前**就能知道。注意上面那条 fp16 噪声底：`tau ≲ 0.01` 的行不要当真。

verify 最初采用**顺序执行 K 次** `predict_velocity`，不是 FLASH 的 `B*K` batch，理由是
`expand_past_key_values` 会把 42.2 MB 的 prefix KV 物化 K 份，而 F2 已证明这个循环本来就受权重带宽约束。
**这个理由是错的，已在 §12.8 实测推翻**：受权重带宽约束恰恰是批量能摊销的条件，而 42.2 MB×K 的拷贝
在 449 GB/s 上只有 ~1 ms。批量版已落地（`spec_verify_batched`，默认开）。

#### 结果 — **通过**。6 个 chunk，fp16，`t_list=(0.10, 0.05)`，`eval_h=12`

```
[spec] accept radius over 12 pose dims [0..11];  gripper dims [28, 29]
[oracle] 完美草稿下，单步近终点估计与 10 步答案的距离：mean 0.0028  p90 0.0051
```

**这是整个方案最关键的一个数字。** verify 机制本身只引入 **0.0028 的按维 RMS 误差** ——
换句话说，在近终点（t=0.05/0.10）做一次 `predict_velocity` 就能把 10 步的答案还原到千分之三。
上界不是限制因素。

接受步数（满分 12）：

| eps＼tau | 0.05 | 0.1 | 0.15 | 0.2 | 0.3 | 0.5 |
|---|---:|---:|---:|---:|---:|---:|
| 0.00 | 12.0 | 12.0 | 12.0 | 12.0 | 12.0 | 12.0 |
| 0.02 | 12.0 | 12.0 | 12.0 | 12.0 | 12.0 | 12.0 |
| 0.05 | 1.8 | 12.0 | 12.0 | 12.0 | 12.0 | 12.0 |
| 0.10 | 0.0 | 1.2 | 11.8 | 12.0 | 12.0 | 12.0 |
| 0.20 | 0.0 | 0.0 | 0.0 | 0.8 | 12.0 | 12.0 |

执行段（12 步）相对 teacher 的 MAE，模型空间：

| eps＼tau | 0.05 | 0.1 | 0.15 | 0.2 | 0.3 | 0.5 |
|---|---:|---:|---:|---:|---:|---:|
| 0.00 | 0 | 0 | 0 | 0 | 0 | 0 |
| 0.02 | 0.0161 | 0.0161 | 0.0161 | 0.0161 | 0.0161 | 0.0161 |
| 0.05 | **0.0076** | 0.0391 | 0.0391 | 0.0391 | 0.0391 | 0.0391 |
| 0.10 | **0.0037** | 0.0101 | 0.0800 | 0.0813 | 0.0813 | 0.0813 |
| 0.20 | **0.0209** | 0.0209 | 0.0209 | 0.0306 | 0.1586 | 0.1586 |

三条可直接用于设计的读数：

1. **`tau ≈ 1.5 × 草稿的按维 RMS 误差`**。表格对角线就是这个关系。
2. **草稿的精度目标：模型空间按维 RMS ≤ 0.02。** 到这个精度，任何 `tau ≥ 0.05` 都全接受，
   执行段相对 teacher 的 MAE 是 0.016。这是 10.1 训练的验收指标，**在训练之前就拿到了**。
3. **拒绝是优雅的，而且是一个 ~10× 的误差收缩器。** eps=0.20 且全部拒绝（tau=0.05）时，
   输出走 `x0_tail = mean_k(x0_hat)`，MAE 0.0209 —— 草稿错 0.20，输出只错 0.021。
   所以即使草稿很差，付 K 次 verify 也能拿到一个远好于草稿的结果。

**反直觉但重要：接受一个"略微不准"的草稿，可能比拒绝它更糟。**
对比 `eps=0.02, tau=0.05`（全接受，MAE 0.0161）与 `eps=0.05, tau=0.05`（几乎全拒，MAE 0.0076）——
后者草稿更差，输出反而更准。`tau` 不是"越大越好"的加速旋钮，它是真实的质量/速度权衡点。

**局限：** 6 个 chunk、单任务、开环。且 `phase10_port_exactness.py` 测到 fp16 下
`x0_hat = x_t − t·v_t` 的重构噪声 max 为 `3.9e-3`，与 oracle 的 mean `2.8e-3` 同量级 ——
**oracle 那一行里有多少是算法、多少是 fp16 舍入，本探针分不开。**
要分开需要一个 fp32 臂，但这不影响结论方向（fp32 只会让上界更好）。

### 探针 4 — `phase10_igpu_draft_cost_probe.py` — **已落地并测完，通过**

两个候选草稿形状，随机权重、fp16、batch=1，两个 `ONEAPI_DEVICE_SELECTOR` 各一个进程。
测的是**形状**不是模型，所以可以在训练之前跑。

* **`wide`** — FLASH 的真实架构（`draft.py:49`）：一整层全宽 VLM decoder，
  over `prefix + state + 50 queries`，**每 tick 重新编码前缀**。
  按 LingBot 的宽度（hidden 2560 / intermediate 9728 / 32 头 / 8 kv 头）是 101 M 参数、193 MiB fp16。
* **`narrow`** — 本计划提的形状：前缀在 **full round 时于 dGPU 上**投影到 512 并缓存，
  每 tick iGPU 只跑 51 个 token 过一层 512 宽的 decoder（attend 到缓存前缀）。2.31 M 参数、4.4 MiB。
  之所以可行，是因为投机轮本来就复用缓存前缀 —— 用的是 verify 已经在用的那个陈旧性假设，不是新增假设。

#### 结果 1：narrow 在 iGPU 上 0.72 ms，闸门大幅通过

| 形状 | dGPU | iGPU | **k** | 参数 | 权重 | vs 一个去噪步(21.3 ms) |
|---|---:|---:|---:|---:|---:|---:|
| **narrow** | 0.32 ms | **0.72 ms** | **2.25×** | 2.31 M | 4.4 MiB | 0.03× |
| `wide` (FLASH) | 1.22 ms | **15.97 ms** | **13.1×** | 101.3 M | 193.3 MiB | 0.75× |
| narrow 的 full-round 刷新 | 0.07 ms | 0.24 ms | 3.4× | — | — | — |

**`k` 不是常数，是形状相关的 —— §K 借来的那个 12.9× 不能套用到草稿上。**
§K 的 12.9×/15.4× 测的是带宽受限的大 MoE GEMM 和读带宽；narrow 只有 4.4 MiB，
iGPU 只落后 **2.25×**。这正是 §K "短核 / 突发是免费的" 那一档，只是 §K 没把这一档量化成 `k`。

**照抄 FLASH 的全宽架构会得到 15.97 ms**，吃掉 75% 的去噪步预算 ——
计划里"不照抄 FLASH 的草稿架构"这个决定，现在有数字了。

#### 结果 2：干扰税 **+2.0 ms（+0.7%）**，即使 iGPU 100% 占空比

§K 的规则是"持续占满 EU 要付 1.75×，绝不要与请求并发"。对这个形状**不成立**。

交替臂 × 3 次重复（P1/P4 的方法，防止漂移伪装成效果），
victim 是 `phase5_latency.py --compile-denoise-step --iters 10`，
load 是 `--role busy --shape narrow --duty 1.0`（实测有效占空比 99.9%，~67k 次调用/50 s）：

| rep | idle total | busy total | Δ | idle denoise | busy denoise | Δ |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 299.6 | 301.6 | +2.0 | 213.9 | 215.2 | +1.3 |
| 2 | 299.4 | 302.1 | +2.7 | 214.1 | 216.0 | +1.9 |
| 3 | 299.6 | 301.2 | +1.6 | 213.8 | 215.2 | +1.4 |
| **中位数** | **299.6** | **301.6** | **+2.0 (+0.7%)** | **213.9** | **215.2** | **+1.3** |

busy 三次全部慢于 idle，所以效应是真的；但 idle 的跑间离散只有 0.2 ms，
而 §K 对 512²/2048² matmul 循环测到的是 **+220 ms（1.74–1.76×）**。
narrow 草稿的行为像 §K 的 128² 那一行（+2.5 ms），不像 512² 那一行。

**§K 规则的修正：代价跟踪的不是*占空比*，而是核大不大到能填满 80 个 EU。**
一个 512 宽、51 个 token 的草稿填不满，连着跑也填不满。

**后果：iGPU 草稿不需要被调度进 dGPU 的空闲窗口。** §K 那句
"never run it concurrently with the request" 是双进程设计上最主要的工程约束，
对这个形状被解除了 —— 计划里 10.3-10 那条（dGPU 做 full round 刷新时 iGPU 继续出草稿维持控制 tick）
的代价从"可能 +75%"降到"+0.7%"。

#### 一次自我更正

本探针第一次跑出的是 "+0.8%"，但那次的 load 进程日志是**空的** ——
Python 输出缓冲，进程被 kill 时没 flush，所以**无法区分"负载在跑但免费"和"负载根本没起来"**，
而这两者在 victim 侧的读数完全一样。该结论已作废并重测：
`--role busy` 现在每 5 秒打印心跳（调用数 / 有效占空比 / ms per call）并 `flush=True`，
`xpu-smi` 独立确认 iGPU 在 2400 MHz。上表是重测后的数字。

#### 基线复现

顺带确认本次环境与 Phase 9 记录一致（不同容器、不同 checkpoint）：

| | PHASE9 记录 | 本次实测 |
|---|---:|---:|
| denoise | 214.7 | 213.9 |
| prefix_fill | 57.1 | 58.1 |
| embed_prefix | 23.7 | 22.6 |
| total (synced) | 301.2 | 299.6 |
| per-step | 21.3 | 21.2 |

#### 用实测值重算投机轮

| | 实测 |
|---|---:|
| iGPU narrow 草稿 | 0.72 ms |
| 跨进程传输（state 220 B 下行 / `x0_draft` 11 KB 上行） | **未测** |
| verify，K 次 `predict_velocity` | K × 21.3 ms |
| **投机轮，K=1** | **~22 ms + 传输** |
| **投机轮，K=2** | **~43 ms + 传输** |
| full round（保持 10 步） | 299.6 ms |
| 摊销 @ `n=4`，K=2 | **94 ms（3.2×）** |
| 摊销 @ `n=4`，K=1 | **78 ms（3.9×）** |

仍未知、且决定这张表成不成立的：**探针 3（verify 到底接受不接受）** 和草稿本身的精度。
传输成本也还没测。

---

## 9. Phase 10.3 — 跨设备传输，`phase10_ipc_probe.py` — **已测完，不是约束**

投机轮的最后一个未测量的量。载荷是**小且固定尺寸**的，这一点不寻常，而且决定了结论：

| 消息 | 形状 | 大小 | 方向 |
|---|---|---:|---|
| `DRAFT` 请求 | `state [1,55]` fp32 | 220 B | dGPU → iGPU |
| `DRAFT` 回复 | `x0_draft [1,50,55]` fp32 | 11.0 KiB | iGPU → dGPU |
| `REFRESH` | `prefix_kv [286,512]` fp16 | 293 KiB | dGPU → iGPU（仅 full round） |

两种传输都测，因为**简单方案没被排除之前，复杂方案不成立**：

| 载荷 | `shm`（POSIX 共享内存 + 自旋标志） | `oneccl`（v2 C API + `libccl_igpu.so`） | vs 21.3 ms |
|---|---:|---:|---:|
| 220 B | 0.150 ms | **0.135 ms** | 0.6% |
| 11 KiB（每 tick 都付） | 0.151 ms | **0.129 ms** | **0.6%** |
| 293 KiB（刷新） | 0.293 ms | **0.267 ms** | 1.3% |

**结论：传输不是设计约束。** oneCCL 快约 15%，但两者都在 verify 步的 1% 以内。
一个投机 tick 付一次往返 ≈ **0.13 ms**。

#### §G 对这条传输的定价需要修正

§G 写的是 "720 collectives at even an optimistic 50 µs is 36 ms"，
并据此把 iGPU 排除。那个估算漏了两件事，读 `oneccl_igpu_communicator.py` 才看得到：

1. **只有 iGPU 那一侧**需要 plugin 管理的 USM host 中转（`_prepare_send_tensor:399`）；
   dGPU 侧直接从 `tensor.data_ptr()` 发。§G 的 "every hop does a USM host round-trip" 只对一半。
2. **`onecclCommRegister` 让 plugin 的 pt2pt fd 握手只做一次、之后跳过**
   （`:104-113`、`_send_packed:429`）。§G/§K 的 50 µs 估算是**未注册路径**的。
   我们的缓冲区固定尺寸、永久复用，正是注册存在的理由。

更重要的是，§G 算的是**每层 720 次**集合通信（TP/PP 把模型切开）。
投机推理的拓扑完全不同：**每个 tick 一次往返**，因为切开的是"草稿/验证"这个职责，不是模型。
720 × vs 1 × —— 这是同一个传输被否决和被接受的全部差别。

> **这条边界后来被从另一侧量过一次。** 专家分片（两卡各读自己那份权重，买聚合带宽）
> 每个 layer-step 付一次往返，36 层 × 10 步 = **360 次**，正好落回被否决的那一侧。
> 实测载荷 `[51,768]` fp16 = 76.5 KiB，往返 **0.155 ms**（oneCCL）/ 0.166 ms（shm），
> 合计 55.8 ms，而聚合带宽**最多**省 12.9 ms —— 传输至少是收益的 **4.3x**。
> 盈亏平衡要求往返 ≤ 0.036 ms。
> iGPU 的 16 MiB cache 救不了：一次往返 0.155 ms，而 dGPU 跑**整个** 32 专家块才
> 0.251 ms，亏在线上不在卡上——把 iGPU 那一半算成免费，layer-step 仍是 0.390 ms
> 对 0.251 ms。见 `PHASE8_LATENCY_PARITY.md` §M §6。
> 该载荷已加入 `phase10_ipc_probe.py`（`--payloads moe_expert_shard_exchange`，
> 默认不跑：oneCCL 路径在同一进程内第 4 个载荷上会挂）。

#### 验证：插件确实加载了

差分测试，因为"插件静默回退到默认传输"和"插件在工作"在读数上无法区分：

```
CCL_PLUGIN=ONECCL_IGPU  -> |INFO| Using CCL_PLUGIN override (ONECCL_IGPU)   两个 rank 都有，0.141 ms
未设 CCL_PLUGIN          -> |INFO| Proceeding without CCL_PLUGIN override.
                            |INFO| Failed to load plugin type: ONECCL_LEGACY_CPU ...   跑不出结果
```

所以上表的 oneCCL 列确实是 iGPU 插件路径。用法：加载 dispatcher `libccl.so`
（**不是**直接加载 `libccl_igpu.so`）并设 `CCL_PLUGIN=ONECCL_IGPU`，
`LD_LIBRARY_PATH` 指向 `…/_install/lib` 和 `…/_install/opt/mpi/lib`。

进程隔离用 **`ZE_AFFINITY_MASK`**（vLLM iGPU 路径的做法），不是 §K 用的
`ONEAPI_DEVICE_SELECTOR`：本机两者等价（0 = B60 dGPU，1 = iGPU），
但 oneCCL 的 `onecclSetDevice()` 走自己的枚举，`ccl_device = rank` 与 `ZE_AFFINITY_MASK = rank` 对齐时可用。

## 10. 实测汇总 — 投机轮的完整成本

除草稿精度外，每一项都已实测：

| 项 | 实测 | 来源 |
|---|---:|---|
| iGPU narrow 草稿 | 0.72 ms | 探针 4 |
| 跨进程往返（oneCCL） | 0.13 ms | 10.3 |
| verify，每次 `predict_velocity` | 21.3 ms | 探针 4 / PHASE9 P4 |
| iGPU 草稿对 dGPU 的干扰（100% 占空比） | +2.0 ms | 探针 4 |
| **投机轮，K=1** | **~22.2 ms** | |
| **投机轮，K=2** | **~43.5 ms** | |
| full round（10 步，硬约束） | 299.6 ms | 探针 4 baseline |

摊销（`n` = 每次 full round 之间的投机轮数）：

| | K=1 | K=2 |
|---|---:|---:|
| `n=4` | 77.7 ms（**3.9×**） | 94.7 ms（**3.2×**） |
| `n=9` | 49.9 ms（**6.0×**） | 71.2 ms（4.2×） |

**`n` 的上限由 KV 陈旧性决定，也就是归档掉的探针 2 —— 所以上表的 `n` 是假设，不是实测。**
这是目前最大的未量化项，超过草稿精度本身。

**上表是各部件单独计时后的拼装estimate，§11 把它换成了整条 tick 的实测：
K=2 投机轮 43.9 ms（预估 43.5）、K=1 投机轮 23.4 ms（预估 22.2）、full round 294.3 ms（预估 299.6）。
拼装误差在 1 ms 量级 —— 但 §11 同时找到一项拼装完全看不见的成本（见 11.4，+215 ms）。**

---

## 11. Phase 10.2 + 10.3 — 双进程运行时，端到端实测

两个新文件，计划里的第 7、8 项；第 9 项（`phase10_two_process_latency.py`）**没有单独成文**，
因为三臂对比只是同一个 tick 循环的一个 flag，独立文件就得复制模型构建与计时逻辑：

* **`phase10_draft_worker.py`** —— 协议 + iGPU 常驻进程 + 两个后端（`LocalDraft` / `RemoteDraft`）。
  `NarrowDraftHead` 从探针 4 import，传输层（`OneCCL` / `Endpoint` / `ShmChannel`）从 §9 的 `phase10_ipc_probe` import，
  所以跑的就是被定过价的那个形状和那条线，不是它们的复制品。
* **`phase10_spec_runtime.py`** —— tick 循环，三臂：`baseline`（今天的路径）/ `local`（草稿在 dGPU，10.2）/
  `igpu`（草稿在 iGPU 的第二个进程，10.3）。verify 直接 import 探针 3 的 `verify_once`。

草稿是**随机初始化**的。这是刻意的：per-tick 延迟不依赖权重（形状固定、verify 是 K 次前向、线上跑的字节数一样），
所以管路可以在训练之前就打通并定价；训练之后变的只有接受率。

### 11.1 协议，以及一次设计上的自我更正

固定尺寸、注册一次的三个槽位，一次只有一条消息在飞：

| 槽位 | dtype | numel | 字节 | 方向 |
|---|---|---|---:|---|
| `req` | fp32 | 2 + 55 | 228 | dGPU → iGPU（`[opcode, seq, state…]`） |
| `prefix` | fp16 | 286 × 512 | 292864 | dGPU → iGPU（只在 full round） |
| `reply` | fp32 | 50 × 55 | 11000 | iGPU → dGPU（`x0_draft`） |

`prefix_proj`（2560→512）留在 **dGPU** 的 full round 里算，只把它的输出发出去 ——
这就是载荷是 293 KiB 而不是 `prefix_embs` 的 1.4 MiB 的原因。iGPU 侧把它变成 k/v 并常驻。

**第一版把 `REFRESH` 设计成 fire-and-forget**，理由是让 iGPU 在 dGPU 自己那 208 ms 去噪循环里重建缓存，
把 0.3 ms 藏掉。oneCCL 下它工作正常（pt2pt send 会排队）；**shm 下两个进程一起挂死** ——
shm 通道是**单槽邮箱**，紧随其后的 `DRAFT` 把还没被读走的 `REFRESH` 覆盖掉了，
worker 永远在等 seq 1，client 永远在等回复。

改法不是去改传输，而是改协议：**每条请求都有回复**，`REFRESH` 的回复就是它携带的那个 state 的草稿。
代价是 0.3 ms 的重建不再被隐藏（在 294 ms 的 full round 上）；换到的是
(a) "worker 拿到新前缀了" 这个真实信号，(b) **每个 full round 免费得到一对 (draft, teacher)** ——
因为 full round 本来就在算同一帧的 teacher 答案，而这正是训练之后预测接受率的那个量（探针 3 的 `eps`）。

**教训值得单独记：两个传输的语义不一样，宽容的那个（oneCCL）会把协议 bug 藏起来。**
简单传输不是只用来当性能对照的，它是用来暴露这类假设的。

### 11.2 `--role selftest` —— 不需要 6B 模型的差分验收

两个后端持有**同一份权重**（都在 CPU 上按 `--draft-seed` 构造后再搬到各自设备），
所以把它们的 `x0_draft` 相减就能验线：槽位错、缓冲区陈旧、注册后被静默截断，都会让结果发散。
纯计时测不出这些。

| 传输 | DRAFT 往返（client） | worker 服务时间 | REFRESH 往返 | max\|iGPU − dGPU\| |
|---|---:|---:|---:|---:|
| oneccl | **0.705 ms** | 0.580 ms | 1.022 ms | 9.77e-04 |
| shm | **0.630 ms** | 0.486 ms | 0.805 ms | 9.77e-04 |

差分是 `9.77e-04`，值域到 2.145，相对 4.6e-4 —— 正好是 fp16 的 `2^-11`。**线是对的。**

两件与 §9 不一样、需要说清楚的事：

1. **顺序反了。** §9 的紧循环里 oneCCL 比 shm 快 15%（11 KiB：0.129 vs 0.151 ms）；
   在真实的请求/回复模式下 shm 反而快 0.075 ms。两者都是 verify 步的 3%，
   所以**传输选择依然不是设计约束** —— 但 §9 那个 15% 不要拿去做别的推论。
2. **shm 的第一版慢了 6 ms**，原因是 `send` 里写了 `value.cpu()`：每跳一次分配 + 一次多余的主机拷贝，
   在 293 KiB 上就是 6 ms。改成拷进预分配的 staging 后是 0.805 ms。§9 的探针本来就是预分配的，
   这个坑是重新实现时踩的。

### 11.3 实测 —— K=2，三臂 × 3 次交替重复，每臂 20 tick

`--compile-denoise-step`、fp16、eager/fp16 attention、`--full-every 4`、`tau 0.15`、
`t_list=(0.10, 0.05)`、`--worker-cpu 11`（见 11.4）。

```
      arm  rep   full ms   spec ms  spec p90  per tick
 baseline    0     294.1       nan       nan     294.2
 baseline    1     294.1       nan       nan     294.3
 baseline    2     294.1       nan       nan     294.4
    local    0     294.4      43.9      44.7      94.1
    local    1     294.5      43.9      44.7      94.2
    local    2     294.5      43.8      44.9      94.6
     igpu    0     296.9      45.6      46.8      96.0
     igpu    1     296.4      45.6      46.7      95.9
     igpu    2     296.1      45.7      46.3      95.8
```

三次重复之间的离散 < 0.5 ms。分阶段中位数：

| | preprocess | embed_prefix | prefix_fill | draft_refresh | denoise | 合计 |
|---|---:|---:|---:|---:|---:|---:|
| full / baseline | 4.0 | 23.3 | 58.4 | — | 208.5 | 294.3 |
| full / local | 3.8 | 23.2 | 58.4 | 0.7 | 208.3 | 294.4 |
| full / igpu | 3.7 | 23.2 | 58.3 | **3.5** | 208.0 | 296.7 |

| | state | draft | verify | accept | 合计 |
|---|---:|---:|---:|---:|---:|
| spec / local | 0.2 | **0.6** | 42.0 | 1.0 | 43.9 |
| spec / igpu | 0.2 | **2.2** | 42.1 | 1.1 | 45.6 |

**基线复现**：full round 294.3 ms vs 探针 4 的 299.6、PHASE9 的 301.2；
denoise 208.5 vs 213.9/214.7；prefix_fill 58.4 vs 58.1/57.1；embed_prefix 23.3 vs 22.6/23.7。
按测得的中位数摊销（`n` = 两次 full round 之间的投机轮数）：

| | n=1 | n=2 | **n=4** | n=9 |
|---|---:|---:|---:|---:|
| local，K=2 | 169.1（1.7×） | 127.4（2.3×） | **94.0（3.1×）** | 68.9（4.3×） |
| igpu，K=2 | 171.0（1.7×） | 129.2（2.3×） | **95.8（3.1×）** | 70.7（4.2×） |
| local，K=1 | 158.8（1.9×） | 113.7（2.6×） | **77.5（3.8×）** | 50.4（5.8×） |
| igpu，K=1 | 160.8（1.8×） | 115.5（2.5×） | **79.2（3.7×）** | 52.0（5.7×） |

K=1 的投机轮实测 **23.4 ms（local）/ 24.8 ms（igpu）**，verify 21.7 ms。
`n=4` 那一列是唯一真正跑满 20 tick 的调度，其余列由中位数推出 —— **推出来的，不是跑出来的**。

§10 的估算表对得相当准（K=1 `n=4` 预估 77.7 ms，实测 79.2 ms）。

### 11.4 本节最重要的发现：**常驻 worker 空转一个核，会让 dGPU 进程的 host 端慢 55×**

第一次跑三臂时得到的是 **3.74×**，比 3.1× 好得多，而且 `baseline` 臂也变慢了 ——
`preprocess` 从 3.9 ms 变成 **218.6 ms**，full round 从 294 ms 变成 515 ms。
`baseline` 臂根本不碰 worker，所以这不是投机路径的问题，而是 **worker 进程存在** 的问题。

单独量了 worker 空闲时的 CPU：

| 传输 | worker 空闲占用 |
|---|---:|
| oneccl | **1.00 核**（`onecclRecv` 硬自旋；`CCL_YIELD=sleep` 无效） |
| shm | 0.05 核（自旋 50 ms 后转 sleep 轮询） |

本机是 12 核的 **Intel Core Ultra 5 338H 混合核**（CPU 0-3 是 4.7 GHz P-core，4-7 是 3.6 GHz E-core，
8-11 是 3.3 GHz LP-E core）。一个自旋核撞上 verifier 进程的 OpenMP 线程池（默认 active-wait 屏障）：
池里 12 个线程要等被抢占的那一个，小的并行 CPU 算子就从 4 ms 变成 218 ms。
**注意这是个 numpy 矩阵乘测不出来的效应**（同样的自旋下 900² matmul 完全不变），
只有真正跑 `processor.preprocess` 才会看到。

三种缓解，都实测过（`baseline` + `igpu` 两臂，20 tick）：

| 配置 | full/preprocess | spec 轮 | per tick | vs baseline |
|---|---:|---:|---:|---:|
| oneccl，什么都不做 | **218.6** | 45.1 | 141.3 | ~~3.74×~~ 假的 |
| oneccl + `--worker-cpu 11` | 3.5 | 45.5 | 95.9 | **3.11×** |
| oneccl + `OMP_WAIT_POLICY=PASSIVE` | 4.6 | 44.6 | 94.9 | **3.10×** |
| shm，什么都不做 | 57.3 | 44.7 | 111.7 | 2.65× |
| shm + `--worker-cpu 11` | 3.5 | 45.0 | 95.4 | **3.08×** |

**读法：**

1. **这个 bug 朝着讨好你的方向失败。** 它同时拖慢 baseline，所以投机的加速比*变大*（3.74× vs 3.11×）。
   如果只看加速比，永远发现不了。发现它靠的是分阶段计时里 `preprocess` 那一格。
   `connect_worker` 现在在没有保留 CPU 且不是 PASSIVE 时会打印警告并明说这一点。
2. **shm 的 sleep 轮询只是减轻，没有解决**：它在 sleep 之前自旋 50 ms，而 full round 有 294 ms，
   于是每个 full round 仍然被抢占一次（57.3 ms）。"空闲时不烧 CPU" 要按 *worst case* 设计，不是平均。
3. **保留一个核是与传输无关的修法**，两个传输都回到 3.1×。留 LP-E core（11）的代价是草稿本身慢 0.3 ms
   （selftest 往返 0.705 → 1.019 ms；给两个核 10-11 没有改善，说明是核的频率而不是核数）——
   0.3 ms 换 215 ms，显然值得。要草稿更快就得给它一个 P-core，那是从 verifier 嘴里抢。

### 11.5 iGPU 到底值不值：目前**不值**，而且原因清楚

| | local（草稿在 dGPU） | igpu（草稿在 iGPU） | 差 |
|---|---:|---:|---:|
| 投机轮 K=2 | 43.9 ms | 45.6 ms | **+1.7** |
| 投机轮 K=1 | 23.4 ms | 24.8 ms | **+1.4** |
| full round | 294.4 ms | 296.7 ms | **+2.3** |
| per tick @ n=4, K=2 | 94.0 ms | 95.8 ms | **+1.8（+1.9%）** |

把草稿搬到 iGPU **每 tick 贵 1.8 ms**，省下的 dGPU 时间只有 0.6 ms（草稿）+ 0.7 ms（刷新投影）。
在这个负载下这是一笔亏本买卖，并且亏得很小、很稳定。

原因不神秘：**投机轮里草稿只占 1.4%（0.6 / 43.9），92% 是 verify。**
草稿跑在哪个设备上，几乎不影响 tick。所以 iGPU 的价值主张不能是"卸载草稿"，只能是计划里 10.3-10 那条：
**full round 那 294 ms 里让 iGPU 用陈旧前缀继续出草稿，维持控制 tick 不中断** ——
单设备方案做不到，而且探针 4 已经测出这个形状的干扰税只有 +2.0 ms（GPU 侧）。
**那一项还没实现**（现在的 `REFRESH` 是同步的，dGPU 在 full round 里不接受草稿请求）。
在它落地之前，诚实的结论是：**双进程 iGPU 路径已经跑通、成本已知（+1.9%/tick），但还没有拿到它独有的那份收益。**

注意 11.4 的 host 端 CPU 账要一起算进 iGPU 的成本：它需要独占一个核，或者要求 verifier 进程放弃 active-wait。
单设备臂没有这项要求。

### 11.6 还没测的，和不能拿这次结果说的话

* **接受率 = 0，96 个投机轮全部拒绝**，`accept distance 0.283`（K=2）/ `0.191`（K=1），
  草稿对 teacher 的按维 RMS 是 **0.888**，而探针 3 要求 ≤ 0.02。
  随机权重下这是**设计如此**，它唯一证明的是 accept 规则接上了、没有误判。
  所以本节的调度用 `--fallback off`：FLASH 的 "拒绝就强制 full round" 规则在随机草稿下会让每个 tick 都变成 full round，
  测不出任何东西。**延迟按 *意图中的* 调度报告，接受率单独报告** —— 训练要修的是后者，前者不会变。
  实测过 `--fallback on` 以确认这不是空话：full/spec 交替，per tick **170.2 ms（1.7×）**，
  staleness 降到 1 轮。这就是**接受率 = 0 时 FLASH 调度的真实性能上限**，
  也是训练之前唯一可以诚实声称的双进程收益。
* **帧级 staleness 仍未量化。** 6-chunk bundle 的步长是 50 帧，本节只能按"轮"报陈旧度（最大 4 轮）。
  上面所有 `n` 的可行性都压在探针 2 上，它仍然没跑。**这仍是最大的未量化项。**
* **开环精度没测。** `--score` 会给每个投机 tick 再跑一次 teacher，但随机草稿下它量的是
  `mean_k(x0_hat)` 这条 fallback 路径（探针 3 说这条路径本身把 0.20 的草稿误差收缩到 0.021），不是草稿。
* **闭环成功率依然完全没有验证**，本机没有仿真环境。
* **FLASH 的 `enable_gripper_verify`（any-K 预检停止）没有移植**，
  `phase10_spec_common.py` 里只有 post-verify 截断那一半。夹爪翻转时两者行为不同，
  移植它要连 `phase10_port_exactness.py` 的验收一起加。
* 编译门限从 phase5 的 `1e-3` 放宽到 `2e-3`：fp16 下编译步的 `max_rel` 是 1.34e-3，
  而 `phase10_port_exactness.py` 测到 verify 自己的重构噪声是 **3.9e-3** ——
  对一步要求 1e-3 比它下游的算术还严。这个放宽是有依据的，但它是个放宽。

### 11.7 复现

```bash
I=/llm/zhuyong/libraries.performance.communication.oneccl-v2/build/_install
export LD_LIBRARY_PATH=$I/lib:$I/opt/mpi/lib:$LD_LIBRARY_PATH CCL_PLUGIN=ONECCL_IGPU

# 协议 + 差分验收 + 单次成本（不需要 6B 模型，~40 s）
ZE_AFFINITY_MASK=0 PYTHONPATH=. python spikes/lingbot_vla_v2/phase10_draft_worker.py \
    --role selftest --iters 100 --worker-cpu 11
ZE_AFFINITY_MASK=0 PYTHONPATH=. python spikes/lingbot_vla_v2/phase10_draft_worker.py \
    --role selftest --transport shm            # 不需要 oneCCL 环境

# 三臂 × 3 重复的 per-tick 实测（~4 min，含权重加载与编译）
ZE_AFFINITY_MASK=0 PYTHONPATH=. python spikes/lingbot_vla_v2/phase10_spec_runtime.py \
    --model /tmp/lingbot-open-loop \
    --dataset /llm/zhuyong/lingbovla/datasets/open_loop/adjust_bottle_3ep_2chunks.npz \
    --reps 3 --ticks 20 --compile-denoise-step --worker-cpu 11 \
    --json-out /tmp/phase10_k2.json
# K=1：加 --t-list 0.05
```

`--model` 必须指向 **RoboTwin 微调** checkpoint；`/tmp/lingbot-vla-v2-perf` 指向的是基础模型（探针 1 记过这个坑）。
延迟与 checkpoint 无关，但混用会让任何精度数字失效。

### 11.8 下一步，按杠杆大小排序

1. **探针 2（KV staleness）**。上面每个 `n` 都是假设。需要 `--stride 1` 的稠密 bundle。
   不需要草稿头、不需要 iGPU。**杠杆最大。**
2. **K=1 vs K=2**：实测差 20.5 ms/轮（43.9 → 23.4），`n=4` 下是 3.1× → 3.8×。
   探针 3 的 `t_list=(0.10,0.05)` 是 FLASH 的默认值，K=1 该用哪个 `t` 没有单独扫过。
3. **10.1 训练**，目标是探针 3 给出的按维 RMS ≤ 0.02；`REFRESH` 的回复已经在每个 full round 上
   免费提供 (draft, teacher) 对，可以直接当训练监控。
4. **10.3-10**（full round 期间用 iGPU 维持 tick）。这是 iGPU 唯一还没兑现的价值，
   需要把 `REFRESH` 改成允许 dGPU 忙时继续服务 `DRAFT`。

---

## 12. Phase 10.4 — 上行到 `vllm_omni/`

按用户要求提前做（不等 10.1 训练）：**一个变量开关投机推理，草稿必须在 iGPU**，
并且 `run_openvino_comparison.sh` 要能扫不同接受率下的性能。

### 12.1 落地面

| 文件 | 内容 |
|---|---|
| `vllm_omni/.../config.py` | `spec_decode`（唯一开关）+ 6 个带实测默认值的调参项 |
| `vllm_omni/.../spec_decode.py` | **新增**：verify 代数（从 spike 搬进来）、`SpecSession`、`SpecDecoder` |
| `vllm_omni/.../draft_igpu.py` | **新增**：`LingbotDraftHead`、oneCCL/shm 传输、`IGpuDraftClient`、iGPU 常驻进程 |
| `vllm_omni/.../pipeline_lingbot_vla_v2.py` | 开关打开时建 decoder；`forward` 按 `session_id` 分流 |
| `vllm_omni/.../processor.py` | 新增公开的 `preprocess_state()`（投机轮不跑图像处理器） |
| `prepare_lingbot_vla_v2.py` | `--spec-decode` / `--spec-draft-path` / `--spec-worker-cpu` 写进 `transformer/config.json` |
| `run_openvino_comparison.sh` | `--spec-decode` 开关 + `--accept-rate` 接受率（外加 5 个次要项） |
| `spikes/.../phase10_spec_accept_sweep.py` | **新增**：接受率扫描，跑的是**生产对象** |
| `spikes/.../phase10_spec_common.py` | 改成 re-export 生产实现 |
| `tests/.../test_spec_decode.py` | **新增** 17 个用例；`test_pipeline.py` 加 3 个 |

**`spec_decode` 没有设备选项**，这是用户的硬性要求，也与本机事实一致：
`torch.xpu.device_count()` 每进程都是 1，"草稿在 iGPU" 本身就是一个进程边界，不是一个 `.to()`。
`extra_args` 里的 `session_id` 本来就一路传到 pipeline（`connection.py:227` 缺省为 `"default"`），
所以 `forward` 只多了一个分支，没有新 hook；**`spec_decode=False` 时走的还是今天那条无状态路径**。

与 §4 计划的三处偏离，都有理由：

1. `LingbotDraftHead` 放在 `draft_igpu.py`，不是 `modeling_lingbot_vla_v2.py`。
   草稿只在 iGPU 进程里跑（服务进程只用它的 `prefix_proj`），不属于被服务的模型图，
   而 modeling 的 docstring 写明自己只放"数学内核"。
2. **前缀长度不是常量**（286 是 release 的 align 布局算出来的），worker 必须在第一个请求前知道它才能开缓冲区。
   做法是启动时跑一次 dummy `embed_prefix` 读形状 —— 比重新推导布局便宜，且不可能与它漂移。
   worker 先 spawn、再收 `setup.json`，所以它的解释器/XPU 初始化与 6B 权重加载是重叠的。
3. `spec_draft_device`、`spec_transport`、草稿宽度/种子、gripper 阈值都**没有**进 config：
   要么是用户明确要求收窄，要么能定成常量或环境变量（`VLLM_LINGBOT_SPEC_TRANSPORT` 等）。

### 12.2 `--accept-rate`：为什么必须是**模拟**的

随机初始化的草稿接受率恒为 0，而 FLASH 的规则是"拒绝就强制下一 tick 走 full round"，
于是每个 tick 都变成 full round，**调度根本没被走到**。
`spec_force_accept_rate` 覆盖 accept 判决，但 **K 次 verify 照跑** ——
所以每个接受率下的**延迟是实测的**；**动作不是**（被接受的那段是草稿自己的输出）。
接受率影响性能只有一条路径（是否迫使下一 tick 重新 ground），所以这条曲线连接的是两个已知端点。

### 12.3 实测 — 接受率 → per-tick 延迟

`/tmp/lingbot-vla-v2-perf`、fp16、`--compile-denoise-step`、K=2、`spec_full_every=4`、
`--worker-cpu 11`、每个接受率 60 tick：

| 接受率（实现值） | per tick | vs 无投机 | full 轮 | spec 轮 | spec 轮耗时 |
|---|---:|---:|---:|---:|---:|
| 无投机（基线） | 293.7 ms | 1.00× | 60 | 0 | — |
| 0.00 | 174.7 ms | **1.68×** | 30 | 30 | 48.5 |
| 0.29 | 153.6 ms | 1.91× | 25 | 35 | 48.2 |
| 0.53 | 133.6 ms | 2.20× | 20 | 40 | 48.5 |
| 0.75 | 115.4 ms | 2.55× | 16 | 44 | 47.6 |
| 0.91 | 102.6 ms | 2.86× | 13 | 47 | 47.8 |
| 1.00 | **98.4 ms** | **2.98×** | 12 | 48 | 48.0 |
| `measured`（今天的真实接受率 = 0） | 174.8 ms | 1.68× | 30 | 30 | 48.5 |

`rate=1` 的 98.4 ms 与 §11.3 三臂实测的 95.8 ms 一致（这里是基础 checkpoint + 合成观测）。

**两条要一起读的结论：**

1. **接受率 0 时也有 1.68×。** 因为被拒绝的投机轮不是白跑：它返回 `mean_k(x0_hat)`，
   即"1 次近终点估计"代替"10 步 Euler"，成本 48 ms 而不是 294 ms。机制是优雅降级的。
2. **但今天这个 1.68× 的动作不可用。** 探针 3 量的优雅降级是 `eps ≤ 0.20` 的草稿
   （输出 MAE 0.021）；随机权重的草稿 `eps ≈ 0.89`，完全在那个范围之外。
   **所以 1.68× 是延迟结论，不是可交付的性能。** 能交付的前提仍然是 10.1 训练。

### 12.4 上行过程中发现并修掉的两个问题

**(1) 强制接受率完全不生效。** 第一版把覆盖放在 gripper 保护**之前**：
随机草稿的夹爪维在阈值两侧乱跳，post-verify 截断把接受长度砍回 0，
于是 rate 0 / 0.5 / 1 三条臂读数完全一样（都是 1.66×），曲线是平的。
覆盖必须是**最后一步**；真实 accept 路径仍然全跑完（成本进测量），只是判决被替换。
回归用例 `test_forced_acceptance_survives_the_gripper_guard` 复现的就是这个失败。

**(2) 模拟接受的分布选型，错了两次才对。**

| 做法 | 问题 |
|---|---|
| 定种子 Bernoulli | 31 轮里请求 0.25 实现成 **0.03** —— 可复现，但坐标轴会漂 |
| Bresenham 均匀铺开 | 实现值精确，却**与 `spec_full_every` 相位锁死**：0.75 时每次拒绝正好落在本来就要 full 的 tick 上（免费），0.9 时落在中间（每次都加一轮），于是 **0.9 比 0.75 慢**。加 tick 不会缓解 |
| **分层**（每 20 轮恰好接受 `round(20p)` 个，定种子打乱） | 实现值精确、无相位共振。**最终采用** |

第二种失败特别值得记：它不是噪声，是系统性的，而且看起来完全合理（"均匀=准确"）。
是曲线非单调把它暴露出来的。

### 12.5 测试与守卫关系

`pytest tests/diffusion/models/lingbot_vla_v2/` — **71 passed, 2 skipped**。新增：

* `test_spec_decode.py`（17）：accept 是前缀而非逐步、`tau` 是按维 RMS（DoF 无关）、
  缝合、夹爪截断、dims 来自 RobotSpec（**不是** FLASH 的 index 6）、
  周期调度、拒绝回退、reset、多会话独立、**worker 死掉降级为 full round 而不是报错**、
  强制接受率越过夹爪保护、实现值=请求值、config 校验只在开关打开时生效、
  草稿头参数量 == 闸门 4 定价的那个形状（per-tick 2,311,168）。
* `test_pipeline.py`（3）：开关关着时 `spec is None`；带 `session_id` 的请求走 decoder
  且**不**同时走无状态路径；不带 `session_id` 的请求（warmup）仍走无状态路径。

`phase10_port_exactness.py` 现在守的是**生产代码**（spike 那个文件变成 re-export）：
200 组随机数据 × 3 个函数 vs FLASH 原实现 **ALL EQUAL**（`torch.equal`），重跑通过。

进程生命周期：`atexit` + `PR_SET_PDEATHSIG` —— 孤儿 worker 会永久阻塞在 `recv` 并占住一个核
（oneCCL 的等待是自旋的），所以它不能活过服务进程。实测跑完无残留进程。

### 12.6 复现

```bash
I=/llm/zhuyong/libraries.performance.communication.oneccl-v2/build/_install
export LD_LIBRARY_PATH=$I/lib:$I/opt/mpi/lib:$LD_LIBRARY_PATH CCL_PLUGIN=ONECCL_IGPU

# 两个变量：开关 + 接受率
bash examples/online_serving/lingbot_vla_v2/run_openvino_comparison.sh \
    --no-prepare --model /tmp/lingbot-vla-v2-perf \
    --spec-decode --accept-rate 0,0.25,0.5,0.75,0.9,1,measured --spec-ticks 60

# 服务路径（开关写进 checkpoint 的 config.json）
python examples/offline_inference/lingbot_vla_v2/prepare_lingbot_vla_v2.py \
    --checkpoint <ckpt> --output /tmp/lingbot-spec --spec-decode --spec-worker-cpu 11
```

### 12.7 上行后仍然没有的东西

* **训练好的草稿头**。没有它，接受率是 0，`--accept-rate` 之外的收益都不能交付。
* **帧级 staleness**（探针 2）。`spec_full_every=4` 是占位值，config 注释里写明了这一点。
* **闭环验证**。本机没有仿真环境。
* FLASH 的 `enable_gripper_verify`（any-K 预检停止）仍未移植，`spec_decode.py` 顶部已注明。
* 服务端只跑过 `SpecDecoder` + `IGpuDraftClient`（生产对象）与 pipeline 分流的单测，
  **没有跑通完整的 OpenPI WebSocket 全链路投机**；那需要起 server + client，尚未做。

### 12.8 批量 verify — `spec_verify_batched` — **已落地，已实测**

§10.1 把 `B*K` 批量 verify 推给 §10.2 并给了一个估算。现在测了，
`phase10_verify_batch_sweep.py` / `test_spec_verify_batch.sh`，
dGPU（`ZE_AFFINITY_MASK=0`）、fp16、compiled、3 次交替重复、单步 = 22.6 ms：

| K | M=51K | 顺序 | **批量** | vs 顺序 | **vs K=1** |
|---:|---:|---:|---:|---:|---:|
| 1 | 51 | 22.8 ms | 22.8 ms | 1.00× | 1.00× |
| 2 | 102 | 44.2 ms | **30.6 ms** | 1.44× | **1.34×** |
| 4 | 204 | 86.9 ms | 46.8 ms | 1.85× | 2.05× |
| 8 | 408 | 172.7 ms | 78.6 ms | 2.20× | 3.39× |
| 10 | 510 | 215.8 ms | 90.5 ms | 2.38× | 3.90× |

**结论：值得做，但"K=2 白拿"是错的。** 默认 K=2 上把每个投机轮的 verify
从 44.2 降到 30.6 ms（−13.6 ms），这是实打实的；但它是 K=1 的 **1.34×**，
不是我先前估的 ~1.0×。

**估算错在哪，记下来免得再犯。** 那个估算只用了 F2 的 MoE GEMM 行
（M=51→102 是 0.251→0.298 ms，即线性的 0.19×）就去推整步。但 MoE GEMM 只占
一个 layer-step 的 **44%**（§PHASE9 §1）。剩下 56%——attention（含 prefix KV 的
`cat`）、融合后的 q/k/v、norm/router——**不受权重带宽约束**，实测按线性的 **~0.78×**
缩放。分解 K=2 的 +7.8 ms：MoE 侧按 F2 只该涨 36×0.047 = +1.7 ms，其余 +6.1 ms
全在非 MoE 部分，与 0.2166 ms/layer-step × 0.78 × 36 = +6.1 ms 吻合。
**不要把 F2 的 GEMM 数字当整步成本引用。**

**K 不是可以随便加大的。** `radius_prefix_acceptance` 对 K 取 `min`（合取），
每多一个 verify 时间步就是多一道否决权，接受率单调下降。K=10 即使批量后也要 90.5 ms
（比 K=1 贵 68 ms），同时把接受率往下压——两头付钱。批量的价值是让 K=2 的稳健性
（防某个近终点 t 恰好让错草稿看起来对）只花 1.34× 而不是 2×。

想让接受率随 batch **上升**，方向是对 N 条候选草稿取 argmax（multi-draft），
而不是对 K 取 min；那需要 10.1 训练多输出头，未做。

落地细节：`predict_velocity` 是 `dynamic=False` 编译的，所以 B=1（full round 的
Euler 循环）和 B=K（verify）是两份特化，两次 warmup、两张图；sweep 脚本据此抬高了
dynamo 的 cache 上限。`prefix_position_ids` 是 `[3,B,S]` 而 `prefix_pad_masks` 是
`[B,S]`，批量展开的轴不同，`test_spec_decode.py` 里有一条专门盯这个的对齐测试。

### 12.9 每 tick 重新 grounding — `spec_reground` — **已落地，已实测，不做默认**

§12.8 之前测的只是"批量 verify"这一条。完整的**新方案**是四条：

```
spec = _ground(新观测) + draft(新 embs) + 一次 B=K 的批量 verify
     + (拒绝 → 同 tick 跑 10 步 denoise)
```

现在四条都落地了，藏在 `config.spec_reground` 后面（默认 off），并且
`test_spec_oneccl.sh` 改成了 **scheme × K × 接受率** 的三轴 sweep：
`cached`（今天出货的：复用上一个 full round 的 prefix KV，拒绝则把 full round 推给
下一 tick）对 `reground`（上面四条）。两个方案谁赢完全是接受率的函数，所以是测出来的
不是吵出来的。容器内、dGPU（`ZE_AFFINITY_MASK=0`）、fp16、compiled、
`--worker-cpu 11`、`spec_full_every=4`、**每格 60 ticks**、baseline（完全不投机）
292.6 ms。

**先看每轮中位数，这些才是结构性的数字：**

| 每轮 | K=1 | K=2 | K=4 |
|---|---:|---:|---:|
| full round（两方案共用） | 300.4 | 301.0 | 301.0 |
| `cached` 投机轮 | 27.4 | 35.2 | 50.1 |
| `reground` 接受轮 | 113.3 | 121.1 | 136.0 |
| `reground` 拒绝轮 | 322.1 | 330.2 | 344.0 |

两条差值把成本分解得很干净，也正好互相验证：

* **接受轮 − `cached` 投机轮 = 85.9 ms，在 K=1/2/4 上完全不变。** 这就是
  `_ground`（§11 的 80.8 ms）加上「refresh 比裸 draft 多的 ~5 ms」，与 K 无关是对的。
* **拒绝轮 − full round = +21.7 / +29.2 / +43.0 ms**，分别约等于 §12.8 表里 K=1/2/4
  的一次批量 verify（22.8 / 30.6 / 46.8）。也就是说，**`reground` 的拒绝轮就是一个
  full round 外加一次白扔的 verify 和一次白扔的 draft**。

**再看 per tick：**

`cached`：

| K | 0 | 0.25 | 0.5 | 0.75 | 0.9 | 1 | measured |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 165.3 | 141.1 | 118.0 | 99.5 | 85.9 | 82.8 | 164.7 |
| 2 | 169.0 | 145.8 | 125.1 | 105.5 | 92.1 | 88.5 | 168.3 |
| 4 | 177.7 | 154.8 | 133.8 | 116.9 | 104.1 | 99.8 | 175.8 |

`reground`：

| K | 0 | 0.25 | 0.5 | 0.75 | 0.9 | 1 | measured |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 352.3\* | 270.4 | 222.8 | 173.7 | 134.6 | 114.5 | 323.5 |
| 2 | 330.5 | 279.1 | 229.9 | 181.7 | 142.1 | 123.0 | 330.4 |
| 4 | 344.1 | 291.8 | 260.5\* | 195.6 | 156.7 | 136.9 | 344.0 |

**结论：`reground` 在每一个格子都更慢，1.37×–1.96×**（把下面带 `*` 的两格按重建值
算；照读 mean 的话上限是 2.13×，但那一格是瞬态），包括接受率 = 1 的格子
（1.39×，因为每 tick 的 80.8 ms grounding 再也摊不掉了）。三个更尖锐的读法：

1. **在今天真实的接受率（0，没有训练好的草稿头）下，`reground` 比"完全不投机"还慢**：
   330.5 对 292.6。原因就是上面那条分解——每个 tick 都是一个 full round 外加白扔的
   draft 和 verify。`reground` 要先追上不投机的 baseline，接受率大约得到 **0.2–0.25**
   （K=1 在 0.25 是 270.4 已经赢，K=4 在 0.25 是 291.8 刚好打平）。
2. **但 `cached` 便宜的那一端是拿动作质量买的，不是赢来的。** 它在接受率 0 时的
   169.0 ms 就是 `(301.0 + 35.2) / 2`：一半的 tick 返回 `x0_tail`——一个
   **刚刚被否掉的**草稿上做一步近终点估计，而且否它的 teacher 读的还是上一帧。
   `reground` 两件事都不干。所以这两栏不是"同一个 decode 的两个价格"，
   1.96× 是那两条保证的价钱，值不值只有闭环能回答（探针 2 更便宜，仍未跑）。
3. **K 在 `reground` 里相对更便宜**，因为它是加在更大基数上的固定加项：接受率 = 1 时
   K=1→4 是 113.3→136.0（1.20×），`cached` 同区间是 26.5→49.4（1.86×），
   verify 本身是 2.05×。这不改变 §12.8 的结论——K 取 `min` 是合取，加大 K 会压低
   **真实**接受率，而强制接受率的 sweep 结构上测不出这一点。

**两个必须记下来的测量假象，别在下一轮被它们骗到。**

* **`--ticks 20` 会让强制接受率非单调。** `_force_accept` 以 20 为一块洗牌，
  20 ticks 就只有一块，于是"被强制拒绝的 tick 恰好落在周期性 full round 本来就要跑的
  位置上"纯靠运气。20 ticks 的那一轮里 0.75 全线比 0.5 还快（三个 K 都是），
  换成 60 ticks 后每一行都恢复单调。这跟 `_force_accept` 自己 docstring 里记的
  Bresenham 变体的假象是同一类。**跑这个 sweep 不要低于 60 ticks。**
* 表里带 `*` 的两格，mean 和"用每轮中位数重建的 per-tick"不一致：
  `reground K=1 @ 0` 读到 352.3 而重建值是 322.1（它自己的 `measured` 列是 323.5，
  所以诚实值是 ~322，352.3 是首个 arm 的一次瞬态），`reground K=4 @ 0.5` 读到 260.5
  而重建值是 243.4。sweep 现在固定输出 `per_tick_ms_from_medians` 这一列专门做这个
  交叉检查，就是它抓出来的。

落地细节：`_full_round` 拆成了 `_ground` / `_denoise` / `_adopt`，两个方案共用
贵的那一半；`reground` 的 draft worker 死亡路径把已经建好的 grounding 交给
`_full_round`，不会付两次 80.8 ms。同 tick 的 denoise 兜底只在
**`accepted_len == 0`** 时触发：夹爪守卫截断出来的非空接受里剩下的是真·被接受的
草稿步，没有 staleness 需要 full round 去修。`spec_reground` 打开后
`spec_full_every` 和 `pending_full` 一起失效（`_wants_full` 直接短路），
每个 session 只剩一个 full round 用来把 prefix 建起来。sweep 里
`regrounded_rounds` 是道明确的引线：如果 `--scheme reground` 的一列报 0，
说明 config 开关没走到 decoder，那一整列是同一个方案测了两遍。

