# LingBot-VLA 2.0 offline inference

This example runs the LingBot-VLA 2.0 policy through the vLLM-Omni diffusion
engine on Intel XPU. The release checkpoint contains the weights but does not
contain the `model_index.json` and full transformer configuration that
vLLM-Omni uses for model discovery.

The repository includes the small deployment-only assets under `deployment/`:

- Qwen3-VL configuration, tokenizer, and image processor files (no VLM weights)
- RobotWin state, action, and camera mapping
- RobotWin training data layout and normalization statistics

## Work log: 2026-09-03, extended 2026-09-04

### Completed

| Milestone | Result |
| --- | --- |
| M1: model kernel | Vendored the 6B inference surface, removed the Qwen2.5-VL training import chain, fixed caller-noise aliasing, added strict checkpoint loading, and passed fp32 stage parity against the repaired upstream reference. |
| M2: observation processor | Added RobotWin camera/state/action mapping, image and language processing, normalization and action reconstruction. All six model inputs match upstream exactly; action conversion has maximum absolute error `1.129e-07`. |
| M3: vLLM-Omni pipeline | Added `LingbotVlaV2Pipeline`, registry integration, `SupportActionOutput`, action result routing, prepared-model generation, repository-local deployment assets, and offline XPU inference. |
| M4: OpenPI protocol | Added the MessagePack/NumPy WebSocket transport, metadata handshake, reset/session handling, AsyncOmni adapter, `/v1/realtime/robot/openpi`, and a RobotWin client. A real XPU WebSocket request returned finite `float32[50,14]` actions. |
| M5: latency and hardening | Attributed the request stage by stage, found the routed `gather` MoE kernel to be 3.7x slower than upstream's dense einsum on the B60, and defaulted to `dense` — 2.43 s to 0.70 s, with fp32 parity becoming exact at all 45 stages. Tested the protocol's failure paths, which found a payload ceiling above what the transport carries and a dead NumPy wire-format compatibility path. |

M1-M3 are committed as `1e157bb7` (`Add LingBot-VLA 2.0 XPU inference support`)
and M4 as `ac22eab4` (`Add OpenPI WebSocket serving for LingBot VLA`). M5 is not
committed yet. `spikes/` holds investigation records and generated numerical
artifacts; the `.npz` files and Python caches stay out of product commits.

### Validation results

- Released checkpoint: 1632 live tensors loaded, zero missing/unexpected, and
  76 training-only align-head tensors dropped.
- XPU model memory: 11.7341 GiB (fp16 and bf16 are the same size).
- Offline output: `type=actions shape=(50, 14) dtype=float32`.
- Offline example, one cold request with no warmup: 0.891 seconds.
- OpenPI WebSocket, eight warm requests on one connection: 0.732-0.935 seconds,
  1.07-1.37 Hz.
- Model kernel alone, warm and in-process: 0.703 seconds, of which the 10-step
  denoise loop is 0.607 seconds and the observation processor is 2 milliseconds.
- These are after the M5 MoE default changed from `gather` to `dense`, which was
  the whole of the speedup; the same paths measured 2.4-2.7 seconds before it.
  `spikes/lingbot_vla_v2/PHASE5_PERF.md` has the plan and the per-step
  conclusions, including which earlier numbers were measured on a loaded host.
- Focused M1-M5 tests: `49 passed, 1 skipped`; the skip is the opt-in real
  checkpoint configuration test.
- Live protocol probe (`spikes/lingbot_vla_v2/phase5_protocol_probe.py`): idle
  close at 30.0 seconds, malformed payloads refused without ending the session,
  a 48 MiB frame closed by the transport, and both NumPy wire formats served.
- Ruff check and format pass for every touched product, example and test file.

Focused regression command:

```bash
docker exec test-image_zy_b8.3.2_lingbot_omni sh -lc '
cd /llm/zhuyong/lingbovla/my/vllm-omni &&
pytest tests/entrypoints/openai_api/test_openpi_connection.py \
  tests/entrypoints/openai_api/test_openpi_serving.py \
  tests/diffusion/models/lingbot_vla_v2/ \
  tests/diffusion/test_diffusion_engine_actions.py \
  -o addopts="" -q'
```

### Container and installed libraries

All runtime validation used container `test-image_zy_b8.3.2_lingbot_omni`, whose
image is `intel/llm-scaler-vllm:0.14.0-b8.3.2`. The host path
`/home/user/zhuyong` is mounted at `/llm/zhuyong` in the container.

Base runtime versions observed after setup:

| Package | Version | Purpose/status |
| --- | --- | --- |
| PyTorch | `2.10.0+xpu` | Intel XPU runtime |
| Transformers | `5.8.0` | Qwen3-VL model and processor |
| vLLM | `0.14.1.dev0+gb17039bcc.d20260626` | Runtime aligned with this vLLM-Omni branch |
| vLLM-Omni | `0.14.0rc1` | Installed editable from this checkout |
| `cache-dit` | `1.2.0` | Required because the v0.14.0 diffusion worker imports it unconditionally |
| `diffusers` | `0.40.0` | Installed/resolved with `cache-dit` |
| `huggingface-hub` | `1.29.0` | Installed/resolved with `cache-dit` |
| `msgspec` | `0.21.1` | Existing dependency used by the OpenPI MessagePack codec |
| `websockets` | `15.0.1` | Existing dependency used by the example client |
| `aenum` | `3.1.17` | Installed while making the v0.14.0 checkout importable |

Commands run inside the container:

```bash
python -m pip install cache-dit==1.2.0
cd /llm/zhuyong/lingbovla/my/vllm-omni
python -m pip install -e . --no-deps
```

Installing `cache-dit==1.2.0` also resolved or upgraded `hf-xet==1.6.0`,
`importlib-metadata==9.0.1`, `oneccl==2021.17.1`,
`oneccl-devel==2021.17.1`, and `zipp==4.1.0`. Pip reported the container's
pre-existing `xgrammar` package expects Triton, which is not installed; this did
not block LingBot model loading or inference.

### Remaining work

1. **Compiled-denoise gate — passed and defaulted.** Five fp16 noise seeds gave
   compiled MAE `1.961e-02` against fp32, below the predeclared OpenVINO INT8
   ceiling `2.882e-02`; eager scored `1.949e-02`. On the six-chunk RobotWin
   bundle, compiled MAE `0.007833` was 0.243% lower than eager `0.007852`.
   `compile_denoise_step` now defaults to true; pass
   `--no-compile-denoise-step` to prepare an eager parity/debug baseline.
2. **Open-loop accuracy benchmark — done.** Predicted chunks are compared with
  dataset ground truth using per-episode MSE/MAE, split joint vs. gripper. On
  the RoboTwin fine-tune the port scores mae **0.0112** against ground truth,
  versus 0.2099 for a "hold the current state" baseline. The mae 0.615 seen
  before this was the *foundation* checkpoint being evaluated on RoboTwin data,
  not a port defect; see step 3 below.
3. **One graph for all ten steps.** The default compiled path still invokes its
  static `predict_velocity` graph ten times per chunk. Extend the compile
  boundary over the fixed Euler loop only after retaining the same fp32/open-loop
  gates; OpenVINO runs the whole loop as one graph.
4. **Machine-readable performance results.** Extend `run_perf_check.sh` to write
  mean/stdev/min/max/p50/p90 plus dtype, MoE mode, compile mode and attention
  backend to JSON.
5. **Prefix stages.** The compiled model path is about 311 ms versus the
  OpenVINO model-only reference of 289 ms, so the main efficiency gap is nearly
  closed. Localize the remaining Inductor numerical drift before enabling the
  optimization by default.
6. **Observation validation.** An observation that decodes but lacks a key the
  processor needs is only caught in the worker, so the client pays a round trip
  and receives `Internal inference error` rather than the missing key.
7. **Reference risk.** Compare mRoPE position IDs against Transformers 4.57 when
  a matching reference bundle is available; current parity is against the
  Transformers 5.8 environment.
8. **Deployment convention.** Evaluate moving handshake/runtime settings from a
  generated `transformer/config.json` to the newer
  `vllm_omni/deploy/<model>.yaml` convention to avoid stale prepared configs.
9. **Grouped/top-4 MoE is deferred.** Parsing the OpenVINO IR proved that its
  reference path also computes all 32 experts densely. Grouped routing remains
  a possible algorithmic improvement, but it is not required to match the
  289 ms reference and is no longer the next optimization.

## Open-loop action accuracy

Open-loop evaluation is split into two environments. The upstream LingBot /
LeRobot environment decodes the held-out dataset and exports a portable NPZ
bundle; the vLLM-Omni container consumes that bundle without installing LeRobot
or a video decoder.

### 1. Download and convert RobotWin data

Run the data preparation in `test-image_zy_b8.3.2_lingbot`. Clone RoboTwin with
its pinned XPolicyLab submodule:

```bash
cd /llm/zhuyong/lingbovla
git clone --recurse-submodules \
  https://github.com/RoboTwin-Platform/RoboTwin.git
```

Activate the upstream LingBot environment and install its pinned LeRobot. The
environment already contains `torchcodec==0.6.0` and `av==15.0.0`.

```bash
source /llm/zhuyong/lingbovla/frameworks.robotics.embodied-intelligence.lingbot-vla-v2/.venv-intel-dev/bin/activate
python -m pip install --no-deps \
  "lerobot @ https://github.com/huggingface/lerobot/archive/refs/tags/v0.4.2.tar.gz"
```

Download one official task through the Hugging Face mirror. This fetches only
`dataset/adjust_bottle/demo_clean.zip` (about 292 MB), not the complete 1.53 TB
RoboTwin release.

```bash
cd /llm/zhuyong/lingbovla/RoboTwin
export HF_ENDPOINT=https://hf-mirror.com
HF_MAX_WORKERS=1 HF_EXTRACT_WORKERS=1 \
  bash scripts/download_xpolicylab_data.sh adjust_bottle
```

The extracted task contains 50 HDF5 trajectories, 50 videos and 50 instruction
files under:

```text
/llm/zhuyong/lingbovla/RoboTwin/data/demo_clean/adjust_bottle/aloha_agilex
```

Convert all 50 episodes to the LeRobot v2.1 layout used by LingBot:

```bash
export HF_LEROBOT_HOME=/llm/zhuyong/lingbovla/datasets/lerobot
python XPolicyLab/scripts/transform_lerobot_v21_format.py \
  "demo_clean.adjust_bottle.aloha_agilex" \
  --repo_id adjust_bottle_demo_clean \
  --max_episode 50
```

The tested XPolicyLab v3 converter passes `streaming_encoding` to
`LeRobotDataset.create()`, which is not supported by the upstream-pinned
`lerobot==0.4.2`. Use `transform_lerobot_v21_format.py`, not the v3 converter,
unless the LeRobot environment is upgraded together with the converter.

The converted dataset is written to:

```text
/llm/zhuyong/lingbovla/datasets/lerobot/adjust_bottle_demo_clean
```

### 2. Export held-out RobotWin episodes

Still in the upstream LingBot environment, export portable chunks. This example
selects episodes 0-2 and two chunks per episode for a quick smoke test:

```bash
cd /llm/zhuyong/lingbovla/my/vllm-omni
python examples/offline_inference/lingbot_vla_v2/export_open_loop_bundle.py \
  --lingbot-root /llm/zhuyong/lingbovla/frameworks.robotics.embodied-intelligence.lingbot-vla-v2/lingbot-vla-v2 \
  --data-path /llm/zhuyong/lingbovla/datasets/lerobot/adjust_bottle_demo_clean \
  --episodes 0 1 2 \
  --max-chunks-per-episode 2 \
  --output /llm/zhuyong/lingbovla/datasets/open_loop/adjust_bottle_3ep_2chunks.npz
```

The exporter reuses upstream `LeRobotDataset`, `FeatureTransform.apply()`, and
`FeatureTransform.unapply()`. Ground truth is therefore stored in raw RobotWin
units with the same feature order as upstream evaluation. Episode-tail padding
is recorded in `valid_steps` and excluded from metrics.

Bundle schema:

| Key | Shape/type |
| --- | --- |
| `images` | `uint8[N,3,H,W,3]`, high/left-wrist/right-wrist |
| `states` | `float32[N,14]` |
| `actions` | `float32[N,50,14]`, raw robot units |
| `prompts` | string `[N]` |
| `episode_ids`, `frame_indices` | integer `[N]` |
| `valid_steps` | integer `[N]`, valid rows in each action chunk |

### 3. Choose the checkpoint for the evaluation goal

Use the same checkpoint on both sides of a comparison. The two released
checkpoints answer different questions:

- `lingbot-vla-v2-6b`: use this to validate that the vLLM-Omni port preserves
  the behavior of the base model used throughout M1-M5. It is also sufficient
  for comparing eager and compiled execution because only the runtime changes.
- `lingbot-vla-v2-6b-robotwin`: use this only when measuring the task accuracy
  of the official RobotWin post-trained policy or comparing against its published
  RoboTwin results.

The open-loop harness accepts either checkpoint. Scores from the two checkpoints
must not be compared as if they measured an implementation regression: their
weights are different.

**Accuracy runs must use the RoboTwin fine-tune.** The two checkpoints have
identical architecture and identical tensor names (1708 keys, same key set), so
the base checkpoint loads and runs without a single warning — the score is the
only thing that tells you which one you loaded. On
`adjust_bottle_3ep_2chunks.npz`, changing nothing but the checkpoint:

| checkpoint | mse | mae |
|---|---:|---:|
| `lingbot-vla-v2-6b` (foundation) | 0.665 | 0.615 |
| `…-6b-robotwin/…/hf_ckpt` | 0.00061 | **0.0112** |

The foundation model was pre-trained on 60k hours of general robot data and
never saw RoboTwin's action space; on RoboTwin data it predicts a generic
bimanual reach, moving the idle arm that the ground truth holds at exactly zero.
`run_open_loop_eval.sh` therefore defaults to the fine-tune. Use the foundation
checkpoint for latency and eager-vs-compiled work, where only the runtime
changes and the weights are irrelevant.

The base checkpoint already used by this port is:

```text
/llm/zhuyong/lingbovla/models/lingbot-vla-v2-6b
```

To optionally download the official RobotWin post-trained checkpoint, run this
inside `test-image_zy_b8.3.2_lingbot`:

```bash
cd /llm/zhuyong/lingbovla/my/vllm-omni
examples/offline_inference/lingbot_vla_v2/download_robotwin_checkpoint.sh
```

The script defaults to `HF_ENDPOINT=https://hf-mirror.com`, pins revision
`0451855729ec904f970600e0aec8b84661423afe`, and downloads only the final
inference checkpoint subtree. The complete download is about 25.5 GB. It is
resumable: rerun the same command to continue existing `.incomplete` files.

The checkpoint passed to the preparation script is:

```text
/llm/zhuyong/lingbovla/models/lingbot-vla-v2-6b-robotwin/checkpoints/global_step_50000/hf_ckpt
```

Prepare eager and compiled directories from those weights:

```bash
python examples/offline_inference/lingbot_vla_v2/prepare_lingbot_vla_v2.py \
  --checkpoint /llm/zhuyong/lingbovla/models/lingbot-vla-v2-6b-robotwin/checkpoints/global_step_50000/hf_ckpt \
  --output /tmp/lingbot-robotwin-eager

python examples/offline_inference/lingbot_vla_v2/prepare_lingbot_vla_v2.py \
  --checkpoint /llm/zhuyong/lingbovla/models/lingbot-vla-v2-6b-robotwin/checkpoints/global_step_50000/hf_ckpt \
  --output /tmp/lingbot-robotwin-compiled \
  --compile-denoise-step
```

### 4. Evaluate through vLLM-Omni

The simplest command, run inside `test-image_zy_b8.3.2_lingbot_omni`, evaluates
the existing six-chunk bundle with the base checkpoint:

```bash
cd /llm/zhuyong/lingbovla/my/vllm-omni
examples/offline_inference/lingbot_vla_v2/run_open_loop_eval.sh
```

Run eager and compiled modes on identical samples/noise and generate a comparison:

```bash
examples/offline_inference/lingbot_vla_v2/run_open_loop_eval.sh --mode both
```

Use `--checkpoint` to select the optional RobotWin post-trained checkpoint,
`--dataset` for another exported bundle, and `--episodes` / `--max-samples` to
select evaluation samples. The script always validates the bundle first and
rebuilds each prepared model directory to avoid stale configuration.

Validate the bundle without loading the model:

```bash
python examples/offline_inference/lingbot_vla_v2/open_loop_eval.py \
  --dataset /llm/zhuyong/lingbovla/datasets/open_loop/adjust_bottle_3ep_2chunks.npz \
  --output-dir /tmp/robotwin-open-loop-dry-run \
  --dry-run
```

Run model inference in the XPU container:

```bash
python examples/offline_inference/lingbot_vla_v2/open_loop_eval.py \
  --model /tmp/lingbot-robotwin-eager \
  --dataset /llm/zhuyong/lingbovla/datasets/open_loop/adjust_bottle_3ep_2chunks.npz \
  --output-dir /tmp/robotwin-open-loop-eager \
  --dtype float16 --seed 1234 --plots
```

The evaluator writes `metrics.json` and `predictions.npz`. Metrics are
micro/macro and per-episode, for all 14 dimensions, the 12 arm joints and the
two grippers, using the same metric set the OpenVINO export repo reports —
cosine, MAE, MSE, max abs diff, p99 abs diff. It prints them as a table:

```text
vs dataset ground truth  |  6 chunks, 3 episodes, 300 valid steps  |  dtype=float16 seed=1234

  metric                all 14    12 joints   2 grippers   macro (all)
  --------------- ------------ ------------ ------------ -------------
  cosine (mean)       0.999673            -            -      0.999673
  cosine (min)        0.998926            -            -      0.999390
  MAE                7.852e-03    8.719e-03    2.653e-03     7.852e-03
  MSE                5.051e-04    5.708e-04    1.110e-04     5.051e-04
  max abs diff       2.999e-01    2.999e-01    7.387e-02     1.661e-01
  p99 abs diff       9.265e-02    9.728e-02    7.094e-02     9.651e-02
```

**The reference here is dataset ground truth**, in robot command units. These
are *not* the same numbers as the identically-named ones in
`spikes/lingbot_vla_v2/PHASE7_NUMERICS.md`, which grade the kernel against a
fp32 run of itself in normalized 55-dim units. Same formulas, different
reference — do not put them in one table.

With `--plots`, it also writes one 14-axis GT/prediction PNG per episode. Noise
is deterministic per bundle sample and its seed/hash is stored with predictions.

Run the same bundle and seed with `/tmp/lingbot-robotwin-compiled`, writing to a
separate output directory, then compare both result sets. Do not judge the
compiled path from synthetic parity alone; its measured action-chunk drift
versus eager is 1.75% in bf16.

```bash
python examples/offline_inference/lingbot_vla_v2/open_loop_eval.py \
  --model /tmp/lingbot-robotwin-compiled \
  --dataset /llm/zhuyong/lingbovla/datasets/open_loop/adjust_bottle_3ep_2chunks.npz \
  --output-dir /tmp/robotwin-open-loop-compiled \
  --dtype float16 --seed 1234 --plots
```

```bash
python examples/offline_inference/lingbot_vla_v2/compare_open_loop.py \
  --baseline /tmp/robotwin-open-loop-eager \
  --candidate /tmp/robotwin-open-loop-compiled \
  --output /tmp/robotwin-open-loop-comparison.json
```

The workflow was smoke-tested with the official `adjust_bottle/demo_clean`
archive downloaded through `HF_ENDPOINT=https://hf-mirror.com`. Fifty episodes
were converted with XPolicyLab's LeRobot v2.1 converter to:

```text
/llm/zhuyong/lingbovla/datasets/lerobot/adjust_bottle_demo_clean
```

A six-chunk bundle (episodes 0-2, frames 0 and 50) was exported to:

```text
/llm/zhuyong/lingbovla/datasets/open_loop/adjust_bottle_3ep_2chunks.npz
```

This small run validates the harness, not the published RoboTwin benchmark. A
representative score requires the RobotWin post-trained checkpoint and a larger,
predeclared held-out split.

### Known remaining evaluation work

1. **The performance check emits no machine-readable artifact.**
   `run_perf_check.sh` prints a verdict and exits non-zero, which gates a
   regression but cannot be diffed across runs. Upstream's
   `examples/offline_inference/internvla_a1/end2end.py` writes
   `forward_latency.json` with mean/stdev/min/max/p50/p90 plus dtype, attention
   implementation and compile flags. Matching that schema — and adding p90, which
   the current median/min/max does not give — would make runs comparable and let
   the numbers feed a perf config the way `tests/dfx/perf/tests/*.json` does
   upstream. Note that no VLA model has such a config upstream either, so there
   is no format to conform to yet, only one to borrow.
2. **Divergence from upstream's VLA conventions.** Upstream v0.28 now ships
   `pi0`, `gr00t`, `internvla_a1` and `dreamzero` against this same OpenPI
   endpoint, and configures them through `vllm_omni/deploy/<model>.yaml`: one
   diffusion stage, `max_num_seqs: 1`, and the handshake metadata in a
   `policy_server_config` block. This port instead has `prepare_lingbot_vla_v2.py`
   serialize the whole config into `transformer/config.json`, which is what makes
   a stale prepared directory silently pin old settings. Worth evaluating whether
   to move to the deploy-config convention before the two diverge further.

These items came from surveying upstream v0.28 (`v0.28.0-3-ga65e89cb`) for a
standard VLA benchmark. There is none: `benchmarks/` has no VLA entry, none of
the 27 perf configs under `tests/dfx/perf/tests/` covers a VLA model, the pi0 and
gr00t e2e tests assert handshake metadata and shapes rather than latency, and
`examples/online_serving/pi0/openpi_client.py` does not time its requests. So
`run_perf_check.sh` is not duplicating something that already exists — but the
two internvla_a1 harnesses above are the conventions to converge on.

M5 is closed. Both performance targets are met — the M4 threshold of one second
per action chunk and the Phase 0 direct-kernel baseline of 0.74 seconds — and the
protocol's failure paths are tested, which found two defects: a payload ceiling
set above what the transport carries, and a dead NumPy wire-format compatibility
path. `spikes/lingbot_vla_v2/PHASE5_PERF.md` holds the plan and every conclusion.

## Deployment asset provenance

The files under `deployment/` are byte-identical copies from the following
online repositories.

### Qwen3-VL configuration and processor

Source repository:
[`YongZhuIntel/frameworks.robotics.embodied-intelligence.lingbot-vla-v2`](https://github.com/YongZhuIntel/frameworks.robotics.embodied-intelligence.lingbot-vla-v2)
at revision `4b6ee0ad9ebac6082a6419953641153aa31a3afb`.

| Repository-local path | Upstream path |
| --- | --- |
| `deployment/qwen3vl_base_config/` | `qwen3vl_base_config/` |

This directory contains `config.json`, generation and chat-template settings,
the tokenizer vocabulary, and image processor configuration. It contains no
Qwen3-VL weight files.

### RobotWin deployment configuration

Source repository:
[`robbyant/lingbot-vla-v2`](https://github.com/robbyant/lingbot-vla-v2)
at revision `be27333c9b5f2663b0ec33f069dd7dfd67fa32b5`.

| Repository-local path | Upstream path |
| --- | --- |
| `deployment/configs/robot_configs/robotwin.yaml` | `configs/robot_configs/robotwin.yaml` |
| `deployment/configs/vla/robotwin/robotwin.yaml` | `configs/vla/robotwin/robotwin.yaml` |
| `deployment/assets/norm_stats/robotwin.json` | `assets/norm_stats/robotwin.json` |

### Policy checkpoint

The 6B policy weights are not copied into this repository. Download them from
[`robbyant/lingbot-vla-v2-6b`](https://huggingface.co/robbyant/lingbot-vla-v2-6b)
at the revision tested here, `11c703bf6a5c1f45b3b69168482da11fdbba53d7`.
The preparation command below creates symlinks to its six safetensors shards.

## One-command performance check

```bash
docker exec test-image_zy_b8.3.2_lingbot_omni sh -lc '
cd /llm/zhuyong/lingbovla/my/vllm-omni &&
examples/online_serving/lingbot_vla_v2/run_perf_check.sh'
```

It regenerates the prepared directory, starts the server, sends warm requests
over one WebSocket connection, discards the first, and checks the median against
the 1 Hz acceptance criterion. It exits non-zero if the target is missed, so it
works as a regression gate. `--attribution` adds the per-stage breakdown,
`--no-offline` skips the cold-start request, `--requests N` changes the sample
count, and `--compile-denoise-step` enables the experimental Inductor path. The
compiled path is default-off because its measured bf16 chunk drift is 1.75%.

Two measurements, and it matters which is which — the 21.6 ms/step denoise below
is only reachable with Inductor:

| path | warm median | denoise loop | model path (synced) |
|---|---:|---:|---:|
| eager, fp16 | 0.861 s (1.16 Hz) | 602.1 ms (60.1 ms/step) | 696.3 ms |
| `--compile-denoise-step`, bf16 | 0.411 s (2.43 Hz) | 215.6 ms (21.6 ms/step) | 311.9 ms |

The compiled figure is the one close to the OpenVINO model-only reference of
289 ms; the default eager path is not. The eager row is the fp16 re-measurement
taken after the dtype flip, on an idle host, and it matches the bf16 eager
record it replaces (0.703 s model path, 0.607 s denoise) to within 1% — fp16
costs nothing here, as Phase 5 predicted.

Two behaviours worth knowing, both learned the hard way:

- It **refuses to measure on a busy host** (exit code 3), listing any orphaned
  Python processes it found. Leaked workers from an earlier run hold host RAM and
  the whole XPU allocation, and once inflated a 1.9 ms stage to 178 ms — which
  sent a whole optimization step after a cost that did not exist.
- It regenerates the prepared directory every time, because
  `prepare_lingbot_vla_v2.py` serializes the entire config into
  `transformer/config.json`. A directory built before a config change silently
  keeps serving the old settings.

## Inference dtype

Every script here defaults to `--dtype float16`. It used to be `bfloat16`, which
was the wrong default: **the checkpoint is stored in fp32** — all 1708 tensors —
so bf16 was throwing away three mantissa bits that the weights actually carried,
in exchange for an exponent range this model never uses.

Measured against our own kernel running at CPU fp32 on identical inputs and
identical initial noise, over five noise draws, using the export repo's
`validate_e2e_split.py` metric verbatim (`spikes/lingbot_vla_v2/phase7_numeric_parity.py`):

| path | cosine | MAE | MSE | max abs | p99 abs |
|---|---:|---:|---:|---:|---:|
| OpenVINO FP16 (n=1, their number) | 0.998544 | 1.608e-02 | 8.716e-04 | 1.846e-01 | 1.291e-01 |
| OpenVINO INT8 (n=1, their number) | 0.995633 | 2.882e-02 | 2.604e-03 | 3.135e-01 | 2.089e-01 |
| vLLM-Omni XPU bf16 | 0.978196 | 6.159e-02 | 1.529e-02 | 1.597e+00 | 4.988e-01 |
| vLLM-Omni XPU fp16 | 0.998187 | **1.949e-02** | 1.229e-03 | 3.377e-01 | 1.321e-01 |

bf16 was worse than the OpenVINO path's *int8*. fp16 lands on its FP16 column.
Task accuracy against dataset ground truth moves the same way — mae 0.01125 to
0.00785 on the RoboTwin bundle.

Two things this is not: it is not a latency lever (Phase 5 measured fp16 against
bf16 at 2.4%, inside harness noise), and fp16's 65504 ceiling is not a risk here
(peak activation 11264, peak weight 44.27, zero non-finite values in a full fp16
run). `spikes/lingbot_vla_v2/PHASE7_NUMERICS.md` has the method and the open
question about the gripper channel's tail.

## One-command XPU test

Run the complete preparation and inference flow from the host:

```bash
examples/offline_inference/lingbot_vla_v2/run_xpu_test.sh
```

The script defaults to container `test-image_zy_b8.3.2_lingbot_omni`, the local
6B checkpoint, fp16, and `/tmp/lingbot-vla-v2-prepared`. Run it with `--help`
to see options for overriding the container, paths, dtype, prompt, and seed.

## Prepare the model directory

The wrapper above runs these two steps. To run them separately, first execute
the preparation command below. It creates two JSON files and symlinks the six
checkpoint shards; it does not copy the 24 GB checkpoint.

```bash
docker exec test-image_zy_b8.3.2_lingbot_omni sh -lc '
cd /llm/zhuyong/lingbovla/my/vllm-omni &&
PYTHONPATH=. python examples/offline_inference/lingbot_vla_v2/prepare_lingbot_vla_v2.py \
  --checkpoint /llm/zhuyong/lingbovla/models/lingbot-vla-v2-6b \
  --output /tmp/lingbot-vla-v2-prepared'
```

The generated directory has this layout:

```text
/tmp/lingbot-vla-v2-prepared/
├── model_index.json
├── transformer/config.json
└── model-0000*-of-00006.safetensors -> release checkpoint shards
```

The script defaults to the repository-local RobotWin assets. Use
`--qwen3vl-path`, `--robot-config`, `--data-config`, or `--norm-stats` to prepare
a different deployment.

## Run XPU inference

The container must have the project runtime dependency `cache-dit==1.2.0`.

```bash
docker exec test-image_zy_b8.3.2_lingbot_omni sh -lc '
cd /llm/zhuyong/lingbovla/my/vllm-omni &&
PYTHONPATH=. python examples/offline_inference/lingbot_vla_v2/lingbot_vla_v2.py \
  --model /tmp/lingbot-vla-v2-prepared \
  --dtype float16'
```

A successful run prints an action chunk in RobotWin command layout:

```text
type=actions shape=(50, 14) dtype=float32
```

## OpenPI WebSocket serving

When running from a source checkout, install its CLI metadata once inside the
container:

```bash
cd /llm/zhuyong/lingbovla/my/vllm-omni
python -m pip install -e . --no-deps
```

Prepare the model as above, then start the vLLM-Omni server inside the container:

```bash
docker exec -it test-image_zy_b8.3.2_lingbot_omni sh -lc '
cd /llm/zhuyong/lingbovla/my/vllm-omni &&
PYTHONPATH=. python -m vllm_omni.entrypoints.cli.main serve \
  /tmp/lingbot-vla-v2-prepared \
  --omni --host 0.0.0.0 --port 8000 \
  --deploy-config /tmp/lingbot-vla-v2-prepared/deploy.yaml \
  --dtype float16 --enforce-eager --disable-log-stats'
```

The server wrapper is run directly inside the container's vLLM-Omni checkout. It
regenerates the prepared directory before the server starts, so it cannot use
stale metadata:

```bash
docker exec -it test-image_zy_b8.3.2_lingbot_omni sh -lc '
cd /llm/zhuyong/lingbovla/my/vllm-omni &&
examples/online_serving/lingbot_vla_v2/run_openpi_server.sh'
```

Enable the compiled denoise step explicitly when the deployment accepts the
measured 1.75% bf16 chunk drift:

```bash
examples/online_serving/lingbot_vla_v2/run_openpi_server.sh \
  --compile-denoise-step
```

In another terminal, run the synthetic RobotWin client:

```bash
docker exec test-image_zy_b8.3.2_lingbot_omni sh -lc '
cd /llm/zhuyong/lingbovla/my/vllm-omni &&
PYTHONPATH=. python examples/online_serving/lingbot_vla_v2/openpi_client.py \
  --host 127.0.0.1 --port 8000'
```

Or run the client wrapper inside the same container:

```bash
docker exec test-image_zy_b8.3.2_lingbot_omni sh -lc '
cd /llm/zhuyong/lingbovla/my/vllm-omni &&
examples/online_serving/lingbot_vla_v2/run_openpi_client.sh'
```

The endpoint is `ws://127.0.0.1:8000/v1/realtime/robot/openpi`. It sends the
policy metadata immediately after connection and returns one float32
`[50, 14]` action chunk for each observation. The measured v0.14.0 XPU WebSocket
round trip is 0.73-0.94 seconds, or 1.07-1.37 Hz, over eight consecutive requests
on one connection. The engine, its worker IPC and the transport together account
for about 110 milliseconds of that at the median.

### `Robot policy not available` response

The OpenPI client expects a binary MessagePack metadata handshake. If the server
starts from an old prepared directory without `policy_server_config`, the endpoint
instead returns this JSON text response:

```json
{"type":"error","error":"Robot policy not available","code":"unsupported"}
```

Older client code then reports `TypeError: a bytes-like object is required, not
'str'`. Regenerate the prepared directory with the current preparation script and
restart the server. `run_openpi_server.sh` performs both steps automatically
inside the container.