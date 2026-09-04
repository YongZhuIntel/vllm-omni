# LingBot-VLA 2.0 offline inference

This example runs the LingBot-VLA 2.0 policy through the vLLM-Omni diffusion
engine on Intel XPU. The release checkpoint contains the weights but does not
contain the `model_index.json` and full transformer configuration that
vLLM-Omni uses for model discovery.

The repository includes the small deployment-only assets under `deployment/`:

- Qwen3-VL configuration, tokenizer, and image processor files (no VLM weights)
- RobotWin state, action, and camera mapping
- RobotWin training data layout and normalization statistics

## Work log: 2026-09-03

### Completed

| Milestone | Result |
| --- | --- |
| M1: model kernel | Vendored the 6B inference surface, removed the Qwen2.5-VL training import chain, fixed caller-noise aliasing, added strict checkpoint loading, and passed fp32 stage parity against the repaired upstream reference. |
| M2: observation processor | Added RobotWin camera/state/action mapping, image and language processing, normalization and action reconstruction. All six model inputs match upstream exactly; action conversion has maximum absolute error `1.129e-07`. |
| M3: vLLM-Omni pipeline | Added `LingbotVlaV2Pipeline`, registry integration, `SupportActionOutput`, action result routing, prepared-model generation, repository-local deployment assets, and offline XPU inference. |
| M4: OpenPI protocol | Added the MessagePack/NumPy WebSocket transport, metadata handshake, reset/session handling, AsyncOmni adapter, `/v1/realtime/robot/openpi`, and a RobotWin client. A real XPU WebSocket request returned finite `float32[50,14]` actions. |

The M1-M3 implementation is committed as `1e157bb7` (`Add LingBot-VLA 2.0 XPU
inference support`). The M4 server, client, tests, metadata, and this work log are
currently working-tree changes. Files under `spikes/` are investigation records
and generated numerical artifacts; do not include the `.npz` files or Python
caches in a product commit.

### Validation results

- Released checkpoint: 1632 live tensors loaded, zero missing/unexpected, and
  76 training-only align-head tensors dropped.
- XPU model memory: 11.7341 GiB in bf16.
- Offline output: `type=actions shape=(50, 14) dtype=float32`.
- Offline post-warmup request: 2.426-2.473 seconds in the measured runs.
- OpenPI server request: 2.744 seconds server-side.
- Focused M1-M4 tests: `34 passed, 1 skipped`; the skip is the opt-in real
  checkpoint configuration test.
- Ruff, Python compilation, editor diagnostics, and `git diff --check` pass for
  the touched M4 files.

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

1. **M5 latency attribution.** Measure processor CPU time, host-to-XPU transfer,
   model execution, postprocessing/device synchronization, worker IPC, and full
   WebSocket time over several post-warmup requests.
2. **MoE A/B benchmark.** Compare the current routed gather implementation with
   the upstream grouped dense-einsum reference. The planned temporary dense
   benchmark has not been run yet.
3. **Performance target.** First reach the M4 acceptance threshold of less than
   one second per action chunk, then match or beat the Phase 0 direct-kernel
   baseline of 0.74 seconds. Current serving throughput is about 0.36 Hz.
4. **Kernel optimization.** Only after attribution, evaluate fixed-shape
   compilation and an XPU grouped MoE kernel while retaining the eager dense path
   as the numerical reference.
5. **Protocol hardening.** Add explicit tests for idle timeout, oversized/invalid
   payloads, sanitized inference errors, and interoperability with the official
   `openpi-client` package.
6. **Reference risk.** Compare mRoPE position IDs against Transformers 4.57 when
   a matching reference bundle is available; current parity is against the
   Transformers 5.8 environment.
7. **Commit M4.** Stage the OpenPI modules, API integration, client, tests,
   prepared metadata, and documentation without staging `spikes/` artifacts.

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

## One-command XPU test

Run the complete preparation and inference flow from the host:

```bash
examples/offline_inference/lingbot_vla_v2/run_xpu_test.sh
```

The script defaults to container `test-image_zy_b8.3.2_lingbot_omni`, the local
6B checkpoint, bf16, and `/tmp/lingbot-vla-v2-prepared`. Run it with `--help`
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
  --dtype bfloat16'
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
  --dtype bfloat16 --enforce-eager --disable-log-stats'
```

The server wrapper is run directly inside the container's vLLM-Omni checkout. It
regenerates the prepared directory before the server starts, so it cannot use
stale metadata:

```bash
docker exec -it test-image_zy_b8.3.2_lingbot_omni sh -lc '
cd /llm/zhuyong/lingbovla/my/vllm-omni &&
examples/online_serving/lingbot_vla_v2/run_openpi_server.sh'
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
round trip was 2.74 seconds; the transport is functional, but reaching 1 Hz is
still an M5 performance task.

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

## Stage M4 files

Stage these M4 files, but do not stage `spikes/` or its `.npz` artifacts:

```bash
git add \
  examples/offline_inference/lingbot_vla_v2/README.md \
  examples/offline_inference/lingbot_vla_v2/prepare_lingbot_vla_v2.py \
  examples/online_serving/lingbot_vla_v2/ \
  tests/entrypoints/openai_api/test_openpi_connection.py \
  tests/entrypoints/openai_api/test_openpi_serving.py \
  vllm_omni/entrypoints/openai/api_server.py \
  vllm_omni/entrypoints/openpi/
```