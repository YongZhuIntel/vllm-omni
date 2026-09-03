# LingBot-VLA 2.0 offline inference

This example runs the LingBot-VLA 2.0 policy through the vLLM-Omni diffusion
engine on Intel XPU. The release checkpoint contains the weights but does not
contain the `model_index.json` and full transformer configuration that
vLLM-Omni uses for model discovery.

The repository includes the small deployment-only assets under `deployment/`:

- Qwen3-VL configuration, tokenizer, and image processor files (no VLM weights)
- RobotWin state, action, and camera mapping
- RobotWin training data layout and normalization statistics

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