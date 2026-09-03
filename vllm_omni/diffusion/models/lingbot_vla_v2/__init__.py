# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""LingBot-VLA 2.0 VLA policy for vllm-omni.

Qwen3-VL-4B vision-language backbone + a Qwen2-shaped action expert with
per-layer token MoE, joined by a flow-matching action head. Multi-camera images +
a language instruction + robot state produce a continuous action chunk
``[chunk_size, max_action_dim]`` rather than tokens.
"""

from vllm_omni.diffusion.models.lingbot_vla_v2.config import (
    DEFAULT_QWEN3VL_MODEL,
    LingbotVlaV2Config,
    dead_inference_tensors,
    infer_architecture,
    read_safetensors_shapes,
)
from vllm_omni.diffusion.models.lingbot_vla_v2.modeling_lingbot_vla_v2 import (
    LingbotVlaV2ForActionPrediction,
)
from vllm_omni.diffusion.models.lingbot_vla_v2.pipeline_lingbot_vla_v2 import (
    LingbotVlaV2Pipeline,
)
from vllm_omni.diffusion.models.lingbot_vla_v2.processor import (
    JointGroup,
    LingbotVlaV2Processor,
    RobotFeatures,
    RobotSpec,
    SourceSlice,
    load_hf_processor,
)

__all__ = [
    "DEFAULT_QWEN3VL_MODEL",
    "JointGroup",
    "LingbotVlaV2Config",
    "LingbotVlaV2ForActionPrediction",
    "LingbotVlaV2Pipeline",
    "LingbotVlaV2Processor",
    "RobotFeatures",
    "RobotSpec",
    "SourceSlice",
    "dead_inference_tensors",
    "infer_architecture",
    "load_hf_processor",
    "read_safetensors_shapes",
]
