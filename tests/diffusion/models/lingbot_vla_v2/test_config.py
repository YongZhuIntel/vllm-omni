# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU unit tests for the LingBot-VLA 2.0 config surface.

The released checkpoint carries a stub ``config.json`` and no training YAML, so
``LingbotVlaV2Config.from_release_checkpoint`` reconstructs the architecture from
the safetensors headers. These tests pin that reconstruction against synthetic
headers, plus an opt-in check against a real checkpoint when one is present.

    pytest tests/diffusion/models/lingbot_vla_v2/test_config.py -v
"""

from __future__ import annotations

import json
import os
import struct

import pytest

from vllm_omni.diffusion.models.lingbot_vla_v2.config import (
    LingbotVlaV2Config,
    dead_inference_tensors,
    infer_architecture,
    read_safetensors_shapes,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

EXPERT_LAYER = "model.qwenvl_with_expert.qwen_expert.model.layers"

# Shapes matching the released 6B checkpoint, trimmed to two expert layers.
_RELEASE_SHAPES = {
    "model.state_proj.weight": [768, 55],
    "model.action_out_proj.weight": [55, 768],
    "model.qwenvl_with_expert.qwenvl.model.language_model.embed_tokens.weight": [151936, 2560],
    f"{EXPERT_LAYER}.0.mlp.experts.gate_proj": [32, 512, 768],
    f"{EXPERT_LAYER}.1.mlp.experts.gate_proj": [32, 512, 768],
    f"{EXPERT_LAYER}.0.mlp.shared_expert.gate_proj.weight": [704, 768],
    f"{EXPERT_LAYER}.0.input_layernorm.gamma.weight": [768, 768],
    "model.depth_align_embs": [256, 2560],
    "model.current_video_align_embs": [256, 2560],
    "model.future_depth_align_embs": [256, 2560],
    "model.future_video_align_embs": [256, 2560],
    "model.depth_align_head.projector.proj_in1.weight": [1024, 1024],
    "model.current_video_align_head.projector.proj_out.weight": [1024, 1024],
}


def test_infer_architecture_matches_release_layout():
    inferred = infer_architecture(_RELEASE_SHAPES)
    assert inferred["expert_hidden_size"] == 768
    assert inferred["max_state_dim"] == 55
    assert inferred["max_action_dim"] == 55
    assert inferred["vocab_size"] == 151936
    assert inferred["use_moe"] is True
    assert inferred["token_moe_layers"] == [0, 1]
    assert inferred["token_num_experts"] == 32
    assert inferred["token_moe_intermediate_size"] == 512
    assert inferred["token_shared_intermediate_size"] == 704
    # The release has no shared_expert_gate tensors but does have AdaRMSNorm ones.
    assert inferred["use_shared_expert_gate"] is False
    assert inferred["adanorm_time"] is True


def test_infer_architecture_detects_align_segments():
    """The align query segments change the prefix length, so they are inferred
    from the presence of their embedding tables rather than assumed."""
    inferred = infer_architecture(_RELEASE_SHAPES)
    assert inferred["use_future_depth"] is True
    assert inferred["use_future_video"] is True

    trimmed = {k: v for k, v in _RELEASE_SHAPES.items() if "future_" not in k}
    inferred = infer_architecture(trimmed)
    assert inferred["use_future_depth"] is False
    assert inferred["use_future_video"] is False


def test_infer_architecture_detects_dense_expert():
    dense = {k: v for k, v in _RELEASE_SHAPES.items() if ".mlp.experts." not in k}
    inferred = infer_architecture(dense)
    assert inferred["use_moe"] is False
    assert inferred["token_moe_layers"] == []


def test_dead_inference_tensors_lists_align_heads_only():
    dead = dead_inference_tensors(_RELEASE_SHAPES)
    assert dead == [
        "model.current_video_align_head.projector.proj_out.weight",
        "model.depth_align_head.projector.proj_in1.weight",
    ]
    # The embedding tables feed the prefix and must NOT be classified as dead.
    assert not any(name.endswith("_align_embs") for name in dead)


def test_from_model_config_filters_unknown_keys_and_coerces_resolution():
    config = LingbotVlaV2Config.from_model_config(
        {
            "chunk_size": 50,
            "image_resolution": [224, 224],
            # Training-only keys that must not reach the dataclass.
            "lr": 1.0e-4,
            "optimizer": "muon",
        }
    )
    assert config.chunk_size == 50
    assert config.image_resolution == (224, 224)
    assert isinstance(config.image_resolution, tuple)
    assert not hasattr(config, "lr")


def test_bias_update_speed_must_be_zero_for_inference():
    """Non-zero bias_update_speed drives the training-time loss-free MoE balancing
    update, which mutates router state during a forward pass."""
    with pytest.raises(ValueError, match="bias_update_speed"):
        LingbotVlaV2Config(bias_update_speed=1e-3)


def test_default_kernels_are_portable():
    """Upstream defaults to flex_attention + flash-attn (CUDA-only); the vllm-omni
    defaults must be the pure-torch paths so non-CUDA targets work."""
    config = LingbotVlaV2Config()
    assert config.attention_implementation == "eager"
    assert config.vit_attn_implementation == "sdpa"


def test_read_safetensors_shapes_reads_headers_only(tmp_path):
    header = {"a.weight": {"dtype": "F32", "shape": [2, 3], "data_offsets": [0, 24]}}
    blob = json.dumps(header).encode()
    path = tmp_path / "model-00001-of-00001.safetensors"
    # Deliberately truncate the tensor data: a header-only reader must not care.
    path.write_bytes(struct.pack("<Q", len(blob)) + blob)

    assert read_safetensors_shapes(str(tmp_path)) == {"a.weight": [2, 3]}


def test_read_safetensors_shapes_requires_a_checkpoint(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_safetensors_shapes(str(tmp_path))


@pytest.mark.skipif(
    not os.path.isdir(os.environ.get("LINGBOT_CKPT", "")),
    reason="set LINGBOT_CKPT to a released lingbot-vla-v2-6b checkpoint",
)
def test_real_release_checkpoint():
    checkpoint = os.environ["LINGBOT_CKPT"]
    config = LingbotVlaV2Config.from_release_checkpoint(checkpoint)
    assert config.expert_hidden_size == 768
    assert (config.max_state_dim, config.max_action_dim) == (55, 55)
    assert config.token_moe_layers == list(range(36))
    assert config.token_num_experts == 32
    assert config.token_moe_intermediate_size == 512
    assert config.token_shared_intermediate_size == 704
    assert config.use_shared_expert_gate is False
    assert config.adanorm_time is True
    assert config.vocab_size == 151936

    dead = dead_inference_tensors(read_safetensors_shapes(checkpoint))
    assert len(dead) == 76
