# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU structural tests for the vendored LingBot-VLA 2.0 kernel.

Numerical parity against the reference implementation needs the 6B checkpoint and
~26 GB of RAM per side, so it lives in ``spikes/lingbot_vla_v2/phase1_parity.py``.
What is pinned here is everything that can be checked on a randomly initialised
*tiny* model in a second: the prefix layout, the invariants the joint walk has to
preserve, and the properties that a previous port got wrong.

    pytest tests/diffusion/models/lingbot_vla_v2/test_modeling.py -v
"""

from __future__ import annotations

import pytest
import torch
from transformers.models.qwen3_vl.configuration_qwen3_vl import (
    Qwen3VLConfig,
    Qwen3VLTextConfig,
    Qwen3VLVisionConfig,
)

from vllm_omni.diffusion.models.lingbot_vla_v2 import (
    LingbotVlaV2Config,
    LingbotVlaV2ForActionPrediction,
)
from vllm_omni.diffusion.models.lingbot_vla_v2.modeling_lingbot_vla_v2 import (
    eager_attention,
    sdpa_attention,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

NUM_LAYERS = 4
HEAD_DIM = 16
NUM_CAMS = 2
BATCH = 1
# Cameras are 4x4 patches; the merger halves each axis, so 4 tokens survive.
GRID = torch.tensor([[1, 4, 4]] * NUM_CAMS)
PATCHES = int(GRID[0].prod())
MASKED_LANG_TOKENS = 3


def test_sdpa_attention_matches_eager_attention():
    torch.manual_seed(0)
    query = torch.randn(1, 5, 4, HEAD_DIM, dtype=torch.float32)
    key = torch.randn(1, 5, 2, HEAD_DIM, dtype=torch.float32)
    value = torch.randn(1, 5, 2, HEAD_DIM, dtype=torch.float32)
    mask = torch.tril(torch.ones(1, 5, 5, dtype=torch.bool))

    eager = eager_attention(query, key, value, mask)
    sdpa = sdpa_attention(query, key, value, mask)

    torch.testing.assert_close(sdpa, eager, rtol=1e-5, atol=1e-6)


def test_eager_attention_fp16_mask_remains_finite():
    torch.manual_seed(1)
    query = torch.randn(1, 5, 4, HEAD_DIM, dtype=torch.float16)
    key = torch.randn(1, 5, 2, HEAD_DIM, dtype=torch.float16)
    value = torch.randn(1, 5, 2, HEAD_DIM, dtype=torch.float16)
    mask = torch.tril(torch.ones(1, 5, 5, dtype=torch.bool))

    output = eager_attention(query, key, value, mask)

    assert torch.isfinite(output).all()


def _vlm_config() -> Qwen3VLConfig:
    vision = Qwen3VLVisionConfig(
        depth=NUM_LAYERS,
        hidden_size=48,
        num_heads=3,
        intermediate_size=64,
        patch_size=16,
        spatial_merge_size=2,
        temporal_patch_size=2,
        in_channels=3,
        num_position_embeddings=64,
        out_hidden_size=64,
        deepstack_visual_indexes=[0, 1, 2],
    )
    text = Qwen3VLTextConfig(
        vocab_size=257,
        hidden_size=64,
        num_hidden_layers=NUM_LAYERS,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=HEAD_DIM,
        intermediate_size=96,
        rms_norm_eps=1e-6,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 1e6,
            "mrope_section": [4, 2, 2],
            "mrope_interleaved": True,
        },
    )
    return Qwen3VLConfig(
        text_config=text,
        vision_config=vision,
        image_token_id=250,
        video_token_id=251,
        vision_start_token_id=252,
        vision_end_token_id=253,
    )


def _policy_config() -> LingbotVlaV2Config:
    return LingbotVlaV2Config(
        vocab_size=257,
        chunk_size=6,
        max_action_dim=7,
        max_state_dim=7,
        num_steps=3,
        expert_hidden_size=32,
        expert_num_layers=NUM_LAYERS,
        expert_num_attention_heads=4,
        expert_num_key_value_heads=2,
        expert_head_dim=HEAD_DIM,
        expert_intermediate_size=48,
        token_moe_layers=[1, 3],
        token_num_experts=6,
        token_top_k=2,
        token_moe_intermediate_size=24,
        token_shared_intermediate_size=20,
        num_task_tokens=4,
        num_backbone_tokens=16,
        align_dim=64,
        tokenizer_max_length=9,
        max_cameras=NUM_CAMS,
    )


@pytest.fixture(scope="module")
def model() -> LingbotVlaV2ForActionPrediction:
    torch.manual_seed(0)
    policy = LingbotVlaV2ForActionPrediction(_policy_config(), vlm_config=_vlm_config()).eval()
    # Zero-initialised align tables and default-initialised norms make several of
    # these invariants pass trivially; give every parameter signal instead.
    for parameter in policy.parameters():
        torch.nn.init.normal_(parameter, std=0.02)
    return policy


@pytest.fixture(scope="module")
def observation(model) -> dict:
    torch.manual_seed(1)
    config = model.config
    vision = model.vlm_config.vision_config
    patch_dim = vision.in_channels * vision.temporal_patch_size * vision.patch_size**2
    lang_masks = torch.ones(BATCH, config.tokenizer_max_length, dtype=torch.bool)
    lang_masks[:, -MASKED_LANG_TOKENS:] = False
    return {
        "images": torch.randn(BATCH, NUM_CAMS, PATCHES, patch_dim),
        "img_masks": torch.ones(BATCH, NUM_CAMS, dtype=torch.bool),
        "lang_tokens": torch.randint(0, 200, (BATCH, config.tokenizer_max_length)),
        "lang_masks": lang_masks,
        "state": torch.randn(BATCH, config.max_state_dim),
        "image_grid_thw": GRID.unsqueeze(0),
        "noise": torch.randn(BATCH, config.chunk_size, config.max_action_dim),
    }


def _prefix(model, observation, **overrides):
    kwargs = {**observation, **overrides}
    return model.embed_prefix(
        kwargs["images"],
        kwargs["img_masks"],
        kwargs["lang_tokens"],
        kwargs["lang_masks"],
        kwargs["image_grid_thw"],
    )


def _sample(model, observation, **overrides):
    kwargs = {**observation, **overrides}
    return model.sample_actions(
        kwargs["images"],
        kwargs["img_masks"],
        kwargs["lang_tokens"],
        kwargs["lang_masks"],
        kwargs["state"],
        kwargs["image_grid_thw"],
        noise=kwargs["noise"],
    )


def _merged_tokens(model) -> int:
    return PATCHES // model.vlm_config.vision_config.spatial_merge_size**2


def test_prefix_layout(model, observation):
    """Cameras, prompt and task queries occupy exactly the slots they claim.

    Each camera contributes its merged patch tokens plus a boundary token on each
    side; the prompt contributes its full padded length; each prefix query segment
    contributes ``num_task_tokens``. Getting this wrong is silent — the model still
    runs, it just attends to the wrong things.
    """
    merged = _merged_tokens(model)
    embs, pad_masks, _, position_ids, visual_masks, deepstack = _prefix(model, observation)

    # ``language`` names the prompt span, not a query block, and the CLS query is a
    # single token; the rest are ``num_task_tokens`` wide.
    assert model.prefix_segments == ("language", "current_depth", "future_depth")
    query_tokens = sum(
        1 if segment == "future_video_cls" else model.config.num_task_tokens
        for segment in model.prefix_segments
        if segment != "language"
    )
    expected = NUM_CAMS * (merged + 2) + model.config.tokenizer_max_length + query_tokens
    assert embs.shape[1] == expected
    assert pad_masks.shape == (BATCH, expected)
    # Only the padded prompt tail is invalid.
    assert int(pad_masks.sum()) == expected - MASKED_LANG_TOKENS
    # Boundary tokens are text, not vision: they must not be marked visual, or the
    # deepstack scatter lands on the wrong rows.
    assert int(visual_masks.sum()) == NUM_CAMS * merged
    assert position_ids.shape == (3, BATCH, expected)
    assert len(deepstack) == len(model.vlm_config.vision_config.deepstack_visual_indexes)
    assert all(feature.shape[0] == NUM_CAMS * merged for feature in deepstack)


def test_sample_actions_shape_and_finiteness(model, observation):
    actions = _sample(model, observation)
    assert actions.shape == (BATCH, model.config.chunk_size, model.config.max_action_dim)
    assert torch.isfinite(actions).all()


def test_sample_actions_does_not_alias_the_caller_noise(model, observation):
    """Upstream integrates in place, so a reused noise buffer resumes from the
    previous request's chunk. A serving pipeline is exactly the caller that reuses
    buffers, so this is pinned rather than merely fixed."""
    reference = observation["noise"].clone()
    _sample(model, observation)
    assert torch.equal(observation["noise"], reference)


def test_moe_kernels_agree(model, observation):
    """``dense`` (the default, upstream's eager kernel) and ``gather`` (8x less
    arithmetic at top-4 of 32) are the same function of the same parameters.

    Written without assuming which one is the default: that has already flipped
    once, when ``gather`` turned out to be 3.7x slower on XPU.
    """
    blocks = [m for m in model.modules() if hasattr(m, "moe_implementation")]
    assert blocks, "the tiny model must instantiate at least one token-MoE block"
    original = [block.moe_implementation for block in blocks]

    def sample_with(implementation):
        for block in blocks:
            block.moe_implementation = implementation
        return _sample(model, observation, noise=observation["noise"].clone())

    try:
        dense = sample_with("dense")
        gather = sample_with("gather")
    finally:
        for block, implementation in zip(blocks, original, strict=True):
            block.moe_implementation = implementation
    assert torch.allclose(gather, dense, atol=1e-6, rtol=0)


def test_masked_camera_drops_its_tokens_and_its_grid(model, observation):
    """A camera can be absent per request. Its slots stay in the sequence (the
    prefix length is static, which is what makes the KV cache reusable) but must be
    masked out, contribute no visual tokens, and consume no rope grid."""
    merged = _merged_tokens(model)
    _, pad_full, _, _, visual_full, deepstack_full = _prefix(model, observation)

    img_masks = observation["img_masks"].clone()
    img_masks[0, 1] = False
    embs, pad_masks, _, _, visual_masks, deepstack = _prefix(model, observation, img_masks=img_masks)

    assert embs.shape[1] == pad_full.shape[1]
    assert int(pad_full.sum()) - int(pad_masks.sum()) == merged + 2
    assert int(visual_full.sum()) - int(visual_masks.sum()) == merged
    assert all(a.shape[0] - b.shape[0] == merged for a, b in zip(deepstack_full, deepstack))

    actions = _sample(model, observation, img_masks=img_masks, noise=observation["noise"].clone())
    assert torch.isfinite(actions).all()
    assert not torch.allclose(actions, _sample(model, observation, noise=observation["noise"].clone()))


def test_batch_rows_are_independent(model, observation):
    """Identical rows must produce identical chunks, and a row-specific camera mask
    must not disturb its neighbours' rope grids — the reference implementation
    iterates one grid iterator across the whole batch and gets this wrong."""
    batch = 3
    repeated = {key: value.expand(batch, *value.shape[1:]).contiguous() for key, value in observation.items()}
    actions = _sample(model, repeated)
    assert actions.shape[0] == batch
    assert torch.allclose(actions[0], actions[2], atol=1e-6, rtol=0)

    img_masks = torch.ones(2, NUM_CAMS, dtype=torch.bool)
    img_masks[1, 0] = False
    two = {key: value.expand(2, *value.shape[1:]).contiguous() for key, value in observation.items()}
    _, pad_masks, _, position_ids, _, _ = _prefix(model, two, img_masks=img_masks)
    assert pad_masks[0].sum() > pad_masks[1].sum()
    row_maxima = position_ids.amax(dim=(0, 2)).tolist()
    assert row_maxima[0] > row_maxima[1]


def test_load_weights_strips_prefix_and_drops_align_heads(model):
    """The module tree mirrors the checkpoint, so loading is a prefix strip plus
    the deliberate exclusion of the align heads (120.68 M parameters that are only
    executed during training)."""
    torch.manual_seed(2)
    target = LingbotVlaV2ForActionPrediction(_policy_config(), vlm_config=_vlm_config()).eval()

    donor = {f"model.{name}": tensor for name, tensor in model.state_dict().items()}
    donor["model.depth_align_head.projector.proj_in1.weight"] = torch.zeros(4, 4)
    donor["model.future_video_align_head.projector.norm_out.bias"] = torch.zeros(4)

    loaded = target.load_weights(donor.items())

    assert "depth_align_head.projector.proj_in1.weight" not in loaded
    assert "state_proj.weight" in loaded
    for name, tensor in model.state_dict().items():
        assert torch.equal(target.state_dict()[name], tensor), name


@pytest.mark.parametrize("defect", ["missing", "unexpected"])
def test_load_weights_rejects_checkpoint_mismatch(model, defect):
    target = LingbotVlaV2ForActionPrediction(_policy_config(), vlm_config=_vlm_config()).eval()
    donor = {f"model.{name}": tensor for name, tensor in model.state_dict().items()}
    if defect == "missing":
        del donor["model.state_proj.weight"]
    else:
        donor["model.not_a_real_weight"] = torch.zeros(1)

    with pytest.raises(ValueError, match=f"{defect}.*key|{defect}.*parameter"):
        target.load_weights(donor.items())
