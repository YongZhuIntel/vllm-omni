# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only LingBot-VLA 2.0 math kernel for vllm-omni.

Only the math that turns a robot observation into an action chunk. No serving or
request glue lives here — that is ``pipeline_lingbot_vla_v2.py``. Graded in fp32
against the upstream ``lingbotvla`` reference, per stage, by
``spikes/lingbot_vla_v2/phase1_parity.py``.

Shape of the computation
------------------------
A Qwen3-VL-4B backbone and a Qwen2-shaped action expert are walked **layer by
layer in lockstep**: at every one of the 36 layers both towers produce their own
q/k/v, the two are concatenated along the sequence axis, one joint attention is
taken over the union, and each tower consumes its own slice of the output. The
towers therefore share attention but not weights.

Inference runs that walk twice:

1. **prefix fill** — ``inputs_embeds=[prefix, None]``: images + language +
   task-query tokens go through the VLM half and leave a per-layer K/V cache.
2. **denoise** — ``inputs_embeds=[None, suffix]`` × ``num_steps``: the action
   expert cross-attends over ``cat(prefix_cache, suffix)`` and predicts a
   velocity. Euler-integrating those velocities from t=1 to t=0 yields the chunk.

Why we own the containers
-------------------------
Every module that touches an attention mask, a KV cache or a position id is
written out here; from ``transformers`` we instantiate only *leaf* modules
(attention projections, MLPs, RMSNorm, patch embed/merger, rotary embeddings)
so the tensor layout and the leaf math stay bit-identical to Qwen3-VL while the
version-dependent ``Model.forward`` machinery — mask preparation, cache classes,
``PreTrainedModel.post_init``, tied-weight bookkeeping — is out of the picture.
This is the same rationale as vllm-omni's π0 kernel, applied one level lower:
π0 keeps HF's top-level ``PaliGemmaForConditionalGeneration`` and pays for it
with weight-name shims that differ per transformers version. LingBot's Phase 0
spike needed nine such shims against transformers 5.8; owning the containers
removes seven of them outright.

Deliberate divergences from the upstream reference, each verified numerically:

* ``sample_actions`` no longer aliases its ``noise`` argument. Upstream does
  ``x_t = noise`` then ``x_t += dt * v_t``, denoising the caller's tensor in
  place; a pipeline that reuses a noise buffer silently starts request *n* from
  request *n-1*'s chunk.
* The MoE runs a **gather** kernel (each expert sees only its routed tokens)
  instead of upstream's dense path (every expert sees every token, zero-weighted
  afterwards). Same value up to fp32 summation order, 8× fewer FLOPs at top-4 of
  32. ``moe_implementation="dense"`` restores the reference path for debugging.
* The align *heads* (Perceiver resamplers + MoGe depth head, 120.68 M params,
  76 tensors) are not built. They only produce training-time alignment targets.
  The learned query *tables* they were trained with are still built and used —
  they are prefix tokens, so dropping them would change the prefix length.

This module deliberately imports nothing from ``vllm`` or ``vllm_omni`` so it can
be loaded and graded standalone, without the engine.
"""

from __future__ import annotations

import itertools
import logging
from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoConfig
from transformers.models.qwen3_vl import modeling_qwen3_vl as hf_qwen3vl
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLTextAttention,
    Qwen3VLTextMLP,
    Qwen3VLTextRMSNorm,
    Qwen3VLTextRotaryEmbedding,
    Qwen3VLVisionBlock,
    Qwen3VLVisionPatchEmbed,
    Qwen3VLVisionPatchMerger,
    Qwen3VLVisionRotaryEmbedding,
    apply_rotary_pos_emb,
)

from vllm_omni.diffusion.models.lingbot_vla_v2.config import LingbotVlaV2Config

logger = logging.getLogger(__name__)

# The additive mask value openpi/LingBot use in place of -inf. Keeping the exact
# constant matters: softmax over a fully-masked row must reproduce the reference
# bit-for-bit, and -inf would give NaN where -2.38e38 gives a uniform row.
BIG_NEG = -2.3819763e38

# Flow-matching time embedding band (upstream FlowMatching.embed_suffix).
TIME_MIN_PERIOD = 4e-3
TIME_MAX_PERIOD = 4.0


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------
def create_sinusoidal_pos_embedding(
    time: torch.Tensor,
    dimension: int,
    min_period: float,
    max_period: float,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Sine-cosine embedding of scalar flow-matching times ``(B,) -> (B, dimension)``."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")
    if time.ndim != 1:
        raise ValueError(f"time must be shape (batch_size,); got {tuple(time.shape)}")

    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=torch.float32, device=device)
    period = min_period * (max_period / min_period) ** fraction
    scaling_factor = 1.0 / period * 2 * torch.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def make_att_2d_masks(pad_masks: torch.Tensor, att_masks: torch.Tensor) -> torch.Tensor:
    """Expand big_vision-style 1D block masks into a 2D visibility matrix.

    ``att_masks`` marks where a new attention block starts; a token may attend to
    every valid token whose cumulative block index is ``<=`` its own. All-ones
    gives pure causal attention, all-zeros gives full bidirectional attention.
    """
    if att_masks.ndim != 2 or pad_masks.ndim != 2:
        raise ValueError(f"expected 2D masks; got {att_masks.ndim}D / {pad_masks.ndim}D")
    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


def eager_attention(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Grouped-query attention over ``[B, L, H, D]`` with a ``[B, Lq, Lk]`` bool mask.

    Portable by construction: no flash-attn, no flex_attention, no SDPA mask
    lowering. Upstream defaults to ``flex_cached``, which needs a CUDA-only
    ``torch.nn.attention.flex_attention`` build; this is the reference path both
    upstream and the export repo use to produce golden values.
    """
    bsize, q_len, num_att_heads, head_dim = query_states.shape
    num_kv_heads = key_states.shape[2]
    groups = num_att_heads // num_kv_heads

    if groups > 1:
        key_states = key_states.repeat_interleave(groups, dim=2)
        value_states = value_states.repeat_interleave(groups, dim=2)

    query = query_states.transpose(1, 2)  # [B, H, Lq, D]
    key = key_states.transpose(1, 2)  # [B, H, Lk, D]
    value = value_states.transpose(1, 2)  # [B, H, Lk, D]

    att_weights = torch.matmul(query, key.transpose(-1, -2)) * (head_dim**-0.5)
    att_weights = torch.where(attention_mask[:, None, :, :], att_weights, BIG_NEG)
    probs = F.softmax(att_weights, dim=-1).to(value.dtype)

    att_output = torch.matmul(probs, value)  # [B, H, Lq, D]
    att_output = att_output.transpose(1, 2).reshape(bsize, q_len, num_att_heads * head_dim)
    return att_output


class RMSNorm(nn.Module):
    """RMSNorm with the fp32 reduction the action expert was trained with.

    Matches upstream ``FixQwen2RMSNorm``: normalize in fp32, cast back, then
    scale. Only the expert's final ``norm`` uses it — every per-layer norm in the
    expert is an :class:`AdaRMSNorm` when ``adanorm_time`` is on.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self) -> str:
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class AdaRMSNorm(nn.Module):
    """RMSNorm + FiLM conditioned on the flow-matching timestep embedding.

    ``out = (1 + gamma(t)) * rmsnorm(x) * weight + beta(t)``. This is how the
    action expert learns "where in the denoising trajectory am I"; the VLM half
    has no such conditioning.

    Note ``cond`` is used in the *parameter* dtype, not forced to fp32 — upstream
    has both variants (``AdaRMSNorm`` / ``FixAdaRMSNorm``) and the released
    checkpoint's per-layer norms are the former.
    """

    def __init__(self, hidden_size: int, cond_dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.gamma = nn.Linear(cond_dim, hidden_size)
        self.beta = nn.Linear(cond_dim, hidden_size)

    def forward(self, hidden_states: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = self.weight * hidden_states

        gamma = self.gamma(cond).unsqueeze(1)  # [B, 1, H]
        beta = self.beta(cond).unsqueeze(1)  # [B, 1, H]
        hidden_states = (1 + gamma.to(torch.float32)) * hidden_states + beta.to(torch.float32)
        return hidden_states.to(input_dtype)


# ---------------------------------------------------------------------------
# Prefix layout
# ---------------------------------------------------------------------------
def prefix_query_segments(
    use_future_depth: bool,
    use_future_video: bool,
    use_future_video_cls: bool,
    use_future_video_patch: bool,
    future_video_share_future_depth_query: bool,
) -> tuple[str, ...]:
    """Prefix segment order after the image block.

    Query tokens always follow the language tokens; current-task queries precede
    future-task queries; ``future_depth`` stays last so tail-span mask surgery
    keeps working. For the released RoboTwin config this returns
    ``("language", "current_depth", "future_depth")`` — future-video shares the
    future-depth query, and the CLS query is disabled.
    """
    segments = ["language", "current_depth"]
    if use_future_video:
        if use_future_video_cls:
            segments.append("future_video_cls")
        if use_future_video_patch and not future_video_share_future_depth_query:
            segments.append("future_video")
    if use_future_depth:
        segments.append("future_depth")
    return tuple(segments)


def prefix_query_token_spans(
    prefix_len: int,
    num_task_tokens: int,
    segments: Iterable[str],
) -> dict[str, tuple[int, int]]:
    """``[start, end)`` of each non-language query segment inside the prefix."""
    counts = {
        "current_depth": num_task_tokens,
        "future_video_cls": 1,
        "future_video": num_task_tokens,
        "future_depth": num_task_tokens,
    }
    query_segments = [name for name in segments if name != "language"]
    cursor = prefix_len - sum(counts[name] for name in query_segments)
    spans: dict[str, tuple[int, int]] = {}
    for name in query_segments:
        spans[name] = (cursor, cursor + counts[name])
        cursor += counts[name]
    return spans


# ---------------------------------------------------------------------------
# Vision tower
# ---------------------------------------------------------------------------
class LingbotVisionTower(nn.Module):
    """Qwen3-VL ViT with a fixed forward and cacheable geometry.

    Differs from ``Qwen3VLVisionModel`` only in that (a) it is a plain
    ``nn.Module``, (b) its forward returns a plain tuple instead of a
    ``BaseModelOutputWithDeepstackFeatures``, and (c) the rope table, ``cu_seqlens``
    and interpolated position embeddings are cached per ``grid_thw``. Robot
    cameras have a fixed resolution, so that cache hits on every request after the
    first.
    """

    # Two pure-geometry helpers borrowed verbatim from HF. They read only
    # ``self.spatial_merge_size`` / ``self.rotary_pos_emb`` / ``self.pos_embed`` /
    # ``self.num_grid_per_side`` / ``self.config``, all constructed identically
    # below. Borrowing beats transcribing ~90 lines of interpolation arithmetic,
    # and the fp32 parity test is what guards against transformers changing them.
    rot_pos_emb = hf_qwen3vl.Qwen3VLVisionModel.rot_pos_emb
    fast_pos_embed_interpolate = hf_qwen3vl.Qwen3VLVisionModel.fast_pos_embed_interpolate

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_size = config.patch_size
        self.spatial_merge_unit = self.spatial_merge_size**2
        self.num_grid_per_side = int(config.num_position_embeddings**0.5)

        self.patch_embed = Qwen3VLVisionPatchEmbed(config=config)
        self.pos_embed = nn.Embedding(config.num_position_embeddings, config.hidden_size)
        self.rotary_pos_emb = Qwen3VLVisionRotaryEmbedding(config.hidden_size // config.num_heads // 2)
        self.blocks = nn.ModuleList([Qwen3VLVisionBlock(config) for _ in range(config.depth)])
        self.merger = Qwen3VLVisionPatchMerger(config=config, use_postshuffle_norm=False)

        self.deepstack_visual_indexes = list(config.deepstack_visual_indexes)
        self.deepstack_merger_list = nn.ModuleList(
            [
                Qwen3VLVisionPatchMerger(config=config, use_postshuffle_norm=True)
                for _ in range(len(self.deepstack_visual_indexes))
            ]
        )
        self._geometry_cache: dict[tuple, tuple] = {}

    def geometry(
        self, grid_thw: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor], torch.Tensor, list[int]]:
        """``(pos_embeds, (cos, sin), cu_seqlens, split_sizes)`` for a patch grid."""
        key = (tuple(map(tuple, grid_thw.tolist())), str(self.pos_embed.weight.device), self.pos_embed.weight.dtype)
        cached = self._geometry_cache.get(key)
        if cached is not None:
            return cached

        pos_embeds = self.fast_pos_embed_interpolate(grid_thw)
        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        seq_len = int(torch.prod(grid_thw, dim=1).sum().item())
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0, dtype=torch.int32
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
        split_sizes = (grid_thw.prod(-1) // self.spatial_merge_unit).tolist()

        value = (pos_embeds, position_embeddings, cu_seqlens, split_sizes)
        self._geometry_cache[key] = value
        return value

    def forward(
        self, pixel_values: torch.Tensor, grid_thw: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[int]]:
        """``(merged_embeds, deepstack_embeds, split_sizes)``.

        ``merged_embeds`` is ``[total_merged_tokens, out_hidden_size]`` covering
        every image in ``grid_thw`` back to back; ``split_sizes`` says how to cut
        it back into per-image chunks.
        """
        pos_embeds, position_embeddings, cu_seqlens, split_sizes = self.geometry(grid_thw)

        hidden_states = self.patch_embed(pixel_values)
        hidden_states = hidden_states + pos_embeds.to(hidden_states.dtype)
        hidden_states = hidden_states.reshape(hidden_states.shape[0], -1)

        cos, sin = position_embeddings
        position_embeddings = (cos.to(hidden_states.dtype), sin.to(hidden_states.dtype))

        deepstack_features: list[torch.Tensor] = []
        for layer_num, block in enumerate(self.blocks):
            hidden_states = block(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
            )
            if layer_num in self.deepstack_visual_indexes:
                merger = self.deepstack_merger_list[self.deepstack_visual_indexes.index(layer_num)]
                deepstack_features.append(merger(hidden_states))

        return self.merger(hidden_states), deepstack_features, split_sizes


# ---------------------------------------------------------------------------
# VLM text tower
# ---------------------------------------------------------------------------
class VlmDecoderLayer(nn.Module):
    """One Qwen3-VL text layer, split into its two halves around joint attention.

    ``compute_qkv`` runs pre-norm + q/k/v projections (including Qwen3's q/k
    RMSNorms); ``apply_attention`` runs o_proj on this tower's slice of the joint
    attention output, the residual, post-norm and the MLP. The joint walk in
    :meth:`LingbotJointModel.forward` calls them in that order with the shared
    attention in between.
    """

    def __init__(self, config, layer_idx: int) -> None:
        super().__init__()
        self.self_attn = Qwen3VLTextAttention(config=config, layer_idx=layer_idx)
        self.mlp = Qwen3VLTextMLP(config)
        self.input_layernorm = Qwen3VLTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3VLTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def compute_qkv(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden_states = hidden_states.to(self.self_attn.q_proj.weight.dtype)
        hidden_states = self.input_layernorm(hidden_states)
        shape = (*hidden_states.shape[:-1], -1, self.self_attn.head_dim)
        query = self.self_attn.q_norm(self.self_attn.q_proj(hidden_states).view(shape))
        key = self.self_attn.k_norm(self.self_attn.k_proj(hidden_states).view(shape))
        value = self.self_attn.v_proj(hidden_states).view(shape)
        return query, key, value

    def apply_attention(
        self, hidden_states: torch.Tensor, att_output: torch.Tensor, start: int, end: int
    ) -> torch.Tensor:
        param_dtype = self.self_attn.o_proj.weight.dtype
        hidden_states = hidden_states.to(param_dtype)
        out_emb = self.self_attn.o_proj(att_output[:, start:end].to(param_dtype))
        out_emb = out_emb + hidden_states
        residual = out_emb
        out_emb = self.post_attention_layernorm(out_emb)
        out_emb = self.mlp(out_emb)
        return out_emb + residual


class VlmTextTower(nn.Module):
    """``embed_tokens`` + the 36 text layers + the final norm + mrope.

    Named to mirror the checkpoint (``qwenvl.model.language_model.*``) so weight
    loading is a prefix strip rather than a rename table.
    """

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList([VlmDecoderLayer(config, idx) for idx in range(config.num_hidden_layers)])
        self.norm = Qwen3VLTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3VLTextRotaryEmbedding(config=config)

    @staticmethod
    def deepstack_process(
        hidden_states: torch.Tensor, visual_pos_masks: torch.Tensor, visual_embeds: torch.Tensor
    ) -> torch.Tensor:
        """Add a deepstack ViT feature onto the visual token rows in place."""
        visual_pos_masks = visual_pos_masks.to(hidden_states.device)
        visual_embeds = visual_embeds.to(hidden_states.device, hidden_states.dtype)
        hidden_states = hidden_states.clone()
        hidden_states[visual_pos_masks, :] = hidden_states[visual_pos_masks, :] + visual_embeds
        return hidden_states


class Qwen3VLInner(nn.Module):
    """Container matching the checkpoint's ``qwenvl.model.*`` level."""

    def __init__(self, vlm_config) -> None:
        super().__init__()
        self.visual = LingbotVisionTower(vlm_config.vision_config)
        self.language_model = VlmTextTower(vlm_config.text_config)


class Qwen3VLTower(nn.Module):
    """Container matching the checkpoint's ``qwenvl.*`` level.

    The released checkpoint has no ``lm_head`` (upstream deletes it before
    saving), so none is built.
    """

    def __init__(self, vlm_config) -> None:
        super().__init__()
        self.config = vlm_config
        self.model = Qwen3VLInner(vlm_config)


# ---------------------------------------------------------------------------
# Action expert
# ---------------------------------------------------------------------------
class ExpertAttention(nn.Module):
    """q/k/v/o projections of the action expert.

    Written out rather than borrowed from ``Qwen2Attention`` because the expert
    never uses HF's attention forward at all: the joint walk owns rope, masking
    and the KV cache. What is left is four Linears whose shapes the checkpoint
    fixes (q/k/v carry bias, o does not — the Qwen2 convention).
    """

    def __init__(self, hidden_size: int, num_heads: int, num_kv_heads: int, head_dim: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=True)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=True)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=True)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)


class ExpertMLP(nn.Module):
    """Dense SwiGLU MLP, for expert layers that are not token-MoE."""

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class SharedExpertMLP(nn.Module):
    """Always-on SwiGLU branch added to every token's MoE output."""

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class GroupedExperts(nn.Module):
    """The E routed experts, stored grouped so the release state dict loads as-is.

    Shapes follow ``nn.Linear.weight`` = ``[out, in]`` with an expert axis in
    front::

        gate_proj [E, intermediate, hidden]
        up_proj   [E, intermediate, hidden]
        down_proj [E, hidden, intermediate]

    Two kernels over the same parameters:

    ``gather`` (default)
        Each expert runs on exactly the tokens routed to it. At top-4 of 32 this
        is 8× less arithmetic than ``dense``.

    ``dense``
        Every expert runs on every token; the routing weights (zero for
        unselected pairs) then contract the expert axis away. This is upstream's
        eager path and therefore what the fp32 golden was produced with, so it
        stays available for exact parity work. The two agree up to fp32
        summation order.
    """

    def __init__(self, num_experts: int, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Parameter(torch.empty(num_experts, intermediate_size, hidden_size))
        self.up_proj = nn.Parameter(torch.empty(num_experts, intermediate_size, hidden_size))
        self.down_proj = nn.Parameter(torch.empty(num_experts, hidden_size, intermediate_size))

    def forward_gather(
        self, hidden_flat: torch.Tensor, routing_weights: torch.Tensor, selected_experts: torch.Tensor
    ) -> torch.Tensor:
        out = torch.zeros_like(hidden_flat)
        # [T, K] -> which (token, slot) pairs picked each expert.
        for expert in range(self.num_experts):
            token_idx, slot_idx = torch.where(selected_experts == expert)
            if token_idx.numel() == 0:
                continue
            x = hidden_flat[token_idx]
            inter = F.silu(F.linear(x, self.gate_proj[expert])) * F.linear(x, self.up_proj[expert])
            y = F.linear(inter, self.down_proj[expert])
            out.index_add_(0, token_idx, y * routing_weights[token_idx, slot_idx].unsqueeze(-1))
        return out

    def forward_dense(
        self, hidden_flat: torch.Tensor, routing_weights: torch.Tensor, selected_experts: torch.Tensor
    ) -> torch.Tensor:
        gate = torch.einsum("td,eid->eti", hidden_flat, self.gate_proj)
        up = torch.einsum("td,eid->eti", hidden_flat, self.up_proj)
        expert_out = torch.einsum("eti,edi->etd", F.silu(gate) * up, self.down_proj)
        expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts).to(routing_weights.dtype)
        weights = (expert_mask * routing_weights.unsqueeze(-1)).sum(dim=1)  # [T, E]
        return torch.einsum("etd,te->td", expert_out, weights.to(expert_out.dtype))


class TokenMoeBlock(nn.Module):
    """Per-token top-k routing over ``GroupedExperts`` plus a shared expert.

    Router details that matter for parity: the gate runs in true fp32 with
    autocast disabled (bf16 logits flip top-k on near-ties), scores are
    ``sigmoid`` rather than softmax, the loss-free balancing bias
    ``e_score_correction_bias`` biases *selection* but not the *weights*, and the
    normalized top-k weights are then scaled by ``routed_scaling_factor``.
    """

    def __init__(self, config: LingbotVlaV2Config) -> None:
        super().__init__()
        hidden_size = config.expert_hidden_size
        self.num_experts = config.token_num_experts
        self.top_k = config.token_top_k
        self.norm_topk_prob = True
        self.router_activation = config.router_activation
        self.routed_scaling_factor = config.routed_scaling_factor
        self.moe_implementation = config.moe_implementation

        # Persistent: it is a trained tensor in the release checkpoint. It is
        # frozen at inference (``bias_update_speed`` must be 0).
        self.register_buffer("e_score_correction_bias", torch.zeros(self.num_experts), persistent=True)
        self.gate = nn.Linear(hidden_size, self.num_experts, bias=False)
        self.experts = GroupedExperts(self.num_experts, hidden_size, config.token_moe_intermediate_size)
        self.shared_expert = SharedExpertMLP(hidden_size, config.token_shared_intermediate_size)
        self.use_shared_expert_gate = config.use_shared_expert_gate
        if self.use_shared_expert_gate:
            self.shared_expert_gate = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_dim = hidden_states.shape
        hidden_flat = hidden_states.reshape(-1, hidden_dim)

        with torch.amp.autocast(hidden_flat.device.type, enabled=False):
            router_logits = F.linear(hidden_flat.float(), self.gate.weight.float())

        if self.router_activation == "sigmoid":
            routing_scores = router_logits.sigmoid()
        else:
            routing_scores = F.softmax(router_logits, dim=1, dtype=torch.float)

        scores_for_choice = routing_scores + self.e_score_correction_bias.unsqueeze(0)
        _, selected_experts = torch.topk(scores_for_choice, self.top_k, dim=-1)
        routing_weights = routing_scores.gather(1, selected_experts)
        if self.norm_topk_prob:
            routing_weights = routing_weights / (routing_weights.sum(dim=-1, keepdim=True) + 1e-20)
        if self.routed_scaling_factor != 1.0:
            routing_weights = routing_weights * self.routed_scaling_factor
        routing_weights = routing_weights.to(hidden_states.dtype)

        if self.moe_implementation == "dense":
            routed = self.experts.forward_dense(hidden_flat, routing_weights, selected_experts)
        else:
            routed = self.experts.forward_gather(hidden_flat, routing_weights, selected_experts)

        shared = self.shared_expert(hidden_flat)
        if self.use_shared_expert_gate:
            shared = F.sigmoid(self.shared_expert_gate(hidden_flat)) * shared
        return (routed.to(hidden_flat.dtype) + shared).reshape(batch_size, seq_len, hidden_dim)


class ExpertDecoderLayer(nn.Module):
    """One action-expert layer, split around joint attention like its VLM twin.

    The only structural difference from :class:`VlmDecoderLayer` is that both
    layernorms are time-conditioned (``AdaRMSNorm``) and the MLP may be a token
    MoE. The expert has no q/k norms.
    """

    def __init__(self, config: LingbotVlaV2Config, layer_idx: int) -> None:
        super().__init__()
        hidden_size = config.expert_hidden_size
        self.self_attn = ExpertAttention(
            hidden_size,
            config.expert_num_attention_heads,
            config.expert_num_key_value_heads,
            config.expert_head_dim,
        )
        moe_layers = config.token_moe_layers or []
        if config.use_moe and layer_idx in moe_layers:
            self.mlp = TokenMoeBlock(config)
        else:
            self.mlp = ExpertMLP(hidden_size, config.expert_intermediate_size)

        if config.adanorm_time:
            self.input_layernorm = AdaRMSNorm(hidden_size, hidden_size, eps=config.expert_rms_norm_eps)
            self.post_attention_layernorm = AdaRMSNorm(hidden_size, hidden_size, eps=config.expert_rms_norm_eps)
        else:
            self.input_layernorm = RMSNorm(hidden_size, eps=config.expert_rms_norm_eps)
            self.post_attention_layernorm = RMSNorm(hidden_size, eps=config.expert_rms_norm_eps)
        self.adanorm_time = config.adanorm_time

    def _norm(self, norm: nn.Module, hidden_states: torch.Tensor, ada_cond: torch.Tensor | None) -> torch.Tensor:
        if self.adanorm_time:
            if ada_cond is None:
                raise ValueError("adanorm_time=True requires the timestep embedding ada_cond.")
            return norm(hidden_states, ada_cond)
        return norm(hidden_states)

    def compute_qkv(
        self, hidden_states: torch.Tensor, ada_cond: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        param_dtype = self.self_attn.q_proj.weight.dtype
        hidden_states = hidden_states.to(param_dtype)
        if ada_cond is not None:
            ada_cond = ada_cond.to(param_dtype)
        hidden_states = self._norm(self.input_layernorm, hidden_states, ada_cond)
        shape = (*hidden_states.shape[:-1], -1, self.self_attn.head_dim)
        query = self.self_attn.q_proj(hidden_states).view(shape)
        key = self.self_attn.k_proj(hidden_states).view(shape)
        value = self.self_attn.v_proj(hidden_states).view(shape)
        return query, key, value

    def apply_attention(
        self,
        hidden_states: torch.Tensor,
        att_output: torch.Tensor,
        start: int,
        end: int,
        ada_cond: torch.Tensor | None,
    ) -> torch.Tensor:
        param_dtype = self.self_attn.o_proj.weight.dtype
        hidden_states = hidden_states.to(param_dtype)
        if ada_cond is not None:
            ada_cond = ada_cond.to(param_dtype)
        out_emb = self.self_attn.o_proj(att_output[:, start:end].to(param_dtype))
        out_emb = out_emb + hidden_states
        residual = out_emb
        out_emb = self._norm(self.post_attention_layernorm, out_emb, ada_cond)
        out_emb = self.mlp(out_emb)
        return out_emb + residual


class ExpertTower(nn.Module):
    """The action expert's ``layers`` + final ``norm``.

    No ``embed_tokens`` (the expert consumes state/action embeddings, never token
    ids) and no ``rotary_emb``: mrope for *both* towers comes from the VLM's
    rotary embedding so the two share one position basis.
    """

    def __init__(self, config: LingbotVlaV2Config) -> None:
        super().__init__()
        self.layers = nn.ModuleList([ExpertDecoderLayer(config, idx) for idx in range(config.expert_num_layers)])
        if config.final_norm_adanorm:
            self.norm = AdaRMSNorm(config.expert_hidden_size, config.expert_hidden_size, eps=config.expert_rms_norm_eps)
        else:
            self.norm = RMSNorm(config.expert_hidden_size, eps=config.expert_rms_norm_eps)
        self.final_norm_adanorm = config.final_norm_adanorm


class ActionExpert(nn.Module):
    """Container matching the checkpoint's ``qwen_expert.*`` level."""

    def __init__(self, config: LingbotVlaV2Config) -> None:
        super().__init__()
        self.model = ExpertTower(config)


# ---------------------------------------------------------------------------
# Joint transformer
# ---------------------------------------------------------------------------
class LingbotJointModel(nn.Module):
    """The lockstep VLM + action-expert walk, with mrope and the K/V cache.

    ``forward`` is the single place attention happens. ``inputs_embeds`` is a
    two-slot list: slot 0 is the VLM prefix, slot 1 is the action-expert suffix,
    and either may be ``None``:

    * ``[prefix, None]`` with ``fill_kv_cache=True`` → prefix pass, fills the cache.
    * ``[None, suffix]`` with ``fill_kv_cache=False`` → one denoise step, reads it.
    """

    def __init__(self, config: LingbotVlaV2Config, vlm_config) -> None:
        super().__init__()
        self.config = config
        self.qwenvl = Qwen3VLTower(vlm_config)
        self.qwen_expert = ActionExpert(config)

        vlm_layers = vlm_config.text_config.num_hidden_layers
        if vlm_layers != config.expert_num_layers:
            raise ValueError(
                "The VLM and the action expert must have the same depth (they are walked in "
                f"lockstep); got vlm={vlm_layers}, expert={config.expert_num_layers}."
            )
        self.num_layers = vlm_layers

    # -- embedding helpers ------------------------------------------------
    def embed_image(
        self, pixel_values: torch.Tensor, grid_thw: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """``(image_embeds [N, tokens, D], deepstack [3 x [N, tokens, D]])``."""
        merged, deepstack, split_sizes = self.qwenvl.model.visual(pixel_values, grid_thw)
        image_embeds = torch.stack(list(torch.split(merged, split_sizes)), dim=0)
        deepstack_embeds = [torch.stack(list(torch.split(feature, split_sizes)), dim=0) for feature in deepstack]
        return image_embeds, deepstack_embeds

    def embed_language_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.qwenvl.model.language_model.embed_tokens(tokens)

    def embed_special_token(
        self, token_id: int, batch: int, count: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """``[batch, count, 1, D]`` copies of one vocabulary embedding."""
        token = torch.tensor([token_id], device=device, dtype=torch.long)
        emb = self.embed_language_tokens(token).to(dtype=dtype)
        return emb.view(1, 1, 1, -1).expand(batch, count, 1, -1)

    def apply_mrope(
        self, query_states: torch.Tensor, key_states: torch.Tensor, position_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Interleaved 3-axis mrope, shared by both towers.

        ``unsqueeze_dim=2`` because our q/k are ``[B, L, H, D]``, not HF's
        ``[B, H, L, D]``.
        """
        rotary_emb = self.qwenvl.model.language_model.rotary_emb
        cos, sin = rotary_emb(query_states, position_ids)
        return apply_rotary_pos_emb(query_states, key_states, cos, sin, unsqueeze_dim=2)

    # -- the walk ---------------------------------------------------------
    def forward(
        self,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        inputs_embeds: list[torch.Tensor | None],
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        fill_kv_cache: bool = False,
        ada_cond: torch.Tensor | None = None,
        visual_pos_masks: torch.Tensor | None = None,
        deepstack_visual_embeds: list[torch.Tensor] | None = None,
    ) -> tuple[list[torch.Tensor | None], list[tuple[torch.Tensor, torch.Tensor]]]:
        towers = [self.qwenvl.model.language_model, self.qwen_expert.model]
        if fill_kv_cache:
            past_key_values = []

        for layer_idx in range(self.num_layers):
            queries, keys, values = [], [], []
            for tower_idx, hidden_states in enumerate(inputs_embeds):
                if hidden_states is None:
                    continue
                layer = towers[tower_idx].layers[layer_idx]
                if tower_idx == 1:
                    q, k, v = layer.compute_qkv(hidden_states, ada_cond)
                else:
                    q, k, v = layer.compute_qkv(hidden_states)
                # Attention is taken in fp32 for both towers regardless of the
                # weight dtype: the joint softmax spans 286 prefix tokens plus the
                # suffix and is where a bf16 run loses the most accuracy.
                queries.append(q.float())
                keys.append(k.float())
                values.append(v.float())

            query_states = torch.cat(queries, dim=1)
            key_states = torch.cat(keys, dim=1)
            value_states = torch.cat(values, dim=1)
            query_states, key_states = self.apply_mrope(query_states, key_states, position_ids)

            if fill_kv_cache:
                past_key_values.append((key_states, value_states))
            else:
                cached_key, cached_value = past_key_values[layer_idx]
                key_states = torch.cat([cached_key, key_states], dim=1)
                value_states = torch.cat([cached_value, value_states], dim=1)

            att_output = eager_attention(query_states, key_states, value_states, attention_mask)

            outputs, start = [], 0
            for tower_idx, hidden_states in enumerate(inputs_embeds):
                if hidden_states is None:
                    outputs.append(None)
                    continue
                end = start + hidden_states.shape[1]
                layer = towers[tower_idx].layers[layer_idx]
                if tower_idx == 1:
                    out_emb = layer.apply_attention(hidden_states, att_output, start, end, ada_cond)
                else:
                    out_emb = layer.apply_attention(hidden_states, att_output, start, end)
                    if deepstack_visual_embeds is not None and layer_idx < len(deepstack_visual_embeds):
                        out_emb = VlmTextTower.deepstack_process(
                            out_emb, visual_pos_masks, deepstack_visual_embeds[layer_idx]
                        )
                outputs.append(out_emb)
                start = end
            inputs_embeds = outputs

        final: list[torch.Tensor | None] = []
        for tower_idx, hidden_states in enumerate(inputs_embeds):
            if hidden_states is None:
                final.append(None)
            elif tower_idx == 1 and self.qwen_expert.model.final_norm_adanorm:
                final.append(self.qwen_expert.model.norm(hidden_states, ada_cond))
            else:
                final.append(towers[tower_idx].norm(hidden_states))
        return final, past_key_values


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------
class LingbotVlaV2ForActionPrediction(nn.Module):
    """LingBot-VLA 2.0: multi-camera observation + instruction + state → action chunk.

    Entry point is :meth:`sample_actions`. Everything it needs beyond the weights
    is fixed by :class:`LingbotVlaV2Config`; the caller supplies already-processed
    tensors (see ``processor.py``), not raw images.
    """

    def __init__(self, config: LingbotVlaV2Config, vlm_config=None) -> None:
        super().__init__()
        self.config = config
        if vlm_config is None:
            vlm_config = AutoConfig.from_pretrained(config.qwen3vl_path)
        if config.vocab_size:
            vlm_config.text_config.vocab_size = config.vocab_size
        vlm_config.text_config._attn_implementation = "eager"
        vlm_config.vision_config._attn_implementation = config.vit_attn_implementation
        self.vlm_config = vlm_config

        self.qwenvl_with_expert = LingbotJointModel(config, vlm_config)

        proj_width = config.expert_hidden_size
        self.proj_width = proj_width
        self.state_proj = nn.Linear(config.max_state_dim, proj_width)
        self.action_in_proj = nn.Linear(config.max_action_dim, proj_width)
        self.action_out_proj = nn.Linear(proj_width, config.max_action_dim)
        self.action_time_mlp_in = nn.Linear(proj_width * 2, proj_width)
        self.action_time_mlp_out = nn.Linear(proj_width, proj_width)

        # Learned task-query tables. These are prefix *tokens*, not heads: each
        # table is mean-pooled from num_backbone_tokens down to num_task_tokens
        # and appended to the prefix, so their presence sets the prefix length.
        align_shape = (config.num_backbone_tokens, config.align_dim)
        self.depth_align_embs = nn.Parameter(torch.zeros(align_shape))
        if config.use_current_video_patch:
            self.current_video_align_embs = nn.Parameter(torch.zeros(align_shape))
        if config.use_current_shared_task_proj:
            self.current_shared_task_proj = nn.Linear(config.align_dim * 2, config.align_dim)
        if config.use_future_depth:
            self.future_depth_align_embs = nn.Parameter(torch.zeros(align_shape))
        if config.use_future_video and config.use_future_video_patch:
            self.future_video_align_embs = nn.Parameter(torch.zeros(align_shape))
        if config.use_shared_future_task_proj:
            self.future_shared_task_proj = nn.Linear(config.align_dim * 2, config.align_dim)
        if config.use_future_video_cls:
            self.future_video_cls_align_emb = nn.Embedding(1, config.align_dim)

        self.prefix_segments = prefix_query_segments(
            use_future_depth=config.use_future_depth,
            use_future_video=config.use_future_video,
            use_future_video_cls=config.use_future_video_cls,
            use_future_video_patch=config.use_future_video_patch,
            future_video_share_future_depth_query=config.future_video_share_future_depth_query,
        )

    # -- prefix -----------------------------------------------------------
    def _pool_align_tokens(self, table: torch.Tensor) -> torch.Tensor:
        """``[num_backbone_tokens, D] -> [num_task_tokens, D]`` by mean pooling.

        The table was trained as one query per backbone patch; the prefix carries
        a pooled summary. Pooling is over ``num_backbone_tokens // num_task_tokens``
        *strided* groups, matching upstream's ``view(T, N // T, D).mean(1)``.
        """
        num_task_tokens = self.config.num_task_tokens
        return table.view(num_task_tokens, table.shape[0] // num_task_tokens, table.shape[1]).mean(dim=1)

    def _current_task_query(self) -> torch.Tensor:
        query = self._pool_align_tokens(self.depth_align_embs)
        if (
            self.config.use_future_video
            and self.config.use_current_video_patch
            and self.config.use_current_shared_task_proj
        ):
            video_query = self._pool_align_tokens(self.current_video_align_embs)
            query = self.current_shared_task_proj(torch.cat([query, video_query], dim=-1))
        return query

    def _future_task_query(self) -> torch.Tensor:
        query = self._pool_align_tokens(self.future_depth_align_embs)
        if (
            self.config.use_future_video
            and self.config.use_future_video_patch
            and self.config.future_video_share_future_depth_query
            and self.config.use_shared_future_task_proj
        ):
            video_query = self._pool_align_tokens(self.future_video_align_embs)
            query = self.future_shared_task_proj(torch.cat([query, video_query], dim=-1))
        return query

    def build_prefix_position_ids(
        self,
        image_token_masks: torch.Tensor,
        attention_mask: torch.Tensor,
        image_grid_thw: torch.Tensor,
        img_masks: torch.Tensor,
    ) -> torch.Tensor:
        """3-axis mrope position ids for the static LingBot prefix.

        A restriction of ``Qwen3VLModel.get_rope_index`` to what this prefix can
        contain — images then text, never video, one grid per visible camera in
        order. Reimplemented rather than called because HF changed the signature
        between the version upstream pins (4.57, modality derived internally) and
        the one vllm-omni ships (5.x, modality passed in as ``mm_token_type_ids``);
        that was the only *behavioural* incompatibility Phase 0 found, and it
        feeds every position id in the model.

        Takes the modality directly as ``image_token_masks`` instead of
        reconstructing it from token ids the way HF does. Only the *inner* patch
        tokens are images — a camera's ``<vision_start>``/``<vision_end>`` and
        every query token count as text — and the caller already knows exactly
        which rows those are, so fabricating placeholder ids to re-derive it would
        only add a way to get it wrong.

        Text advances position by one per token on all three axes. An image
        advances it by ``max(h, w) // spatial_merge_size`` in total while laying
        out ``(t, h, w)`` indices across its tokens, so a 16x16 grid at merge 2
        costs 8 positions rather than 64. Positions masked out by
        ``attention_mask`` are left at 0, as in HF.

        ``image_grid_thw`` is ``(B, num_cameras, 3)`` and ``img_masks`` is
        ``(B, num_cameras)``: grids are consumed per batch row and only for that
        row's *visible* cameras, since an invisible camera contributes no tokens.
        """
        spatial_merge_size = self.vlm_config.vision_config.spatial_merge_size
        device = image_token_masks.device

        position_ids = torch.zeros(3, *image_token_masks.shape, dtype=torch.long, device=device)
        for batch_idx, row_masks in enumerate(image_token_masks):
            keep = attention_mask[batch_idx].bool()
            is_image = row_masks[keep].tolist()

            grids = iter(image_grid_thw[batch_idx][img_masks[batch_idx]])
            current_pos = 0
            pieces = []
            for image_run, group in itertools.groupby(is_image):
                length = len(list(group))
                if not image_run:
                    pieces.append(torch.arange(length, device=device).view(1, -1).expand(3, -1) + current_pos)
                    current_pos += length
                else:
                    grid = next(grids)
                    pieces.append(self._vision_position_ids(current_pos, grid, spatial_merge_size, device))
                    current_pos += max(int(grid[1]), int(grid[2])) // spatial_merge_size
            position_ids[:, batch_idx, keep] = torch.cat(pieces, dim=1).reshape(3, -1).to(position_ids.dtype)
        return position_ids

    @staticmethod
    def _vision_position_ids(
        start_position: int, grid_thw: torch.Tensor, spatial_merge_size: int, device: torch.device
    ) -> torch.Tensor:
        """``(3, t*h*w / merge^2)`` temporal/height/width indices for one image."""
        grid_t = int(grid_thw[0])
        grid_h = int(grid_thw[1]) // spatial_merge_size
        grid_w = int(grid_thw[2]) // spatial_merge_size

        position_temporal = torch.arange(grid_t, device=device)
        position_width = torch.arange(grid_w, device=device) + start_position
        position_height = torch.arange(grid_h, device=device) + start_position

        position_width = position_width.repeat(grid_h * grid_t)
        position_height = position_height.repeat_interleave(grid_w).repeat(grid_t)
        position_temporal = position_temporal.repeat_interleave(grid_h * grid_w) + start_position
        return torch.stack([position_temporal, position_height, position_width], dim=0)

    def embed_prefix(
        self,
        images: torch.Tensor,
        img_masks: torch.Tensor,
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        """Assemble the VLM prefix.

        Layout, per camera: ``<vision_start> patch×64 <vision_end>``; then the
        padded language block; then the task-query segments. For the released
        3-camera / 224px / 72-token config that is ``3*66 + 72 + 8 + 8 = 286``.

        Returns ``(embs, pad_masks, att_masks, position_ids, visual_pos_masks,
        deepstack_embeds)``.
        """
        if image_grid_thw is None:
            raise ValueError("LingBot-VLA 2.0 needs image_grid_thw from the Qwen3-VL image processor.")

        bsize = images.shape[0]
        num_images = images.shape[1]
        device = images.device

        # ``(N, 3)`` is accepted as shorthand for a single request; anything else
        # must be per-(request, camera) or the vision tower would silently slice
        # one request's patches with another's grid.
        if image_grid_thw.ndim == 2:
            image_grid_thw = image_grid_thw.unsqueeze(0).expand(bsize, -1, -1)
        if image_grid_thw.shape[:2] != (bsize, num_images):
            raise ValueError(
                f"image_grid_thw must be (batch, num_cameras, 3) matching images "
                f"({bsize}, {num_images}, ...); got {tuple(image_grid_thw.shape)}."
            )
        if img_masks.ndim == 1:
            img_masks = img_masks.unsqueeze(0)
        img_masks = img_masks.bool()

        flat_images = images.reshape(bsize * num_images, *images.shape[2:])
        flat_grid_thw = image_grid_thw.reshape(-1, 3)

        img_emb, deepstack_embs = self.qwenvl_with_expert.embed_image(flat_images, flat_grid_thw)
        embed_dtype = img_emb.dtype
        num_patch = img_emb.shape[1]
        img_emb = img_emb.reshape(bsize, num_images, num_patch, -1)
        deepstack_embs = [x.reshape(bsize, num_images, num_patch, -1) for x in deepstack_embs]

        cfg = self.vlm_config
        start_emb = self.qwenvl_with_expert.embed_special_token(
            cfg.vision_start_token_id, bsize, num_images, device, embed_dtype
        )
        end_emb = self.qwenvl_with_expert.embed_special_token(
            cfg.vision_end_token_id, bsize, num_images, device, embed_dtype
        )
        img_chunks = torch.cat([start_emb, img_emb, end_emb], dim=2)
        image_token_len = num_patch + 2

        image_pad_masks = img_masks[:, :, None].expand(bsize, num_images, image_token_len)
        # Only the inner patch tokens are "visual". The two boundary tokens count
        # as text: they receive no deepstack feature and advance mrope like text.
        image_visual_masks = torch.zeros_like(image_pad_masks)
        image_visual_masks[:, :, 1 : 1 + num_patch] = img_masks[:, :, None].expand(bsize, num_images, num_patch)

        parts = [img_chunks.reshape(bsize, num_images * image_token_len, -1)]
        masks = [image_pad_masks.reshape(bsize, -1)]
        visual_masks = [image_visual_masks.reshape(bsize, -1)]

        lang_emb = self.qwenvl_with_expert.embed_language_tokens(lang_tokens).to(dtype=embed_dtype)
        query_pad_masks = torch.ones(bsize, self.config.num_task_tokens, device=device, dtype=lang_masks.dtype)

        def append(tokens: torch.Tensor, token_masks: torch.Tensor) -> None:
            parts.append(tokens)
            masks.append(token_masks)
            visual_masks.append(torch.zeros_like(token_masks))

        for segment in self.prefix_segments:
            if segment == "language":
                append(lang_emb, lang_masks)
            elif segment == "current_depth":
                query = self._current_task_query().to(device=device, dtype=embed_dtype)
                append(query.expand(bsize, -1, -1), query_pad_masks)
            elif segment == "future_depth":
                query = self._future_task_query().to(device=device, dtype=embed_dtype)
                append(query.expand(bsize, -1, -1), query_pad_masks)
            elif segment == "future_video":
                query = self._pool_align_tokens(self.future_video_align_embs).to(device=device, dtype=embed_dtype)
                append(query.expand(bsize, -1, -1), query_pad_masks)
            elif segment == "future_video_cls":
                cls_emb = self.future_video_cls_align_emb.weight.to(device=device, dtype=embed_dtype)
                cls_masks = torch.ones(bsize, 1, device=device, dtype=lang_masks.dtype)
                append(cls_emb.expand(bsize, -1, -1), cls_masks)
            else:  # pragma: no cover - prefix_query_segments emits nothing else
                raise ValueError(f"Unsupported prefix query segment: {segment}")

        embs = torch.cat(parts, dim=1)
        pad_masks = torch.cat(masks, dim=1).bool()
        visual_pos_masks = torch.cat(visual_masks, dim=1).bool()

        fill_value = self.config.vlm_causal
        att_masks = torch.full((bsize, embs.shape[1]), fill_value, device=device, dtype=torch.bool)

        position_ids = self.build_prefix_position_ids(visual_pos_masks, pad_masks, image_grid_thw, img_masks)

        # Deepstack features are indexed by the flat visual-token mask, so they
        # must be filtered down to the visible cameras' patch tokens only.
        visible_patches = img_masks[:, :, None].expand(bsize, num_images, num_patch)
        deepstack_visual_embeds = [feature[visible_patches] for feature in deepstack_embs]

        return embs, pad_masks, att_masks, position_ids, visual_pos_masks, deepstack_visual_embeds

    # -- suffix -----------------------------------------------------------
    def embed_suffix(
        self, state: torch.Tensor, noisy_actions: torch.Tensor, timestep: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """``(time_emb, embs, pad_masks, att_masks)`` for the action-expert suffix.

        One state token followed by ``chunk_size`` action tokens. ``att_masks``
        is ``[True, True, False, ...]``: the state token opens a block (so the
        prefix cannot see the suffix), the first action token opens another (so
        the state token cannot see the actions), and the remaining action tokens
        share that block, i.e. attend to each other bidirectionally.
        """
        bsize = state.shape[0]
        device = state.device
        dtype = state.dtype

        state_emb = self.state_proj(state)
        time_emb = create_sinusoidal_pos_embedding(
            timestep, self.proj_width, min_period=TIME_MIN_PERIOD, max_period=TIME_MAX_PERIOD, device=device
        ).to(dtype=dtype)

        action_emb = self.action_in_proj(noisy_actions)
        action_time_emb = torch.cat([action_emb, time_emb[:, None, :].expand(-1, action_emb.shape[1], -1)], dim=-1)
        action_time_emb = self.action_time_mlp_in(action_time_emb)
        action_time_emb = F.silu(action_time_emb)
        action_time_emb = self.action_time_mlp_out(action_time_emb)

        embs = torch.cat([state_emb[:, None], action_time_emb], dim=1)
        suffix_len = embs.shape[1]
        pad_masks = torch.ones((bsize, suffix_len), device=device, dtype=torch.bool)
        att_masks = torch.zeros((bsize, suffix_len), device=device, dtype=torch.bool)
        att_masks[:, :2] = True
        return time_emb, embs, pad_masks, att_masks

    def _build_full_position_ids(
        self, prefix_position_ids: torch.Tensor, prefix_pad_masks: torch.Tensor, suffix_pad_masks: torch.Tensor
    ) -> torch.Tensor:
        """Continue the prefix's mrope positions across the suffix.

        The suffix is 1D — it has no spatial extent — so all three axes advance
        together, starting one past the largest valid prefix position. Padded
        prefix slots are excluded from that maximum so a short prompt does not
        drag the suffix's positions backwards.
        """
        valid_prefix_pos = prefix_position_ids.masked_fill(~prefix_pad_masks.unsqueeze(0), 0)
        prefix_offsets = valid_prefix_pos.amax(dim=(0, 2)) + 1
        suffix_1d = prefix_offsets[:, None] + torch.cumsum(suffix_pad_masks.long(), dim=1) - 1
        suffix_1d = suffix_1d.masked_fill(~suffix_pad_masks, 1)
        suffix_position_ids = suffix_1d.unsqueeze(0).expand(3, -1, -1)
        return torch.cat([prefix_position_ids, suffix_position_ids], dim=-1)

    def _block_query_columns(self, att_2d_masks: torch.Tensor, prefix_len: int) -> torch.Tensor:
        """Hide selected prefix query segments from every suffix row.

        ``make_att_2d_masks``' cumulative scheme can express "cannot see what
        comes after me" but not "cannot see this earlier segment", so the
        rectangular blocks are zeroed on the built matrix instead. Both flags are
        off in the released RoboTwin config; they are not derivable from the
        checkpoint and must come from the deploy config.
        """
        spans = prefix_query_token_spans(prefix_len, self.config.num_task_tokens, self.prefix_segments)
        blocked: list[tuple[int, int]] = []
        if self.config.block_future_depth_to_action and "future_depth" in spans:
            blocked.append(spans["future_depth"])
        if self.config.block_suffix_to_future_video:
            for name in ("future_video_cls", "future_video"):
                if name in spans:
                    blocked.append(spans[name])
            if self.config.future_video_share_future_depth_query and "future_depth" in spans:
                blocked.append(spans["future_depth"])
        for start, end in blocked:
            att_2d_masks[:, :, start:end] = False
        return att_2d_masks

    # -- denoising --------------------------------------------------------
    @torch.no_grad()
    def sample_actions(
        self,
        images: torch.Tensor,
        img_masks: torch.Tensor,
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
        state: torch.Tensor,
        image_grid_thw: torch.Tensor,
        noise: torch.Tensor | None = None,
        num_steps: int | None = None,
    ) -> torch.Tensor:
        """Observation → action chunk ``[B, chunk_size, max_action_dim]``.

        Convention: ``t=1`` is noise and ``t=0`` is the action, integrated by
        explicit Euler in ``num_steps`` equal steps. The caller's ``noise`` is
        never written to — upstream aliases it, which makes a pipeline that
        reuses a noise buffer resume from the previous request's chunk.
        """
        num_steps = self.config.num_steps if num_steps is None else num_steps
        bsize = state.shape[0]
        device = state.device
        dtype = state.dtype

        if noise is None:
            noise = torch.randn((bsize, self.config.chunk_size, self.config.max_action_dim), device=device, dtype=dtype)

        (
            prefix_embs,
            prefix_pad_masks,
            prefix_att_masks,
            prefix_position_ids,
            visual_pos_masks,
            deepstack_visual_embeds,
        ) = self.embed_prefix(images, img_masks, lang_tokens, lang_masks, image_grid_thw)

        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        _, past_key_values = self.qwenvl_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            inputs_embeds=[prefix_embs, None],
            past_key_values=None,
            fill_kv_cache=True,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
        )

        # Time is accumulated rather than recomputed as ``1 + step * dt`` so the
        # fp32 rounding of the timestep matches the reference implementation the
        # parity gate compares against.
        dt = torch.tensor(-1.0 / num_steps, dtype=dtype, device=device)
        time = torch.tensor(1.0, dtype=dtype, device=device)
        x_t = noise
        for _ in range(num_steps):
            v_t = self.predict_velocity(
                state=state,
                prefix_pad_masks=prefix_pad_masks,
                prefix_position_ids=prefix_position_ids,
                past_key_values=past_key_values,
                x_t=x_t,
                timestep=time.expand(bsize),
            )
            x_t = x_t + dt * v_t
            time = time + dt
        return x_t

    def predict_velocity(
        self,
        state: torch.Tensor,
        prefix_pad_masks: torch.Tensor,
        prefix_position_ids: torch.Tensor,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]],
        x_t: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """One denoising step over the cached prefix."""
        time_emb, suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(state, x_t, timestep)

        bsize, prefix_len = prefix_pad_masks.shape
        suffix_len = suffix_pad_masks.shape[1]

        # Suffix rows see every valid prefix column, plus the block structure
        # within the suffix itself.
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(bsize, suffix_len, prefix_len)
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
        full_att_2d_masks = self._block_query_columns(full_att_2d_masks, prefix_len)

        full_position_ids = self._build_full_position_ids(prefix_position_ids, prefix_pad_masks, suffix_pad_masks)
        position_ids = full_position_ids[:, :, -suffix_len:]

        outputs, _ = self.qwenvl_with_expert.forward(
            attention_mask=full_att_2d_masks,
            position_ids=position_ids,
            inputs_embeds=[None, suffix_embs],
            past_key_values=past_key_values,
            fill_kv_cache=False,
            ada_cond=time_emb if self.config.adanorm_time else None,
        )
        suffix_out = outputs[1][:, -self.config.chunk_size :]
        return self.action_out_proj(suffix_out.to(self.action_out_proj.weight.dtype))

    # -- weights ----------------------------------------------------------
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load a released ``lingbot-vla-v2`` checkpoint.

        The module tree above mirrors the checkpoint, so the only rewrite needed
        is stripping the ``model.`` prefix that the upstream ``LingbotVlaV2Policy``
        wrapper adds. Everything else is a straight name match — which is the
        point of mirroring rather than remapping: a rename table has to be kept in
        step with two moving targets, a mirrored tree fails loudly instead.

        The align *heads* in the checkpoint have no home here on purpose (see the
        module docstring); they are counted and reported, not warned about. Any
        *other* unmatched key, or any parameter that received no tensor at all,
        is a real problem and fails the load.
        """
        params = dict(self.named_parameters())
        buffers = dict(self.named_buffers())

        loaded: set[str] = set()
        dead: list[str] = []
        skipped: list[str] = []
        for name, tensor in weights:
            key = name[len("model.") :] if name.startswith("model.") else name
            if "align_head" in key or "resampler" in key:
                dead.append(key)
                continue
            if key in params:
                params[key].data.copy_(tensor)
            elif key in buffers:
                buffers[key].data.copy_(tensor)
            else:
                skipped.append(key)
                continue
            loaded.add(key)

        missing = [
            name
            for name in params
            if name not in loaded and "rotary_emb" not in name and not name.endswith(".inv_freq")
        ]
        if missing or skipped:
            raise ValueError(
                "LingBot-VLA 2.0 checkpoint does not match the runtime model: "
                f"{len(skipped)} unexpected key(s) (first 5: {skipped[:5]}); "
                f"{len(missing)} missing parameter(s) (first 5: {missing[:5]})."
            )
        logger.info(
            "LingBot-VLA 2.0 load_weights: %d tensors loaded, 0 missing, %d dead align-head tensors dropped.",
            len(loaded),
            len(dead),
        )
        return loaded


__all__ = [
    "BIG_NEG",
    "AdaRMSNorm",
    "LingbotJointModel",
    "LingbotVisionTower",
    "LingbotVlaV2ForActionPrediction",
    "RMSNorm",
    "create_sinusoidal_pos_embedding",
    "eager_attention",
    "make_att_2d_masks",
    "prefix_query_segments",
    "prefix_query_token_spans",
]
