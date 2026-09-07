# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Config surface for LingBot-VLA 2.0 in vllm-omni.

LingBot-VLA 2.0 is a Qwen3-VL-4B vision-language backbone plus a Qwen2-shaped
action expert with per-layer token MoE, joined by a flow-matching action head:
multi-camera images + a language instruction + robot state produce a continuous
action chunk ``[chunk_size, max_action_dim]``.

Why this file infers so much
----------------------------
The released ``robbyant/lingbot-vla-v2-6b`` checkpoint is **not** self-describing.
Its ``config.json`` is a three-line stub::

    {"vlm_family": "qwen3_vl"}

and, unlike the upstream training layout, it ships no ``lingbotvla_cli.yaml``. The
upstream deploy script reconstructs the config by reading a training YAML that
sits three directories above the weights — which does not exist for a plain
``git clone`` of the HF repo.

So ``from_release_checkpoint`` recovers the architecture from the **safetensors
headers** (tensor names and shapes only, no tensor data is read). Every field it
infers creates parameters, so getting one wrong shows up immediately as a
state-dict mismatch rather than as silently wrong numerics.

A deploy YAML may override anything via ``from_model_config``.
"""

from __future__ import annotations

import glob
import json
import os
import re
import struct
from dataclasses import dataclass, field
from dataclasses import fields as dataclass_fields
from typing import Any

# Observation key conventions (LeRobot-style), shared with the processor.
ACTION = "action"
OBS_STR = "observation"
OBS_STATE = OBS_STR + ".state"
OBS_IMAGES = OBS_STR + ".images"

# Qwen3-VL-4B-Instruct supplies the tokenizer, image processor and the vision /
# text transformer dimensions. The policy checkpoint carries the weights but none
# of these JSONs, so a base-model directory must be provided.
DEFAULT_QWEN3VL_MODEL = "Qwen/Qwen3-VL-4B-Instruct"


@dataclass
class LingbotVlaV2Config:
    """Runtime config for the LingBot-VLA 2.0 policy.

    A plain dataclass rather than an HF ``PretrainedConfig``: the fields below are
    the ones that shape serving behaviour. Transformer dimensions for the VLM
    tower come from the Qwen3-VL base config at ``qwen3vl_path``.
    """

    # --- Base VLM -----------------------------------------------------------
    # Local dir (or HF repo id) holding the Qwen3-VL config/tokenizer/processor.
    qwen3vl_path: str = DEFAULT_QWEN3VL_MODEL
    vocab_size: int = 151936

    # --- Action chunk -------------------------------------------------------
    chunk_size: int = 50
    max_action_dim: int = 55
    max_state_dim: int = 55

    # --- Flow matching ------------------------------------------------------
    num_steps: int = 10

    # --- Action expert (Qwen2 shape) ----------------------------------------
    expert_hidden_size: int = 768
    # Must equal the VLM's depth: the two towers are walked in lockstep, one
    # joint attention per layer.
    expert_num_layers: int = 36
    expert_num_attention_heads: int = 32
    expert_num_key_value_heads: int = 8
    expert_head_dim: int = 128
    # SwiGLU width of the *dense* MLP, used only by layers that are not token-MoE.
    expert_intermediate_size: int = 2752
    expert_rms_norm_eps: float = 1e-6

    # --- Action expert token MoE --------------------------------------------
    use_moe: bool = True
    token_moe_layers: list[int] | None = None
    token_num_experts: int = 32
    token_top_k: int = 4
    token_moe_intermediate_size: int = 512
    token_shared_intermediate_size: int = 704
    router_activation: str = "sigmoid"
    routed_scaling_factor: float = 4.0
    use_shared_expert_gate: bool = False
    # Loss-free MoE balancing; training-only, must be 0 at inference.
    bias_update_speed: float = 0.0
    # "dense" runs every expert on every token and zero-weights the rest — three
    # einsums per layer, and the upstream eager path the fp32 golden was produced
    # with. "gather" runs each expert on only its routed tokens: ~8x less
    # arithmetic at top-4 of 32, but as a Python loop over all 32 experts it is
    # ~70k tiny kernel launches per request, and it measured 3.7x *slower* on the
    # B60 (262 ms/denoise step vs 70 ms; see spikes/lingbot_vla_v2/PHASE5_PERF.md).
    # Kept because the arithmetic argument still holds behind a real grouped
    # kernel or on a device where launches are cheap. Same value up to fp32
    # summation order.
    moe_implementation: str = "dense"

    # --- Prefix / attention -------------------------------------------------
    adanorm_time: bool = True
    # Whether the expert's *final* norm is also time-conditioned. Off in the
    # released checkpoint (its ``model.norm`` carries only ``weight``).
    final_norm_adanorm: bool = False
    vlm_causal: bool = True
    tokenizer_max_length: int = 72
    # Qwen3-VL checkpoints use their tokenizer's native conversation format.
    use_chat_template: bool = True
    # Number of learned task-query tokens per align segment in the prefix.
    num_task_tokens: int = 8
    # Which align query segments are appended to the prefix. These change the
    # prefix length, so they must match the checkpoint.
    use_future_depth: bool = True
    use_future_video: bool = True
    use_future_video_patch: bool = True
    use_future_video_cls: bool = False
    use_current_video_patch: bool = True
    # The two task projections that fuse a depth query with a video query.
    use_current_shared_task_proj: bool = True
    use_shared_future_task_proj: bool = True
    # When set, future-video reuses the future-depth query instead of getting a
    # segment of its own.
    future_video_share_future_depth_query: bool = True
    # Shape of the learned query tables: one row per backbone patch, pooled down
    # to ``num_task_tokens`` rows for the prefix.
    num_backbone_tokens: int = 256
    align_dim: int = 2560

    # Mask surgery: hide a query segment from every action row. Pure mask flags,
    # so they leave no trace in the checkpoint and cannot be inferred from it —
    # they must come from the deploy config. Both off in the release.
    block_future_depth_to_action: bool = False
    block_suffix_to_future_video: bool = False

    # --- Inputs -------------------------------------------------------------
    image_resolution: tuple[int, int] = (224, 224)
    max_cameras: int = 3
    # Camera order the model attends to.
    image_feature_keys: list[str] | None = None
    image_key_map: dict[str, str] = field(default_factory=dict)

    # --- Kernel selection ---------------------------------------------------
    # Upstream defaults to "flex_cached" (torch flex_attention) for the joint
    # VLM+expert attention and "flash_attention_2" for the vision tower. Neither
    # is available on every vllm-omni target (flash-attn is CUDA-only), so the
    # portable pure-torch paths are the default here.
    attention_implementation: str = "eager"
    vit_attn_implementation: str = "sdpa"
    # Phase 8 FP16 attention gate passed: 5-seed fp32-reference mean MAE 1.990e-2
    # and RobotWin task MAE did not regress. Use fp32 for parity diagnosis.
    attention_precision: str = "fp16"
    # Inductor reduces the fixed-shape 10-step denoise loop from ~602 ms to
    # ~216 ms on B60. Phase 8 gates fp16 compiled MAE at 1.961e-2 against fp32
    # (below the vendor INT8 ceiling of 2.882e-2) and found no open-loop MAE
    # regression, so compiled is the deployment default. Keep eager available
    # for parity diagnosis through --no-compile-denoise-step.
    compile_denoise_step: bool = True
    # Full Prefix graph compile is experimentally measured at ~41 ms versus
    # ~58 ms eager on B60, but its accuracy gate is still pending.
    compile_prefix: bool = False

    def __post_init__(self) -> None:
        resolution = self.image_resolution
        if not isinstance(resolution, (tuple, list)) or len(resolution) != 2:
            raise ValueError(f"image_resolution must be a 2-tuple; got {resolution!r}.")
        self.image_resolution = (int(resolution[0]), int(resolution[1]))
        if self.token_moe_layers is not None:
            self.token_moe_layers = [int(layer) for layer in self.token_moe_layers]
        if self.bias_update_speed:
            raise ValueError(
                "bias_update_speed drives the training-time loss-free MoE balancing "
                f"update and must be 0 for inference; got {self.bias_update_speed}."
            )
        if self.moe_implementation not in ("gather", "dense"):
            raise ValueError(f"moe_implementation must be 'gather' or 'dense'; got {self.moe_implementation!r}.")
        if self.attention_precision not in ("fp32", "fp16"):
            raise ValueError(
                f"attention_precision must be 'fp32' or 'fp16'; got {self.attention_precision!r}."
            )
        if self.num_backbone_tokens % self.num_task_tokens:
            raise ValueError(
                f"num_backbone_tokens ({self.num_backbone_tokens}) must be divisible by "
                f"num_task_tokens ({self.num_task_tokens}) to pool the align query tables."
            )
        head_width = self.expert_num_attention_heads * self.expert_head_dim
        if head_width % self.expert_num_key_value_heads:
            raise ValueError(
                f"expert_num_attention_heads ({self.expert_num_attention_heads}) must be a "
                f"multiple of expert_num_key_value_heads ({self.expert_num_key_value_heads})."
            )

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------
    @classmethod
    def from_model_config(cls, model_config: dict[str, Any] | None) -> LingbotVlaV2Config:
        """Build from a deploy-yaml ``model_config`` block, ignoring unknown keys."""
        if not model_config:
            return cls()
        raw = dict(model_config)
        if "image_resolution" in raw:
            raw["image_resolution"] = tuple(raw["image_resolution"])
        allowed = {item.name for item in dataclass_fields(cls)}
        return cls(**{key: value for key, value in raw.items() if key in allowed})

    @classmethod
    def from_release_checkpoint(
        cls,
        checkpoint_dir: str,
        qwen3vl_path: str | None = None,
        **overrides: Any,
    ) -> LingbotVlaV2Config:
        """Recover the architecture from a released checkpoint's safetensors headers.

        Only the JSON headers are read, so this is fast and needs no GPU/host memory
        for the ~25 GB of weights.
        """
        shapes = read_safetensors_shapes(checkpoint_dir)
        inferred = infer_architecture(shapes)
        if qwen3vl_path is not None:
            inferred["qwen3vl_path"] = qwen3vl_path
        inferred.update(overrides)
        return cls(**inferred)


# ----------------------------------------------------------------------------
# Checkpoint introspection
# ----------------------------------------------------------------------------
def read_safetensors_shapes(checkpoint_dir: str) -> dict[str, list[int]]:
    """Return ``{tensor_name: shape}`` by reading only the safetensors headers.

    A safetensors file starts with a little-endian u64 header length followed by
    that many bytes of JSON, so the tensor data is never touched.
    """
    shapes: dict[str, list[int]] = {}
    for path in sorted(glob.glob(os.path.join(checkpoint_dir, "*.safetensors"))):
        with open(path, "rb") as handle:
            header_len = struct.unpack("<Q", handle.read(8))[0]
            header = json.loads(handle.read(header_len))
        for name, meta in header.items():
            if name != "__metadata__":
                shapes[name] = meta["shape"]
    if not shapes:
        raise FileNotFoundError(f"No readable *.safetensors headers under {checkpoint_dir}")
    return shapes


def infer_architecture(shapes: dict[str, list[int]]) -> dict[str, Any]:
    """Derive the config fields that create parameters, from tensor shapes.

    Restricted to fields that affect the module tree: if any of them is wrong the
    checkpoint fails to load, which is a loud failure rather than a quiet one.
    """
    expert_prefix = r"model\.qwenvl_with_expert\.qwen_expert\.model\.layers\.(\d+)\."

    expert_hidden_size, max_state_dim = shapes["model.state_proj.weight"]
    max_action_dim = shapes["model.action_out_proj.weight"][0]

    # The release stores MoE experts grouped as (num_experts, intermediate, hidden)
    # rather than as a ModuleList of per-expert Linears.
    moe_layers: set[int] = set()
    expert_layers: set[int] = set()
    token_num_experts = 0
    moe_intermediate = None
    shared_intermediate = None
    dense_intermediate = None
    for name, shape in shapes.items():
        layer_match = re.match(expert_prefix, name)
        if layer_match:
            expert_layers.add(int(layer_match.group(1)))
        match = re.match(expert_prefix + r"mlp\.experts\.gate_proj$", name)
        if match:
            moe_layers.add(int(match.group(1)))
            token_num_experts, moe_intermediate = shape[0], shape[1]
        elif re.search(r"mlp\.shared_expert\.gate_proj\.weight$", name):
            shared_intermediate = shape[0]
        elif re.match(expert_prefix + r"mlp\.gate_proj\.weight$", name):
            dense_intermediate = shape[0]

    inferred: dict[str, Any] = {
        "expert_hidden_size": expert_hidden_size,
        "max_state_dim": max_state_dim,
        "max_action_dim": max_action_dim,
        "use_moe": bool(moe_layers),
        "token_moe_layers": sorted(moe_layers),
        "token_num_experts": token_num_experts,
        "use_shared_expert_gate": any("shared_expert_gate" in name for name in shapes),
        # AdaRMSNorm stores per-norm gamma/beta projection weights.
        "adanorm_time": any(re.search(r"layernorm\.(gamma|beta)\.weight$", name) for name in shapes),
        # The expert's final norm is separately conditioned (or not).
        "final_norm_adanorm": any(
            re.search(r"qwen_expert\.model\.norm\.(gamma|beta)\.weight$", name) for name in shapes
        ),
    }
    if expert_layers:
        inferred["expert_num_layers"] = max(expert_layers) + 1
    if moe_intermediate is not None:
        inferred["token_moe_intermediate_size"] = moe_intermediate
    if shared_intermediate is not None:
        inferred["token_shared_intermediate_size"] = shared_intermediate
    if dense_intermediate is not None:
        inferred["expert_intermediate_size"] = dense_intermediate

    inferred.update(_infer_expert_attention(shapes, expert_hidden_size))
    inferred.update(_infer_align_layout(shapes))

    embed_tokens = shapes.get("model.qwenvl_with_expert.qwenvl.model.language_model.embed_tokens.weight")
    if embed_tokens:
        inferred["vocab_size"] = embed_tokens[0]
    return inferred


def _infer_expert_attention(shapes: dict[str, list[int]], hidden_size: int) -> dict[str, Any]:
    """Head counts of the action expert, from its q/k projection widths.

    ``head_dim`` is not recoverable on its own — only the products
    ``num_heads * head_dim`` and ``num_kv_heads * head_dim`` are stored — so the
    default (128, the Qwen2 action-expert value) is assumed and the counts are
    derived from it. A wrong ``head_dim`` therefore shows up as a non-integer
    head count here rather than as wrong numerics later.
    """
    prefix = "model.qwenvl_with_expert.qwen_expert.model.layers.0.self_attn."
    q_shape = shapes.get(prefix + "q_proj.weight")
    k_shape = shapes.get(prefix + "k_proj.weight")
    if not q_shape or not k_shape:
        return {}

    head_dim = LingbotVlaV2Config.expert_head_dim
    q_width, k_width = q_shape[0], k_shape[0]
    if q_width % head_dim or k_width % head_dim:
        raise ValueError(
            f"Action-expert q/k widths ({q_width}, {k_width}) are not multiples of the assumed "
            f"head_dim {head_dim}; pass expert_head_dim explicitly."
        )
    if q_shape[1] != hidden_size:
        raise ValueError(f"Action-expert q_proj input {q_shape[1]} disagrees with hidden size {hidden_size}.")
    return {
        "expert_num_attention_heads": q_width // head_dim,
        "expert_num_key_value_heads": k_width // head_dim,
        "expert_head_dim": head_dim,
    }


def _infer_align_layout(shapes: dict[str, list[int]]) -> dict[str, Any]:
    """Which align query segments the prefix carries, and how wide they are.

    Each segment exists iff its embedding table does, and each adds
    ``num_task_tokens`` tokens to the prefix — so these flags set the prefix
    length and a wrong one is a shape error, not a silent regression.

    One flag is not directly visible: ``future_video_share_future_depth_query``
    leaves both ``future_depth_align_embs`` and ``future_video_align_embs`` in the
    checkpoint either way. It is recovered from ``future_shared_task_proj``, which
    upstream only builds when sharing is on (``init_video_heads`` asserts the
    implication), and defaults to sharing when that projection is absent.
    """
    depth_table = shapes.get("model.depth_align_embs")
    inferred: dict[str, Any] = {
        "use_future_depth": "model.future_depth_align_embs" in shapes,
        "use_future_video": "model.future_video_align_embs" in shapes
        or "model.future_video_cls_align_emb.weight" in shapes,
        "use_future_video_patch": "model.future_video_align_embs" in shapes,
        "use_future_video_cls": "model.future_video_cls_align_emb.weight" in shapes,
        "use_current_video_patch": "model.current_video_align_embs" in shapes,
        "use_current_shared_task_proj": "model.current_shared_task_proj.weight" in shapes,
        "use_shared_future_task_proj": "model.future_shared_task_proj.weight" in shapes,
    }
    inferred["future_video_share_future_depth_query"] = (
        inferred["use_shared_future_task_proj"] or "model.future_video_align_embs" not in shapes
    )
    if depth_table:
        inferred["num_backbone_tokens"], inferred["align_dim"] = depth_table
    return inferred


def dead_inference_tensors(shapes: dict[str, list[int]]) -> list[str]:
    """Tensor names that the checkpoint carries but ``sample_actions`` never reads.

    The depth / video align *heads* (Perceiver-style resamplers, plus the MoGe
    depth head) exist to produce the training-time alignment targets. Inference
    only consumes the learned task-query embedding tables that feed the prefix
    (``*_align_embs``) and the two shared task projections — never the heads
    themselves. Traced with ``spikes/lingbot_vla_v2/phase1_trace_surface.py``:
    120.68 M parameters (~460 MiB fp32) across 76 tensors are constructed and
    loaded but never executed.

    Kept as a helper (not applied automatically) so a port can drop them
    deliberately and assert the saving, rather than silently changing what loads.
    """
    return sorted(name for name in shapes if "align_head" in name or "resampler" in name)
