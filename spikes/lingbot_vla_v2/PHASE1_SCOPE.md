# Phase 1 — vendoring scope, measured

Phase 0 answered "can it run". Phase 1 vendors the inference subset into
`vllm_omni/diffusion/models/lingbot_vla_v2/`. The upstream `lingbot_vla` package
is ~5000 lines, most of it training, so the subset was **measured** rather than
guessed: `phase1_trace_surface.py` profiles a real `sample_actions` call and
records every upstream function that executes.

Run it on the 6B checkpoint (`--full`), not the structural model: the structural
build sets `adanorm_time=False, align_params={}` and therefore misses the AdaRMSNorm
and align-token paths the release config enables.

## The surface to vendor

**Inference — 29 functions across 5 files** (`surface_6b.json`):

| file | what runs |
|---|---|
| `modeling_lingbot_vla_v2.py` | `FlowMatchingV2`: `sample_actions`, `embed_prefix`, `predict_velocity`, `_build_full_position_ids` · `QwenvlWithExpertV2Model`: `forward`, `get_image_features`, `embed_image`, `embed_language_tokens`, `embed_special_token`, `apply_mrope`, `handle_kv_cache`, `_apply_deepstack` |
| `qwen3vl_in_vla.py` | `Qwen3VLVisionAttention.forward`, `Qwen3VLVisionBlock.forward`, `Qwen3VLTextDecoderLayer.forward`, `preprcess_grid_thw`, `forward_without_grid_thw` |
| `qwen2_action_expert.py` | `Qwen2DecoderLayer.forward`, `Qwen2TokenMoeBlock.forward`, `Qwen2MoeSharedExpertMLP.forward`, `FixQwen2RMSNorm.forward` |
| `modeling_lingbot_vla.py` (v1) | `AdaRMSNorm.forward`, `FlowMatching.embed_suffix`, `_block_suffix_to_future_video_if_enabled_` |
| `utils.py` | `make_att_2d_masks`, `create_sinusoidal_pos_embedding`, `prefix_query_segments` |

Construction adds 33 more functions (module `__init__`s, config classes,
`Qwen2FusedExperts.reset_parameters`, `build_processor`).

### Two consequences worth acting on

**1. The v1 dependency is three functions — inline them.** `modeling_lingbot_vla.py`
is 1610 lines and, at import, drags in `qwenvl_in_vla.py` (Qwen2.5-VL). Inference
uses exactly `AdaRMSNorm.forward`, `FlowMatching.embed_suffix` and
`_block_suffix_to_future_video_if_enabled_`. Porting those three into the vendored
v2 file deletes the whole v1 + Qwen2.5-VL import chain — and with it shim #5 from
Phase 0 (`Qwen2RMSNorm` → `Qwen2_5_VLRMSNorm`) and shim #9's trigger site
(the unguarded `from flash_attn.layers.rotary import ...` lives in `qwenvl_in_vla.py`).

**2. 120.68 M parameters are loaded but never executed.** The depth / video align
*heads* (Perceiver-style resamplers + the MoGe depth head, 76 tensors, ~460 MiB
fp32) produce training-time alignment targets. Inference only reads the learned
task-query tables that feed the prefix:

| used at inference | params |
|---|---|
| `depth_align_embs`, `current_video_align_embs`, `future_depth_align_embs`, `future_video_align_embs` | 4 × 0.66 M |
| `current_shared_task_proj`, `future_shared_task_proj` | 2 × 13.1 M |
| **never executed**: `*_align_head.projector.*` | **120.68 M** |

Careful: all four `*_align_embs` *are* used — they are appended as prefix query
segments (`embed_prefix`, the `prefix_query_segments` loop), so dropping them
changes the prefix length. It is only the heads that are dead.
`LingbotVlaV2Config.dead_inference_tensors()` lists them; it is a helper, not
applied automatically, so a port drops them deliberately and asserts the saving.

## Landed so far

- `vllm_omni/diffusion/models/lingbot_vla_v2/config.py` — `LingbotVlaV2Config`
  plus `from_release_checkpoint()`, which recovers the architecture from the
  **safetensors headers** (the released checkpoint's `config.json` is just
  `{"vlm_family": "qwen3_vl"}` and it ships no `lingbotvla_cli.yaml`). Verified
  against the real 6B checkpoint: hidden 768, state/action 55, 36 MoE layers,
  32 experts, inter 512 / shared 704, no shared-expert gate, adanorm on,
  vocab 151936, both future-align segments present, 76 dead tensors.
- `tests/diffusion/models/lingbot_vla_v2/test_config.py` — synthetic-header unit
  tests plus a real-checkpoint test gated on `LINGBOT_CKPT`.

## Blocker for Phase 3 (serving), not for Phase 1

> **Resolved.** The base is now vllm-omni **`v0.14.0`** (branch `lingbot-vla-v2`),
> which imports and tests cleanly against the container's vLLM 0.14.1.dev. The
> cost — v0.14.0 predates vllm-omni's robot-policy layer entirely — and the
> resulting design are written up in [`ARCHITECTURE.md`](ARCHITECTURE.md).
> The section below records the state before that decision.

**`vllm_omni` cannot be imported in this container.** Two separate problems:

1. `aenum` was missing — fixed (`pip install aenum`).
2. `vllm_omni/patch.py:18` needs `vllm.v1.request.StreamingUpdate`, which does not
   exist in the installed **vLLM 0.14.1.dev** (`intel/llm-scaler-vllm:0.14.0-b8.3.2`).
   This checkout is on `main` (HEAD `e51fe6ec`, 2026-09-02), which targets the
   vLLM 0.26 line. The `/llm/zhuyong/vllm-omni` copy has the same requirement.

So the repo's pytest suite cannot run here at all — even a pure-stdlib test fails
at collection, because importing anything under `vllm_omni` executes
`vllm_omni/__init__.py`. The config work above was validated by loading the module
directly by path.

Phase 1 is unaffected (the vendored kernel is plain torch + transformers), but
Phase 3 needs one of:

- **upgrade vLLM** in the container to the 0.26 line vllm-omni `main` targets, or
- **check out the vllm-omni `v0.14.0` release tag**, which is the one aligned with
  this container's vLLM 0.14 (vllm-omni publishes a release per even upstream minor).

The second is much less disruptive but means porting onto an older vllm-omni API;
the first keeps the port on `main` where it would eventually be upstreamed. This
is a fork in the road that changes where the pipeline/registry code has to land,
so it should be decided before Phase 3 starts.

## Status — M1 is closed

All three items this section used to list are done:

1. **Kernel vendored** at `vllm_omni/diffusion/models/lingbot_vla_v2/`, per the
   surface table, with the three v1 functions inlined and the align heads dropped
   (120.68 M parameters that only run during training). The module tree mirrors
   the checkpoint exactly: 0 keys with no home, 0 parameters with no weight, 0
   shape mismatches.
2. **Parity gate passes.** `python phase1_parity.py --num-steps 2` →
   `PASS - 45 stage(s) agree within 0.0001 relative`. Prefix / vision / KV-cache /
   time / suffix / position stages are bit-identical; the expert probes at layers
   0, 1, 18, 35 and the outputs sit at 1e-7 relative (fp32 summation order:
   gather-vs-dense MoE, einsum vs. matmul). Grading is against a *repaired*
   upstream — see the bf16 finding in [`FINDINGS.md`](FINDINGS.md), which also
   explains why `golden_cpu_fp32.npz` is no longer the fp32 target.
3. **Noise aliasing fixed** in the vendored `sample_actions`, and pinned by
   `test_sample_actions_does_not_alias_the_caller_noise`.

Structural coverage landed alongside it, on a tiny randomly-initialised model so
it needs neither the checkpoint nor 26 GB of RAM:

```bash
pytest tests/diffusion/models/lingbot_vla_v2/ -o addopts="" -q   # 16 passed, 1 skipped
```

(the skip is the real-checkpoint config test, gated on `LINGBOT_CKPT`; `-o
addopts=""` works around `--cov` in `pyproject.toml` with pytest-cov absent.)

## Status — M2 is closed

`processor.py` now bridges `robot_obs` to the kernel's six tensors and converts
the normalized action chunk back to the robot's source-key layout and units. It
loads the real robot config and norm-stats chain, preserves fixed camera slots,
builds `image_grid_thw`, and supports the normalization modes used upstream.

Two gates cover it:

```bash
pytest tests/diffusion/models/lingbot_vla_v2/test_processor.py -o addopts="" -q
# 4 passed

cd spikes/lingbot_vla_v2
python phase2_processor_parity.py
# six model inputs exact; action max|d|=1.129e-07
```

The parity gate uses the real RobotWin mapping, norm stats, and Qwen3-VL image /
token processors without loading the 6B checkpoint. Images, masks, language,
state, and `image_grid_thw` agree exactly with upstream `FeatureTransform`; the
float64-upstream / float32-runtime action conversion agrees within fp32 rounding.

## Status — M3 is closed

The pipeline, registry entry, `SupportActionOutput`, engine actions branch,
prepared-directory builder, and offline example are landed. The real 6B release
checkpoint was prepared as six symlinks and loaded through `OmniDiffusion` on
XPU bf16 with strict weight totality:

```text
1632 tensors loaded, 0 missing, 76 dead align-head tensors dropped
model memory: 11.7341 GiB
type=actions shape=(50, 14) dtype=float32 elapsed=2.426s
```

The engine's mandatory dummy warmup also completes. The container needs the
repository-pinned `cache-dit==1.2.0`; without it, v0.14.0's diffusion worker
fails at import even when cache acceleration is disabled.

## Status — M4 protocol is closed

The OpenPI-compatible WebSocket endpoint, MessagePack/NumPy codec, metadata
handshake, reset/session handling, AsyncOmni adapter, and RobotWin client are
landed. A real XPU request returned finite `float32[50,14]` actions through
`/v1/realtime/robot/openpi`.

The measured server-side request took 2.744 s at the time, so the M4 acceptance
criterion of at least 1 Hz was not met by transport work alone.

**Since closed in M5.** Warm round trips are 0.73–0.94 s (1.07–1.37 Hz) and the
kernel alone is 0.703 s, below the Phase 0 direct-kernel baseline of 0.74 s. The
whole difference came from switching `moe_implementation` from `gather` to
`dense`; see `PHASE5_PERF.md`.
