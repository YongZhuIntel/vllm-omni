# Phase 0 — LingBot-VLA-2.0 in the vLLM-Omni environment

Go/no-go spike for porting LingBot-VLA-2.0 into vLLM-Omni as a diffusion-stage
robot policy. Runs the **unmodified upstream model** inside vLLM-Omni's Python
environment, produces the fp32 golden reference that later phases are graded
against, and records every incompatibility hit along the way.

Nothing here touches `vllm_omni/`. This directory is a scaffold, not a port.

## Verdict: GO

The upstream 6B policy runs end-to-end under vLLM-Omni's transformers 5.8 on the
Intel Arc Pro B60, at **0.74 s per action chunk** (bf16), and the released
checkpoint loads with **zero missing / zero unexpected** tensors.

Nine compatibility shims were needed; all are small and documented in
`bootstrap.py`. Only one (`get_rope_index`) is behavioural rather than a rename.

## Environment

| | |
|---|---|
| Host | Intel Arc Pro B60 Graphics, 23.9 GiB; 60 GB RAM |
| torch / transformers | 2.10.0+xpu / 5.8.0 (upstream pins 4.57.3) |
| Checkpoint | `/llm/zhuyong/lingbovla/models/lingbot-vla-v2-6b` (6.38 B params, fp32, ~25 GB) |
| Base VLM config | the export repo's `qwen3vl_base_config/` — the `models/Qwen3-VL-4B-Instruct/` dir has only `vocab.json` + `merges.txt` |
| Model shape | 3 cams @224 → prefix_len 286, chunk 50, state/action dim 55, 36 expert layers, 32 experts top-4, 10 denoise steps |

## Results

### Performance

| device / dtype | model load | `sample_actions` (warm) |
|---|---|---|
| CPU fp32 | 45 s | **31.05 s** |
| XPU bf16 | 34 s | **0.74 s** |

fp32 does not fit on the B60 (25.5 GB weights vs 23.9 GiB), so XPU runs are bf16.

### Numerics — all vs. the CPU fp32 golden

| comparison | mean abs | max abs | mean rel |
|---|---|---|---|
| CPU bf16 vs CPU fp32 | 4.22e-2 | 5.56e-1 | 10.3 % |
| XPU bf16 vs CPU fp32 | 4.62e-2 | 6.07e-1 | 11.3 % |
| **XPU bf16 vs CPU bf16** | **1.09e-2** | 1.09e-1 | **2.6 %** |

Read this as: **XPU costs almost nothing beyond bf16 itself.** The device-to-device
gap (2.6 %) is one bf16 ulp; the 10 % is the price of bf16, and it is the same on
both devices. `phase0_layer_probe.py` confirms the per-stage picture — CPU and XPU
agree to ~1e-2 relative at *every* stage (vision tower, prefix embedding, KV cache
at layers 0/18/35, one velocity step), and mrope position ids match exactly.

`phase0_step_growth.py` shows the divergence does **not** compound through the
denoise loop: 1.1e-2 / 3.0e-2 / 2.1e-2 / 2.6e-2 relative at 1 / 2 / 4 / 10 steps.

Consequence for the port: **gate numerical parity in fp32, per stage.** A bf16
action chunk is a 10 %-relative object and cannot resolve a real regression.

## Incompatibilities found

All shims live in `bootstrap.py` with the reasoning inline.

### transformers 4.57 → 5.8

1. **`flash_attn` must NOT be stubbed.** The export repo's
   `export_patches._STUB_TOP_LEVEL` fabricates a `flash_attn` module. Under
   transformers 5.8 that makes `is_flash_attn_2_available()` take a branch that
   indexes `PACKAGE_DISTRIBUTION_MAPPING["flash_attn"]` → `KeyError`, because the
   stub has no installed distribution. Without the stub the `and` short-circuits
   and FA is correctly reported absent.
2. **`AutoModelForVision2Seq`** → renamed `AutoModelForImageTextToText`.
   Patching the `transformers` module namespace does **not** work: 5.x installs a
   `_LazyModule` and a later lazy attribute access swaps in a *new* module object
   with a fresh `__dict__`, silently dropping the injected attribute. Patch
   `_LazyModule.__getattr__` (the class survives) instead.
3. **`modeling_utils.no_init_weights`** removed → no-op context manager.
4. **`import_utils.is_safetensors_available`** removed → `True`.
5. **`modeling_qwen2_5_vl.Qwen2RMSNorm`** → `Qwen2_5_VLRMSNorm`. Only reached
   because the v2 model file imports the v1 (Qwen2.5-VL) module for `AdaRMSNorm` /
   `FlowMatching`; a real port should cut that dependency instead of shimming it.
6. **`_tied_weights_keys` changed from `list[str]` to `dict`.** Upstream overrides
   it with the 4.x list on three classes, which makes 5.8's `post_init()` fail in
   `get_expanded_tied_weights_keys` with `'list' object has no attribute 'keys'`.
7. **`Qwen3VLForConditionalGeneration.visual`** proxy property dropped in 5.x;
   the v2 model reaches through it in `get_image_features`.
8. **`Qwen3VLModel.get_rope_index` gained a required `mm_token_type_ids` arg.**
   4.57 derived image/video token positions internally; 5.x makes the caller pass
   a per-token modality tensor (text 0, image 1, video 2). **This is the one
   behavioural change** — it feeds the mrope position ids, so a port must diff
   them against a 4.57 reference. (Our probe shows CPU/XPU agree exactly, but that
   does not prove agreement with 4.57.)

### XPU-specific

9. **`is_flash_attn_available()` returns `True` on any XPU box** in transformers
   5.8 — it is `... or is_torch_npu_available() or is_torch_xpu_available()`,
   because transformers routes XPU through its own FA kernels. Upstream reads it
   as "the `flash_attn` PyPI package is importable" and does an unguarded
   `from flash_attn.layers.rotary import apply_rotary_emb` behind it
   (`qwenvl_in_vla.py:31-33`) → `ModuleNotFoundError`. The same code imports fine
   on a CPU-only host, which is why the OpenVINO export work never hit this.

### Upstream bug — `sample_actions` aliases and destroys its `noise` argument

`FlowMatchingV2.sample_actions` does `x_t = noise` then `x_t += dt * v_t`, so it
denoises **the caller's tensor in place** and returns that same storage.

Upstream never notices: `select_action` passes `noise=None`, so a fresh tensor is
allocated per call. A vLLM-Omni pipeline will notice — π0's pipeline already
accepts a caller-supplied `extra_args["noise"]` (`pipeline_pi0.py:219-224`), and
any pipeline holding a preallocated noise buffer hits it too. Request 2 silently
starts from request 1's action chunk.

`phase0_aliasing_check.py` demonstrates it: reusing the buffer changes the output
by `max|d| = 1.79`, while cloning per call is bit-exact across runs. This bug also
faked a scary "XPU diverges by 0.42" result during this spike before it was found.

**A port must copy the noise before the denoise loop** (or keep the upstream
`noise=None` contract and never accept caller noise).

### Upstream bug — the action expert is built in bf16, so "fp32" isn't

*Found in Phase 1, while the vendored kernel was being graded against upstream.
Recorded here because it is an upstream defect, not a porting question.*

`QwenvlWithExpertV2Config` hardcodes `torch_dtype="bfloat16"` for the expert
(`modeling_lingbot_vla_v2.py:106`), so `Qwen2ForCausalLM._from_config` constructs
all 36 layers in bf16 — **regardless of the dtype the caller asked for**. Only the
modules that are *replaced* after construction (`replace_lnorm_with_adanorm`,
`_install_moe_blocks`) come back as fresh fp32 modules. Everything else stays
half precision, `load_state_dict` silently rounds the fp32 checkpoint into it, and
`build_model._finalize`'s `.to(torch.float32)` widens the **already rounded**
values back out. The dtype looks right afterwards; the mantissa is gone.

329 tensors arrive bf16-rounded (`--all` mode groups them):

| count | tensors | live at inference? |
|---|---|---|
| 252 | 36 x `layers.*.self_attn.{q,k,v}_proj.{weight,bias}` + `o_proj.weight` | yes |
| 1 | `qwen_expert.model.norm.weight` | yes |
| 76 | align-head projections | no (training only) |

Every one of them is *exactly* its own `.to(bfloat16).to(float32)` — nothing is
corrupted, it is pure rounding. Cost on an "fp32" run, measured against a
genuinely-fp32 build of the same weights:

| stage | max abs | max rel |
|---|---|---|
| velocity, step 0 | 2.076e-2 | 5.70e-3 |
| action chunk | 3.612e-2 | 1.10e-2 |

Consequences for the port:

- **The vendored kernel loads the checkpoint bit-exactly**, so it does *not*
  reproduce this, and grading it against a stock upstream build fails at any
  tolerance below ~1e-2. `phase1_parity.py` therefore repairs the reference
  in-process (`repair_upstream_precision`) before comparing; with that, all 45
  probed stages agree to ~1e-7 relative.
- **`golden_cpu_fp32.npz` was produced before this was known**, so it carries the
  rounding. It is still a valid bf16-era smoke reference but is *not* an fp32
  parity target. Regenerate it if one is needed.
- An fp32 deployment of upstream is not actually running fp32 attention in the
  expert; a bf16 deployment is unaffected (the rounding is what it wanted).

```bash
python phase1_weight_diff.py --side upstream --all   # exact=1379 rounded=329 other=0
python phase1_weight_diff.py --side vendored --all   # (none)
python phase1_parity.py --raw-upstream               # measure the defect end to end
```

## Files

| file | purpose |
|---|---|
| `bootstrap.py` | sys.path wiring + all compatibility shims |
| `phase0_torch_spike.py` | main run: build, infer, save/compare an action chunk |
| `phase0_layer_probe.py` | per-stage CPU-vs-XPU divergence |
| `phase0_step_growth.py` | divergence vs. denoise step count |
| `phase0_aliasing_check.py` | regression check for the noise-aliasing bug |
| `golden_cpu_fp32.npz` | saved fp32 chunk (carries the bf16 defect above — see the caveat) |
| `cpu_bf16.npz`, `xpu_bf16_fixed.npz` | bf16 runs used in the table above |

Phase 1 added two more, listed here because the finding above depends on them:

| file | purpose |
|---|---|
| `phase1_parity.py` | per-stage fp32 gate, vendored kernel vs. repaired upstream |
| `phase1_weight_diff.py` | which side's weights disagree with the checkpoint on disk |

## Reproduce

```bash
cd spikes/lingbot_vla_v2

# no checkpoint needed — validates the shims and the code paths (~10 s)
python phase0_torch_spike.py --structural

# fp32 golden reference (~25 GB RAM, ~45 s load + 31 s inference)
python phase0_torch_spike.py --device cpu --dtype float32 --repeat 0 --out golden_cpu_fp32.npz

# XPU, graded against the golden
python phase0_torch_spike.py --device xpu --dtype bfloat16 --repeat 2 \
    --ref golden_cpu_fp32.npz --out xpu_bf16_fixed.npz

python phase0_layer_probe.py --dtype bfloat16
python phase0_step_growth.py --dtype bfloat16
python phase0_aliasing_check.py
```

Paths default to this machine's layout; override with `LINGBOT_EXPORT_REPO`,
`LINGBOT_VLA_SRC`, `QWEN3VL_PATH`, `LINGBOT_CKPT`.

## What Phase 1 inherits

- **Go.** No blocking incompatibility; the model runs and the checkpoint loads exactly.
- The fixed seeds (`IMAGE_SEED=0`, `NOISE_SEED=1234`, prompt
  `"pick up the object"`) as the shared input for parity work.
  `golden_cpu_fp32.npz` was expected to be the fp32 parity *target*; it turned out
  to carry upstream's bf16 rounding, so Phase 1 grades live against a repaired
  upstream instead (see the bug above).
- The nine shims above become real edits in the vendored kernel rather than
  monkeypatches — items 5 and 8 deserve design attention (drop the v1 Qwen2.5-VL
  dependency; verify mrope ids against a 4.57 reference).
- Fix the noise aliasing in the vendored `sample_actions`.
- Parity gate in fp32 per stage, not on bf16 action chunks.
- Perf baseline to beat: 0.74 s/chunk, eager attention, no torch.compile.

## Not covered by Phase 0

Deliberately out of scope; these are Phase 2+ work:

- **Real observations.** Inputs here are random camera frames plus a fixed prompt,
  so this grades numerics, not robot behaviour. The `FeatureTransform` /
  `norm_stats` chain (`assets/norm_stats/robotwin.json`,
  `configs/robot_configs/robotwin.yaml`) is untouched, so nothing here says the
  actions are in real joint units.
- **fp32 on XPU** — does not fit in 23.9 GiB; would need a second card or offload.
- **Agreement with transformers 4.57.** The golden is a 5.8 number. If the OpenVINO
  repo's reference bundle is available, diffing against it would close shim item 8.
