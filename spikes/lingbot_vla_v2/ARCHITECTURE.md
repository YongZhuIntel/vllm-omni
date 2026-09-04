# LingBot-VLA 2.0 in vLLM-Omni — architecture

Design document for serving LingBot-VLA 2.0 as a vLLM-Omni **diffusion-stage
robot policy**. Written against the chosen base, **vllm-omni `v0.14.0`**, whose
API differs substantially from `main`; every contract below was read out of the
v0.14.0 tree and every claim marked *measured* was checked on this machine.

Reads in order after [`FINDINGS.md`](FINDINGS.md) (Phase 0: does it run at all?)
and [`PHASE1_SCOPE.md`](PHASE1_SCOPE.md) (what exactly has to be vendored).

---

## 1. Scope

**In.** A single-camera-rig, single-request action-chunk policy: multi-camera
images + a language instruction + robot state → one continuous action chunk
`[chunk_size=50, action_dim=55]`, served from a checkpoint that loads with zero
missing / zero unexpected tensors, numerically graded in fp32 against the Phase 0
golden.

**Out (for now).** Batching across robots, LoRA, quantization, Cache-DiT,
tensor/sequence parallelism. A 6.4 B policy on one XPU at 0.74 s/chunk is
latency-bound, not throughput-bound, and none of those features have a
robot-policy story in v0.14.0. Section 10 keeps the seams open.

---

## 2. The base: v0.14.0, and what it costs

The container ships **vLLM 0.14.1.dev** (`intel/llm-scaler-vllm:0.14.0-b8.3.2`).
vllm-omni tracks upstream vLLM's minor version, so v0.14.0 is the aligned
release. Confirmed: `import vllm_omni` succeeds, and the repo's own pytest suite
runs (the lingbot config tests pass 10/10 under plain `pytest`), which was
impossible on `main`.

The cost of that alignment is specific and worth stating plainly: **v0.14.0
predates every robot-policy feature in vllm-omni.** Measured by walking the tags:

| tag | `deploy/*.yaml` + `pipeline_registry` | `entrypoints/openpi` (robot websocket) | VLA precedents |
|---|---|---|---|
| **v0.14.0** ← chosen | — | — | — |
| v0.20.0 | yes | — | `internvla_a1` |
| v0.22.0 | yes | yes | `internvla_a1` |
| v0.24.0 | yes | yes | + `gr00t` |
| v0.28.0 | yes | yes | + `pi0` |

So this is not "port onto an older API"; it is "build the robot-policy layer,
because it does not exist yet here". The upside is that v0.14.0's *generic*
diffusion machinery already has every hook the job needs (§5) — the missing
pieces are the declarative deploy config and the websocket server, and only the
latter is real work.

### Why not backport instead

Running a newer vllm-omni on vLLM 0.14 was measured, not assumed. Static scan of
every `import vllm…` in each tag against the *installed* vLLM:

| tag | vllm modules missing | vllm symbols missing |
|---|---|---|
| v0.14.0 | 3 | 2 |
| v0.20.0 | 38 | 33 |
| v0.22.0 | 44 | 45 |
| v0.24.1 | 54 | 46 |
| v0.28.0 | 62 | 65 |

Narrowed to just the subtrees a robot policy needs (`diffusion/`, `config/`,
`core/`, `inputs/`), v0.20.0 looks tempting at 6 modules / 6 symbols — but
resolving them by hand walks straight into vLLM **private** internals
(`vllm.model_executor.models.registry._resolve_module_name`,
`vllm.v1.core.sched.interface.PauseState`) and then into semantic drift, not just
renames (vLLM's `@config` decorator signature changed). Chasing private APIs
across eight upstream minors is a worse bet than writing one websocket server.

Bulk of the package-wide gap is the OpenAI entrypoints layer, which vLLM
reorganized heavily between 0.14 and 0.28 — irrelevant to us, but it is what
`vllm_omni/__init__.py` drags in.

---

## 3. What the model computes

```
                     images[B,3,3,224,224]   lang_tokens[B,72]   state[B,55]
                              │                     │              │
        ┌─────────────────────▼─────────────────────▼──────────────▼────────┐
        │ embed_prefix:  ViT(24L) per camera → 3×64 img tokens              │
        │                + text tokens + 4 align-query segments             │
        │                → prefix_len 286                                   │
        └─────────────────────────────┬────────────────────────────────────┘
                                      │  ONE pass, Qwen3-VL text tower (36L)
                                      ▼
                              KV cache (36 layers)         ← computed once
                                      │
        ┌─────────────────────────────┴────────────────────────────────────┐
        │ 10× Euler flow-matching step (the only loop):                     │
        │   suffix = [state, noisy_action_chunk, time]                      │
        │   v_t = action_expert(suffix, kv_cache)   Qwen2-shaped, 36L,      │
        │                                           per-layer token MoE     │
        │                                           (32 experts, top-4,     │
        │                                            sigmoid router,        │
        │                                            routed_scale 4.0)      │
        │   x_t += dt · v_t                                                 │
        └─────────────────────────────┬────────────────────────────────────┘
                                      ▼
                          actions [B, 50, 55]
```

Two properties drive the whole design:

1. **The VLM runs once, the expert runs ten times.** This is the opposite of a
   text LLM (one prefill, N cheap decodes over a growing cache): here the cache
   is fixed-size and the "decode" is a full 36-layer MoE pass over 51 tokens.
   No paged KV cache, no scheduler, no continuous batching — hence
   *diffusion*-stage, not AR-stage.
2. **There is no VAE and no text encoder.** The v0.14.0 diffusion stack assumes
   both in places (§6).

Shapes are fixed by the release checkpoint and recovered from it at load time
(§8): prefix 286, chunk 50, state/action dim 55, 3 cameras @224,
`tokenizer_max_length` 72, 10 steps, expert hidden 768.

---

## 4. Module layout

Following the skill's **Path B** (custom repo, not diffusers) and its placement
rules — ported code lives in-tree, no dependency on the upstream training repo:

```
vllm_omni/diffusion/models/lingbot_vla_v2/
├── __init__.py                 exports the pipeline + factories        [landed]
├── config.py                   LingbotVlaV2Config, checkpoint          [landed]
│                               introspection, dead-tensor list
├── modeling_lingbot_vla_v2.py  FlowMatchingV2 + QwenvlWithExpertV2Model
│                               (the 29-function inference surface)       [landed]
├── qwen3vl_in_vla.py           Qwen3-VL vision/text layers as the VLA uses them
├── qwen2_action_expert.py      Qwen2 decoder + token-MoE block
├── processor.py                robot_obs → tensors; norm_stats           [landed]
└── pipeline_lingbot_vla_v2.py  LingbotVlaV2Pipeline contract             [landed]

tests/diffusion/models/lingbot_vla_v2/
├── test_config.py              synthetic-header + real-checkpoint      [landed]
├── test_modeling.py            tiny-model structural kernel coverage    [landed]
├── test_processor.py           obs→tensor contract, normalization       [landed]
│                               round-trip
└── test_pipeline.py            request, warmup, noise, load contracts    [landed]

examples/offline_inference/lingbot_vla_v2/
├── prepare_lingbot_vla_v2.py   builds the prepared model dir (§7.3)      [landed]
└── lingbot_vla_v2.py           offline action-chunk example              [landed]
```

Splitting the vendored kernel across three files rather than one mirrors upstream's
own file boundaries, which keeps a future diff against upstream readable. The v1
(`modeling_lingbot_vla.py`) and Qwen2.5-VL import chain is **not** vendored: only
three functions from it actually execute (`AdaRMSNorm.forward`,
`FlowMatching.embed_suffix`, `_block_suffix_to_future_video_if_enabled_`), and
inlining them removes two of Phase 0's nine shims outright — see
[`PHASE1_SCOPE.md`](PHASE1_SCOPE.md).

---

## 5. What v0.14.0 already gives us

Every contract the port needs, and where it lives:

| need | v0.14.0 mechanism | file |
|---|---|---|
| pipeline object | `nn.Module` subclass, `forward(req) -> DiffusionOutput` | `diffusion/worker/diffusion_model_runner.py:174` |
| arch → class | `_DIFFUSION_MODELS[arch] = (folder, module, cls)` | `diffusion/registry.py` |
| carry robot obs in | `sampling_params.extra_args: dict[str, Any]` | `inputs/data.py:235` |
| carry actions out | `multimodal_output` + free-form `final_output_type` | `outputs.py:90` |
| declare a capability | `Protocol` markers, e.g. `SupportAudioOutput` | `diffusion/models/interface.py:17` |
| weight loading | `model.weights_sources = [ComponentSource(...)]` then `model.load_weights(iter)` | `diffusion/model_loader/diffusers_loader.py:43,190,210` |
| per-model pre/post hooks | `_DIFFUSION_PRE/POST_PROCESS_FUNCS` | `diffusion/registry.py` |
| dtype / device placement | `set_default_torch_dtype` + `with target_device:` | `diffusion/model_loader/diffusers_loader.py:210` |

`extra_args` and a free-form `final_output_type` are the two that matter most, and
both are already generic. The robot-policy shape fits v0.14.0's diffusion stage
without bending it.

---

## 6. Core-repo touchpoints

Five places in `vllm_omni/` assume image/audio diffusion. **Three are avoidable
from inside the pipeline; two need small patches.**

| # | assumption | where | resolution |
|---|---|---|---|
| 1 | `model.vae` is accessed unconditionally after construction | `diffusion/registry.py:136-139` | **no patch** — set `self.vae = None`; `hasattr(None, "use_slicing")` is `False` |
| 2 | model is discovered via `model_index.json` + `transformer/config.json`, else only BAGEL is recognized | `entrypoints/omni_diffusion.py:58-85`, `entrypoints/async_omni_diffusion.py:96` | **no patch** — ship both JSONs in a prepared model dir (§7.3). Verified: `get_hf_file_to_dict` resolves both from a local directory, and `TransformerConfig` is a free-form dict container (`diffusion/data.py:94`) |
| 3 | warmup sends a 1024×1024 text-to-image request with no `robot_obs` | `diffusion/diffusion_engine.py:316` | **no patch** — pipeline synthesizes a dummy observation when `extra_args["robot_obs"]` is absent. Doubles as a genuine warmup of the 10-step loop |
| 4 | output is `images=[...]` unless `supports_audio_output(...)` | `diffusion/diffusion_engine.py:115,156` | **patch** — add an `actions` branch mirroring the audio one |
| 5 | no capability protocol for action output | `diffusion/models/interface.py` | **patch** — add `SupportActionOutput` + `supports_action_output()` beside `supports_audio_output` (`diffusion/diffusion_engine.py:38`) |

Touchpoints 4 and 5 are ~30 lines total and are deliberately written as the
*same* pattern v0.14.0 already uses for audio, so they read as a natural
extension rather than a robot-specific special case. That also makes them the
only part of this port that a future rebase has to reconcile — upstream solved
the same problem later, differently.

Deciding 1–3 in the pipeline rather than in core is the main structural choice in
this document: it keeps the blast radius of a VLA policy inside its own
directory.

---

## 7. Data contracts

### 7.1 Observation in

`extra_args["robot_obs"]`, a plain dict so it survives pickling to the worker
process:

```python
{
  "images": {                      # HWC uint8, RGB, per camera
      "cam_high":       np.ndarray[224, 224, 3],
      "cam_left_wrist": np.ndarray[224, 224, 3],
      "cam_right_wrist":np.ndarray[224, 224, 3],
  },
  "state":  np.ndarray[55],        # float32, raw robot units
  "prompt": "pick up the object",
}
```

Camera order comes from `config.image_feature_keys`; `image_key_map` renames a
robot's own topic names onto them. Missing cameras are zero-filled and masked
off via `img_masks` rather than dropped, because the prefix length is baked into
the checkpoint's align-query layout.

`processor.py` owns everything between this dict and model tensors: resize to
224², normalize with `norm_stats`, pad state to `max_state_dim`, tokenize to
`tokenizer_max_length=72`, build `image_grid_thw`. Normalization is *not*
optional — Phase 0 fed random frames and therefore proved numerics only, never
that actions land in real joint units.

### 7.2 Actions out

```python
DiffusionOutput(output={"actions": np.ndarray[50, 55]})   # from pipeline.forward
    ↓  DiffusionEngine.step, new actions branch (touchpoint 4)
OmniRequestOutput(final_output_type="actions",
                  multimodal_output={"actions": np.ndarray[50, 55]},
                  images=[])
```

Actions are un-normalized back to robot units inside the pipeline, so the
consumer receives directly executable values. `final_output_type="actions"` and
the `multimodal_output["actions"]` key are chosen to match what upstream
vllm-omni settled on later (§10), so a rebase is a rename-free move.

### 7.3 The prepared model dir

The released checkpoint is not self-describing — its `config.json` is
`{"vlm_family": "qwen3_vl"}`, and it ships no training YAML. Per the skill's
Path B convention, `prepare_lingbot_vla_v2.py` assembles a directory:

```
<prepared>/
├── model_index.json           {"_class_name": "LingbotVlaV2Pipeline"}
├── transformer/config.json    the architecture inferred by config.py, frozen
└── *.safetensors              symlinks to the release checkpoint
```

This buys three things: model discovery with no core patch (touchpoint 2), the
architecture resolved once at prep time instead of on every server start, and a
diffable record of what a given deployment actually loaded. The Qwen3-VL base
config/tokenizer path stays a separate `qwen3vl_path` setting, since it is a
different upstream artifact.

---

## 8. Weight loading

Pattern 2 from the skill (standard loader + custom `load_weights`), because the
release is ordinary safetensors that only needs name remapping:

```python
self.weights_sources = [ComponentSource(model_or_path=od_config.model,
                                        subfolder=None, prefix="",
                                        fall_back_to_pt=False)]
```

`load_weights` returns the set of names it consumed and additionally:

* **drops the dead align heads.** 120.68 M params / 76 tensors (~460 MiB fp32)
  are loaded-but-never-executed — the depth/video align *heads*. The four
  `*_align_embs` tables they sit next to **are** live (they are prefix query
  segments), so this is a curated list, not a prefix filter:
  `config.dead_inference_tensors()`.
* **asserts totality.** Phase 0 established `missing=0, unexpected=0` against
  this checkpoint; the port keeps that as a hard check rather than the
  commented-out soft check in `diffusers_loader.py:231-241`. Every field
  `infer_architecture()` guesses creates parameters, so a wrong guess surfaces
  here as a loud state-dict error instead of quietly wrong numerics.

---

## 9. Numerical parity

Phase 0's numbers dictate the method. bf16 costs ~10 % mean relative error
against fp32 *on both CPU and XPU*, while XPU-vs-CPU at equal dtype is 2.6 % —
one bf16 ulp. A bf16 action chunk therefore cannot resolve a real regression.

**Gate in fp32, per stage, on CPU.**

| stage | compared against |
|---|---|
| `embed_image` (ViT, per camera) | `phase0_layer_probe.py` stage output |
| `embed_prefix` (+ align segments, prefix_len 286) | ditto |
| prefix forward → KV cache at layers 0 / 18 / 35 | ditto |
| one `predict_velocity` | ditto |
| full 10-step chunk | `golden_cpu_fp32.npz` |

Fixed inputs: `IMAGE_SEED=0`, `NOISE_SEED=1234`, prompt `"pick up the object"`.
Per-stage rather than end-to-end because the flow-matching loop mixes errors from
every component into one 50×55 array — a whole-chunk check tells you *that*
something broke, never *what*.

Two known caveats carried forward: the golden is a **transformers 5.8** number
(shim 8, `get_rope_index`'s new `mm_token_type_ids`, is the one behavioural
change and is unverified against 4.57), and fp32 does not fit on the B60, so
XPU stays bf16 and is graded on the CPU-bf16 delta rather than on fp32.

The vendored `sample_actions` must also **copy `noise` before the loop.**
Upstream does `x_t = noise; x_t += dt·v_t`, denoising the caller's tensor in
place and returning that storage; a pipeline that reuses a noise buffer silently
starts request N from request N−1's output (measured: `max|Δ| = 1.79`).

---

## 10. Milestones

Ordered so each one is independently verifiable, and so the piece with no
v0.14.0 precedent comes last.

| # | deliverable | done when | depends on vllm_omni? |
|---|---|---|---|
| **M1** | vendored kernel + `config.py` | fp32 per-stage parity vs golden (§9); v1/Qwen2.5-VL chain gone; noise aliasing fixed | no — plain torch + transformers |
| **M2** | `processor.py` + norm_stats | obs dict → tensors matches upstream `FeatureTransform`; actions round-trip to joint units | no |
| **M3** | `pipeline_lingbot_vla_v2.py` + registry + touchpoints 4–5 + prepared dir | `OmniDiffusion(...).generate(...)` returns an action chunk offline, XPU bf16 | yes |
| **M4** | robot serving | OpenPI websocket round-trip works; ≥1 Hz remains gated by M5 | yes — backported OpenPI protocol |
| **M5** | XPU perf | beat 0.74 s/chunk (eager attn, no compile) | yes |

M1+M2 are the bulk of the code and carry all the numerical risk, while touching
nothing in `vllm_omni/` — they can land and be reviewed on their own. M3 is
where the port becomes a vllm-omni model. M4 is the only milestone that has to
invent an interface, which is exactly why it is not blocking anything before it;
if it turns out to be the priority, that is the moment to revisit the base
version, because v0.22.0+ ships `entrypoints/openpi/serving.py` for free.

M3 measured complete on the release checkpoint: 1632 live tensors loaded with
zero missing/unexpected, 76 dead align-head tensors dropped, dummy warmup passed,
and one RobotWin request returned `actions[50,14]` in robot units. The first
post-warmup pipeline request took 2.426 s, so performance remains an M5 item.

M4 protocol measured complete: the server sends RobotWin metadata on connect and
returns finite `float32[50,14]` actions over MessagePack WebSocket. Server-side
request latency was 2.744 s, so the ≥1 Hz part of the milestone remains an M5
performance gate rather than a transport gap.

---

## 11. Rebase-forward plan

v0.14.0 is a deliberate detour off `main`, so the port is written to make the
return trip cheap:

* **Names match upstream's later choices**: `robot_obs`, `final_output_type="actions"`,
  `multimodal_output["actions"]`, `policy_server_config`. Free to do now,
  expensive to retrofit.
* **The kernel is version-independent.** `lingbot_vla_v2/` (minus the pipeline)
  imports only torch + transformers, so M1/M2 rebase untouched.
* **Core patches are confined to touchpoints 4–5** and are shaped like the audio
  path they sit beside. Everything else stays inside the model directory.
* Reference for what upstream did later: `/llm/zhuyong/vllm-omni` (on `main`)
  has `pi0`, `internvla_a1`, `gr00t` and `entrypoints/openpi/` — `internvla_a1`
  is the closest architecture (Qwen3-VL + action expert), `pi0` the closest
  pipeline structure.

---

## 12. Risks

| risk | why it matters | mitigation |
|---|---|---|
| **M4 has no precedent here** | the robot client protocol must be invented, and getting it wrong means rework in the layer users actually touch | copy upstream v0.22.0's `openpi` websocket shape rather than designing fresh |
| `get_rope_index` behavioural shim (Phase 0 #8) | mrope position ids feed everything downstream; the golden itself may be wrong vs 4.57 | diff position ids against a 4.57 reference bundle if one can be obtained; CPU/XPU agreement does not settle it |
| fp32 doesn't fit on the B60 (25.5 GB vs 23.9 GiB) | the fp32 gate can only run on CPU (31 s/chunk) | keep the gate CPU-only and cheap by testing per stage; XPU is graded on the bf16 delta |
| MoE kernel choice | Phase 0 ran the export repo's grouped-einsum `_eager_fused_experts_forward`; 32 experts × 36 layers × 10 steps is the hot loop | treat as M5; keep the einsum path as the parity reference |
| checkpoint introspection guesses wrong | every inferred field creates parameters | strict `missing=0/unexpected=0` assert at load (§8) |
| 6.4 B weights at bf16 = 12.8 GB on a 23.9 GiB card | leaves little headroom for activations + the worker process | measured working in Phase 0; revisit if batching is ever added |

---

## 13. House rules, as they exist at v0.14.0

The `add-diffusion-model` skill documents `main`'s lint gates; v0.14.0's
pre-commit set is much smaller, and writing to the wrong one wastes effort.
Checked against this tree:

| rule (on `main`) | at v0.14.0 |
|---|---|
| SPDX must say `vLLM-Omni project` | **no SPDX hook**; tree uses `Copyright contributors to the vLLM project` |
| `import regex as re`, `pybase64` | **no check-imports hook**; stdlib `re` is fine (`regex` is installed, so use it anyway for rebase-friendliness) |
| no `torch.cuda.*`, use `current_omni_platform` | **not enforced**; still worth following — this is an XPU box |
| tests need CI-level + hardware marks | `--strict-markers` is on; `core_model` and `cpu` exist, **`advanced_model` / `full_model` do not** |
| L1–L4 Buildkite wiring | that pipeline layout postdates v0.14.0 |

Active hooks: ruff check + format (line length 120; `E,W,F,I,N,UP`), typos,
actionlint, check-pickle-imports, signoff, whitespace/EOF. The landed
`config.py` / `test_config.py` pass all of them.

`pytest-cov` is not installed in this container, and `pyproject.toml` puts
`--cov` in `addopts`, so tests need:

```bash
python -m pytest tests/diffusion/models/lingbot_vla_v2/ -q -o addopts=""
```
