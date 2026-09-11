# Phase 9 — what FlashRT's LingBot path has that ours does not

Asked on 2026-09-11: read `/llm/zhuyong/FlashRT` and say whether there is
optimization space left, and if so plan it.

Answer: **yes, roughly 27 ms of bit-exact work and another 40 ms behind accuracy
gates**, which would take the model path from 286.5 ms to ~220 ms — past the
246 ms OpenVINO reference. Two of FlashRT's mechanisms are already on our list
(F4 FiLM hoist, F5 gate/up merge); three are new; and two of its headline
choices are measurably *wrong* on this hardware.

*Revised 2026-09-11 after P1 and P4 landed: the original figures were 35 ms and
~210 ms. Both landed items came in at ~0.6x their micro-benchmark estimate, and
the remaining estimates are haircut by the same factor. 8.3 ms of the 27 is now
banked rather than predicted.*

Everything below was measured on the B60 today, at the real shapes, under
`torch.compile(dynamic=False)` — not read off FlashRT's CUDA numbers. The
probes are `/tmp/probe_*.py`; the ones that survive should move into `spikes/`
as committed probes when the work starts.

## 0. Read the differences first

FlashRT is a hand-written CUDA realtime engine. Its LingBot integration
(`flash_rt/models/lingbot/`, 5390 lines) is **LingBot-VLA 1.0**, not ours:

| | FlashRT's LingBot | ours |
|---|---|---|
| backbone | Qwen2.5-VL, 4B | Qwen3-VL, 6B |
| action expert | 36 layers, **dense SwiGLU** | 36 layers, **token-MoE, 32 experts top-4** |
| action chunk | `[1, 50, 75]` | `[1, 50, 55]` |
| device | Jetson AGX Thor sm_110 | Arc Pro B60, Xe2 |
| precision | FP8 activations + NVFP4 gate_up, static calibration | fp16 |
| headline | 64.1 ms @ 10 steps | 286.5 ms model path |

**Do not adopt 64.1 ms as a target.** It is a smaller model without our MoE,
on different silicon, at FP8/FP4 against our fp16. The comparable reference
remains OpenVINO's 246 ms (Phase 8, "which OpenVINO number is the target").

What transfers is the *mechanism list*, and the reason it is worth reading is
that FlashRT independently converged on the same diagnosis we did — its
`denoise_step_layer` is one long argument that the loop is dispatch- and
bandwidth-bound at small `M`, which is F1/F2 written in CUDA.

## 1. Per-layer-step budget — new, and it accounts for the loop

Nothing in Phase 8 broke the denoise loop down below the stage level. Compiled,
fp16, real shapes, 200–300 iterations after warmup:

| component | ms / layer-step | ×360 | share |
|---|---:|---:|---:|
| routed MoE (4 einsums: gate, up, down, combine) | 0.2492 | 89.7 | 44% |
| joint attention incl. the KV `cat` | 0.1102 | 39.7 | 20% |
| ~~shared expert (3 GEMMs)~~ | ~~0.0792~~ | ~~28.5~~ | **wrong shape — see below** |
| expert q/k/v (3 GEMMs) | 0.0644 | 23.2 | 12% |
| o_proj + AdaRMSNorm + router + residuals | ~0.042 | ~15 | 8% |
| measured loop (F1) | 0.5594 | **201.4** | |

F2's 90.4 ms for the MoE GEMMs reproduces exactly (89.7), which is the
cross-check that makes the rest of this file usable: these micro numbers are in
the same regime as the real loop, so savings derived from them are credible
**as upper bounds** (F4's lesson — Inductor has often already paid part of the
cost).

**Correction, 2026-09-11 (found while starting P4).** The shared-expert row was
measured at `expert_intermediate_size = 2752`. That is the wrong constant: the
released RoboTwin checkpoint's `token_moe_layers` covers all 36 layers, so
`ExpertMLP` is never instantiated and the only SwiGLU branch in the loop is the
shared expert at `token_shared_intermediate_size = **704**` — 3.9x narrower.
(Confirmed on the real model: `load_weights` reports `36 fused gate_up`, one per
layer, all of them shared experts.) The row is therefore an overcharge of
unknown size, the original "97% accounted for" claim is **withdrawn**, and the
sum row is deleted rather than patched with a guess: isolated compiled micros at
`[51,768]×[768,704]` are noise-dominated (§6, P4) and are not the instrument to
re-derive it with. What survives is the ranking of the top two rows, which is
what the plan is built on.

The second row is the surprise. **Attention is 20% of the loop** and no phase
has ever sized it.

## 2. What FlashRT does, and whether it survives on the B60

| FlashRT mechanism | where | our verdict |
|---|---|---|
| batched FiLM γ/β precompute (`precompute_expert_film`, one GEMM for all `[step, layer, slot]`) | `forward.py:440` | **adopt** — it is exactly step I / F4, already sized at −8–9 ms on the real model. FlashRT supplies the implementation shape: cache one stacked `[4·L·H, H]` weight, one batched GEMM over all 10 timesteps. |
| merged `gate_up` weight | `prepare_expert_merged_weights`, `forward.py:131` | **adopted, landed** — our F5, at the shared expert's real 704 width: **−1.8 ms**, bit-exact. P4. |
| merged QKV projection | `compute_kqv_expert` | **adopted, landed** — F5 only ever considered gate/up. **−6.5 ms**, bit-exact. P1. |
| preallocated KV buffer, k/v written straight into the suffix slot (no per-step copy) | `forward.py:512-536` | **reject** — see §4. Real in eager, zero under Inductor. |
| FP8 weights / NVFP4 `gate_up` with static calibration | `fp4_ops.py`, `calibration.py` | **adopt the idea, reject the format** — on B60 `_scaled_mm` fp8 is *slower* than fp16; `torch._int_mm` int8 is 1.6x faster. §4. |
| maskless fused attention (zero the pad K/V rows, accept the softmax distortion) | `mask_kv_cache_pad_rows` | **replace with an exact version** — our mask is column-uniform, so the pads can be *compacted away* instead of approximated. §3. |
| FA4 / CuTe attention kernel, 15 hand-written fused kernels | `csrc/` | **unavailable** — no equivalent binding on this stack (F3). Not re-litigated. |
| graph capture of the denoise loop | `graph_runner.py`, `sample_actions_graph` | **deferred, not dead** — `torch.xpu.XPUGraph` exists and is worth ~5.5%, but capturing the *compiled* step corrupts across replays (F3b). Reopens after P1/P4/P5; see P10. |
| split prefix graph / decoder graph | `sample_actions_split_graph` | **already have it** — `compile_prefix` + `compile_denoise_step`. |
| one ViT call for all cameras | — | already done (J, §"already done"). |

## 3. The new finding: the denoise mask is column-uniform

`predict_velocity:1633` builds the prefix half of the denoise mask as

```python
prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(bsize, suffix_len, prefix_len)
```

and `_block_query_columns` (`:1505`) then zeroes whole *columns*. So **every one
of the 51 suffix query rows sees exactly the same set of prefix columns.** Two
consequences, both exact — no approximation, no accuracy gate on the algebra:

1. **The invalid prefix columns can be dropped once**, after `prefix_fill`,
   instead of being masked 360 times. 286 → 223 valid slots on the RobotWin
   config (Phase 8 verified `bad_positions=0` when compressing the prefix), so
   the denoise attention runs at `Lk = 274` instead of 337.
2. **The prefix half of the mask then disappears entirely.** What is left is a
   51×51 block mask on the suffix, which is small enough to be free.

Measured (eager attention kernel, the deployed one):

| | ms / layer-step | ×360 |
|---|---:|---:|
| today, `Lk=337`, full mask | 0.1310 | 47.2 |
| compacted, `Lk=274`, full mask | 0.1007 | 36.3 |
| SDPA, `Lk=337` | 0.0955 | 34.4 |
| SDPA, `Lk=274` | 0.0673 | 24.2 |
| SDPA + `enable_gqa`, `Lk=274` (no `repeat_interleave`) | 0.0624 | 22.5 |
| SDPA maskless, `Lk=274` | 0.0272 | 9.8 |

Compaction alone is **−10.9 ms and exact**. It also unblocks the thing Phase 8
closed: the `flash_suffix` two-call split was rejected because "constructing
compressed `cu_seqlens` requires a data-dependent token count" inside the
compiled graph. Compacting once *before* the loop moves that data dependency
out of the graph.

The cost is that the valid prefix length varies with the instruction length, so
`dynamic=False` needs the compacted length bucketed (round up to a multiple of
32 and pad; ≤3 graph variants in practice).

### And the recorded SDPA accuracy failure does not reproduce

Phase 8 shelved `suffix_sdpa` on a five-seed MAE of `1.538e-01` against fp32,
versus eager's `1.949e-02`. On the denoise attention shape in isolation, against
an fp32 reference, with the real mask structure (63 padded prefix columns, the
state-token block):

| path | MAE | max |Δ| |
|---|---:|---:|
| eager fp16 (deployed) | 5.020e-05 | 1.165e-03 |
| **SDPA fp16, bool mask** | **2.082e-05** | 1.753e-04 |
| SDPA fp16, additive fp16 mask | 2.082e-05 | 1.753e-04 |
| SDPA fp32 | 4.332e-09 | 1.490e-07 |

**SDPA is 2.4x more accurate than the kernel we ship.** So the model-level
`1.538e-01` cannot be the attention math; it is a bug somewhere in that path
(most likely its interaction with the compiled graph or the cached prefix, which
is where `prefix_sdpa` also produced NaNs). That reopens 13–23 ms as a defect to
find rather than a tradeoff to accept, and it should be found before anyone
writes a custom XE2 attention kernel.

## 4. Two of FlashRT's choices are wrong on this hardware

Worth recording because both look obviously right on paper.

### The preallocated KV buffer buys nothing under Inductor

FlashRT allocates `[B, Lp+Ls, ...]` once and writes k/v directly into the suffix
region, calling the per-step copy out. Ours does `torch.cat([cached, new])` per
layer per step — 720 concatenations of a 0.69 MB tensor. In eager that is worth
~19 ms. Compiled it is worth nothing:

| | eager | **compiled** |
|---|---:|---:|
| `cat` + attention | 0.1208 | **0.1102** |
| preallocated buffer + attention | 0.1136 | **0.1075** |

Within noise at 300 iterations, and an earlier 50-iteration pass had the buffer
*losing* by 50%. Inductor already elides the copy; mutating a captured buffer
fights its functionalization. **Rejected** — this is the F4 lesson again, and
this time the sign flipped.

### FP8 is the wrong format; int8 is the right one

FlashRT runs FP8 activations with NVFP4 `gate_up`. Both exist on this stack —
`torch._scaled_mm` and `torch._int_mm` are present and functional on XPU. At our
shapes fp8 is a regression and int8 is the largest single lever in the loop:

| `[51,768] × [768,32768]` (the whole routed gate+up bank) | ms |
|---|---:|
| fp16 | 0.1360 |
| fp8 `_scaled_mm` | 0.1500 |
| **int8 `_int_mm`** | **0.0836** |
| `_weight_int8pack_mm` (weight-only, 2-D) | 0.5492 — unusable |
| dequant-then-bmm (weight-only, naive) | 0.5215 — unusable |

int8 delivers 1.63x on a bank whose bytes halve, i.e. almost exactly the
bandwidth ratio F2's roofline predicts. Weight-only variants are *slower* than
fp16 because the dequant materializes fp16 weights; the win needs **W8A8**, so
it needs a Phase 7 gate.

## 5. The reformulation that makes int8 reachable at all

`forward_dense` (`modeling_lingbot_vla_v2.py:849`) is four einsums:

```python
gate = torch.einsum("td,eid->eti", hidden_flat, self.gate_proj)
up   = torch.einsum("td,eid->eti", hidden_flat, self.up_proj)
expert_out = torch.einsum("eti,edi->etd", F.silu(gate) * up, self.down_proj)
return torch.einsum("etd,te->td", expert_out, weights)
```

`hidden_flat` is **the same `[T, 768]` for every expert** — the dense path
broadcasts it. So gate and up are not batched GEMMs at all; they are one 2-D
GEMM with a wide N. And the down projection can absorb the combine step, because
the routing weight is a per-`(token, expert)` scalar that commutes with the
`down_proj` matmul: scale the activations by `w[t,e]` first and the expert sum
becomes a plain contraction. The whole routed block collapses to **two 2-D
GEMMs**:

```python
o   = (x @ GU).view(T, E, 2, I)             # [51,768] @ [768, 32·2·512]
act = F.silu(o[:,:,0]) * o[:,:,1] * w[..., None]
out = act.reshape(T, E*I) @ DW              # [51,16384] @ [16384,768]
```

`GU = cat(gate_proj, up_proj)` reshaped to `[768, 32768]`, `DW = down_proj`
reshaped to `[16384, 768]`, both built once at load time. This matters for three
separate reasons:

- it merges gate/up (F5) across all 32 experts at once,
- it removes two einsum reshapes and the separate combine,
- **and it turns both matmuls into 2-D GEMMs, which is the only form
  `torch._int_mm` accepts.** There is no batched int8 GEMM on XPU; without this
  reformulation the int8 lever is simply unavailable.

Measured, compiled, at the real shapes:

| routed-expert block | ms / layer-step | ×360 | Δ |
|---|---:|---:|---:|
| today, 4 einsums | 0.2492 | 89.7 | — |
| two 2-D GEMMs, fp16 | 0.2285 | 82.3 | **−7.5 ms** |
| two 2-D GEMMs, int8 W8A8 | 0.1717 | 61.8 | **−27.9 ms** |

GEMM time alone: fp16 0.136 + 0.070 = 0.206 → int8 0.084 + 0.034 = 0.117, i.e.
1.76x, against an int8 roofline of 2x. Consistent with a bandwidth-bound kernel.

Accuracy: the fp16 reformulation changes summation order (the expert sum moves
inside the contraction), measured at `max|Δ| = 1.5e-5`, relative `4.5e-4` on
random data — small but **not bit-exact**, so it goes through the Phase 7 gate
like everything else. The int8 step obviously does.

This is also the item F2 pointed at and nobody could reach: *"the one lever where
we can go past OV rather than catch up to it"* — the reference keeps all 1359.0 M
expert weights at fp16.

## 6. The plan

Ordered by return per unit of risk. Every number is a compiled micro-measurement
at the real shape, so **treat each as an upper bound until it is re-measured on
the real model** — F4 found a 3x overstatement doing exactly this, §4 found two
items whose sign flipped, and both landed items came in at ~0.6x their estimate
(P1 0.62x, P4 0.56x against the shape-corrected figure). Scale the unmeasured
rows accordingly.

| # | item | est. | numerics | where |
|---|---|---:|---|---|
| ~~P1~~ | Merge expert `q_proj`/`k_proj`/`v_proj` into one `[768, 6144]` GEMM, split the output | est. −10.4, **actual −6.5 ms** | **bit-exact, landed** | `ExpertAttention`, `compute_qkv:957`, `load_weights` |
| **P2** | Compact the prefix KV to its valid columns once after `prefix_fill`; drop the prefix half of the denoise mask | **−10.9 ms** | exact | `sample_actions`, `predict_velocity:1633`; needs length bucketing for `dynamic=False` |
| **P3** | Hoist γ/β FiLM + masks + position ids out of the loop (step I / F4, FlashRT's `precompute_expert_film` as the shape) | **−8 to −9 ms** | bit-comparable | `AdaRMSNorm:471`, `denoise_actions:1586` |
| ~~P4~~ | Merge the shared expert's `gate_proj`/`up_proj` (F5) — plain `[gate‖up]` concat | est. −3.2, **actual −1.8 ms** | **bit-exact, landed** | `GatedExpertMLP`, `load_weights` |
| | *subtotal, no accuracy risk* | **≈ −27 ms** (−8.3 banked, ~−19 estimated) | | → model path ~259 ms |
| **P5** | Reformulate the routed MoE into two 2-D GEMMs (§5) | **−7.5 ms** | rel 4.5e-4, Phase 7 gate | `GroupedExperts.forward_dense:846` |
| **P6** | Debug the `suffix_sdpa` model-level MAE, then ship SDPA (`enable_gqa=True`, no `repeat_interleave`) on the compacted mask | **−12 to −14 ms** on top of P2 | SDPA is *more* accurate than eager in isolation; the 1.5e-1 is a defect | `sdpa_attention:180`, the compiled-graph interaction |
| **P7** | int8 W8A8 on the two routed GEMMs, with FlashRT's calibration contract as the blueprint | **−20 ms** on top of P5 | Phase 7 gate is the whole job | new quant path + `load_weights` |
| | *subtotal, all gates passed* | **≈ −67 ms** | | → model path **~220 ms**, still past OV's 246 |

Two follow-ons, unsized, worth doing only after the above:

- **P8.** The same P1/P4 merges on the VLM tower. They act on `prefix_fill`
  (57.1 ms, 36 layers × 1 pass instead of × 10), so the ceiling is a few ms —
  but the code is already written by then.
- **P9.** int8 on the expert attention projections (15.73 MB/layer-step). Merged
  QKV is already a 2-D GEMM after P1, so this is a small delta on top of P7.

### P10 — XPU Graph, reopened as a *consequence* of P1/P4/P5

F3b closed this: `torch.xpu.XPUGraph` on `2.13.0+xpu` is real and capturing the
compiled ten-step loop is worth `210.1 → 198.2 ms` (~5.5%), but the five-seed
gate exposed cross-replay contamination (mean MAE `5.388e-01`) because the probe
replays Inductor's reusable output buffers. F3b also recorded the escape hatch
and dismissed it on cost:

> Capturing the eager step avoids this corruption and passes ten alternating-request
> replays bit-exact, but costs `275.7 ms`, much slower than the direct compiled path.

**P1/P4/P5 attack precisely the reason eager is slow.** The eager penalty
(275.7 − 210.1 ≈ 65 ms) is Inductor fusing many small ops; the merges delete
those ops at the source rather than fusing them. Measured on the three merge
targets, eager, per layer-step:

| | eager before | eager after | Δ |
|---|---:|---:|---:|
| routed MoE | 0.2577 | 0.2492 | −0.0085 |
| shared expert (at the wrong 2752 width — see §1) | 0.0544 | 0.0428 | −0.0116 |
| expert q/k/v | 0.0423 | 0.0181 | −0.0242 |
| **sum** | **0.3544** | **0.3101** | **−0.0443 (−16 ms ×360)** |

and the op count on those three drops from ~10 GEMM-class ops to 4. So the
question F3b answered as "eager capture is 65 ms too slow" should be re-asked
after P5, when a large part of that 65 ms no longer exists.

**Do not add P10's ~12 ms to the −75 ms.** Graph replay recovers exposed *host
dispatch* time, and P1/P3/P4 remove dispatches too — it is partly the same money.
The correct sequence is: land P1–P5, then re-measure.

Decision test, cheap, ~an afternoon:

1. After P1/P4/P5, time the real denoise step eager vs compiled.
2. If eager is within ~10% of compiled: capture **eager** with
   `phase8_xpu_graph_denoise_probe.py`, which F3b already showed is bit-exact
   across ten alternating-request replays. That is a deployable graph with no
   accuracy argument to make.
3. If eager is still far behind: P10 stays closed. Fixing the compiled-capture
   buffer aliasing is an upstream Inductor-XPU job (there is no `cudagraph_trees`
   equivalent on XPU — F3 measured `triton.cudagraphs=True` as an accepted no-op),
   not something to take on inside this port.

One caveat on the table above: each row was compiled as its own region, so the
compiled column carries a per-call wrapper cost that the real single-graph step
pays once. **It does not show that eager beats compiled** — F3b's whole-step
measurement is the authoritative one. It shows only that the merges move eager
toward compiled, which is the precondition step 2 depends on.

### Sequencing

~~P1 → P4~~ (both landed 2026-09-11) → P3 → P2 first: all four are independent, none needs a new numeric
argument, and together they are the difference between 286.5 and ~251 ms. P2
should land before P6 because it is what makes the mask cheap enough for the
SDPA question to be worth asking.

P6 is a **bug hunt, not an optimization** — start it by reproducing the
`1.538e-01` with the committed probe and bisecting eager-vs-SDPA per layer. If
it turns out to be real after all, P6 is dead and P7 is unaffected.

P5 and P7 are one piece of work: do not build the int8 path on the einsum form,
because the einsum form cannot call `_int_mm` at all.

### Gates

Rule 3 from Phase 8 applies unchanged: nothing lands until it has been through
`phase7_numeric_parity.py` at ≥5 seeds against the fp32 reference (ceiling
`2.882e-02`, the OV INT8 column) **and** `run_open_loop_eval.sh` on the RobotWin
six-chunk bundle without task-MAE regression. P1/P4 should come back bit-exact;
if they do not, something in the weight rewrite is wrong.

## 7. Execution log

Each item gets its steps written down before the code, and its result written
back here after. `est.` is the §6 micro-measurement; `actual` is the real model.

### P1 — merge expert q/k/v — **in progress, started 2026-09-11**

Est. −10.4 ms, expected bit-exact.

Where the fusion belongs: `DiffusersPipelineLoader.load_weights` snapshots
`weights_to_load = self._get_expected_parameter_names(model)` at
`diffusers_loader.py:866`, *before* calling `model.load_weights` at `:870`, and
its strict check is `weights_to_load - loaded_weights`. So a model that rewrites
its own tree inside `load_weights` and returns the original checkpoint names
passes the check untouched. The loader's own comment at `:1003` — "Load weights
first using model's load_weights (handles QKV fusion etc.)" — says this is the
intended seam. No new hook is needed.

Steps:

1. `config.py`: add `fuse_expert_qkv: bool = True`, with the measurement in the
   comment, following the `compile_denoise_step` / `attention_backend`
   convention. No `__post_init__` validation needed (plain bool).
2. `ExpertAttention`: declare `qkv_proj = None` in `__init__` so the checkpoint
   mirror is unchanged at construction time, and add `fuse_qkv()` that builds
   `[q‖k‖v]` along dim 0 (weight *and* bias — the expert's q/k/v all carry bias)
   and deletes the three originals.
3. `ExpertDecoderLayer.compute_qkv`: when `qkv_proj` is present, one GEMM then
   `split` into the three head-shaped views. The branch is on a Python attribute,
   so `fullgraph=True` specializes it at trace time.
4. `LingbotVlaV2ForActionPrediction.load_weights`: after the copy loop and after
   the missing/skipped check, fuse if the flag is set. Log it.
5. `prepare_lingbot_vla_v2.py`: `--fuse-expert-qkv` / `--no-fuse-expert-qkv`,
   so the A/B and the Phase 7 gate can both be driven from the CLI.
6. Verify: fused vs unfused outputs on one request (expect bit-exact), then the
   Phase 7 five-seed gate, then `run_open_loop_eval.sh`, then latency.

Only the *expert* tower. The VLM tower's `compute_qkv` (`:676`) interleaves
Qwen3 q/k norms between the projection and the view, so it needs a different
split point — that is P8, and it pays off in `prefix_fill`, not the loop.

**Result: landed. −6.5 ms on the denoise loop, bit-exact.**

Latency, `phase5_latency.py --model /tmp/lingbot-vla-v2-perf --compile-denoise-step
--iters 10 --warmup 3`, the two arms interleaved over three repetitions so drift
cannot masquerade as the effect:

| rep | denoise, split | denoise, fused |
|---|---:|---:|
| 1 | 224.3 | 216.8 |
| 2 | 223.2 | 216.8 |
| 3 | 223.3 | 216.8 |
| **median** | **223.3** | **216.8** |

−6.5 ms on the loop, −5.6 ms on the request (309.1 → 303.5 synced total), and the
per-step line moves 22.2 → 21.5 ms. The fused arm is the more repeatable of the
two, which is what removing 720 dispatches should do.

**The estimate was 1.6x too high** (−10.4 predicted, −6.5 delivered). Same
direction and same cause as F4: Inductor was already recovering part of the
three-GEMM cost, so an isolated micro-benchmark of the *unfused* form
over-charges it. Apply that 0.6 haircut to the remaining §6 estimates until each
is measured — it puts the "no accuracy risk" subtotal nearer −21 ms than −35 ms.

Numerics — exact, not merely within tolerance:

| check | result |
|---|---|
| tiny CPU model, `test_fused_expert_qkv_is_bit_exact` | `torch.equal` |
| 6B, xpu/bfloat16, 5 seeds, eager | `max|Δ| = 0` |
| 6B, xpu/bfloat16, 5 seeds, Inductor | `max|Δ| = 0` |
| 6B, cpu/float32, 2 seeds | `max|Δ| = 0` |
| Phase 7 five-seed gate, fused vs split | **every field byte-identical** (`mae` 2.9744935408234596e-02 bf16, 1.5662188455462456e-02 fp16 on both sides) |

`run_open_loop_eval.sh` was **not** re-run. The sampled chunk is bit-identical on
both sides of the change, so the task MAE cannot move; re-running it would
measure the harness, not P1. Recorded here rather than left implied.

New scripts: `phase9_p1_qkv_fusion.py` (loads unfused, samples, fuses in place,
samples again -- two 6B models do not fit in 23.9 GiB together).
`--fuse-expert-qkv/--no-` added to `phase5_latency.py` and
`phase7_numeric_parity.py` for the A/B.

### P4 — merge the shared expert's gate/up — **landed 2026-09-11**

Est. −3.2 ms, expected bit-exact. Delivered **−1.8 ms**, bit-exact.

Two things had to be corrected before any code was written:

1. **The §6 estimate was measured at the wrong width.** It used
   `expert_intermediate_size = 2752`, but `token_moe_layers` covers all 36
   layers, so `ExpertMLP` is never constructed and the branch that actually runs
   is the shared expert at `token_shared_intermediate_size = 704`. See the §1
   correction. `use_shared_expert_gate = False`, so there is no third projection
   to fold in either.
2. **The re-measurement at 704 was unusable.** The compiled micro reported
   `[gate‖up]` merged (0.0815 ms) as *faster than the unmerged pair plus the
   gate it does not have* and inconsistent across repeats — at `[51,768]×[768,704]`
   the kernel is smaller than the measurement noise. So the estimate was anchored
   on P1's real-model result instead: P1 removed 2 GEMMs per layer-step for
   6.5 ms, i.e. ~9 µs per removed GEMM per 360 layer-steps; P4 removes 1.

Implementation: `ExpertMLP` and `SharedExpertMLP` were byte-identical duplicate
classes, so both now derive from a shared `GatedExpertMLP` carrying
`fuse_gate_up()` and the branching `forward`. Nothing in the repo does an
`isinstance` check on either name, so the re-parenting is inert. The fusion runs
from `load_weights` behind `config.fuse_expert_gate_up`, the same seam and the
same flag convention as P1.

Latency, `phase5_latency.py --model /tmp/lingbot-vla-v2-perf --compile-denoise-step
--compile-max-relative-error 0.05 --iters 10 --warmup 3 --fuse-expert-qkv`, arms
interleaved over three repetitions on top of P1:

| rep | denoise, split | denoise, fused |
|---|---:|---:|
| 1 | 216.8 | 215.7 |
| 2 | 216.5 | 213.9 |
| 3 | 215.7 | 214.7 |
| **median** | **216.5** | **214.7** |

−1.8 ms on the loop and −1.8 on the request (303.0 → 301.2 synced median), with
the fused arm ahead in all three pairs. Per-step 21.5 → 21.3 ms. Small, and it is
the honest size of one 768×704 GEMM out of ~10 GEMM-class ops per layer-step —
the estimate was 1.8x high, the same direction as P1.

Cumulative after P1+P4: denoise **223.3 → 214.7 ms**, request **309.1 → 301.2**.

Numerics — exact:

| check | result |
|---|---|
| tiny CPU model, `test_fused_expert_gate_up_is_bit_exact` | `torch.equal`, both subclasses |
| tiny CPU model, `test_both_expert_fusions_compose` (P1+P4 together) | `torch.equal` |
| 6B, xpu/bfloat16, 5 seeds, eager | `max|Δ| = 0` |
| 6B, xpu/bfloat16, 5 seeds, Inductor | `max|Δ| = 0` |
| Phase 7 five-seed gate, fused vs split | **every field byte-identical** |

The tiny test config is the useful one here: `token_moe_layers = [1, 3]` out of 4
layers, so it instantiates both `ExpertMLP` and `SharedExpertMLP` and covers the
dense subclass that the released checkpoint never builds.

`run_open_loop_eval.sh` **not** re-run, for the same reason as P1: the sampled
chunk is bit-identical on both sides.

`phase9_p1_qkv_fusion.py` was renamed `phase9_fusion_exactness.py` and given
`--fusion {qkv,gate_up,both}` rather than copied. `--fuse-expert-gate-up/--no-`
added to `phase5_latency.py`, `phase7_numeric_parity.py` and
`prepare_lingbot_vla_v2.py`.

### Side finding: the prepared checkpoint moves under the Phase 7 cache, 2026-09-11

Running the gate rebuilt `phase7_golden_fp32.npz` instead of using it. That is
the cache doing its job: it stores a `checkpoint` fingerprint of the resolved
first shard, the prepared `/tmp` model dirs are symlink farms, and those symlinks
were re-pointed at `global_step_50000/hf_ckpt` on 2026-09-10 — after the npz
(and after `phase7_numeric_parity.json`, written 2026-09-07). Any phase-7 run
today rebuilds it, with or without P1.

**Resolved during P4, later the same day — the recorded baseline was right and
the P1-run numbers were the anomaly.** By the P4 gate the links had been
re-pointed once more, at 02:57 on 2026-09-11, this time back to the canonical
`/llm/zhuyong/lingbovla/models/lingbot-vla-v2-6b/`. The reference rebuilt again,
and the numbers came back to the committed json:

| | recorded 2026-09-07 | P1 run, `global_step_50000/hf_ckpt` | P4 run, canonical |
|---|---:|---:|---:|
| xpu:bfloat16 MAE | 6.159e-02 | 2.974e-02 | **6.115e-02** |
| xpu:float16 MAE | 1.949e-02 | 1.566e-02 | **1.899e-02** |
| OpenVINO FP16 (reference) | 1.608e-02 | 1.608e-02 | 1.608e-02 |

So `phase7_numeric_parity.json` is **not** stale: it describes the checkpoint we
actually deploy, reproduced today to within the fp32 reference's own rebuild
noise (~0.7% and ~2.6%). The middle column measured a training checkpoint that
happened to be linked in for a few hours. The accuracy budget for P5/P7 stands
as recorded — fp16 at 1.9e-02 against the 2.882e-02 ceiling — and does **not**
need re-basing.

The real lesson is about the harness, not the model: `checkpoint_fingerprint`
resolves through the symlink, which is what caught this, but nothing pins *which*
checkpoint a prepared `/tmp` dir points at. Re-check the link target before
quoting any absolute Phase 7 number. A/B comparisons are unaffected — both arms
of P1 and of P4 ran against the same reference and came back byte-identical.

## 8. What this does not change

- **The iGPU stays out of the request path.** Nothing in FlashRT bears on
  G/G2/G4/G5/K; it is a single-device CUDA engine.
- **Graph capture is not on the critical path.** F3b's cross-replay corruption
  is unaffected by any of this — the fix for *that* is upstream. What changes is
  that the safe eager-capture variant may become affordable once P1/P4/P5 land;
  see P10. FlashRT's CUDA-graph path still has no direct counterpart.
- **Batching is still the throughput lever, not a latency one** (G3). P5/P7 make
  batching *better* — a 2-D GEMM at `M = 51·B` scales the way F2's crossover
  table predicts — but they do not change the single-request argument.
