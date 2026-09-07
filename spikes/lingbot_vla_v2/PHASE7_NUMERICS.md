# Phase 7 — numerical accuracy against the fp32 reference

Phase 5/6 chased latency. This phase chases a different number, and the first
job is to keep the two apart from a third one they are constantly confused with.

## Three different "accuracy" numbers

| number | what it compares | units | where it lives |
|---|---|---|---|
| **task accuracy** | predicted chunk vs *dataset ground truth* | physical (rad, m) | `open_loop_eval.py`, mae **0.0112** |
| **implementation equivalence** | our chunk vs *PyTorch fp32* on identical inputs and identical noise | normalized (55-dim) | this phase |
| **stage parity** | per-tensor, layer by layer, same dtype both sides | activations | `phase1_parity.py`, ~1e-7 |

The OpenVINO table the request came with is the **middle** one. It is produced by
`validation/validate_e2e_split.py` in the export repo: same observation, same
fixed initial noise, PyTorch fp32 `sample_actions` as reference, OV IR as
candidate, metric on the final `(1, 50, 55)` chunk. It says nothing about
whether the policy completes the task — the export IR currently has *foundation*
weights baked in, so that path cannot produce a RoboTwin score at all.

The reference protocol, read off `make_reference_bundle.py` and
`wrappers.make_prefix_example`: 3 cameras of 224x224 uniform-random pixels,
prompt `"pick up the object"`, `state = zeros(1, max_state_dim)`,
`noise = randn(1, 50, 55)` under `manual_seed(0)`, 10 Euler steps. Synthetic
inputs on purpose — this is a numerics test, not a policy test.

## First estimate (2026-09-07, superseded — kept for the lesson)

Before building the real harness, the phase-0 artifacts already on disk
(`golden_cpu_fp32.npz`, `cpu_bf16.npz`, `xpu_bf16_fixed.npz`) were scored with
`metric_stats`: CPU bf16 4.223e-02, XPU bf16 4.619e-02, against OV fp16's
1.608e-02. Read as "we are worse than the reference's int8".

The direction was right and the magnitude was not, because every one of those
numbers is a **single noise draw** and this metric has a ~1.8x spread across
draws (below). Same trap the reference table falls into — it is n=1 too.

## Measured (2026-09-07, `phase7_numeric_parity.py`)

Export-repo protocol exactly: 3x 224x224 uniform-random cameras under generator
seed 0, prompt `"pick up the object"`, `state = zeros`, 10 Euler steps, our
kernel at fp32 on CPU as the reference. **Five noise draws (seeds 0-4)**, mean
reported, MAE range shown because a single draw cannot distinguish a real 2x
from a lucky one.

| path | cosine | MAE | MSE | max abs | p99 abs | MAE range |
|---|---:|---:|---:|---:|---:|---:|
| OV **FP16** (n=1) | 0.998544 | 1.608e-02 | 8.716e-04 | 1.846e-01 | 1.291e-01 | — |
| OV **INT8** (n=1) | 0.995633 | 2.882e-02 | 2.604e-03 | 3.135e-01 | 2.089e-01 | — |
| ours **XPU bf16** | 0.978196 | 6.159e-02 | 1.529e-02 | 1.597e+00 | 4.988e-01 | 4.68e-2 – 8.79e-2 |
| ours **XPU fp16** | 0.998187 | **1.949e-02** | 1.229e-03 | 3.377e-01 | 1.321e-01 | 1.58e-2 – 2.82e-2 |

**fp16 alone reaches the target.** cosine 0.998187 against the reference's
0.998544; MAE 1.949e-02 against 1.608e-02, and the reference's single draw sits
*inside* our fp16 range. Within the variance this metric actually has, the two
are the same number. bf16 → fp16 is a 3.2x improvement for a dtype flag.

Two things that fall out and are worth keeping:

* **The spread is the story.** 1.8x across five draws, both dtypes. The
  reference's own fp16-vs-int8 gap (1.6e-2 vs 2.9e-2) is barely outside
  single-draw noise, so it should not be over-read either. Anything reported
  against this metric from one sample is not a measurement.
* The reference table is *below* the threshold its own script gates on
  (`mae < 1e-3` at `validate_e2e_split.py:258`) — 1.608e-02 is 16x over. The bar
  being matched is the reference's *reported* number, not a passing one.

### The gap was a dtype choice, not a port defect

bf16 carries 8 mantissa bits; fp16 carries 11 — 8x coarser per element before
anything else happens, and the measured ratio is 3.2x. The port contributes
*less* error than the dtype ratio alone predicts. There was no missing bug.

The same 3 mantissa bits show up in the Euler clock. `sample_actions`
accumulates `time = time + dt` at model dtype
(`modeling_lingbot_vla_v2.py:1231-1294`) rather than recomputing `1 - i/n`, so
after 10 steps it lands on **-0.004883 in bf16** and **-0.000610 in fp16** —
exactly 8x — instead of 0. Switching dtype fixes this for free; no separate
change is needed for it.

Phase 5 already resolved the dtype question the *other* way, for a reason that
no longer applies: its step 1 closed fp16 because it bought only 2.4% of
latency. It was never evaluated as an accuracy lever, and 2.4% is the price, not
the product.

### fp16's dynamic range is not a problem here — measured

The one real risk in switching was fp16's 65504 ceiling: Qwen3-VL trained in
bf16, whose 8-bit exponent absorbs activation outliers that fp16 would clip.
`--activation-audit` hooks every leaf module and reports peak |activation|:

```text
activation audit -- fp16 ceiling 65504, observed peak 11264.0 (5.8x headroom)
       11264.0  qwenvl.model.language_model.layers.6.mlp.down_proj
        9088.0  qwenvl.model.language_model.layers.35.mlp.down_proj
        2112.0  qwenvl.model.language_model.layers.16.mlp.down_proj
        1752.0  qwenvl.model.visual.blocks.23.mlp.linear_fc2
```

**5.8x of headroom, and zero non-finite activations in the actual fp16 run.**
The audit was taken on the bf16 run on purpose — bf16 carries fp32's exponent
range, so its magnitudes are the true ones; auditing fp16 only tells you whether
it already overflowed, which is the same question asked too late. The fp16 run
reports the same peaks (11224) to within rounding, confirming nothing clipped.

So the contingency plan — vision tower in bf16, action expert in fp16 — is not
needed. Two heavy tails sit in `down_proj` of language layers 6 and 35; if a
future checkpoint pushes those past ~65k, that is where to look first.

### Task accuracy improves too

Rule 2 below: a dtype change is not accepted on the equivalence metric alone.
`run_open_loop_eval.sh --dtype float16`, RoboTwin fine-tune, same bundle, same
seed, everything else unchanged:

| dtype | mse | **mae** | mae_joint | mae_gripper | mse_gripper |
|---|---:|---:|---:|---:|---:|
| bf16 (current default) | 6.08e-04 | 0.01125 | 0.01263 | 0.002953 | **1.438e-05** |
| fp16 | 5.05e-04 | **0.00785** | 0.00872 | 0.002653 | 1.110e-04 |

Overall mae improves 30% and every mean-absolute channel improves. One thing
does not, and it should not be buried: **`mse_gripper` gets ~8x worse** (1.44e-05
→ 1.11e-04) while `mae_gripper` improves. Mean down, squared-mean up means fp16
has a heavier tail on the gripper channel — a few large outliers rather than
broadly worse tracking. The gripper is the one near-binary dimension in the
14-DoF layout, so an outlier there is a spurious open/close rather than a small
tracking error. Worth a look before this ships.

#### Chased it — it is a one-step phase lag, and fp16 is 8.4x *better* without it

All 15 of the outliers are in one place: sample 1 (episode 0, frame 50),
gripper dim 6, steps 1-14. That segment is a single linear closing ramp,
1.0 → 0.0 at −0.0714 per step. fp16 predicts the same ramp *one step late*:

| | mae over the ramp |
|---|---:|
| fp16 vs `gt[t]` | 0.06722 |
| fp16 vs `gt[t-1]` | **0.00430** |

The error is the ramp slope, to two digits. It is not a spurious open/close; it
is the model choosing to start closing one control step later, and once the ramp
ends fp16 tracks the closed gripper 4x tighter than bf16 does. Excluding that
one 15-step ramp:

| | mse_gripper (all) | mse_gripper (minus the ramp) |
|---|---:|---:|
| bf16 | 1.438e-05 | 1.424e-05 |
| fp16 | 1.110e-04 | **1.691e-06** |

So the whole regression is one sample's timing, and everywhere else fp16 is
8.4x better on exactly the metric that raised the alarm.

Why bf16 cannot win the settled segment: with the gripper held at 0, bf16's
predictions land on `{-7.811e-03, -3.905e-03, ~0, +1.953e-03}` — multiples of
2^-9, its representable spacing there. bf16's error floor on a closed gripper is
its own quantization grid, about ±0.004. fp16 sits at ~0.001. Reading the
ramp-lag sample as "bf16 is more faithful here" gets it backwards: bf16 matched
ground truth on that one transition at this one seed while being 3.2x worse
overall on numerical equivalence. That is a coin landing heads, not fidelity.

**Caveat closed in fp16's favour.** One-step gripper phase on one of six samples
is worth knowing about for closed-loop work — it is a discrete decision near a
boundary and will flip under any perturbation — but it is not a reason to keep
bf16.

## Steps

1. ~~**Build the measurement first.**~~ **Done** — `phase7_numeric_parity.py`.
   The export repo's "fp16-source vs fp32-source" diagnostic turned out to be
   unnecessary rather than merely unrecorded: it exists to separate the dtype
   floor from OpenVINO's implementation error because those are two codebases.
   With reference and candidate being the same code, the measured number *is*
   the dtype floor. What did have to be added instead was **multiple noise
   draws** — see the superseded first estimate.
2. ~~**Switch the default dtype to fp16.**~~ **Done.** Reaches the target on the
   equivalence metric, improves task accuracy, costs 2.4% of latency, and the
   dynamic-range risk is retired with 5.8x of headroom. Flipped in
   `run_open_loop_eval.sh`, `open_loop_eval.py`, `lingbot_vla_v2.py`,
   `run_xpu_test.sh`, `run_openpi_server.sh`, `run_perf_check.sh` (which now
   also prints `dtype` in its verdict block, so a recorded latency is no longer
   ambiguous) and the offline README, which gained an **Inference dtype**
   section carrying the comparison table. `run_openpi_server.sh` was the one I
   nearly missed and the one that matters most — it is the deployment entry
   point, so leaving it at bf16 would have shipped the worse dtype while every
   test script reported the better one. The `mse_gripper` tail is chased and
   closed above.

   **The checkpoint is stored in fp32.** Scanning the shards: all 1708 tensors
   are `torch.float32`, 6.38B elements. This was the opposite of my assumption —
   I had reasoned about a bf16→fp16 exponent-range trade, and there is no trade.
   bf16 was discarding three mantissa bits the weights actually carried, buying
   an exponent range this model never approaches. Weight side of the fp16 range
   question, same scan: max `|w|` = 44.27
   (`...layers.0.self_attn.k_norm.weight`), 1479x under the 65504 ceiling, zero
   tensors over it; 0.0002% of elements flush to zero and 0.1786% land in
   subnormals. Negligible against a metric that improved 3.2x.

   Deliberately **not** flipped: the spike harnesses (`phase5_latency.py`,
   `phase6_denoise_profile.py`, `phase6_moe_micro.py`,
   `open_loop_steps_sweep.py`, `open_loop_conditioning_probe.py`, `phase0_*.py`)
   still default to bf16, because their recorded numbers in `PHASE5_PERF.md` and
   `PHASE6_*.md` were measured at bf16 and a silent default change would make
   those documents lie. Pass `--dtype float16` explicitly when re-measuring.
   Also not flipped: `deployment/qwen3vl_base_config/config.json`, which
   describes the HF checkpoint config rather than a runtime choice — every
   script passes `--dtype` explicitly and the pipeline reads `od_config.dtype`
   (`pipeline_lingbot_vla_v2.py:42`), so that field does not govern inference.
3. **Not needed** — promote the flow-matching loop state to fp32. Kept on record
   because it is still the cheapest remaining lever if a tighter bar ever
   appears: `x_t`, `time`, `dt` only, a 50x55 tensor over 10 steps against a
   loop that spends 1065 ms in host dispatch. It would have to be a flag
   defaulting off, since it diverges from upstream bit-for-bit (upstream does
   the same dtype thing at `lingbotvla/.../modeling_lingbot_vla_v2.py:971,973`)
   and `phase1_parity.py` grades against upstream. fp16 already removed 8x of
   the clock drift on its own, so the remaining upside here is small.
4. **Not needed** — selective fp32 on RMSNorm, softmax, the time embedding and
   adaLN modulation, and `action_out_proj`. Only revisit if the bar moves below
   the reference's fp16.

## Verification after the flip (2026-09-07)

Three things had to hold, and all three were re-run rather than assumed.

**`phase1_parity.py`: PASS, 45/45 stages at exactly 0.000e+00.** Bit-identical,
which is the expected result — the flip touched script defaults, not kernel
code — but Phase 5 rule 2 says run it anyway and this is why: "obviously
unaffected" is how a real regression gets shipped. One observation not chased:
the run's non-fatal cross-check line reports `upstream re-run vs
golden_cpu_fp32.npz: max|d|=4.051e-02 rel=1.24e-02`. Two fp32 CPU runs should
not differ by 1.2e-2, so that golden was captured under something other than
today's inputs. It does not gate anything and it is not new to this change, but
it means `golden_cpu_fp32.npz` should not be trusted as a reference until
someone re-derives it.

**`run_perf_check.sh --attribution` at fp16, idle host:** warm WebSocket median
**0.861 s (1.16 Hz)** over 8 requests (0.746-0.938 s), model path 696.3 ms,
denoise loop 602.1 ms at 60.1 ms/step, cold offline request 0.863 s. PASS
against the 1 Hz M4 gate. The bf16 eager record this replaces was 0.703 s model
path / 0.607 s denoise — the same to within 1%, comfortably inside the harness's
own 0.746-0.938 s spread. Phase 5's 2.4% estimate holds; fp16 is free here.

The first attempt at this measurement was **refused by the script** (exit 3,
`load1=5.23`) because the parity run's load had not decayed yet. That refusal is
the feature working: an absolute latency taken at load 5.2 would have been
recorded next to numbers taken at load 0.2 and quietly poisoned the comparison.

**Task accuracy** was already re-run before the flip (`run_open_loop_eval.sh`,
RoboTwin fine-tune) — mae 0.01125 → 0.00785, table above.

`open_loop_eval.py` now reports the **full OpenVINO metric set** — cosine, MAE,
MSE, max abs, p99 abs — over all 14 dims and both splits, printed as a table
instead of a raw JSON dump. Recomputed on the stored predictions:

| metric (all 14, micro) | bf16 | fp16 |
|---|---:|---:|
| cosine (mean) | 0.999637 | 0.999673 |
| cosine (min) | 0.998752 | 0.998926 |
| MAE | 1.125e-02 | **7.852e-03** |
| MSE | 6.083e-04 | 5.051e-04 |
| max abs diff | 3.215e-01 | 2.999e-01 |
| p99 abs diff | 9.538e-02 | 9.265e-02 |

fp16 wins every one of them. The only place bf16 leads anywhere is the gripper
split's max/p99, which is the one-step ramp lag dissected above.

**Rule 1 applies to this addition and is the reason it needed care.** These
share formulas with the table at the top of this document but not the
reference: here it is dataset ground truth in robot command units, there it is
a fp32 run of the same kernel in normalized 55-dim units. `format_report`
carries that warning in its docstring and the printed header says
`vs dataset ground truth`, because a cosine of 0.9997 sitting next to a cosine
of 0.9985 with no label is an invitation to conclude something false.

## Rules

1. **Same metric or no comparison.** Any number reported against the reference
   goes through `metric_stats` unchanged, on the final chunk, with the same
   fixed noise on both sides. A differently-defined MAE is not a smaller MAE.
2. **Do not trade task accuracy for equivalence.** The open-loop mae 0.0112 is
   the number that matters to a robot; this phase must re-run
   `run_open_loop_eval.sh` after any dtype change and report both.
3. **Record every conclusion here, including the negatives.**
