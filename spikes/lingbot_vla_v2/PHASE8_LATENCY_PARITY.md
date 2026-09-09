# Phase 8 — closing the last 2.8x to the OpenVINO reference

This is the execution plan for the one Phase 6 step that was left open. Phase 6
identified the cause and proved the fix works; it stopped because enabling the
fix needed an accuracy gate that did not exist yet. Phase 7 built that gate.
This file is separate from `PHASE5_PERF.md` only because that file already
carries two phases and this one is the whole remaining latency story.

## First: which OpenVINO number is the target

Four different reference numbers are in circulation and they are not
interchangeable. Every figure below is `avg` from a log in the export repo, on
**B60 dGPU, `[precision] f16`, fp16-compressed weights** — despite the `int8` in
the filenames, none of these runs are int8:

| log | vit | text | loop (10 steps) | per step | **total** |
|---|---:|---:|---:|---:|---:|
| `run_info_demo_dgpu_int8.log` | 17 | 59 | 213 | 21 | 288 |
| `run_info_demo_dgpu_int8_kvf16.log` | 17 | 47 | 209 | 21 | 273 |
| `run_info_demo_dgpu_int8_fmha.log` | 16 | 42 | **188** | **19** | **246** |
| `run_info_demo_pipeline.log` (vit on GPU.0) | 6 | 51 | 193 | 19 | 250 (min 235) |

**There is no 200 ms run in any log I can find.** The best documented
single-sample total is **246 ms** (fmha, one device); the pipeline log's 251
ms/sample steady state is the same work with the ViT overlapped onto a second
GPU, which lowers *its* vit column to 6 ms without making anything faster. The
PTL table quoted earlier in this project (733 ms FP16 / 655 ms INT8) is a
different, slower machine and is not the target.

**Target: 246 ms of model time.** If someone has a 200 ms run, the log needs to
be produced before it becomes the goal — the difference decides whether step C
below is optional or mandatory.

## Where our 790-861 ms actually goes

`run_perf_check.sh --attribution`, fp16, eager, idle host, 2026-09-07. Warm
WebSocket median 0.861 s; the user's re-run reported 0.790 s, inside this
harness's own 0.746-0.938 s spread.

| stage | ours (fp16 eager) | OV best (fmha) | delta |
|---|---:|---:|---:|
| pre.* + h2d + post | 4.7 | — | +4.7 |
| `embed_prefix` (their vit) | 24.7 | 16 | +8.7 |
| `prefix_fill` (their text) | 64.5 | 42 | +22.5 |
| **`denoise` (10 steps)** | **602.1** | **188** | **+414.1** |
| per denoise step | 60.1 | 19 | +41.1 |
| **model path (synced)** | **696.3** | **246** | **+450.3** |
| wall − model path | ~165 | n/a | +165 |

**92% of the model-side gap is the denoise loop.** Everything else combined is
36 ms. Any work that is not about the denoise loop is rounding error until the
denoise loop is fixed.

## Why, and what is already proven

Phase 6 step 3 settled the cause and it is not arithmetic
(`phase6_denoise_profile.py`, one request's loop):

```text
227465 aten ops
summed self DEVICE time:     51.0 ms
summed self CPU  time:     1065.2 ms
glue ops: 210557 calls (93%), 855 ms of the host time
```

The card is idle waiting for Python. The reference submits the whole 10-step
loop as **one IR call**; we submit ~227k individual ops. Phase 6 also retired
every competing hypothesis by measurement: the MoE einsums already run at 13.06
TFLOPS standalone (`phase6_moe_micro.py`), fp16 vs bf16 is 2.4%, and the
reference is dense too, so top-4 routing is not the difference.

And the fix is already measured to work:

| path | denoise loop | per step | model path | warm median |
|---|---:|---:|---:|---:|
| eager (default today) | 602.1 ms | 60.1 ms | 696.3 ms | 0.861 s |
| `--compile-denoise-step` (recorded, bf16) | 215.6 ms | 21.6 ms | 311.9 ms | 0.411 s |
| OV fmha | 188 ms | 19 ms | 246 ms | — |

**Per-step Inductor already lands in the reference's latency class.** Phase 8
now supplies the gate that was missing in September; the default is compiled
fp16, with eager available as an explicit parity/debug opt-out.

## Steps

Ordered by expected return. A is most of the gap; do not start C before A.

| # | step | expected | status |
|---|---|---|---|
| A | **Gate and land the compiled denoise path** | −385 ms model | **done** |
| B | Locate the Inductor drift (only if A's gate fails) | correctness | skipped; A passed |
| C | One graph for all 10 steps, as the reference does | −25 to −50 ms | **closed 2026-09-08**: no XPU graph API exists and AOTInductor gives 0.99x on the real step, see F3 |
| D | Prefix stages: `prefix_fill` 64.5→42, `embed_prefix` 24.7→16 | −31 ms | compiled Prefix gate passed; default blocked by cold start |
| E | The ~165 ms outside the model path | −? served rate | not started |
| F | Phase 6 step 4 — make the harness able to track all of this | none directly | pending since Phase 6 |
| H | **Cut MoE weight bytes** — the loop is memory-bound at 51 FLOP/byte against a 207 machine balance | −30 ms to roofline, −30 more at int8 | not started; **the largest reachable item**, see F2 |

Step H displaced a grouped-MoE kernel, which F2 measured and rejected: the FLOP
argument promises 6.38x and delivers 1.27x, because the MoE streams the same
75.5 MB of weights whatever `M` is.

### C. Full-loop capture experiment — not deployable with current Inductor

The Euler loop is now isolated as `denoise_actions()` and the attribution probe
has an explicit `--compile-denoise-loop` experiment. On the B60, compiling the
complete 10-step graph with XPU Inductor remained in fullgraph generation for
more than six minutes without reaching the timed warmup. The existing compiled
single-step path starts measuring in seconds and delivers 215.6 ms denoise.

This rules out enabling the full-loop `torch.compile` path in the serving
default: its cold-start cost is not acceptable and no latency result was
produced. The loop boundary remains useful for a future XPU graph capture or
precompiled backend, but the next implementation should avoid asking Inductor
to lower the entire unrolled 36-layer x 10-step graph at runtime.

**Closed 2026-09-08 — see F3.** Both successors were then tried and both fail.
There is no XPU graph API in `torch 2.10.0+xpu` or `ipex 2.10.10.post1+xpu`, and
Inductor's `mode="reduce-overhead"` / `triton.cudagraphs` are accepted but are
measured no-ops (0.97x, 0.99x). AOTInductor *is* deployable and does answer the
cold-compile objection — it compiles offline in about a minute and loads in
seconds — but
on the real `predict_velocity` it runs 20.3 ms/step against the JIT path's 20.1,
because precompilation removes Python overhead and not per-kernel submission.
Step C
therefore needs a C++/SYCL extension binding `ext_oneapi_graph`, which the
device does advertise; until then step H is the better-ranked lever.

### D. Prefix attention experiments — partial improvements, no default change

The portable `eager_attention` path was compared with an opt-in
`--attention-backend sdpa` probe. With denoise compilation disabled, the B60
measurement was:

| backend | embed_prefix | prefix_fill | denoise | total |
|---|---:|---:|---:|---:|
| eager | 24.9 ms | 64.5 ms | 602-ish ms | 680.5 ms |
| SDPA | 24.8 ms | 58.2 ms | 592.0 ms | 680.5 ms |

SDPA saves about 6.3 ms in prefix fill, but the IPEX runtime warns that xetla
is unsupported. Combining SDPA with the existing compiled `predict_velocity`
path fails during Inductor tracing in the XPU scaled-dot-product-attention
backend. Therefore the production default remains eager attention plus compiled
denoise; a useful follow-up requires an IPEX xetla-capable SDPA backend or a
separate prefix-only attention selection so denoise can retain its known-good
compiled path.

The prefix-only selection is available as `--attention-backend prefix_sdpa`.
It runs without denoise compilation (`prefix_fill` about 50.6 ms in the probe),
but its cached prefix produces NaNs when passed through the compiled denoise
parity path. It remains a diagnostic probe, not a deployment option; the
current XPU stack needs a stable SDPA KV-cache contract before this route can
be combined with compiled denoise.

`prefix_sdpa_safe` then repaired the all-masked padding rows by allowing only
each padding query to attend to itself; valid query visibility is unchanged.
It removed the NaNs and combined with compiled denoise, but measured
`prefix_fill=59.7 ms` and `total=296.5 ms`, slightly slower than the existing
FP16 eager Prefix baseline (`57.6 ms` / `294.0 ms`). This closes the SDPA Prefix
route on the current stack: it is numerically stable after masking repair but
does not improve latency.

### Denoise-loop SDPA and FMHA — not deployable on the current XPU stack

The Prefix and loop were then tested independently. `suffix_sdpa` keeps the
Prefix eager and applies SDPA only to denoise, where there are no fully masked
query rows. It is finite and saves about `4.3 ms` on the ten-step loop in a
20-iteration probe (`200.6 -> 196.3 ms`), but its five-seed fp32-reference
result is MAE `1.538e-01` (range `1.347e-01-1.909e-01`), far above the
`2.882e-02` vendor ceiling. It must not be enabled as a serving backend.

The RobotWin six-chunk open-loop gate confirms the numerical regression is a
real task regression: eager scored MAE `0.00785235` while compiled
`suffix_sdpa` scored `0.03108115` (`+295.8%`), with MSE increasing from
`0.000505105` to `0.002748884`. The open-loop runner can reproduce this with
`--mode both --attention-backend suffix_sdpa`; it keeps the eager baseline on
the portable eager backend.

The available XE2 FlashAttention varlen API cannot directly express the mixed
suffix mask: the state token sees valid Prefix plus itself, while every action
token sees valid Prefix plus the complete state/action block. A semantically
exact `flash_suffix` split into two unmasked calls was tested. It is finite but
slower without Inductor (`816 ms` denoise), and cannot enter the compiled
denoise graph because constructing compressed `cu_seqlens` requires a
data-dependent token count. FMHA therefore also remains an experimental
diagnostic path, not a deployment option.

### vLLM IPEX varlen Prefix experiment

The reference `/llm/vllm` path exposes `vllm._ipex_ops.ipex_ops.varlen_attention`.
The real LingBot mask was inspected before using it: the Prefix is `[286,286]`
with 223 valid tokens, and removing padding gives an exact 223-token causal
mask (`bad_positions=0`). This makes an IPEX varlen implementation semantically
possible, so `--attention-backend ipex_prefix` now compresses valid tokens,
calls the XPU causal kernel, and restores the padded layout.

It is stable but slower on the current B60 runtime:

| path | prefix_fill | denoise | total |
|---|---:|---:|---:|
| FP16 eager attention + compiled denoise | 57.6 ms | 205.9 ms | 294.0 ms |
| IPEX varlen Prefix + compiled denoise | 75.4 ms | 206.3 ms | 311.8 ms |

The IPEX path also showed about 1.1% compiled chunk relative drift. The kernel
is retained as a diagnostic/reference implementation, but it is not a default
optimization. The likely next kernel work is a fixed-shape XPU FMHA path that
avoids per-layer token compression and supports the LingBot GQA layout directly.

`/llm/vllm-xpu-kernels` contains an existing XE2/XE3
`cutlass_chunk_prefill_interface` with causal varlen prefill,
`cu_seqlens_q/k`, static maximum sequence lengths, and
`[seq, heads, head_size]` inputs. This is a better long-term target than the
current IPEX adapter because it can avoid per-layer compression and restore
operations.

It is already reachable through the existing
`vllm_xpu_kernels.flash_attn_varlen_func` binding, so a `flash_prefix` probe was
added without a new C++ extension. On the current B60 it is stable but not
faster: `prefix_fill` was about `75.8 ms` versus `57.6 ms` for the existing
FP16 eager path. The compiled combination also showed about `0.9%` chunk
relative drift. The remaining optimization is therefore kernel/layout tuning
inside the XE2 path, not another Python-level backend switch.

A `flash_prefix_gqa` probe also passed native Q/KV head counts directly
(`32` query heads, `8` KV heads), avoiding the explicit KV repeat. The kernel
accepted the layout but still measured `prefix_fill` around `73.9 ms`, so KV
head materialization is not the only bottleneck. The dominant remaining cost
is the batch-1 short-sequence kernel schedule and per-layer invocation/layout
overhead; a useful next kernel change must fuse or amortize that overhead.

The second probe changed only the attention accumulation dtype while keeping
the eager attention implementation and compiled denoise path:

| attention precision | embed_prefix | prefix_fill | denoise | total |
|---|---:|---:|---:|---:|
| fp32 (default, 20 iters) | 24.8 ms | 66.9 ms | 217.4 ms | 314.6 ms |
| fp16 (experiment, 20 iters) | 24.8 ms | 57.5 ms | 207.1 ms | 294.7 ms |

Under identical 5-warmup/20-iteration conditions this saves about 19.9 ms
(`6.3%`) and reduces the vLLM/OpenVINO total ratio from `1.28x` to `1.20x`.

### Compiled Prefix gate — passed, opt-in only

The fixed-shape 36-layer Prefix walk was compiled as a separate graph through
`--compile-prefix`. After the first graph build, it reduced `prefix_fill` from
about `57.6 ms` to `41.2 ms`. Combined with compiled denoise, the synchronized
model path measured `278.4 ms`, compared with `294.0 ms` without compiled
Prefix; the OpenVINO comparison is `1.13x` (`278.4 / 246.0`).

Both accuracy gates passed:

| gate | result |
|---|---:|
| fp32 reference, 5 noise seeds | `2.015e-02` mean MAE |
| RobotWin bundle, 6 chunks | `0.00784710` MAE vs eager `0.00785235` |
| real WebSocket serving | `0.347 s` median, `2.88 Hz` |

The cost is cold start. The real serving harness required `138 s` to become
ready because Inductor compiled the Prefix graph during initialization. The
graph is therefore exposed through `--compile-prefix` in the preparation,
open-loop, and performance scripts, but `compile_prefix` remains false by
default. Making this the default requires persistent/precompiled graph reuse or
a substantially cheaper segmented compilation strategy.

### FP16 attention gate — passed 2026-09-07

The five-seed fp32-reference check gave compiled FP16 attention mean MAE
`1.990e-02`, below the `2.882e-02` vendor ceiling. The RobotWin six-chunk
task check also passed:

| candidate | MAE | MSE | cosine |
|---|---:|---:|---:|
| eager FP32 attention | `0.00785235` | `0.000505105` | `0.999672920` |
| compiled FP16 attention | `0.00784798` | `0.000505800` | `0.999672443` |

FP16 attention is now the default in `LingbotVlaV2Config` and prepared models.
Pass `--attention-precision fp32` for the parity/debug baseline. The eager
open-loop baseline explicitly remains FP32 so future comparisons do not change
meaning when the product default changes.

### Compiled Prefix gate — passed numerically, opt-in for cold-start cost

The Prefix fullgraph path passed the five-seed fp32-reference gate with mean
MAE `2.015e-02` and the six-chunk RobotWin task gate with MAE `0.00784710`
versus eager `0.00785235`. Its steady-state model path measured `278.4 ms`
and the real WebSocket harness measured `0.347 s` median (`2.88 Hz`).

The tradeoff is startup: the real server took `138 s` to become ready because
the 36-layer Prefix graph compiles during initialization. `--compile-prefix`
is therefore available in `run_perf_check.sh` and `run_open_loop_eval.sh`, but
`compile_prefix` remains false by default until the compiled graph can be
persisted/reused without imposing this startup cost.

### A. Gate and land the compiled denoise path — passed 2026-09-07

The gate was defined before reading results: compiled ships at fp32-referenced
MAE ≤ `2.882e-02` (the vendor OpenVINO INT8 ceiling) with no open-loop task-MAE
regression. It passed both checks:

| gate | eager fp16 | compiled fp16 | verdict |
|---|---:|---:|---|
| fp32 reference, 5 noise seeds | `1.949e-02` MAE | `1.961e-02` MAE | pass |
| RobotWin bundle, 6 chunks / 300 steps | `7.852e-03` MAE | `7.833e-03` MAE | pass |

The compiled path has been made the default (`compile_denoise_step=True`). It
measured 215.6 ms denoise / 311.9 ms synchronized model path, versus 602.1 ms /
696.3 ms eager. `--no-compile-denoise-step` is the parity/debug opt-out.

### Reproducible OpenVINO comparison

Use `run_openvino_comparison.sh` when measuring from the vLLM-Omni environment.
It runs only the vLLM in-process attribution probe with the selected warmup and
repeat counts, then compares `vit`, `text`, `denoise`, and `total` against the
recorded OpenVINO reference:

```bash
examples/online_serving/lingbot_vla_v2/run_openvino_comparison.sh \
  --model /tmp/lingbot-vla-v2-perf \
  --warmup 5 --repeat 20 \
  --output /tmp/lingbot-openvino-comparison.json
```

The script never launches OpenVINO and does not require the OpenVINO repository
or virtualenv. Add `--eager` to measure the vLLM eager baseline. The reference
values are `vit=16 ms`, `text=42 ms`, `denoise=188 ms`, and `total=246 ms`, from
`run_info_demo_dgpu_int8.sh` with `action-mode=loop`.

The OpenVINO side is fixed to `GPU.1`, `suffix=int8`, and `action-mode=loop`,
which means one IR call contains all ten denoise steps. The vLLM side reports
the equivalent prefix stages and ten `predict_velocity` calls with device
synchronization. The script prints `vLLM/OpenVINO` ratios and records the
boundary difference in JSON. Neither side includes WebSocket, MessagePack,
engine scheduling, or server IPC; use `run_perf_check.sh` separately for that
end-to-end serving measurement.

### Original A plan

The whole phase. Everything needed to judge it now exists.

- **A1. Re-measure the drift at fp16.** The 1.75% chunk drift and the 3.18%
  velocity drift were both measured in bf16. fp16 has three more mantissa bits
  and changed the numerics everywhere else in Phase 7; this number cannot be
  carried over. Re-run the eager-vs-compiled comparison at `--dtype float16`
  before anything else, because if it drops materially the gate is trivial.
- **A2. Grade compiled against the fp32 reference, not against eager.** Add a
  compiled candidate to `phase7_numeric_parity.py --candidates`. This is the
  measurement that makes the decision, because "1.75% from eager" is not a
  quantity anyone can act on, whereas "mae 2.1e-2 against fp32, where OV FP16
  is 1.61e-2 and OV INT8 is 2.88e-2" says immediately whether the compiled path
  is inside the envelope the vendor already ships. Use ≥5 noise seeds — Phase 7
  established this metric has ~1.8x spread across single draws.
- **A3. Task accuracy, eager vs compiled.** `run_open_loop_eval.sh --mode both`
  then `compare_open_loop.py`. Both now report cosine / MAE / MSE / max abs /
  p99 abs, so the comparison is no longer two MSE scalars.
- **A4. Write the gate down before reading A2 and A3.** Proposed: compiled
  ships if its fp32-referenced mae is **≤ 2.88e-2** (the OV INT8 column — the
  vendor ships that precision, so it is a defensible ceiling rather than an
  invented one) **and** task mae does not regress beyond the noise of the
  open-loop bundle. Deciding the threshold after seeing the number is how a 3x
  speedup talks its way past a real regression.
- **A5. Land it.** Flip `compile_denoise_step` to `True` in
  `vllm_omni/diffusion/models/lingbot_vla_v2/config.py:162`, and re-examine
  `--enforce-eager` in `run_openpi_server.sh:58` and `run_perf_check.sh:154` —
  it is on both server paths and was never revisited after it was added.
  Warm-up cost has to be measured too: Inductor compiles on first request, and
  a 7 s first chunk is a deployment problem even if the warm median is 0.4 s.

**Risk to state plainly:** if A4's gate fails, the 3x is not available on this
route and B becomes mandatory rather than conditional. Do not pre-commit to
landing A.

### B. Locate the Inductor drift — conditional

Only if A4 fails. Phase 6 established `aot_eager` is bit-exact with no speedup
and that `force_same_precision` / `emulate_precision_casts` do not remove the
drift, so the cause is a lowering decision, not a global precision mode. The
work is a bisection: compile subsets of `predict_velocity` until the first
lowered op that introduces the drift has a name.

### C. One graph for all 10 steps

Even compiled, we launch the step graph 10 times with Python between launches;
the reference makes one call. The residual after A is 21.6 vs 19 ms/step plus
whatever the inter-step glue costs. Options: extend the compile boundary over
the loop at static shapes, or capture the loop as an XPU graph. Worth 25-50 ms —
real, but a tenth of A, so it must not be started first.

*(Both options have since been tried and both are closed — F3. Kept here as the
original reasoning; the "capture the loop as an XPU graph" option turned out to
have no Python binding on this stack at all.)*

### D. Prefix stages

`prefix_fill` at 64.5 ms against 42 ms is the bigger of the two, and the
reference's own logs show where its win came from: **fmha took it 59 → 42 ms**
and the loop 213 → 188 ms. Check which attention backend the 36-layer, 286-token
prefill actually selects and whether the mask is blocking SDPA fusion, which is
the same class of problem the reference solved with its `Select → Add` rewrite.

### E. The ~165 ms outside the model path

Wall 861 ms − model 696 ms. Never attributed. It is not part of the 246 ms
target, which is model-only, but it is part of the rate a robot actually sees:
after A it would be roughly a third of the total. Split it into WebSocket
transport, serialization, and engine scheduling before assuming which one it is.

### F. Harness work (Phase 6 step 4, still pending)

`run_perf_check.sh` prints dtype, MoE mode and compile mode as of 2026-09-07 but
still lacks: the model-only subtotal as a first-class line comparable to 246 ms,
the OV reference printed next to Phase 0's 0.74 s, p90, and a JSON artifact that
can be diffed across runs. Every step above produces numbers this harness should
be recording automatically.

### F1. Compiled fp16 per-stage profile — done 2026-09-08

`phase8_stage_device_profile.py`, `/tmp/lingbot-vla-v2-perf`, fp16,
`compile_denoise_step` on, `eager`/`fp16` attention, `moe=dense`,
`inference_mode`, 20 repeats — i.e. exactly the knobs
`run_openvino_comparison.sh` passes to `phase5_latency.py`, which is what
produced the recorded 294.0 ms:

```text
stage                                 wall  host issue     drain  repeat/it  verdict
embed_prefix                         23.7m       23.6m      0.1m      23.7m  dispatch-bound
prefix_forward                       57.2m       56.6m      0.7m      57.1m  dispatch-bound
predict_velocity (1 step)            22.4m       20.0m      2.5m      20.1m  mixed
denoise_actions (10 steps)          201.5m      195.1m      6.4m     201.4m  dispatch-bound
sample_actions (whole request)      286.5m      282.6m      3.8m     286.5m  dispatch-bound
```

Stages sum to 282.2 against a whole-request 286.5 (1.5% unaccounted, the glue
between them), so the decomposition accounts for the request. `prefix_forward` at
57.1 ms reproduces the recorded `prefix_fill` 57.6 ms and the loop's 201.4
reproduces 205.9, which is the cross-check that matters: this is the same
configuration measured a different way, and the two agree.

**A first run of this profile reported 24.7 / 67.7 / 213.3 = 305.0 ms and I
wrote it into this file as if the older 294.0 had been measured under a different
attention backend. That was wrong, and the error was mine, not the older
number's.** `LingbotVlaV2WithExpertModel.__init__` hardcodes
`attention_precision = "fp32"` (`modeling_lingbot_vla_v2.py:942`); the deployed
`fp16` arrives by assignment from the config in `pipeline_lingbot_vla_v2.py:61`,
and `phase5_latency.py:460-462` mirrors it. The profile script did neither, so it
was measuring fp32 attention, and it also ran outside `torch.inference_mode()`
(`pipeline_lingbot_vla_v2.py:141`), paying autograd bookkeeping on every op —
which for a dispatch-bound path lands entirely in the measurement. fp32 attention
accounts for the +10.6 ms on `prefix_forward`; the rest is the missing
`inference_mode`. Both are now script flags defaulting to the deployed values,
and the run prints `attention=` and `inference_mode=` so a mismatch is visible in
the output.

The lesson worth keeping: **a config knob that a module sets in `__init__` and
the pipeline overrides afterwards will be silently wrong in any harness that
constructs the module directly.** `attention_backend`, `attention_precision`,
`compile_prefix` and `compile_denoise_step` are all in that category here.

#### Method, and why not the profiler

The profiler is not asked. It reports per-operator device time, and on this
backend `mm`/`einsum` report zero — the mistake that produced the retracted
"51 ms device / 243 ms host" split. Instead each stage is submitted without
synchronising, and the host's return time is separated from the device's:

```
sync; t0;  run stage;  t_issued;  sync;  t_done
host_issue = t_issued - t0     drain = t_done - t_issued
```

`drain` is a hard lower bound on device time — the device was demonstrably busy
that long with nothing more being fed to it.

**The control that makes this readable.** A near-zero `drain` could equally mean
"the path synchronises internally", which would make the whole table meaningless.
It does not: on this stack 117 queued 4096×4096 fp16 matmuls — ~298 ms of device
work — submit in **1.8 ms** and leave **168.3 ms** of drain outstanding.
Submission is genuinely asynchronous, so the host *could* have run ahead here and
did not.

#### Result: still dispatch-bound, but the host tax is ~40 ms, not ~243 ms

Every stage is dispatch-bound. The host's submission is the binding constraint
end to end: 282.6 ms of the 286.5 ms request, with only 3.8 ms of device work
outstanding when submission returns.

**And here is the limit of this method, stated plainly: a dispatch-bound workload
cannot have its device time measured by timing.** The host never lets the queue
build up, so there is no drain to read. This test bounds device work to
[3.8, 286.5] ms, which is useless as an estimate.

The external estimate is OpenVINO, which submits each stage as one IR call and is
therefore close to pure device time. Against it:

| stage | ours | OV (fmha) | host tax |
|---|---|---|---|
| embed_prefix / vit | 23.7 | 16 | +7.7 |
| prefix_forward / text | 57.1 | 42 | +15.1 |
| denoise loop | 201.4 | 188 | +13.4 |
| | **286.5** | **246** | **+40.5** |

So the retracted claim was **right in direction and wrong in magnitude**: the
path is dispatch-bound, but the device is doing the large majority of the work
and we pay roughly a 16% submission tax on top of it — not 243 ms of host time
against 51 ms of device time.

#### What this settles

- **Step C's remaining upside is ~13 ms**, the loop's distance from OV's 188 ms —
  not the large win implied when 243 ms looked like host time. The prefix is ~15
  and the ViT ~8. Every stage is within ~15 ms of the device floor, so there is
  no single large win left anywhere in the model path. Parity with 246 ms now
  means winning all three, or finding it outside the model path (step E).
- **Batching converges on the device floor, and that is all.** Raising the batch
  does not change the op count, so host submission stays ~283 ms while device
  work scales. At batch=2, 2 × 246 = 492 ms > 283 ms, so the workload flips to
  device-bound and lands near **246 ms/request against 286.5** — about **1.16x**,
  not the 1.7x projected in G3 mechanism 1. Worth doing for a fleet, much less
  dramatic.
- **G2's ~10.5% dual-device ceiling stands and is now on firmer ground**, since
  device work really is the bulk of the request. The retraction of "1.7%" was
  correct.
- **G4's conclusion gets slightly stronger.** Its arithmetic used stage wall
  times, which this measures directly: the dGPU ViT is 23.7 ms, not the 30.5 ms
  assumed, so an iGPU ViT prefetch wins *less* period (8.3%, was 10.4%) and loses
  *more* staleness (53.3 ms, was 46.5 ms). Both tables updated in place.

#### `moe=dense` is the right thing to have profiled, on three counts

Worth spelling out, because the whole comparison against OV rests on it:

1. **It is what ran, not what was assumed.** The script passes `moe=None` to
   `build()`, so the value comes from `/tmp/lingbot-vla-v2-perf/transformer/config.json`
   (`"moe_implementation": "dense"`), and the run prints `moe=dense`.
2. **It matches the 294.0 ms run.** `phase5_latency.py:406` defaults `--moe` to
   `None`, and neither `run_openvino_comparison.sh` nor `compare_openvino.py`
   passes it. Both runs used the same kernel.
3. **The reference is dense too, verified rather than inferred.**
   `PHASE5_PERF.md`'s 2026-09-04 correction parses
   `converter/action_expert_loop_int8.xml`: four einsums per MoE layer per step
   over all 32 experts, `TopK` present once but only building routing weights,
   and all 1359.0 M expert weights fp16. **Both sides compute the same 1.836
   TFLOP.** So the 201.4 vs 188 ms comparison is arithmetic-for-arithmetic.

That makes one more number computable, and it is the most encouraging in this
file:

| | loop | effective |
|---|---|---|
| ours, eager bf16 (Phase 5) | 607.3 ms | 3.02 TFLOPS |
| **ours, compiled fp16 (F1)** | **201.4 ms** | **9.12 TFLOPS** |
| OV, `dgpu_int8` | 213.0 ms | 8.62 TFLOPS |
| OV, fmha | 188.0 ms | 9.77 TFLOPS |

**Phase 5's "2.85x per-FLOP efficiency gap" is closed.** We are now within 7% of
the reference's per-FLOP rate on the denoise loop, and we are *faster* per FLOP
than the non-fmha reference run. Whatever is left in the loop is not arithmetic
efficiency.

### F2. The grouped-MoE kernel — measured and rejected 2026-09-08

Immediately after F1 I wrote here that a grouped top-4 MoE was "the only
remaining item larger than the ~40 ms host tax", projecting a 68 ms loop and a
149 ms request from the 2.95x FLOP ratio. **That was wrong.** It applied a FLOP
ratio to a loop that is not FLOP-bound, and assumed the per-FLOP efficiency
would survive the shape change. Four measurements, all cheap, kill it.

#### 1. Top-4 routing does not reduce weight traffic — 30.4 of 32 experts are hit

`phase8_moe_routing_probe.py` hooks all 36 `TokenMoeBlock`s and reproduces the router
exactly (fp32 gate, sigmoid, `e_score_correction_bias`), over one real request:

```text
invocations: 360  (36 layers x 10 steps)     tokens per invocation: 51   E=32, top_k=4
distinct experts selected: min=22 max=32 mean=30.6
invocations touching ALL 32 experts: 139/360 (38.6%)
```

51 tokens × top-4 = 204 selections over 32 experts, so nearly every expert is
selected by *someone*. Its weights get read either way. **Top-4 cuts arithmetic
2.95x and weight bytes by 4%.** (Across runs the mean is 30.4–30.6; the sampled
noise differs, the conclusion does not.)

##### "But only 4 of 32 are active per token — why move all 32?"

Because an expert's weights are read **once per invocation, not once per token**,
so what sets traffic is the *union* of the choices of the tokens that share the
read, not the per-token count. The same probe measures that union at every group
size, splitting the 51 tokens into consecutive groups (one expert = 2.36 MB of
gate+up+down at fp16):

| tokens/group | groups | distinct experts | MB read per invocation | vs dense |
|---|---|---|---|---|
| 1 | 51 | 4.0 | 481.3 | 6.38x |
| 2 | 26 | 7.0 | 427.6 | 5.66x |
| 4 | 13 | 11.5 | 353.3 | 4.68x |
| 8 | 7 | 16.4 | 271.7 | 3.60x |
| 16 | 4 | 20.1 | 189.4 | 2.51x |
| 26 | 2 | 27.0 | 127.5 | 1.69x |
| **51 (today)** | **1** | **30.4** | **71.8** | **0.95x** |

Per-token sparsity is real — a single token needs exactly 4 experts, 9.4 MB, 12%
of the dense read. But it is not *extractable*, because the union saturates fast:
the chance an expert is missed by all 51 tokens is about `(1 − 4/32)^51 ≈ 0.001`,
so essentially every expert is wanted by somebody. Chasing the sparsity by
processing fewer tokens per invocation makes it worse in exactly the proportion
the table shows — each group re-reads its own experts from DRAM, and the product
`groups × distinct` rises monotonically as the group shrinks.

So **the dense path is already the byte-minimal grouping at T=51**, and a perfect
gather kernel would save 5% of bytes (75.5 → 71.8 MB). That is the whole reason
F2 rejects the grouped kernel, and it is also why the lever has to be the bytes
per expert (int8) or more tokens per read (batching), not the routing.

#### 2. The dense MoE is memory-bound, by 4x

B60 achievable, both measured (`phase8_moe_roofline.py`,
`phase8_bandwidth_probe.py`): **93.1 TFLOPS** fp16 (4096³ GEMM in 1.48 ms) and
**449 GB/s** (`sum` over a 2.72 GB fp16 tensor in 6.1 ms). Machine balance =
**207 FLOP/byte**.

One MoE layer-step reads 75.5 MB of expert weights (`32 × 3 × 768 × 512` fp16)
to do 3.85 GFLOP: **arithmetic intensity 51 FLOP/byte**, four times below
balance. Roofline time is `max(mem 0.168, compute 0.041) = 0.168 ms`; measured
`bmm` is 0.251 ms, i.e. **67% of roofline**.

*(Corrected 2026-09-08: this said 0.290 ms and 58%. The first `bench()` call in
the probe process reads ~15% slow — oneDNN primitive setup that five warmup
iterations do not absorb — and the original sweep measured `M=51` first. As the
sole call it still reproduces 0.290; after one prior call 0.264; warm 0.251,
stable to ±0.001 over three runs. The probe now discards a warm-up call and uses
200 iterations. Everything downstream of this number shrank by ~14 ms.)*

So FLOPs are not the constraint and never were. Bytes are.

##### Where the weights are moved *to*, and why nothing stays there

The B60's hierarchy, from `clinfo` and `torch.xpu.get_device_properties`:

| level | size | note |
|---|---|---|
| VRAM (GDDR6) | 22.71 GiB | 449 GB/s measured; 2400 MHz, 160 Xe-cores |
| L2 / last-level cache | **18 MiB** | 256 B cache line |
| SLM (per work-group) | 128 KiB | |
| registers → XMX | — | `has_subgroup_matrix_multiply_accumulate` |

Set against the working set, this is the whole story:

- one expert's gate+up+down is **2.36 MB** — fits L2 comfortably
- one invocation needs **75.5 MB** — **4x larger than L2**
- all 36 layers' routed experts are **2.7 GB** — **143x L2**

So the destination is not a place anything *stays*. Each weight element is
fetched from VRAM, held in registers/SLM just long enough to be multiplied
against the token rows, and evicted; the next denoise step re-fetches all 2.7 GB
from VRAM because there is nowhere to keep it. Ten steps, ten full passes,
27.2 GB, 60.5 ms.

And the reuse that *does* happen is exactly `M`: while a weight element is
resident it serves all 51 token rows, which is 2·51 = 102 FLOP per 2 bytes —
**precisely the measured AI of 51 FLOP/byte.** The current kernel is therefore
already fetching each element once per invocation, which is the minimum. That
leaves only two ways to move the ratio, and they are the two in step H: fewer
bytes per element (int8), or more rows per element (batching).

It also locates the batching crossover exactly. AI = `M` = 51·B, and the machine
balance is 207, so:

| batch | M | AI | regime |
|---|---|---|---|
| 1 | 51 | 51 | memory-bound by 4x |
| 2 | 102 | 102 | memory-bound by 2x |
| **4** | **204** | **204** | **≈ balance (207) — the crossover** |
| 8 | 408 | 408 | compute-bound |

Which is why per-request GEMM time falls steeply to B=4 (0.251 → 0.110 ms) and
then flattens (B=8: 0.088). Past batch 4 there is no bandwidth dividend left to
collect.

#### 3. So the FLOP saving does not materialise — 6.38x promised, 1.27x delivered

`phase8_moe_gemm_probe.py`, the real shapes (`E=32, H=768, I=512`), gate + up +
`silu`× + down as three `bmm`s, × 360 invocations:

| M rows/expert | ms/layer-step | TFLOPS | loop projection |
|---|---|---|---|
| 408 (batch 8) | 0.700 | 44.00 | 252.0 ms |
| 204 (batch 4) | 0.438 | 35.15 | 157.8 ms |
| 102 (batch 2) | 0.298 | 25.81 | 107.4 ms |
| **51 (dense, batch 1)** | 0.251 | **15.33** | **90.4 ms** |
| 16 | 0.205 | 5.90 | 73.7 ms |
| **8 (grouped top-4, capacity 8; avg need 6.4)** | 0.198 | **3.04** | **71.5 ms** |
| 4 | 0.194 | 1.56 | 69.8 ms |
| 2 | 0.191 | 0.79 | 68.6 ms |

**The time is nearly independent of `M`.** Going from 51 rows to 2 — a 25x FLOP
cut — buys 24%. Effective throughput collapses from 15.33 to 0.79 TFLOPS in
almost exact proportion, because what the kernel is actually doing is streaming
75.5 MB of weights regardless. A perfect grouped kernel saves **~19 ms**
(90.4 → 71.5), not the ~133 ms the FLOP ratio promises.

The rows above `M=51` were added later and say something the rest of this file
had wrong: **batching is nearly free on the dominant arithmetic.** Doubling the
tokens costs 1.19x, not 2x, and per request the same GEMMs fall 0.251 → 0.149 →
0.110 → 0.088 ms at B=1/2/4/8. Same weights, more rows against them. G3's
batching estimate assumed device work scales with B and is therefore too
pessimistic; it needs re-deriving, and the answer will favour batching.

#### 4. And the deployment constraint forces the bad shape

`pipeline_lingbot_vla_v2.py:69-75` compiles `predict_velocity` with
`fullgraph=True, dynamic=False`. Data-dependent group sizes cannot appear in that
graph, so a grouped kernel must pad every expert to a fixed capacity — which is
precisely the `M=8`/`M=16` rows measured above. The 1.27x is not a pessimistic
proxy for the real thing; it *is* the real thing.

#### What building it would actually take, for the record

If someone wants it anyway: replace `forward_gather`'s Python loop over 32
experts (`modeling_lingbot_vla_v2.py:737-750`) with (a) `bincount`/`cumsum` over
`selected_experts` for per-expert offsets, (b) a gather of routed rows into a
contiguous `[E, C, H]` capacity-padded buffer, (c) three `bmm`s or a variable-M
grouped GEMM, (d) a weighted `index_add_` scatter back. On XPU there is no
exposed cutlass grouped-GEMM binding, so a variable-M kernel means Triton
through the XPU backend or oneDNN directly — and a variable-M kernel is the one
thing `dynamic=False` forbids. Plus a Phase 7 accuracy gate, since dense is the
path the fp32 golden was produced with (`GroupedExperts` docstring) and any
grouped kernel changes summation order. Days of work for ~19 ms, at best.

#### The right lever is bytes, and nobody has pulled it

The same roofline that kills the grouped kernel hands over a better target. MoE
GEMMs are **90.4 ms of the 201.4 ms loop**, against a fp16 roofline of 60.5 ms:

| MoE GEMMs, ×360 invocations | time |
|---|---|
| measured today | 90.4 ms |
| roofline, fp16 weights | 60.5 ms |
| roofline, int8 weights | 30.3 ms |
| roofline, int4 weights | 15.1 ms |

Two independent gains, neither touching a FLOP:

1. **Close the 67%-of-roofline gap** — 30 ms, from memory-access shape alone
   (layout, `bmm` vs a fused kernel, avoiding the `[E,T,I]` intermediates that
   `forward_dense` materialises). No numerics change at all.
2. **Quantize the routed experts** — the roofline is proportional to bytes, so
   int8 halves it. And the reference has *not* done this: `PHASE5_PERF.md`'s XML
   parse found all 1359.0 M expert weight elements are fp16, with the `_int8` in
   the filename covering only attention projections and the shared expert. This
   is the one lever where we can go past OV rather than catch up to it, the
   accuracy gate for it already exists from Phase 7, and weight-only
   quantization is far less work than a grouped GEMM.

Ranked against everything else in this file: bytes (up to ~60 ms: 30 to the
fp16 roofline, 30 more at int8) > host dispatch tax (~40 ms, but **no way to
reach it from Python** — F3) > grouped MoE (~19 ms, days of work, forbidden
shape). That leaves bytes as the only large item that is both real and
reachable.

Still open from step F, and now the top item — added 2026-09-08 (from G3):
record the **control rate**. `chunk_size: 50` is in the config but the Hz that
turns 50 steps into a time budget is nowhere in this repo, and it is the number
that decides whether 286 ms is comfortable or marginal. Without it every latency
figure here is unanchored.

### F3. XPU graph capture — unavailable, and precompilation does not substitute, 2026-09-08

F1 proved the request is dispatch-bound: 282.6 ms of the 286.5 ms is the host
submitting work and the device drains in 3.8 ms. On CUDA that diagnosis has a
standard cure — capture the kernel sequence into a graph, then replay it with one
submission — so this was worth settling before spending anything on step H.
Three questions, two probes: `phase8_graph_probe.py` (API surface, device
capability, synthetic chain) and `phase8_aoti_step_probe.py` (the real step).

**1. The hardware and driver support SYCL graphs. PyTorch does not expose them.**

`sycl-ls --verbose` lists `ext_oneapi_graph` and `ext_oneapi_limited_graph`
among the B60's aspects (driver 1.14.37435+12, oneAPI Unified Runtime over
Level-Zero V2); the iGPU advertises them too. So the capability is real at the
runtime level. Nothing binds it into Python:

| probe | result |
|---|---|
| `torch.xpu` attrs matching `graph` | none |
| `torch.xpu.XPUGraph` / `graph` / `graph_pool_handle` | absent |
| `torch.xpu.make_graphed_callables` / `is_current_stream_capturing` | absent |
| `torch.xpu.graphs` module | `ModuleNotFoundError` |
| `torch._C._CUDAGraph` / `torch._C._XPUGraph` | present / **absent** |
| `intel_extension_for_pytorch` attrs matching `graph` | none |
| `torch._inductor.config.triton.cudagraphs` | `False` |

That is `torch 2.10.0+xpu` and `ipex 2.10.10.post1+xpu`.

**2. The Inductor graph knobs are accepted and are no-ops.**

`mode="reduce-overhead"` does not raise, and neither does setting
`triton.cudagraphs = True` by hand — the flag even reads back `True` afterwards,
and `cudagraph_trees` defaults on. It changes no timing. On 60 chained
`8x64 @ 64x64` fp16 matmuls, a shape chosen so that per-kernel submission is
essentially the entire cost, over three runs of 300 iterations:

| variant | steady | vs `torch.compile` |
|---|---|---|
| eager | 1.22 – 1.25 ms | |
| `torch.compile` (default) | 1.38 – 1.40 ms | 1.00x |
| `torch.compile mode="reduce-overhead"` | 1.41 – 1.43 ms | **0.97 – 1.00x** |
| `torch.compile` + forced `triton.cudagraphs` | 1.40 – 1.42 ms | **0.97 – 1.00x** |
| AOTInductor `.pt2` | 1.21 – 1.27 ms | 1.11 – 1.15x |

A silently-ignored flag is worse than a missing one, which is why this needed a
measurement and not just a `hasattr` check.

Two cautions about this table, both learned the hard way. First, at the 50
iterations used on the first pass the chain moves ~15% run to run — larger than
every effect here — and that noise produced a "1.20x for AOTInductor" that did
not survive 300 iterations. The probe's default is now 300. Second, even the
stable 1.13x for AOTInductor is not a win over *eager*: it recovers what
`torch.compile`'s guard and wrapper machinery costs on a chain this small, and
lands back where eager already was.

**3. AOTInductor works, deploys, and does not help the real step.**

This is the interesting half. Step C failed on *cold compile* — Inductor stayed
in fullgraph generation for over six minutes — and AOTInductor answers precisely
that objection: `torch.export` plus `aoti_compile_and_package` move lowering
offline and emit a `.pt2` that loads in seconds. Both entry points exist on this
stack and `torch.export` handles XPU tensors, so it was applied to the real
`predict_velocity`:

| | |
|---|---|
| `torch.export` | 18.0 s, 7364 nodes |
| `aoti_compile_and_package` | 51 – 63 s, **3624 MB** package |
| AOTInductor steady | **20.3 ms/step** |
| JIT-compiled baseline (F1) | 20.1 ms/step |

**No gain — 0.99x.** And the package is 3.6 GB because the prefix KV cache and
every weight closed over by the exported step are frozen in as constants.

The two results agree, and this is the conclusion that matters: AOTInductor
removes *Python-side* overhead — guards, wrapper, dispatcher — which on 60
trivially small kernels is worth 1.13x against `torch.compile` and nothing at
all against eager. It does not remove *submission*: it still enqueues each
kernel into the Level-Zero queue one at a time, from C++ instead of from Python.
The real step's kernels are already Inductor-fused, so the Python fraction of
its host time is small; what remains is per-kernel driver submission, and only a
genuine graph collapses many submissions into one. Precompilation and graph
capture are different mechanisms, and only the second one addresses F1's
diagnosis. That is also why the 3.6 GB package is not worth engineering around —
even a lean export would not change the timing.

**Verdict for step C.** The host tax cannot be recovered from Python today.
Doing it needs a C++/SYCL extension that binds `ext_oneapi_graph`, records the
command list once and replays it — real work, against an extension that is still
moving, for a ceiling that is smaller than it first looks: step C is the loop
alone, whose host tax over the OV device floor is **~13 ms** (G2), and capturing
every stage caps out at the whole ~40 ms. Step H is better ranked on both counts:
larger (up to ~60 ms), and it needs no new runtime binding, with its first 30 ms
changing no numerics. Revisit if a later `torch-xpu-ops` ships an `XPUGraph`; the
check is one run of `phase8_graph_probe.py --skip-chain`.

## G. dGPU + iGPU — evaluated and rejected for latency, 2026-09-08

Asked whether `/llm/zhuyong/vllm`'s iGPU pipeline-parallel path — the one that
runs Qwen3.6-35B-A3B across both cards with `VLLM_XPU_IGPU_PP=1`,
`VLLM_XPU_IGPU_PP_MASKS=0,1`, `VLLM_PP_LAYER_PARTITION=30,10`,
`CCL_PLUGIN=ONECCL_IGPU` — should be ported into vLLM-Omni for
LingBot-VLA-V2-6B. **No.** It makes this model slower on both metrics, and the
vendor already measured it.

### What the vLLM path actually does

`multiproc_executor.py:575-625` sets, per worker, before spawn:

```python
masks = os.getenv("VLLM_XPU_IGPU_MASKS", os.getenv("VLLM_XPU_IGPU_PP_MASKS", "0,1")).split(",")
process_env = {"ZE_AFFINITY_MASK": masks[local_rank].strip(), "CCL_LOCAL_RANK": ..., ...}
```

so each rank sees exactly one physical XPU and both use process-local `xpu:0`
(`xpu_worker.py:195-245`, which also skips the global oneCCL all-reduce warmup in
igpu_mode). `VLLM_PP_LAYER_PARTITION=30,10` then puts 30 decoder layers on rank 0
and 10 on rank 1, and the hidden-state tensor dict hops between them through
`oneccl_igpu_communicator.py`. Note the enumeration is the inverse of
OpenVINO's: here `ZE_AFFINITY_MASK=0` is the B60 dGPU and `1` is the iGPU
(`xpu-smi discovery`: id 0 = Arc Pro B60 at `0000:03:00.0`, id 1 = Intel Graphics
at `0000:00:02.0`), whereas OV calls the iGPU `GPU.0`.

### Why it works there and not here

| | Qwen3.6-35B-A3B | LingBot-VLA-V2-6B |
|---|---|---|
| Why PP is used | **capacity** — 35B does not fit one card | nothing: 11.73 GiB of the B60's 23.9 GiB |
| Workload | continuous batching, many in-flight micro-batches | one robot, batch=1, one chunk at a time |
| Metric | tokens/s aggregate | **single-request latency** (control rate) |
| Stage cost | roughly balanced by tuning the split | 8-9x imbalanced by hardware |

PP does not reduce single-request latency under any partition: for one request the
stages run strictly in sequence, so latency is `t_stage0 + transfer + t_stage1`.
It only pays when the pipeline is full, and a closed-loop robot has exactly one
observation in flight.

### The 8-9x imbalance, measured on this exact model

OpenVINO iGPU-only, same IRs, same fp16 weights:

| | vit | text | loop (10 steps) | step | total |
|---|---|---|---|---|---|
| `run_info_demo_igpu.log` (GPU.0) | 77 | 377 | 1746 | 175 | **2200 ms** |
| `run_info_demo_igpu_int8.log` | 62 | 356 | 1605 | 161 | **2023 ms** |
| `run_info_demo_dgpu_int8_fmha.log` (B60) | 16 | 42 | 188 | 19 | **246 ms** |

Any layer moved to the iGPU costs ~8x what it costs on the dGPU. A 30/10 split
would put roughly a quarter of the work on hardware that is an order of magnitude
slower — the arithmetic never recovers, whatever the partition.

### The vendor already ran both dual-device experiments, and both lost

Single-device B60 baseline: 246 ms/sample = **4.06 samples/s**.

`run_info_demo_pipeline.log` — ViT on the iGPU, text+action on the dGPU, one
sample prefetched:

```
[pipe] 20 samples in 5.02 s -> 3.98 samples/s (251 ms/sample steady-state)
[pipe] per-sample LATENCY is unchanged-to-worse; only samples/s improves, and
       only offline (a closed-loop robot has no next observation to prefetch).
[pipe] [vit] below is the UNHIDDEN residual ... not the vit's device time
[vit] avg=6 ms
```

The `avg=6 ms` ViT line is overlap residual, not iGPU speed — the iGPU's real ViT
is 77 ms. Throughput went 4.06 → 3.98 samples/s. It lost.

`run_info_demo_dp.log` — two full data-parallel replicas, one per device:

```
[dp] 24 samples in 7.79 s -> 3.08 samples/s
[dp]   r0 (GPU.1, dGPU)  20 samples (83%)  avg  336 ms/sample = 2.97 samples/s
[dp]   r1 (GPU.0, iGPU)   4 samples (17%)  avg 1946 ms/sample = 0.51 samples/s
```

This is the decisive number. **Adding the iGPU dragged the dGPU from 246 ms to
336 ms per sample — 37% slower** — because the two contend for host memory
bandwidth and PCIe, and the iGPU has no memory of its own. The iGPU's 0.51
samples/s does not pay for the 1.09 samples/s it cost the dGPU. Aggregate
throughput went 4.06 → 3.08 samples/s. It lost worse.

So on this box: dGPU alone 4.06 > pipeline 3.98 > DP 3.08 samples/s, and the
dGPU-alone number is also the best latency. Both dual-device configurations are
dominated on both axes.

### Also true, and enough on their own

- ViT is not the bottleneck. `embed_prefix` is 24.7 ms of our 294 ms model path
  (8%); denoise is 205.9 ms (70%). Offloading the ViT can save at most 24.7 ms
  and only by overlapping across *consecutive* requests, which for a closed-loop
  controller means acting on a stale observation — a control problem, not a win.
- The oneCCL iGPU plugin is narrow: XPU tensors only
  (`oneccl_igpu_communicator.py:393`), plugin-managed USM host send/recv buffers
  with a host round-trip on every hop (`:400`, `:415`, `_copy_host_ptr_to_tensor`),
  packed tensor dicts must be single-dtype (`:631`, `:668`), no send-allgather
  (`parallel_state.py:922`, `:1035`), and a documented hang on multiple sequential
  transfers (`:633`). Our denoise loop would cross it 10 times per request.
- vLLM-Omni's diffusion side has `pipeline_parallel_size` plumbing
  (`diffusion/data.py:27`, `distributed/parallel_state.py:285`) but the LingBot
  pipeline binds a single `get_local_device()`
  (`pipeline_lingbot_vla_v2.py:41`) and has no stage split. Building one is real
  work for a result already measured as negative.

### G2. Intra-request parallelism — the axes exist, the ceiling is ~10.5%

The follow-up question was the right one: forget pipelining across requests, is
there *data-independent* work **inside one request** that the iGPU could run
concurrently so wall time becomes `max(dGPU, iGPU)` instead of a sum? The
dataflow answers it.

`sample_actions` (`modeling_lingbot_vla_v2.py:1434`) is three stages in strict
sequence, each consuming the previous one's only output:

```
embed_prefix(images, lang)      → prefix_embs            #  23.7 ms
prefix_forward(fill_kv_cache)   → past_key_values        #  57.1 ms
denoise_actions: 10 × predict_velocity                   # 201.4 ms
```

(Measured per stage in F1; the stages sum to 282.2 against a 286.5 ms
whole-request wall, so this decomposition is the request.)

so there is no stage-level concurrency to find. Enumerating the axes *within*
the stages:

| Axis | Exists? | Notes |
|---|---|---|
| ViT camera batch in `embed_prefix` | **yes, and free** | `flat_images = images.reshape(bsize * num_images, ...)` (`:1286`) — the cameras are independent rows of one ViT call. One transfer back, no collective. |
| Tensor parallel in `prefix_forward` | yes, not free | 36 layers × 2 all-reduces |
| Sequence parallel over the prefix | yes, not free | 286 padded / 223 valid tokens, all-gather per layer |
| Tensor parallel in `predict_velocity` | yes, very not free | 36 layers × 2 all-reduces × **10 steps** = 720 collectives |
| CFG / guidance branch parallel | **no** | Flow matching predicts one velocity; there is no cond/uncond pair, so `diffusion/distributed/cfg_parallel.py` has nothing to split |
| Across denoise steps | **no** | `x_t = x_t + dt * v_t` (`:1512`) — Euler is sequential by definition |

#### The heterogeneous split ceiling

For divisible work costing `t` on the dGPU with the iGPU `k`x slower, splitting
fraction `x` to the iGPU gives `wall = max(t(1-x), t·k·x)`, minimised at
`x* = 1/(1+k)` for `wall* = t·k/(1+k)`. **The saving is `t/(1+k)` — at best
`1/(1+k)` of the stage.** Per-stage `k` from the OV fp16 logs above:

| stage | k = iGPU/dGPU | best saving | our stage (F1) | max saved |
|---|---|---|---|---|
| vit | 77/16 = 4.81 | 17.2% | 23.7 ms | 4.1 ms |
| text prefix | 377/42 = 8.98 | 10.0% | 57.1 ms | 5.7 ms |
| denoise loop | 1746/188 = 9.29 | 9.7% | 201.4 ms | 19.5 ms |
| | | | **286.5 ms** | **29.3 ms (10.2%)** |

So ~10.5% is the naive ceiling: perfect divisibility, zero communication, zero
contention, zero launch overhead, and TP inside the denoise loop. (Recomputed on
F1's measured stage times; it was 31.0 ms of 294.0 = 10.5% before, so the ceiling
is insensitive to which decomposition is used — it is set by `k`, not by us.)

#### Corrected 2026-09-08: 10.5% is the right ceiling. An earlier "1.7%" was wrong.

This section first argued the ceiling was far lower than 10.5%, on the grounds
that only 51.0 ms of the 294 ms model path is device work and the remaining
~243 ms is host serialisation a second device cannot absorb. **That was an error,
and `PHASE5_PERF.md:702-704` states it outright:**

> Device self time is a **floor**, not a total: `mm`/`einsum` report zero on this
> backend, so combine it with the micro above (106 ms of MoE alone) and real
> device work is ~150–250 ms

`mm`/`einsum` report zero device time on this backend, so 51.0 ms undercounts
badly; the same note estimates real device work in the denoise loop alone at
**~150–250 ms**. Subtracting a documented floor from a wall clock and calling the
difference host time was not a valid derivation.

It also mixed configurations: that profile is the **eager** path, where the loop's
wall was ~600 ms.

**Now measured — see F1.** The compiled fp16 path was profiled per stage on
2026-09-08 with an issue/drain test, and it confirms the correction. The gap to
OV's 246 ms is spread across all three stages, and comes to ~40 ms of host
submission tax on top of a device floor that dominates the request:

| stage | ours (measured F1) | OV (fmha) | gap |
|---|---|---|---|
| embed_prefix / vit | 23.7 | 16 | +7.7 |
| prefix_forward / text | 57.1 | 42 | +15.1 |
| denoise / loop | 201.4 | 188 | +13.4 |
| | **286.5** | **246** | **+40.5** |

So the 10.5% k-factor ceiling above stands as the ceiling, and the reason
dual-device still loses is the one in the next subsection — the iGPU is 9x
slower and the mechanisms cost more than 10.5% — **not** "it is all host time."

Two downstream corrections, both toward less optimism: **G3 mechanism 1's
batching estimate** was built on the same flat-host assumption and is retracted
there, and **step C's upside is ~13 ms, not the 243 ms this section implied.**

#### And the mechanisms cost more than the ceiling

1. **Communication.** A denoise-loop TP all-reduce carries 51 × 768 fp16 =
   **76.5 KiB** — latency-bound, not bandwidth-bound, which is the worst case for
   a transport whose every hop does a USM host round-trip
   (`_copy_tensor_to_host_ptr` / `_copy_host_ptr_to_tensor`,
   `oneccl_igpu_communicator.py:380-420`). 720 collectives at even an optimistic
   50 µs is **36 ms** — more than the entire 20 ms denoise ceiling, and the
   plugin is documented to hang on multiple sequential transfers (`:633`).
2. **Contention, measured.** `run_info_demo_dp.log` has the dGPU at 336 ms/sample
   instead of 246 while the iGPU is busy: **+37%**. A balanced split keeps the
   iGPU busy by construction, so this is fully in play. It is 3.5x the naive
   ceiling and 20x the real one.
3. **Whatever host time remains is adversarial.** The eager profile had 1065.2 ms
   of host dispatch against a ~600 ms loop, i.e. the host was the binding
   constraint before compilation. How much survives compilation is unmeasured,
   but the iGPU path needs *more* host work per unit of device work (USM host
   round-trip on every hop), so any residual host pressure works against it. OV
   does not have this exposure: it submits the whole loop as one IR call.

#### Conclusion

The only free axis (ViT camera split) is worth at most 5.2 ms — under the
harness's own run-to-run spread, so not worth building. Every axis large enough
to matter is inside the denoise loop and needs per-layer collectives through a
host-round-trip transport, at 720 collectives against a 20 ms ceiling.

The ceiling is ~10.5% under ideal conditions; the measured contention penalty
alone is +37%. That is the whole argument, and it does not depend on the host/
device split that this section previously got wrong.

### G3. Multiple requests — one real latency lever, and the iGPU is not in it

Having ruled out splitting one request, the question becomes whether having
several in flight helps. Four mechanisms, and they are not the same thing.

#### 1. Batching (batch > 1) — big win, but on throughput, not latency

Structurally blocked today. `diffusion_worker.py:322-356` is

```python
while self._running:
    msg = self.mq.dequeue(indefinite=True)
    output = self.worker.execute_model(msg, self.od_config)
```

and `execute_model(self, req: OmniDiffusionRequest)` (`:142`,
`diffusion_model_runner.py:144`) takes **one** request, not a list. vLLM-Omni's
diffusion path runs strictly one request at a time, start to finish. There is no
batching to turn on.

The *model* is batch-capable: `bsize = state.shape[0]` threads through
`sample_actions`, `denoise_actions` and `predict_velocity`, and every mask is
built `[B, ...]`.

**Retracted 2026-09-08:** this subsection originally projected ~173 ms/request at
batch=2 and ~112 ms at batch=4, from a `243 ms host + B × 51 ms device` model. That
model came from treating `PHASE5_PERF.md`'s 51.0 ms device-time **floor** as a
total — see the correction in G2.

**Re-sized 2026-09-08 from the F1 measurement.** The mechanism is real and its
size is now bounded. Raising the batch does not change the aten op count (227465
at batch=1), only tensor heights, so host submission stays at the measured
~283 ms while device work scales with the batch. Against OV's 246 ms device
reference:

| batch | host submit | device work | wall | per request |
|---|---|---|---|---|
| 1 | 283 | 246 | 286.5 (measured) | **286.5** |
| 2 | ~283 | ~492 | ~492 | **~246** |
| 4 | ~283 | ~984 | ~984 | **~246** |

Batching buys the submission tax back — about **1.16x**, converging on the device
floor — and no more, because past batch=2 the workload is device-bound and device
time grows with the batch. Worth doing for a fleet; nothing like the 1.7x this
subsection originally implied.

The other two points stand unchanged:

- The GEMMs are tiny — 51 suffix tokens at hidden 768 — so they are latency-bound
  rather than throughput-bound, and taller tensors are close to free on the
  device up to some batch. This is why batch=2 lands near 2× device rather than
  worse.
- Per-request *latency still gets worse* under any batch; only the aggregate
  control rate across a fleet improves. For one robot batching is a regression
  regardless of how the numbers land.

Prerequisite if pursued: batched requests must agree on prefix length (already
padded to 286, so likely fine) and on `image_grid_thw`.

#### 2. Concurrent requests on one dGPU — blocked by memory

The device idles ~83% of a request, so two in-flight streams could in principle
overlap A's device work with B's host dispatch. Two obstacles: the worker loop
above is serial, and the GIL means two request threads in one process would not
overlap dispatch anyway. Two *processes* would — but 2 × 11.73 GiB = 23.5 GiB of
the B60's 23.9 GiB before activations and the 40.2 MiB × 2 KV cache. **Two fp16
replicas do not fit on one B60.** Throughput again, and blocked.

#### 3. Chunk-level pipelining — the one that actually reduces latency, and it is free

This is the answer. The model emits a **50-step** action chunk (`chunk_size: 50`,
`config.py:68`). The robot executes those 50 steps over `50/f` seconds at control
rate `f`. As long as inference fits inside that window, chunk N+1 can be computed
while the robot is still executing chunk N — the control loop never stalls and the
latency observed *at the actuator* is zero. This is what openpi calls async
inference.

At any plausible `f` the window is one to two seconds, so both 294 ms and the OV
246 ms fit inside it with room to spare. Which reframes the whole phase: the cost
of our 294 ms is not a stalled robot, it is that each chunk is computed from an
observation **294 ms old**. Getting to 246 ms buys 48 ms less staleness. That is
the honest value of Phase 8, and it should be stated that way rather than as
"1.16 Hz".

##### What it actually buys, corrected 2026-09-08

Framing this as "48 ms less staleness" undersold it. The real benefit is that
**the robot never runs out of actions.** If chunk N+1 is requested only after
chunk N is exhausted, the arm must freeze or hold its last commanded pose for a
full inference period every 50 steps — a visible per-chunk stall worth the whole
294 ms model path plus step E's unattributed residual, not 48 ms. Requesting at
step k of chunk N makes the motion continuous.

Whether the deployed controller stalls today is unknown from this repo, and it is
the first thing to establish, because it decides whether this is the largest
available win or a no-op.

##### Where the code changes land

Checked 2026-09-08. Split by owner:

**Server: no change needed.** `connection.py:166-204` is, per connection,
strictly `receive() → serving.infer() → send_bytes()`. It does not read ahead
while inferring — but it does not have to. A request sent early sits in the
transport buffer and is picked up the instant the previous response is sent, so
the engine runs back-to-back with no idle gap. The serial worker loop
(`diffusion_worker.py:322`) is fine for this: pipelining here means "the next
request is already queued", not "two run at once". `session_id` continuity is
already handled (`connection.py:194-196` resets on change, and a pipelining
client must reuse one id — which the demo client already does).

**`openpi_client.py`: not the thing to change.** It is a 79-line synthetic demo —
`rng.integers` for images, and `websocket.send(...)` immediately followed by
`websocket.recv()` on lines 63-64, fully blocking. It measures round-trip
latency; it is not a control loop and should not become one.

**The controller: this is where the work is, and we may not own it.** Whoever
runs the robot loop (RoboTwin / the integrator's stack) needs: an async or
threaded send/recv instead of the blocking pair; an action-chunk buffer; the
trigger to request N+1 at step k of chunk N; a choice of k, which needs the
control rate `f` that this repo does not record; and a policy for splicing the
arriving chunk onto the executing one, since switching at an arbitrary step is
discontinuous and the usual fix is a weighted blend over the overlap.

**Worth adding here:** a pipelining reference client and a sustained-rate mode in
the harness. `run_perf_check.sh` measures a blocking round-trip median, which
cannot observe whether the control loop stalls — the number this mechanism is
supposed to move. Step F.

Three things follow:

- It needs no iGPU, no batching, and no change under `vllm_omni/`.
- The tradeoff is observation staleness, which is a control decision, not a
  compute one. It should be surfaced to whoever owns the controller, not decided
  here.
- **The control rate `f` is recorded nowhere in this repo.** `chunk_size: 50` is
  in the config; the Hz that turns it into a time budget is not. Without it we
  cannot say whether 294 ms is comfortable or marginal. Add it to step F.

#### 4. Speculative duplicate execution — strictly negative

Send the same request to both devices, take whichever returns first. The dGPU
finishes in 294 ms and the iGPU in ~2200 ms, so the dGPU wins every race by 7x
while the loser burns the +37% contention penalty measured in
`run_info_demo_dp.log`. There is no variance story that rescues this.

#### Conclusion

Multiple requests *do* open a real latency lever — mechanism 3 — and it removes
the control-loop stall entirely at the cost of observation staleness. Mechanism 1
is the throughput lever and is worth more than anything dual-device, but it is not
latency. The iGPU appears in none of them: batching has to happen on one device,
chunk pipelining needs no second device, and the only multi-request shape that
does use the iGPU is the vendor's prefetch pipeline, already measured at 251 ms
against 246 ms single-device.

### G4. "Run request N+1's ViT on the iGPU while the dGPU denoises N"

Asked directly, so worth answering with arithmetic rather than by pointing at the
vendor's log. This is the **best** dual-device idea on the table — the only one
that is structurally sound. It is still a latency regression, and that turns out
to be provable rather than merely measured.

#### What is right about it

Unlike layer-split PP or in-request TP, this construction has no per-layer
collectives, splits work that is genuinely independent (different requests), and
the window fits with room to spare:

| | ours (measured F1) | OV's version |
|---|---|---|
| ViT to hide (dGPU cost) | 23.7 ms of 286.5 | 16 ms of 246 |
| iGPU ViT cost | 77 ms | 77 ms |
| dGPU window to hide it in | 201.4 ms denoise | 188 ms |
| fits? | yes, 2.6x margin | yes |
| period gain if free | 23.7 ms (**8.3%**) | 16 ms (6.5%) |

Our margin is better than the one OV measured, because our eager ViT is more
expensive relative to our total. So this is not dismissible on the window.
(Figures updated from F1's per-stage profile; they were 30.5 of 294.0 for a
10.4% gain before, which does not change the conclusion below.)

#### Why it still loses: period improves, staleness gets worse

A robot's metric is **staleness** — the age of the observation the emitted action
was computed from — not the period between actions. Chunk pipelining (G3) already
removes the period from the picture entirely. Steady state:

```
single device      period 286.5    staleness = 23.7 + 57.1 + 201.4        = 282.2 ms
iGPU ViT prefetch  period 262.8    staleness = 77 (iGPU) + 57.1 + 201.4   = 335.5 ms
```

Even starting the iGPU ViT as late as possible (finishing exactly when the dGPU
frees up), the observation still has to walk through a 77 ms ViT instead of a
23.7 ms one before anything else can touch it. **Period improves by
`t_dGPU_vit` = 23.7 ms; staleness worsens by `t_iGPU_vit − t_dGPU_vit` = 53.3 ms.**
The F1 profile made this worse, not better: the dGPU's ViT is cheaper than
previously recorded, so there is less period to win and more staleness to lose.

That generalises, and it is the useful result: **moving any stage to a slower
device worsens staleness even under perfect overlap**, because the first stage
that touches an observation is on that observation's critical path by definition.
Overlap converts a slower device into throughput. It cannot convert it into
latency. This is what the vendor's log means by "per-sample LATENCY is
unchanged-to-worse; only samples/s improves."

#### And the period gain is not free either

The ViT output that must cross the link, at the released 3-camera / 224px config
(`embed_prefix` docstring: `3*66 + 72 + 8 + 8 = 286`), with hidden 2560 and three
deepstack mergers (`deepstack_merger_list.{0,1,2}`, verified in the checkpoint
index):

| tensor | shape | fp16 bytes |
|---|---|---|
| `prefix_embs` | [1, 286, 2560] | 1.40 MiB |
| `deepstack_visual_embeds` × 3 | [1, 3, 64, 2560] each | 2.81 MiB |
| `pad_masks`, `att_masks`, `visual_pos_masks` | bool | small |
| `prefix_position_ids` | int64, mrope [3, 1, 286] | small |
| | | **≈ 4.21 MiB + masks** |

Every hop goes `_copy_tensor_to_host_ptr` → USM host buffer →
`_copy_host_ptr_to_tensor` (`oneccl_igpu_communicator.py:380-420`), so that is
~8.4 MiB of **host** memcpy per request on the CPU that is already our
bottleneck. And the masks are bool/int64 while the embeddings are fp16, but
"packed tensor dict requires one dtype" (`:631`, `:668`) — so this is at least
three separate hops, into a plugin documented to hang on multiple sequential
transfers (`:633`).

Empirically OV paid ~21 ms of contention for this arrangement: they should have
gained 16 ms (246 → 230) and measured 251. If our contention cost is the same
~21 ms we net +9.5 ms of period (3%); if it is larger — likely, since we are host
dispatch-bound and they are not — we net negative. Even the throughput case is a
coin flip.

#### The precondition nobody has met yet

There must *be* a request N+1 to prefetch. In a closed loop there is not: obs N+1
does not exist until the robot has acted on chunk N. This only becomes possible
*after* G3's chunk pipelining is in place — and once it is, the whole 294 ms is
already off the control-loop critical path, so the thing being optimised is 30 ms
of staleness that this construction makes worse by 46.5 ms.

#### Verdict

Feasible, and the least bad dual-device design — but it buys period and sells
staleness, which is the wrong direction for a robot. The one context where it is
the right call is **offline dataset evaluation** (`open_loop_eval.py` over a large
bundle), where staleness is meaningless and samples/s is everything. Even there,
batching (G3 mechanism 1, ~1.16x at batch=2 as re-sized from F1) beats its best
case of 1.09x without a second device, a second process, or a 4.21 MiB host
round-trip.

Cost to build, for the record: a second worker process with its own
`ZE_AFFINITY_MASK`, device placement in `OmniDiffusionConfig`, a cross-device
transfer for the six `embed_prefix` outputs across at least three dtype-separated
hops, and request lookahead in a worker loop that is currently
`while: dequeue(); execute_model()`. That is a lot of machinery for a negative
latency result.

### The condition under which this becomes worth revisiting

Only if the goal changes from *latency* to *aggregate throughput across several
robots*, **and** the contention in `run_info_demo_dp.log` is shown to be an
artifact rather than a property of a memory-less iGPU. Then the right shape is
still not PP — it is data parallelism (one independent full replica per device,
no cross-device tensor traffic at all), because the model fits on both. That is
also the configuration the vendor measured at 3.08 samples/s, so the burden of
proof is on making that number better, not on the design.

Until then the answer to "best performance" is the single-device work already in
this document: A landed (294 ms model path), and C/D/E are what remains between
us and 246 ms. There is no dual-device shortcut past them.

## Rules

Phase 5 and 6 rules carry over. Two that this phase will be tempted to break:

1. **Read the reference before theorising about it.** Phase 5 inferred the
   reference's algorithm from a FLOP ratio and got it backwards. Phase 8 already
   nearly inherited a 200 ms target that no log supports.
2. **Never compare absolute latencies across host states.** `run_perf_check.sh`
   refuses to measure above load 2.0 and that refusal has already caught one bad
   number in this phase (`load1=5.23`, a parity run that had not decayed).

And one new one:

3. **A latency change is not landed until it has been through the Phase 7
   metric.** "N% different from eager" is not an accuracy result. The question
   is always how far from fp32, next to where the vendor's own shipped
   precisions sit.
