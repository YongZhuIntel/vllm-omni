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
| C | One graph for all 10 steps, as the reference does | −25 to −50 ms | attempted; Inductor cold compile not deployable |
| D | Prefix stages: `prefix_fill` 64.5→42, `embed_prefix` 24.7→16 | −31 ms | SDPA probe measured; backend blocked |
| E | The ~165 ms outside the model path | −? served rate | not started |
| F | Phase 6 step 4 — make the harness able to track all of this | none directly | pending since Phase 6 |

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

The second probe changed only the attention accumulation dtype while keeping
the eager attention implementation and compiled denoise path:

| attention precision | embed_prefix | prefix_fill | denoise | total |
|---|---:|---:|---:|---:|
| fp32 (default, 20 iters) | 24.8 ms | 66.9 ms | 217.4 ms | 314.6 ms |
| fp16 (experiment, 20 iters) | 24.8 ms | 57.5 ms | 207.1 ms | 294.7 ms |

Under identical 5-warmup/20-iteration conditions this saves about 19.9 ms
(`6.3%`) and reduces the vLLM/OpenVINO total ratio from `1.28x` to `1.20x`.

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
