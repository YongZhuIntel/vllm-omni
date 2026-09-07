# Phase 5 — latency

M1–M4 made the policy *correct* and *servable*. M5 is the only remaining
milestone and it has one number in it.

| path | measured | where |
|---|---|---|
| Phase 0 bare kernel | **0.74 s** | `phase0_torch_spike.py`, B60 bf16 |
| M3 offline pipeline through `OmniDiffusion` | **2.426–2.473 s** | `examples/offline_inference/lingbot_vla_v2/` |
| M4 OpenPI WebSocket, server-side | **2.744 s** | `/v1/realtime/robot/openpi` |

All three are the same machine (single `Intel(R) Arc(TM) Pro B60 Graphics`,
23.91 GiB), the same dtype (bf16 — fp32 does not fit, and fp16 is off the
upstream-validated path at identical XMX throughput), and the same 10-step
flow-matching loop. So the ~1.7 s spread is pure software overhead.

**Acceptance:** M4's criterion of ≥1 Hz end-to-end, then Phase 0's 0.74 s/chunk.
Starting serving rate was ~0.36 Hz.

**Both are met.** Final state, idle host, B60 bf16, `moe_implementation="dense"`:

| path | measured | rate |
|---|---|---|
| in-process, warm (`phase5_latency.py`) | **0.703 s** | 1.42 Hz |
| OpenPI WebSocket, warm, 8 requests | **0.73–0.94 s** (median 0.81) | **1.07–1.37 Hz** |
| offline example, cold single request | 0.891 s | 1.12 Hz |

One change produced all of it: `moe_implementation` `gather` → `dense`. The rest
of the phase was measurement, including one measurement that had to be thrown
away — see the step 4 entry.

## Rules for this phase

1. **Attribute before optimising.** No kernel work lands until a measurement
   says which stage it would help. Two of the ideas on the table (fixed-shape
   compile, an XPU grouped-MoE kernel) only pay off if the denoise loop actually
   dominates, and nothing has established that.
2. **Keep the numerical gate.** The eager dense MoE path stays reachable
   (`moe_implementation="dense"`) and `phase1_parity.py` must still pass after
   every change. A faster policy that moved is not a faster policy.
3. **Record every conclusion here**, including the negative ones. A measurement
   that rules an idea out is the cheapest result in the phase.

## Steps

| # | step | question it answers | status |
|---|---|---|---|
| 1 | **Stage attribution** (`phase5_latency.py`) | Of the 2.43 s, how much is the processor, H2D, prefix fill, denoise, postprocess — and how much is left over for the engine/IPC? | **done** — denoise was 90% |
| 2 | **MoE A/B** | Routed gather vs. upstream dense einsum, on XPU. The gather kernel is 8× fewer FLOPs at top-4/32 but launches many small ops; on a GPU that may lose to one big dense matmul. | **done** — dense wins 3.7× |
| 3 | **Land the dense default** | Flip `moe_implementation`, confirm `phase1_parity.py` still passes, re-measure through the engine and the WebSocket. | **done** — 1.13 Hz offline, 1.05–1.35 Hz served |
| 4 | **Image preprocessing** | 180 ms of CPU per request, apparently the second-largest stage. Three cameras through a Python HF image processor. | **dropped** — it is 1.9 ms; the 180 ms was host contention |
| 5 | **Kernel work** | Fixed-shape compile and/or a real grouped MoE kernel — only if steps 3–4 leave a gap to 0.74 s. | **reopened** — met the 0.74 s baseline, but the OpenVINO reference does the same work in 289 ms; see the last log entry |
| 6 | **Protocol hardening** | Independent of latency: idle timeout, oversized/invalid payloads, sanitized errors, official `openpi-client` interop. | **done** — 2 bugs found and fixed |

## Log

Newest last. Each entry: what was run, what came out, what it rules in or out.

### 2026-09-04 — step 1 tooling

`phase5_latency.py` written. It rebuilds what `LingbotVlaV2Pipeline.__init__`
builds and runs what its `forward` runs, in-process, with a device sync at every
stage boundary — so it measures everything *below* the engine. The residual
against the engine's 2.43 s is then the engine/IPC share, without needing to
instrument the worker process.

Two design notes worth keeping:

- The `mp` executor is the **only** diffusion executor in v0.14.0
  (`vllm_omni/diffusion/executor/` has just `multiproc_executor.py`), so the
  pipeline always runs in a separate process. Monkeypatching from the parent
  cannot reach it; the in-process reconstruction is what makes step 1 possible
  at all.
- Syncing at stage boundaries serialises work that would otherwise overlap, so
  the script also reports an unsynced end-to-end total. If the two totals
  disagree materially, the per-stage split is hiding overlap and should not be
  read as a budget.

Building the model fp32-on-CPU and moving it afterwards gets the process
OOM-killed (25.5 GB of parameters on top of a 24 GB mmapped checkpoint, on a
60 GB host). The script builds straight onto the device in the target dtype
under `torch.set_default_dtype`, the way the engine's loader does;
`load_weights`' `.data.copy_` casts each shard tensor as it lands.

### 2026-09-04 — step 1: the denoise loop is 90%, and it is slower than Phase 0

`python phase5_latency.py --iters 5` (B60, bf16, 10 steps, medians of 5
post-warmup runs):

| stage | ms | share |
|---|---|---|
| `pre.state` | 0.4 | 0.0% |
| `pre.images` | 178.0 | 6.1% |
| `pre.language` | 0.6 | 0.0% |
| `h2d` | 7.8 | 0.3% |
| `model.embed_prefix` | 30.2 | 1.0% |
| `model.prefix_fill` | 66.8 | 2.3% |
| **`model.denoise`** | **2624.6** | **90.2%** |
| `post.d2h_unnormalize` | 0.3 | 0.0% |
| total (synced) | 2908.6 | |
| total (unsynced) | 2899.9 | |

Per denoise step: 267.6 ms for step 0, 262.1 ms for the rest — flat, so there is
no warmup artefact hiding in the loop.

Two conclusions, one of them not the one the phase was set up to find:

1. **There is no engine/IPC problem.** The synced and unsynced totals agree to
   0.3%, and the whole processor + transfer + postprocess side is 187 ms. The
   engine's 2.43 s is *less* than this in-process 2.90 s, so worker IPC is not
   merely small — it is invisible next to run-to-run variation in the kernel.
   Step 2 as originally planned (engine overhead) is **dropped**: there is
   nothing there to find.
2. **The kernel itself regressed against Phase 0.** Phase 0's bare upstream
   kernel was 0.74 s/chunk on this same card at this same dtype. The vendored
   kernel is doing the same work in ~2.7 s. That is not overhead around the
   model, it is the model — which makes the MoE A/B (planned as step 3, on the
   theory that it *might* matter) the immediate next measurement rather than a
   later one.

### 2026-09-04 — step 2: the gather MoE is a 3.7× pessimisation on XPU

`python phase5_latency.py --iters 5 --moe dense`, same conditions:

| stage | gather (ms) | dense (ms) |
|---|---|---|
| `model.denoise` | 2624.6 | **706.4** |
| per denoise step | 262.1 | **70.5** |
| total (synced) | 2908.6 | **990.8** |

Everything outside the denoise loop is unchanged (`pre.images` 178.0 → 180.5,
`prefix_fill` 66.8 → 66.7), which is the control this A/B needed.

**The FLOP argument was right and irrelevant.** `forward_gather` really is 8×
less arithmetic at top-4 of 32 — and it loses anyway, because it is a Python
`for` loop over all 32 experts containing a `torch.where`, two `F.linear`s, a
`silu`, a multiply and an `index_add_`. At 36 layers × 10 steps that is ~11,500
loop iterations and on the order of 70,000 kernel launches per request, each on
a slice of ~6 tokens. `forward_dense` issues three `einsum`s per layer over a
51-token suffix; the wasted arithmetic costs less than the launches it avoids.

This also closes the Phase 0 gap cleanly. Dense in-process total is 991 ms, of
which 180 ms is image preprocessing that the Phase 0 spike never ran — leaving
~810 ms of model against Phase 0's 740 ms, i.e. the vendored kernel is within
~10% of the reference once it uses the reference's MoE path.

Consequence: **`moe_implementation` should default to `"dense"`.** The gather
path stays available — it is genuinely fewer FLOPs and could win on a device
where launches are cheap or with a real grouped kernel behind it — but it is no
longer what ships. Note this makes the default path the one the fp32 golden was
produced with, so it is also the better parity default.

### 2026-09-04 — step 3: landed, and 1 Hz is met

`moe_implementation` now defaults to `"dense"` in `config.py`, with the
docstrings in `modeling_lingbot_vla_v2.py` corrected (they still described
`gather` as the default and sold the FLOP argument without the launch cost).
`test_moe_kernels_agree` was rewritten to read the current default rather than
assume `gather`, since that has now flipped once.

**The parity gate got stricter, not looser.** `phase1_parity.py --num-steps 2`
now reports **all 45 stages at exactly 0.000e+00**. Before the flip the expert
probes and the outputs sat at ~1e-7 relative, and the write-up attributed that to
"fp32 summation order: gather-vs-dense MoE, einsum vs. matmul". That was right:
running the reference's own kernel removes the residual entirely. There is no
longer any numerical difference between the vendored kernel and repaired
upstream at fp32.

**End to end through the engine, B60 bf16:**

| | before | after |
|---|---|---|
| offline `OmniDiffusion` request | 2.517 s | **0.899 / 0.899 / 0.876 s** |
| rate | 0.40 Hz | **~1.13 Hz** |

M4's ≥1 Hz acceptance criterion is met.

One trap worth recording: the first re-measurement still showed 2.517 s. The
default flip had no effect because `prepare_lingbot_vla_v2.py` **serialises the
whole config** into `transformer/config.json`, so the prepared directory built
17 hours earlier still pinned `"moe_implementation": "gather"` and the engine
faithfully honoured it. Regenerating the prepared directory is what produced the
0.899 s. `run_openpi_server.sh` already regenerates before serving, for exactly
this class of reason; the offline path does not, so a stale prepared directory
will silently keep serving old settings.

This also re-confirms step 1's conclusion at the new operating point: in-process
dense totals 0.99 s (synced) / 1.00 s (unsynced) against the engine's 0.88 s, so
the engine and its worker IPC remain below measurement noise. Nothing to
optimise there.

**Over the OpenPI WebSocket**, five consecutive requests on one connection
(`openpi_client.py --num-steps 5`), client-side round trip including
MessagePack encode/decode of three frames:

```
step=0  0.753 s (1.33 Hz)
step=1  0.739 s (1.35 Hz)
step=2  0.948 s (1.05 Hz)
step=3  0.953 s (1.05 Hz)
step=4  0.948 s (1.06 Hz)
```

2.744 s → 0.74–0.95 s, i.e. **0.36 Hz → 1.05–1.35 Hz**. The transport adds
nothing measurable on top of the engine.

### 2026-09-04 — incidental: the handshake advertised the wrong frame size

Found while reading the metadata in the run above, not looked for. The OpenPI
handshake reported `image_resolution: [224, 224]`, taken from
`LingbotVlaV2Config.image_resolution`, whose default is `(224, 224)`. But the
processor resizes to `RobotSpec.image_size`, which comes from the training
config's `img_size` and is **256** for RobotWin. A robot honouring the handshake
would have sent 224² frames that the server then upsampled to 256², throwing
away detail the policy was trained on — and it would never error.

`config.image_resolution` turns out to be read by nothing except this metadata,
which is why the M2 parity work never caught it: parity fed the processor
directly. Fixed at the source rather than by changing the default —
`prepare_lingbot_vla_v2.py` now builds the `RobotSpec` and reports
`spec.image_size`, and derives `action_dim` from `spec.action_slices` instead of
hardcoding 14. `openpi_client.py` now sizes its frames and state vector from the
handshake instead of hardcoding 224 and 14, so a future mismatch shows up as a
server-side error rather than silent resampling.

### 2026-09-04 — step 4: there is no image-preprocessing problem, and steps 1–3 were measured on a loaded host

Step 4 was supposed to attack the 178 ms `pre.images` stage. It does not exist.
Timing the processor directly — three 256×256 RobotWin frames through
`_build_images` — gives **0.63 ms**, about 280× less than the stage attribution
reported. `_process_frame` is 0.20 ms per camera, of which the Hugging Face
`Qwen2VLImageProcessor` call is 0.15 ms; the frames arrive already at
`spec.image_size`, so the `F.interpolate` branch never runs.

So the instrument was measuring the host, not the code. `phase5_latency.py` now
runs `_build_images` **twice** per request and reports both laps as a control. Re-run
on an idle host (`--iters 5`, dense, same command as step 1):

| stage | step 1 (loaded host) | now (idle host) |
|---|---|---|
| `pre.images` | 178.0 | **1.9** |
| `pre.images_again` | — | 1.3 |
| `h2d` | 7.8 | **0.9** |
| `model.embed_prefix` | 30.2 | **25.0** |
| `model.prefix_fill` | 66.8 | 65.5 |
| `model.denoise` (dense) | 706.4 | **607.3** |
| per denoise step | 70.5 | **60.7** |
| total (synced) | 990.8 | **702.8** |
| total (unsynced) | — | 703.4 |

**Cause.** Steps 1 and 2 ran while three orphaned `PPID=1` vllm-omni worker
trees from the M3/M4 validation runs were still alive, holding ~8 GB RSS and
11.3 GiB of B60 memory and spinning on the CPU. They were killed partway through
step 3, to get the fp32 parity gate past an OOM — which silently re-based every
number measured afterwards. Host load average at the time of the step-1 run is
consistent with this: immediately after the kill it was still decaying through
6.83 (15 min) to 0.78 (1 min).

Two lessons, and the second one is the expensive one:

1. **Record host state alongside every timing.** A CPU-side stage inflating 100×
   under contention is what made a 1.9 ms stage look like the second-biggest cost
   in the request and put a whole step on the plan.
2. **A/B under contention can still be trusted; absolute numbers cannot.** Both
   arms of step 2 ran under the same load, so the *ratio* survived — see below.

**The A/B verdict is unchanged.** Re-run of `--moe gather` on the idle host:
denoise **2255.0 ms** (225.3 ms/step) against dense's **607.3 ms**
(60.7 ms/step) — **3.71×**, versus 3.72× measured under load. Everything outside
the loop is identical between the two arms (`pre.images` 1.8 vs 1.9, `prefix_fill`
65.3 vs 65.5). Dense stays the default.

**Phase 0's target is met, not merely approached.** In-process dense is
**702.8 ms** against the Phase 0 bare-kernel baseline of 740 ms — and that
702.8 ms includes the observation processor and postprocess, which the Phase 0
spike never ran. The vendored kernel is now slightly *faster* than the reference
it was ported from, at exact fp32 parity. Step 5 (fixed-shape compile, a grouped
MoE kernel) is therefore **not needed for acceptance** and is not being started;
the FLOP argument for a real grouped kernel still stands as future work, worth
~8× of the denoise loop's arithmetic if a kernel ever makes the launches cheap.

### 2026-09-04 — end-to-end re-measurement, and a methodology note on the offline number

**Over the OpenPI WebSocket**, eight consecutive requests on one connection,
idle host, client-side round trip including MessagePack encode/decode:

```
0.760  0.741  0.910  0.906  0.732  0.744  0.935  0.855   (seconds)
1.32   1.35   1.10   1.10   1.37   1.34   1.07   1.17    (Hz)
```

Median 0.81 s, i.e. **1.07–1.37 Hz**. Against the in-process 0.703 s, the engine
plus the worker IPC plus the transport cost ~110 ms at the median and ~30 ms at
the best request. The spread is bimodal — a ~0.74 s cluster and a ~0.91 s
cluster, ~170 ms apart, reproducible across both the five-request and
eight-request runs. Not chased: 1 Hz holds in the slow cluster too.

**The offline example's number is a cold-start number.** `lingbot_vla_v2.py`
times exactly one `engine.generate` with no warmup, so its 0.891 s (idle host)
is not comparable to any of the warm figures above, and the earlier
"0.899 / 0.899 / 0.876 s" was three separate cold single-request runs rather
than three requests. It is a fair number for what a robot sees on its first
chunk after start-up, and nothing else. Warm serving is the 0.73–0.94 s above.

Acceptance status: M4's ≥1 Hz is met warm and cold, and Phase 0's 0.74 s/chunk
is met in-process. The only open step in this phase is protocol hardening, which
has nothing to do with latency.

### 2026-09-04 — step 6: the behaviours were all there; two of them did not work

Idle timeout, the payload ceiling, malformed-payload refusal and error
sanitization were all written in M4 and none of them were tested. Writing the
tests found two real defects.

Coverage added: 16 cases in `tests/entrypoints/openai_api/test_openpi_connection.py`
(idle close, oversized payload, three malformed-payload shapes, stray text frame,
sanitized inference error, session survival after failure, reset without an
observation, both NumPy wire formats, rejected dtypes), plus
`phase5_protocol_probe.py`, which runs the same failure paths against a live
uvicorn server — the half the unit tests structurally cannot see.

**Bug 1: the payload ceiling was above the transport's.**
`MAX_OPENPI_PAYLOAD_BYTES` was 64 MiB. Uvicorn's `ws_max_size` default is 16 MiB
and neither vLLM nor vLLM-Omni overrides it, so a frame between the two limits
never reached the application check: the robot got a 1009 close instead of an
error frame, mid-session. Lowered to 16 MiB so the application check is the
binding one. Confirmed live — a 48 MiB observation still closes with 1009, which
is now correct behaviour rather than a limit that lied. Working payloads are
~600 KiB, so this costs nothing real.

**Bug 2: the msgpack-numpy decode path could never succeed.**
The endpoint accepts two NumPy wire formats: openpi-client's `__ndarray__`
markers, and msgpack-numpy's `nd`/`type`/`kind`/`data` markers. The second
validated `kind` by comparing it to `dtype.kind` — but in that format `kind` is
`""` for every ordinary dtype and `"V"` only for a structured one. So every
float and integer array arriving in msgpack-numpy form was rejected with
"Invalid request payload", and the compatibility path was dead code that looked
like a feature. A round trip through this repo's own `_pack` never caught it
because `_pack` emits the *other* format. Fixed to the actual convention, with
structured dtypes still refused. The live probe now gets `actions (50, 14)` from
an `nd`-marker observation.

On `openpi-client` interop: the package is not installed in this container, so
this is verified at the wire-format level rather than against the library. That
is the meaningful test either way — `_pack_numpy`/`_decode_openpi_numpy_marker`
implement exactly `openpi_client.msgpack_numpy`'s `__ndarray__` and
`__npgeneric__` markers, and both directions are covered.

**Reverted: validating the observation's keys at the transport.** An observation
that decodes but lacks `state` is only caught in the worker, so the client pays
an IPC round trip to be told "Internal inference error" instead of which key is
missing. Checking the three keys in `connection.py` fixed the message and broke
the state-only observation the existing tests use — which is the argument against
it: this transport serves any robot policy, and requiring images or a state
vector at that layer would reject a policy that takes neither. Left as a recorded
limitation with the reasoning in the code, rather than fixed in the wrong layer.

**Live results** (`phase5_protocol_probe.py`, server via `run_openpi_server.sh`):

```
undecodable bytes            -> error "Invalid request payload"      0.001 s
a list, not a mapping        -> error "Invalid request payload"      0.000 s
an empty mapping             -> error "Internal inference error"     0.004 s
an observation with no state -> error "Internal inference error"     0.028 s
valid observation afterwards -> actions (50, 14)                     0.757 s
nd/type/kind markers         -> actions (50, 14)                     0.926 s
48 MiB observation           -> transport close, code 1009
idle connection              -> server close after 30.0 s, code 1000
```

Every refusal leaves the session usable, which is the property that matters on a
robot: one bad chunk must not cost the connection and its warm worker.

## Phase 5 is closed

Both performance targets are met, all six steps are resolved, and the phase
produced three fixes and one reverted change:

| change | effect |
|---|---|
| `moe_implementation` default `gather` → `dense` | 2.43 s → 0.70 s, and fp32 parity became exact at 45/45 stages |
| handshake `image_resolution` derived from `RobotSpec` | stopped advertising 224² for a policy trained at 256² |
| `MAX_OPENPI_PAYLOAD_BYTES` 64 MiB → 16 MiB | the advertised limit is now the enforced one |
| msgpack-numpy `kind` check corrected | a dead compatibility path now works |
| transport-level observation validation | tried, reverted, recorded — wrong layer |

Two of those were found by looking at output that was not the thing being
measured (the handshake metadata during a latency run; the `kind` field while
writing a test for something else), and one whole planned step turned out to be
an artifact of a dirty machine. That ratio is the argument for rule 3.

### 2026-09-04 — a single command to re-verify

`examples/online_serving/lingbot_vla_v2/run_perf_check.sh`. Everything above took
two terminals and manual reading of numbers; this collapses it into one command
that regenerates the prepared directory, serves, sends warm requests on one
connection, discards the first, and exits non-zero if the median misses 1 Hz.

It encodes the two traps this phase produced, so they cannot be re-hit silently:

* **It refuses to measure on a busy host** (exit 3), printing the orphaned
  processes it found. This is the step-1/step-2 failure turned into a guard.
* **It always regenerates the prepared directory**, because the config is
  serialized into it.

Verified in both directions: a normal run reports `PASS 1.32 Hz`, and with a
deliberately orphaned Python process present it refuses with exit 3. Sample runs:
median 0.760 s (1.32 Hz) and 0.921 s (1.09 Hz) — the same bimodality noted above,
so a single run should not be read to three digits.

One hazard found while writing it and fixed: the cleanup path first reaped with
`pgrep -f "$OUTPUT"`, which also matches any shell whose command line contains
the model path — including the one that invoked the script. Narrowed to PPID 1,
python, and never the script or its parent.

### 2026-09-04 — is there a standard VLA benchmark to conform to? No.

Asked before building anything further, so that `run_perf_check.sh` does not
duplicate an existing convention. Surveyed both checkouts.

This fork (`0.14.0rc1`) has no VLA anything beyond this port. The other checkout,
`/llm/zhuyong/vllm-omni` at `v0.28.0-3-ga65e89cb`, is far ahead and **does** ship
VLA models — `pi0`, `gr00t`, `internvla_a1`, `dreamzero` — against this same
`/v1/realtime/robot/openpi` endpoint. But it still has no VLA benchmark:

| looked at | found |
|---|---|
| `benchmarks/` (now 8 subdirs incl. `accuracy`, `kernels`, `lingbot_video`) | no VLA entry |
| `tests/dfx/perf/tests/*.json` (27 perf configs — this dir does not exist in our fork) | none for pi0/gr00t/internvla/dreamzero |
| `tests/e2e/online_serving/test_pi0_expansion.py`, `test_gr00t_openpi_expansion.py` | `full_model`+`diffusion` functional marks; assert handshake metadata and shapes, no latency |
| `examples/online_serving/pi0/openpi_client.py` | 119 lines, no `perf_counter` — it does not time |
| `benchmarks/diffusion/diffusion_benchmark_serving.py` | `--task` is `t2v/i2v/ti2v/ti2i/i2i/t2i`, posts to `/v1/chat/completions`; wrong transport and wrong load shape (N concurrent requests vs. one long-lived connection at concurrency 1) |

Two upstream harnesses are worth converging on, both in
`examples/offline_inference/internvla_a1/`:

* `end2end.py --benchmark-forward` — in-process forward latency: one cold start,
  `--warmup-iters`, then `--benchmark-iters`, syncing around each, and
  `_latency_summary()` writes mean/stdev/min/max/**p50/p90** to
  `forward_latency.json` alongside dtype, attn implementation and compile flags.
  Same shape as `phase5_latency.py` + `run_perf_check.sh`, but with a diffable
  artifact and a p90 we do not compute.
* `internvla_a1_common.py::run_open_loop_evaluation` — open-loop **accuracy**:
  per-episode MSE/MAE against dataset ground truth, split joint vs. gripper, with
  plots.

The second one names the real hole. Everything this port has validated is a port
check: fp32 parity against golden tensors says the translation is faithful, not
that the policy acts well. Recorded as remaining-work items 4-6 in the example
README.

### 2026-09-04 — the OpenVINO reference does the same work in 289 ms

> **The stage comparison below stands. The cause does not — the "wasted MoE
> arithmetic" hypothesis is wrong, and the next entry shows why.** Kept because
> reading the reference's IR is what refuted it, and the refutation is the useful
> part.

`frameworks.robotics.embodied-intelligence.lingbot-vla-v2/run_info_demo_dgpu_int8.log`,
same B60 (`[device] GPU.1`), 5 warmup iterations, same 10 denoise steps, same
`(1, 50, 55)` action chunk. It is a far stronger reference than Phase 0's PyTorch
spike and it changes this phase's closing conclusion.

Comparing model stages only — the OV total of 288 ms is exactly
`vit 17 + text 59 + loop 213`, with no observation preprocessing, no H2D and no
unnormalize, so ours must be compared with those excluded too:

| stage | OV | ours (eager bf16, dense) | ratio |
|---|---|---|---|
| vit prefix / `embed_prefix` | 17 ms | 25.0 ms | 1.47x |
| text prefix / `prefix_fill` | 59 ms | 65.5 ms | 1.11x |
| **10 denoise steps** | **213 ms** (21.3/step) | **607.3 ms** (60.7/step) | **2.85x** |
| model-only total | **289 ms** | **697.8 ms** | **2.41x** |

**The prefix side is at parity.** 76 ms vs 90.5 ms. Whatever is wrong is not the
backbone, not the KV cache fill and not the processor. Of the 409 ms difference,
**394 ms (96%) is the denoise loop**.

**And the arithmetic accounts for it almost exactly.** Counting the action expert
at 36 layers, 51 suffix tokens, 286 prefix keys, hidden 768, 32 heads x 128 with
8 KV heads, routed MoE 32 experts of intermediate 512, shared expert 704:

| | per layer per step | 36 layers x 10 steps |
|---|---|---|
| attention projections | 0.802 GFLOP | |
| attention scores | 0.282 GFLOP | |
| shared expert | 0.165 GFLOP | |
| routed MoE, **all 32 experts** | **3.850 GFLOP** | **1.836 TFLOP** |
| routed MoE, **top-4 only** | **0.481 GFLOP** | **0.623 TFLOP** |

Dense is **2.95x** the arithmetic. Measured, it is **2.85x** the time. Effective
throughput lands at 3.02 TFLOPS for our dense path and 2.92 TFLOPS for the OV
loop if it routes top-4 — i.e. **the two implementations run at the same
efficiency, and the entire denoise gap is the 28 experts we compute and throw
away.** (If the OV loop were also dense it would have to be hitting 8.62 TFLOPS,
nearly 3x our per-FLOP rate, which no fusion story explains at 51 tokens.)

Neither side is bandwidth-bound: the action expert is 1.70 B parameters, 3.40 GB
in bf16 if every expert is read, 1.02 GB for top-4 — 7.5 ms and 2.2 ms
respectively at the B60's memory bandwidth, against 60.7 and 21.3 ms measured.
Both are an order of magnitude off, so this is small-GEMM efficiency, not DRAM.

**This falsifies step 5's "not needed".** The step-2 conclusion — dense beats
gather 3.7x — was correct *for eager PyTorch*, where 70,000 tiny launches cost
more than 8x the arithmetic. It was never a statement that dense is the right
kernel. With a real grouped MoE kernel (one batched GEMM over the 4 selected
experts, not a Python loop over 32) the FLOP argument comes back, and the ceiling
it points at is ~289 ms, i.e. **~3.4 Hz**. Grouped MoE moves from "optional" to
the single highest-value remaining item.

Two things to confirm before treating the 289 ms as settled:

* The IR filenames say `_int8` but `[precision]` says `vit=f16 text=f16
  action=f16` and `[weights]` says `fp16-compressed` for all three. If the
  weights really are int8 rather than fp16, part of the gap is quantization
  rather than routing, and the top-4 inference above is contaminated.
* `[device] GPU.1` needs to be confirmed as the same B60, not a second adapter.

A third difference is real but probably second-order: `[loop] 10 denoise steps
(1 IR call)` means all 36 layers x 10 steps are one compiled static graph, built
in 7.3 s and cached in `/root/.ov_cache`, while our server runs `--enforce-eager`
with no graph capture at all. That removes dispatch overhead — but dispatch
overhead cannot be the main term, because the FLOP ratio already explains 2.85x
of the 2.85x.

**Discriminating experiment for the routing hypothesis:** slice the checkpoint to
4 experts per layer and re-run `phase5_latency.py`. If denoise falls to ~210 ms,
routing is the whole story and a grouped kernel is worth building. If it falls
much less, the compiled graph is doing something else and this needs another look.

### 2026-09-04 — correction: the reference is dense too. The gap is efficiency, not FLOPs.

The previous entry inferred the reference must route top-4 because a 2.95x
arithmetic ratio matched a 2.85x time ratio. Reading
`converter/action_expert_loop_int8.xml` instead of inferring shows that is wrong.

Per MoE layer, per denoise step, the reference emits exactly four einsums:

| equation | inputs | output |
|---|---|---|
| `th,eih->tei` (gate) | `51x768`, `32x512x768` | `51x32x512` |
| `th,eih->tei` (up) | `51x768`, `32x512x768` | `51x32x512` |
| `tei,ehi->teh` (down) | `51x32x512`, `32x768x512` | `51x32x768` |
| `teh,te->th` (combine) | `51x32x768`, `51x32` | `51x768` |

That is `forward_dense`, operator for operator. `TopK` (`51x32 -> 51x4`) appears
once per layer per step and only builds the routing weights; the expert GEMMs run
over all 32. **Both implementations compute the same 1.836 TFLOP.** Three facts
worth recording from the same parse:

* **The routed experts are not quantized.** All 1359.0 M elements of
  `.mlp.experts.` weight constants are `fp16` — exactly `36 x 3 x 32 x 768 x 512`.
  The `_int8` in the filename covers 0.438 GB of `u8` constants elsewhere
  (attention projections and the shared expert, ~19% of the FLOPs). The log's own
  `[precision] action=f16` was the accurate line; the filename was not. The
  precision caveat in the previous entry is resolved: quantization is not the
  explanation either.
* **The graph is fully unrolled.** 40 einsums and 10 TopKs per layer — 4 and 1
  per step across 10 steps — with no `Loop` or `TensorIterator` op anywhere. All
  36 layers x 10 steps are one static graph, which is what `[loop] ... (1 IR
  call)` means.
* **The reference independently chose dense.** Step 2's decision was not a
  workaround for eager PyTorch's launch overhead; it is what the vendor's own
  export does at this shape. That retroactively confirms it.

So the gap is **per-FLOP efficiency, and nothing else**:

| | FLOPs | time | effective |
|---|---|---|---|
| ours, eager PyTorch bf16 | 1.836 TFLOP | 607.3 ms | **3.02 TFLOPS** |
| OpenVINO, compiled fp16 | 1.836 TFLOP | 213.0 ms | **8.62 TFLOPS** |

**2.85x on identical arithmetic.** Neither is bandwidth-bound (3.40 GB of bf16
weights is 7.5 ms at the B60's bandwidth), so both are leaving the card mostly
idle; the reference just leaves it less idle. Candidate causes, in the order they
should be tested:

1. **fp16 vs bf16.** The reference runs `f16`; this port runs bf16 because that
   was the upstream-validated dtype, never because it was measured faster here.
   On Intel Xe XMX, fp16 is the first-class path. This is a one-flag A/B.
2. **Compiled static graph vs eager.** One submitted graph against ~14,400
   einsum dispatches plus everything around them, per request, through the Python
   interpreter — and the server additionally runs `--enforce-eager`, so there is
   no graph capture at all.
3. **Fusion.** The reference's IR has `Swish`, `ReduceMean`/`Sqrt`/`Divide`
   chains and the `Multiply`/`Add` around them available to the GPU plugin as
   fusable groups; eager materialises each one.

**Grouped/top-4 MoE is demoted again**, and this time for a better reason than
before: the reference reaches 289 ms *without* it, so it is not on the path to
matching the reference. It remains a genuine 2.95x arithmetic saving that neither
implementation collects — worth revisiting only after the efficiency gap closes,
because a 2.95x FLOP cut on a kernel running at 35% of the reference's efficiency
is worth less than fixing the 35%.

**Corrected ceiling:** the reference shows 289 ms of model time is achievable on
this card at this precision with this algorithm, i.e. **~3.4 Hz** — same number as
before, different route to it.

---

# Phase 6 — closing the efficiency gap to the OpenVINO reference

Phase 5 met its targets (≥1 Hz, and Phase 0's 0.74 s). Phase 6 exists because a
better reference appeared: the vendor's OpenVINO export does the *same algorithm*
on the *same card* in 289 ms of model time, at 8.62 TFLOPS against our 3.02.
There is no algorithmic idea left to find — this is 2.85x of implementation
efficiency on identical arithmetic.

**Target:** 289 ms of model time (~3.4 Hz). **Floor to beat first:** anything
under 500 ms would already double the served rate.

The rules from Phase 5 carry over unchanged, with one addition: **read the
reference before theorising about it.** Phase 5's last-but-one entry inferred the
reference's algorithm from a FLOP ratio and got it backwards; parsing 45 MB of
IR took two minutes and settled it.

## Steps

| # | step | question it answers | status |
|---|---|---|---|
| 1 | **fp16 vs bf16 A/B** | The reference runs `f16`; this port runs bf16 because that was the upstream-validated dtype, never because it was measured. On Intel Xe XMX fp16 is the first-class path. One flag, no code. | **done — negative, 2.4%** |
| 2 | **Escape eager** | The reference submits one static graph for all 36 layers x 10 steps; we dispatch every op from Python, and the server adds `--enforce-eager` on top. Try `torch.compile` on the denoise loop at fixed shapes, and drop `--enforce-eager` if the engine allows it. | **in progress — now the whole phase** |
| 3 | **Sub-stage attribution inside one layer** | The 2.85x is currently attributed to the loop as a whole. Split one layer into attention / router / MoE einsums / shared expert / norms so the remaining gap has an address. | **done — it is dispatch, not any stage** |
| 4 | **Perf-script changes** | Make the harness able to track 1-3: a model-only subtotal comparable to the reference's 289 ms, the reference itself printed alongside Phase 0's 0.74 s, p90, and a JSON artifact that can be diffed across runs. | pending |
| 5 | **Grouped/top-4 MoE** | Deliberately last. It is a real 2.95x arithmetic saving that *neither* implementation collects, but the reference hits 289 ms without it, so it is not on the path to parity. Revisit only once efficiency is fixed. | deferred |

## Log

Newest last.

### 2026-09-04 — per-step Inductor reaches the OpenVINO latency class, with drift

`predict_velocity` is a viable fixed-shape compile boundary. Compiling it with
`torch.compile(backend="inductor", dynamic=False, fullgraph=True)` produced no
graph breaks and reduced the 10-step denoise loop from ~607 ms to **215 ms**.
The opened-up request measured **311 ms** total versus the OpenVINO model-only
reference of 289 ms.

The real OpenPI performance harness, with the option serialized into the
prepared config and consumed by `LingbotVlaV2Pipeline`, measured:

```text
warm WebSocket, 8 requests: median 0.381 s (2.62 Hz), min 0.345, max 0.539
offline one request: 0.482 s
```

Numerics prevent enabling it by default without an accuracy gate:

| backend/options | velocity max rel | 10-step chunk max rel | result |
|---|---:|---:|---|
| `aot_eager`, fullgraph | 0 | 0 | exact, no speedup (58.7 ms/step) |
| Inductor default | 3.18% | 1.75% | 21.4 ms/step |
| + `force_same_precision` | 3.18% | — | no effect |
| + `emulate_precision_casts` | 2.71% | — | still outside strict parity |

Both eager and compiled paths are internally deterministic. XPU is already in
`FP32MathMode.FP32`, so TF32/BF32 attention is not the cause. The product option
`compile_denoise_step` is therefore **default false** and the performance script
exposes `--compile-denoise-step` for controlled A/B runs. The next correctness
step is open-loop action evaluation; the next compiler step is locating the
first Inductor-lowered operation that creates the drift.

### 2026-09-04 — step 1: fp16 is not the answer (2.4%)

`--dtype float16` against the bf16 baseline, same harness, same host state.
The denoise loop moved by **2.4%** — inside run-to-run noise for this harness.
The reference's `f16` is therefore incidental to its 289 ms, not causal. Closed
negative; bf16 stays the default because it is the upstream-validated dtype.

> **Revisited 2026-09-07.** Closed negative *as a latency lever*, and that still
> holds. But fp16 turns out to be the main **accuracy** lever: bf16's 8 mantissa
> bits put us at mae 4.6e-2 against a fp32 reference, worse than the OpenVINO
> path's int8. 2.4% is fp16's price, not its product. See `PHASE7_NUMERICS.md`.

### 2026-09-04 — step 5 (early, as a micro): the MoE einsums are not the problem

`phase6_moe_micro.py` runs the action expert's two MoE einsums standalone at the
real shapes, with no surrounding model. They sustain **13.06 TFLOPS**, which for
the whole 10-step loop's MoE arithmetic works out to **106.2 ms**. That is real
device work and it is not recoverable by a better kernel — the einsum
formulation is already near the card's achievable rate for these shapes. It also
bounds step 5's upside: the arithmetic saving is real but the kernel is not
leaving time on the table.

### 2026-09-04 — step 3: the loop is dispatch-bound, and that is the whole gap

`phase6_denoise_profile.py` over one request's denoise loop:

```text
227465 aten ops
summed self DEVICE time:     51.0 ms   (what the card actually spent)
summed self CPU time:      1065.2 ms   (what the host spent dispatching)
implied per-op floor at 5.1 us submit cost: 1160.1 ms

bucket   calls    dev ms    cpu ms   share
glue    210557     48.40    855.35   80.3%
gemm     16908      2.58    209.84   19.7%
```

93% of the dispatches are glue — `copy_`, `mul`, `cat`, `add`, `view`,
`as_strided`. Device self time is a **floor**, not a total: `mm`/`einsum` report
zero on this backend, so combine it with the micro above (106 ms of MoE alone)
and real device work is ~150–250 ms — i.e. roughly the OpenVINO reference's
entire 213 ms loop. **The reference is not computing faster. It removed the host
from the inner loop.** That retires every "what work is done" hypothesis at
once: int8, top-4 routing, dtype, MoE kernel choice, einsum formulation. Step 2
(escape eager) is not one option among five; it is the only one.

A methodology note worth keeping: the first run of this profile reported
"192.9 ms device time", which would have inverted the conclusion. `_self_device_us`
fell through to `self_cpu_time_total` when the backend reported no device time,
silently relabelling host time as device time. It was caught because
`aten::as_strided` showed 0.2 us/call and `as_strided` does no device work. The
helper now returns `0.0` rather than falling back, and the report prints device
and CPU as separate columns.

### 2026-09-07 — the open-loop mae 0.62 was the wrong checkpoint, not a port bug

`run_open_loop_eval.log` reported mse 0.665 / mae 0.615. Repointing the runner
at the RoboTwin fine-tune and changing nothing else:

| checkpoint | mae all | mae active arm | mae idle arm | jerk |
|---|---:|---:|---:|---:|
| ground truth | 0 | 0 | 0 | 0.0023 |
| hold-state baseline | 0.2099 | — | — | 0 |
| `lingbot-vla-v2-6b` (what ran) | 0.6150 | 0.6533 | 0.7272 | 0.3317 |
| `…-6b-robotwin/…/hf_ckpt` | **0.0112** | 0.0185 | 0.0068 | 0.0118 |

`lingbot-vla-v2-6b` is the **6B foundation model** — 60k hours of general
pre-training, never fine-tuned on RoboTwin. `lingbot-vla-v2-6b-robotwin` is the
RoboTwin 2.0 fine-tune (`checkpoints/global_step_50000/hf_ckpt`), whose own
README claims 100% on `adjust_bottle`. The two have **identical architecture and
identical tensor names** (1708 keys, same key set), so the foundation checkpoint
loads and runs without a single warning. Accuracy is the only signal that tells
them apart. `run_open_loop_eval.sh` now defaults to the fine-tune and says why.

Every symptom is explained by the checkpoint and nothing else:

* the foundation model predicted **both** arms sweeping — 0.727 mae on the six
  idle-arm dims where ground truth is exactly `0.0`, and the predicted idle arm
  was near a mirror of the predicted active arm. The fine-tune keeps it parked
  (mean `|.|` 0.007). Half the 0.615 came from those six dims alone.
* the first predicted action sat **0.89 away from the current state**; a policy
  that knows the action space starts where the robot is. The fine-tune: 0.008
  (ground truth 0.0017).
* the chunk was temporally white — jerk 0.33 against ground truth's 0.0023, flat
  in `num_steps` from 5 to 100 while mae never improved. The fine-tune's jerk is
  0.0118. That flatness was the tell that the *velocity field* was wrong rather
  than under-integrated, but it pointed at the weights, not at the integrator.

Things ruled out along the way, each by measurement rather than by reading:

| checked | result |
|---|---|
| metric interpretation | mse/mae are **error**; 0.6 is not "60% correct" |
| bundle export vs raw parquet | exact, max diff 2.22e-16 |
| ground-truth idle arm all-zero | genuine; this dataset is single-arm per episode |
| zeroed idle arm out of distribution? | **no** — back-solve the norm-stat means and ~⅓ of training frames also have an arm zeroed; `q01` for those joints is ~0 |
| normalization round-trip | 4.44e-16 |
| joint-group slot packing | each group padded to its own `max_dim`; matches upstream |
| image size / aspect ratio | `img_size` defaults to 256 in the training dataset builder (`lingbotvla/data/dataset.py:89`) *and* the deploy policy, and the RoboTwin yaml sets neither |
| our resize vs torchvision's | bit-identical, max\|d\| 0 |
| chat template | `LingbotVLAV2Config` defaults `use_qwen3_chat_template=True` |
| task in the training mixture | `adjust_bottle` is entry #1 of `robotwin.txt` |
| Euler direction / suffix attention mask | `t: 1→0`, `dt` negative; `att_masks=[1,1,0,…]` gives all 50 action tokens bidirectional attention |
| 45-stage forward parity | ~1e-7 relative, fp32 |
| processor parity on **real** observations | `open_loop_processor_real.py`: bit-exact images/tokens/state, action round-trip 2e-7 |

That last one is new tooling and closes a real gap. `phase2_processor_parity.py`
graded the processor only on synthetic inputs — a 256x256 noise frame, a
`standard_normal(14)` state, and the prompt `"pick up the object"` — which
dodges the resize entirely (the real frames are 240x320), never lands on the
`q01` floor, and never exercises truncation against `tokenizer_max_length=72`.
`open_loop_processor_real.py` grades all six kernel inputs plus the action
round-trip on each of the six real bundle observations.

One incidental quirk found, deliberately **not** changed: in `sample_actions`,
`dt` and `time` are built in the **model** dtype, so a bf16 run accumulates the
timestep in bf16. `-1/num_steps` is exactly representable only for `num_steps`
in {1, 2, 4, …}; at `num_steps=10` the loop's final `t` is `-0.0049` instead of
`0`, and at 100 it is `-0.0618`. Upstream does exactly the same
(`modeling_lingbot_vla_v2.py:971,973`, `dtype=dtype`), so promoting the timestep
to fp32 would make this port *more* correct arithmetically and *less* faithful
to the reference the parity gate grades against. It is not the cause of anything
here — the fine-tune scores 0.011 with the same drift. Left as-is; if it is ever
changed it has to change on both sides at once.
