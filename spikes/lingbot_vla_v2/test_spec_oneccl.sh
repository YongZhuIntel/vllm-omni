#!/usr/bin/env bash
# Per-tick cost of speculative decoding over scheme x K x accept rate.
#
# Runs the shipped SpecDecoder with the real iGPU draft worker over oneCCL, and
# sweeps the three axes that actually move the number:
#
#   SCHEMES        cached   -- reuse the last full round's prefix KV; on rejection
#                              defer a full round to the next tick (today's ship).
#                  reground -- config.spec_reground: fresh embed_prefix +
#                              prefix_fill every speculative round, draft off the
#                              fresh embeddings, and rejection runs the Euler loop
#                              inside the rejecting tick.
#   KS             verify count. Batched into one B*K forward, so cheaper than it
#                  looks -- but `min` over K lowers *real* acceptance, which a
#                  forced-rate sweep cannot show. See config.spec_verify_batched.
#   ACCEPT_RATES   forced acceptance. The verify passes run at every rate, so the
#                  latency is real; the actions are not, because an accepted chunk
#                  is the untrained draft's own. `measured` uses the real rule,
#                  which is 0 until a draft head is trained.
#
# Which scheme wins is entirely a function of acceptance, so both arms are run at
# every cell rather than argued about. Override any axis from the environment:
#
#   SCHEMES=reground KS=2 ACCEPT_RATES=0,1 bash spikes/lingbot_vla_v2/test_spec_oneccl.sh
#
# --spec-worker-cpu is not optional in practice: the worker's oneCCL recv hard
# spins a core, and unreserved it collides with this process's OpenMP pool and
# adds ~215 ms to every full round -- which slows the baseline too, so it inflates
# the speedup instead of showing up as a regression (PHASE10_SPECULATIVE.md §11.4).

set -euo pipefail

I=/llm/zhuyong/libraries.performance.communication.oneccl-v2/build/_install
export LD_LIBRARY_PATH=$I/lib:$I/opt/mpi/lib:${LD_LIBRARY_PATH:-}
export CCL_PLUGIN=ONECCL_IGPU

MODEL="${MODEL:-/tmp/lingbot-vla-v2-perf}"
SCHEMES="${SCHEMES:-cached,reground}"
KS="${KS:-1,2,4}"
ACCEPT_RATES="${ACCEPT_RATES:-0,0.25,0.5,0.75,0.9,1,measured}"
# Per arm, and an arm is one (scheme, K, rate) cell, so keep an eye on the
# product: the defaults above are 2 x 3 x 7 = 42 arms.
#
# Do not lower this. --accept-rate stratifies in shuffled blocks of 20, so at 20
# ticks there is exactly one block and whether a forced rejection lands on a tick
# the periodic schedule wanted anyway is luck -- which showed up as 0.75
# measuring *faster* than 0.5 across all three K. 60 ticks restored monotonicity
# in every row (PHASE10_SPECULATIVE.md §12.9).
TICKS="${TICKS:-60}"
WORKER_CPU="${WORKER_CPU:-11}"
FULL_EVERY="${FULL_EVERY:-4}"
OUT="${OUT:-/tmp/lingbot-spec-scheme-sweep.json}"

# Pins this process to the dGPU and leaves the iGPU to the worker it spawns.
export ZE_AFFINITY_MASK="${ZE_AFFINITY_MASK:-0}"
# Capped, not inherited, so the arms are comparable to §11's numbers whatever the
# caller's shell exports.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

cd /llm/zhuyong/lingbovla/my/vllm-omni
PYTHONPATH=. python spikes/lingbot_vla_v2/phase10_spec_accept_sweep.py \
    --model "$MODEL" \
    --compile-denoise-step \
    --scheme "$SCHEMES" \
    --k-list "$KS" \
    --accept-rate "$ACCEPT_RATES" \
    --ticks "$TICKS" \
    --full-every "$FULL_EVERY" \
    --worker-cpu "$WORKER_CPU" \
    --json-out "$OUT"
