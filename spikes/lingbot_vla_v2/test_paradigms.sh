#!/usr/bin/env bash
# Does ParaDiGMS (Picard parallel sampling) beat the ten sequential Euler steps,
# on the dGPU alone or with the iGPU taking a share?
#
# Three arms, and the first one can settle it on its own:
#
#   picard      How many sweeps until Picard reproduces the sequential answer,
#               and what one sweep costs. Break-even is ~2.36 sweeps (one B=10
#               batched forward against the loop it replaces), so three sweeps
#               means the scheme loses before the iGPU is even considered.
#   step-cost   One eager predict_velocity per device. Run twice, once per card.
#               Prices the iGPU's share of a split sweep -- §L's ~1140 ms is MoE
#               only and therefore a lower bound.
#   contention  What the dGPU pays while the iGPU runs real denoise. Splitting a
#               sweep across two cards requires concurrency, so this is the
#               scheme's structural cost, not avoidable overhead. §K measured
#               1.74-1.76x with an EU-bound matmul; denoise is memory-bound, so
#               it gets re-measured at this shape rather than assumed -- and it
#               comes out at 1.01x, which is why §K's tax is now stated over
#               occupancy rather than duration (§M §4).
#
# Verdict, PHASE8_LATENCY_PARITY.md §M: ParaDiGMS loses on the dGPU alone (9
# sweeps against a 2.19 break-even, 0.24x), and the iGPU cannot take even one of
# the ten points (233 ms for one against the dGPU's 153 ms for all ten). Re-run
# this to re-derive that, not to discover it.
#
# Rules #2: wants `load < 2.0` and no other container holding a GPU. Every arm
# prints the load average it started at; §L had to discard a full set of
# absolutes for ignoring this.

set -euo pipefail

MODEL="${MODEL:-/tmp/lingbot-vla-v2-perf}"
ARMS="${ARMS:-picard,step-cost,contention}"
DTYPE="${DTYPE:-float16}"
ITERS="${ITERS:-10}"
WARMUP="${WARMUP:-3}"
OUTDIR="${OUTDIR:-/tmp/lingbot-paradigms}"
# step-cost on the iGPU is eager on a 6B model; it is slow by construction, which
# is the finding, so keep the iteration count low.
IGPU_ITERS="${IGPU_ITERS:-3}"
IGPU_WARMUP="${IGPU_WARMUP:-1}"

# Capped rather than inherited, so these arms stay comparable to §11's numbers
# whatever the caller's shell exports.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

cd /llm/zhuyong/lingbovla/my/vllm-omni
mkdir -p "$OUTDIR"
PROBE=spikes/lingbot_vla_v2/phase12_paradigms_probe.py

run() {  # run <affinity> <arm> <json-name> [extra args...]
    local affinity="$1" arm="$2" name="$3"; shift 3
    echo
    echo "=============================================================="
    echo "  arm=$arm  ZE_AFFINITY_MASK=$affinity"
    echo "=============================================================="
    ZE_AFFINITY_MASK="$affinity" PYTHONPATH=. python "$PROBE" \
        --arm "$arm" --model "$MODEL" --dtype "$DTYPE" \
        --json-out "$OUTDIR/$name.json" "$@"
}

for arm in ${ARMS//,/ }; do
    case "$arm" in
        picard)
            # Compiled: the break-even it is scored against (213.5 ms sequential,
            # 90.5 ms per sweep) is a compiled number.
            run 0 picard picard --iters "$ITERS" --warmup "$WARMUP" --compile-denoise-step
            ;;
        step-cost)
            # Eager on both, so the ratio is apples to apples. §L's probe splits
            # per device the same way, for the same reason (§K1: torch-xpu sees
            # one Level-Zero platform per process).
            run 0 step-cost step_cost_dgpu --iters "$ITERS" --warmup "$WARMUP"
            run 1 step-cost step_cost_igpu --iters "$IGPU_ITERS" --warmup "$IGPU_WARMUP"
            ;;
        contention)
            # Parent on the dGPU; it spawns its own iGPU load generator.
            run 0 contention contention --iters "$ITERS" --warmup "$WARMUP" \
                --compile-denoise-step --load-affinity 1
            ;;
        *)
            echo "unknown arm: $arm" >&2; exit 2
            ;;
    esac
done

echo
echo "[json] $OUTDIR"
