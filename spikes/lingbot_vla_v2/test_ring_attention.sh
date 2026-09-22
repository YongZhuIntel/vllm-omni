#!/usr/bin/env bash
# Phase 14 — can ring attention put the iGPU inside the denoise loop?
#
# Six arms, in the order that makes a failure readable. The first is a gate and
# the second can settle the question on its own:
#
#   ring-math    Does the ring reproduce `eager_attention` at the real denoise
#                shapes (q [1,51,32,128], kv [1,337,8,128], the real block
#                mask)? One device, no transport. Until this passes, nothing
#                downstream is measuring ring attention.
#   attn-share   What fraction of the loop *is* attention? That is the ceiling
#                for any attention-only split, and §F2's layer-step
#                decomposition never isolated it. Compare the printed ceiling
#                against 360 x 2 x 0.155 ms of transport before believing any
#                two-process number is worth chasing.
#   wire         The transport alone: the real ring payloads, with the worker
#                echoing a preallocated buffer -- no weights, no kernels. Run it
#                BEFORE the two split arms, because without it the only way to
#                price the wire is a residual or an interpolation, and PHASE14's
#                two attempts at those disagreed by 2.6x (§8 ③ ④).
#   ring-2p kv   Context parallel: the 286-token prefix KV cache split 143/143,
#                queries replicated, the iGPU doing attention **and nothing
#                else** (it loads no weights at all).
#   ring-2p seq  Sequence parallel over the 51 suffix rows, 26/25. Both cards
#                hold the 6B model. This is the literal ask, and it is the
#                expensive one to run: two 6B loads, one of them into host DRAM.
#   share-sweep  "Then give the iGPU a smaller share." Three curves. Rows and
#                keys run once per card: the iGPU's row curve saturates at
#                1704 ms, because sequence parallelism replicates the weights and
#                a rank with one row still streams all 27.2 GB of them. The third
#                curve is the important one -- the **compiled** loop against real
#                prefix KV length, dGPU only. It is the only trustworthy source
#                for the KV-split ceiling (206.3 ms at 286 keys, 181.7 at 143),
#                because the eager path is launch-bound and reads flat, and an
#                ablation lets inductor dead-code-eliminate the q/k/v GEMM.
#
# Priors, all measured on this host and all in PHASE8_LATENCY_PARITY.md:
# §K1 two GPUs = two processes; §K2 the iGPU's MoE layer-step is 12.9x the
# dGPU's; §M.6 one round trip is 0.155 ms against a 0.251 ms expert block, and
# the loop has 360 layer-steps; §G5 row splitting is capped near 13% even with a
# free iGPU. Re-run this to re-derive the verdict, not to discover it.
#
# What it actually came out as (PHASE14_RING_ATTENTION.md): context parallel
# 0.38-0.40x, sequence parallel 0.18-0.19x, and the binding constraint is **not**
# the 360 collectives §G/§M.6 framed this class of scheme on -- those are 13% and
# 0.7% of the added cost. It is the iGPU's compute, and it does not shrink when
# you give the iGPU less (share-sweep).
#
# Rules #2: wants `load < 2.0` and no other container holding a GPU. Every arm
# prints the load average it started at; §L had to discard a full set of
# absolutes for ignoring this.

set -euo pipefail

MODEL="${MODEL:-/tmp/lingbot-vla-v2-perf}"
ARMS="${ARMS:-ring-math,attn-share,wire,ring-2p-kv,ring-2p-seq,share-sweep}"
DTYPE="${DTYPE:-float16}"
ITERS="${ITERS:-10}"
WARMUP="${WARMUP:-3}"
OUTDIR="${OUTDIR:-/tmp/lingbot-ring}"
# shm, not oneCCL, and that is a finding rather than a convenience: PHASE14 §5
# measured shm faster on both topologies and 15x faster on the one-time setup
# transfer (16.9 vs 524.0 ms at 20.1 MiB). PHASE10 §9's equality was measured at
# 293 KiB and does not extend to these payloads. `TRANSPORT=oneccl` re-derives it.
TRANSPORT="${TRANSPORT:-shm}"
# PHASE10 §11.4: `onecclRecv` hard-spins, and an unreserved spinner collides
# with this process's OpenMP pool for ~215 ms per round. Not optional.
WORKER_CPU="${WORKER_CPU:-10-11}"

# oneCCL v2 with the iGPU plugin, same install and same two variables
# `test_spec_oneccl.sh` uses.
I="${ONECCL_PREFIX:-/llm/zhuyong/libraries.performance.communication.oneccl-v2/build/_install}"
export LD_LIBRARY_PATH="$I/lib:$I/opt/mpi/lib:${LD_LIBRARY_PATH:-}"
export CCL_PLUGIN=ONECCL_IGPU

# Capped rather than inherited, so these arms stay comparable to §11's numbers
# whatever the caller's shell exports (§F6: OpenMP oversubscription on a hybrid
# CPU cost 160 ms before it was found).
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

cd "${REPO:-/llm/zhuyong/lingbovla/my/vllm-omni}"
mkdir -p "$OUTDIR"
PROBE=spikes/lingbot_vla_v2/phase14_ring_attention_probe.py

# Rules #2 is about the load average at the *start* of an arm, and an arm that
# has just finished leaves the 1-minute average at ~1.7. Waiting is cheaper than
# discarding the run: §L threw away a full set of absolutes over this, and one of
# PHASE14's own runs had to be repeated for starting at 1.75.
SETTLE="${SETTLE:-90}"

load1() {  # the 1-minute average in hundredths, so this needs no `bc` (the
           # container has none) and no locale-dependent float compare
    awk '{printf "%d", $1 * 100}' /proc/loadavg
}

settle() {
    local waited=0
    while [ "$(load1)" -gt 100 ] && [ "$waited" -lt "$SETTLE" ]; do
        sleep 10; waited=$((waited + 10))
    done
    [ "$waited" -gt 0 ] && echo "[settle] waited ${waited}s; load is now $(cut -d' ' -f1 /proc/loadavg)"
    return 0
}

run() {  # run <affinity> <json-name> [extra args...]
    local affinity="$1" name="$2"; shift 2
    settle
    echo
    echo "=============================================================="
    echo "  $name   ZE_AFFINITY_MASK=$affinity"
    echo "=============================================================="
    ZE_AFFINITY_MASK="$affinity" PYTHONPATH=. python "$PROBE" \
        --model "$MODEL" --dtype "$DTYPE" \
        --json-out "$OUTDIR/$name.json" "$@"
}

for arm in ${ARMS//,/ }; do
    case "$arm" in
        ring-math)
            # The gate. fp32 must match eager to ~1e-6; fp16 is allowed to beat
            # it, since the ring accumulates the softmax in fp32 and eager does
            # not. Cheap: one device, no transport, no timing.
            run 0 ring_math --arm ring-math --shards 1,2,4,8
            ;;
        attn-share)
            # Eager on purpose. The share is a ratio and the projection onto the
            # compiled 213.5 ms baseline is stated as a projection in the output.
            run 0 attn_share --arm attn-share --iters "$ITERS" --warmup "$WARMUP"
            ;;
        wire)
            run 0 wire_shm --arm ring-2p --split wire --transport shm \
                --worker-cpu "$WORKER_CPU" --wire-iters "${WIRE_ITERS:-300}"
            run 0 wire_oneccl --arm ring-2p --split wire --transport oneccl \
                --worker-cpu "$WORKER_CPU" --wire-iters "${WIRE_ITERS:-300}"
            ;;
        ring-2p-kv)
            run 0 ring_2p_kv --arm ring-2p --split kv \
                --transport "$TRANSPORT" --worker-cpu "$WORKER_CPU" \
                --iters "$ITERS" --warmup "$WARMUP"
            ;;
        share-sweep)
            # Both cards, sequentially. Never concurrently: §L §6 measured the
            # iGPU's throughput swinging 2.4x with host activity, and §K's tax
            # runs the other way. `--iters 2` because the iGPU arm is ~1.7-2.2 s
            # per point by construction -- that slowness is the finding.
            # dGPU gets the compiled row curve too: its slope is everything
            # that scales with the 51 suffix rows (the fused q/k/v that rebuilds
            # their K/V every layer-step, o_proj, the norms). <= 10% of the loop.
            run 0 share_sweep_dgpu --arm share-sweep --sweep rows,keys \
                --compile-denoise-step --iters "${SWEEP_ITERS:-3}"
            settle
            run 1 share_sweep_igpu --arm share-sweep --sweep rows,keys \
                --iters "${SWEEP_ITERS_IGPU:-2}"
            settle
            # The prefix-length curve, compiled. This is the only trustworthy
            # source for the KV-split ceiling: no stub, so no dead-code
            # elimination, and no eager launch-bounding to flatten it. dGPU only.
            run 0 loop_kv_sweep --arm share-sweep --sweep loop --compile-denoise-step \
                --loop-prefix 286,214,143,72,8 --iters "${SWEEP_ITERS:-8}"
            ;;
        ring-2p-seq)
            # Two 6B loads, the second into host DRAM at 29 GB/s. Give it
            # minutes, and fewer iterations than the other arms: every layer-step
            # is a real iGPU MoE block at §K2's 3.246 ms.
            run 0 ring_2p_seq --arm ring-2p --split seq \
                --transport "$TRANSPORT" --worker-cpu "$WORKER_CPU" \
                --iters "${SEQ_ITERS:-3}" --warmup "${SEQ_WARMUP:-1}"
            ;;
        *)
            echo "unknown arm: $arm" >&2; exit 2
            ;;
    esac
done

echo
echo "[json] $OUTDIR"
