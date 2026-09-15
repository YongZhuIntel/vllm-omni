#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
set -euo pipefail

# Run the vLLM model-side latency probe and compare its result with the recorded
# OpenVINO one-IR-call reference. This script never launches OpenVINO; it only
# needs the vLLM-Omni runtime and the vLLM checkpoint.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd -- "$SCRIPT_DIR/../../.." && pwd)
CHECKPOINT="/llm/zhuyong/lingbovla/models/lingbot-vla-v2-6b"
MODEL="/tmp/lingbot-vla-v2-perf"
VLLM_DEVICE="xpu"
VLLM_DTYPE="float16"
ATTENTION_BACKEND="eager"
ATTENTION_PRECISION="fp16"
WARMUP="5"
REPEAT="20"
COMPILE_MAX_RELATIVE_ERROR="0.05"
OUTPUT="/tmp/lingbot-openvino-comparison.json"
COMPILE_DENOISE_STEP="1"
COMPILE_PREFIX="0"
PREPARE_MODEL="1"
# Speculative decoding (Phase 10). SPEC_DECODE is the switch; the draft head
# always runs on the iGPU in a second process, because torch.xpu.device_count()
# is 1 per process on this host. ACCEPT_RATE forces the accept decision so the
# schedule can be priced at a chosen acceptance rate: the verify passes still
# run, so the latency is real, but an accepted chunk is the untrained draft's own
# and the actions are not valid. "measured" uses the real accept rule, which is 0
# until a draft head is trained. Measured: 294 ms per tick at rate 0 (every tick
# re-grounds) against 95.8 ms at rate 1 with SPEC_FULL_EVERY=4, K=2.
SPEC_DECODE="0"
ACCEPT_RATE="0,0.5,1"
SPEC_FULL_EVERY="4"
SPEC_T_LIST="0.10 0.05"
SPEC_TICKS="20"
# Not optional in practice: the draft worker's oneCCL recv hard-spins a core, and
# unreserved it collides with this process's OpenMP pool and adds ~215 ms to
# every full round -- which also slows the non-speculative arm, so it inflates
# the speculative speedup instead of showing up as a regression. §11.4.
SPEC_WORKER_CPU="11"
SPEC_DRAFT_PATH=""
SPEC_OUTPUT="/tmp/lingbot-spec-accept-sweep.json"
# Pinned, not inherited, so this script measures the same thing whatever the
# caller's shell happens to export. F6 found that an uncapped intra-op pool
# costs ~160 ms of *served* median on this hybrid CPU, but that finding does not
# apply here and this is deliberately recorded rather than assumed: measured
# 2026-09-09, this script reports total 294.3 ms uncapped and 294.2 ms at 4
# threads. There is no asyncio loop or websocket thread competing in-process, so
# the idle pool never takes the dispatch thread's core. The 294 ms reference
# number in `PHASE8_LATENCY_PARITY.md` is therefore unaffected by the serving
# fix, and the two scripts stay comparable to their own histories.
OMP_THREADS="${OMP_NUM_THREADS:-4}"

usage() {
    cat <<EOF
Usage: $0 [options]

Run vLLM latency and compare against the recorded OpenVINO reference timings.

Options:
  --checkpoint PATH     vLLM checkpoint directory (default: $CHECKPOINT)
  --model PATH          Prepared model output directory (default: $MODEL)
  --vllm-device D       vLLM device (default: $VLLM_DEVICE)
  --dtype DTYPE         vLLM dtype (default: $VLLM_DTYPE)
    --attention-backend D vLLM attention backend (default: $ATTENTION_BACKEND)
    --attention-precision P attention precision: fp32 or fp16 (default: $ATTENTION_PRECISION)
  --warmup N            vLLM warmup runs (default: $WARMUP)
  --repeat N            vLLM timed runs (default: $REPEAT)
    --compile-max-relative-error X  Compiled parity tripwire (default: $COMPILE_MAX_RELATIVE_ERROR)
  --output PATH         JSON report path (default: $OUTPUT)
  --eager               Use eager vLLM denoising instead of compiled denoising
    --compile-prefix      Compile the fixed-shape Prefix walk with Inductor
  --omp-threads N       Intra-op thread cap (default: $OMP_THREADS; measured neutral here)
  --no-prepare          Reuse --model instead of preparing it from --checkpoint
  -h, --help            Show this help

Speculative decoding (draft head on the iGPU, second process):
  --spec-decode         Also run the speculative per-tick sweep after the comparison
  --accept-rate LIST    Forced acceptance rates, comma separated, or "measured"
                        (default: $ACCEPT_RATE). Implies nothing on its own; pass
                        --spec-decode too. Latency is real at every rate; the
                        actions are not, because an accepted chunk is the draft's.
  --spec-full-every N   Speculative rounds between full rounds (default: $SPEC_FULL_EVERY)
  --spec-t-list "A B"   Verify timesteps; K is their count (default: $SPEC_T_LIST)
  --spec-ticks N        Control ticks per rate (default: $SPEC_TICKS)
  --spec-worker-cpu C   CPUs reserved for the draft worker (default: $SPEC_WORKER_CPU)
  --spec-draft-path P   Trained draft head; random weights if unset
  --spec-output PATH    JSON report for the sweep (default: $SPEC_OUTPUT)
EOF
}

while (($#)); do
    case "$1" in
        --checkpoint) CHECKPOINT="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --vllm-device) VLLM_DEVICE="$2"; shift 2 ;;
        --dtype) VLLM_DTYPE="$2"; shift 2 ;;
        --attention-backend) ATTENTION_BACKEND="$2"; shift 2 ;;
        --attention-precision) ATTENTION_PRECISION="$2"; shift 2 ;;
        --warmup) WARMUP="$2"; shift 2 ;;
        --repeat) REPEAT="$2"; shift 2 ;;
        --compile-max-relative-error) COMPILE_MAX_RELATIVE_ERROR="$2"; shift 2 ;;
        --output) OUTPUT="$2"; shift 2 ;;
        --eager) COMPILE_DENOISE_STEP="0"; shift ;;
        --compile-prefix) COMPILE_PREFIX="1"; shift ;;
        --omp-threads) OMP_THREADS="$2"; shift 2 ;;
        --no-prepare) PREPARE_MODEL="0"; shift ;;
        --spec-decode) SPEC_DECODE="1"; shift ;;
        --accept-rate) ACCEPT_RATE="$2"; ACCEPT_RATE_SET="1"; shift 2 ;;
        --spec-full-every) SPEC_FULL_EVERY="$2"; shift 2 ;;
        --spec-t-list) SPEC_T_LIST="$2"; shift 2 ;;
        --spec-ticks) SPEC_TICKS="$2"; shift 2 ;;
        --spec-worker-cpu) SPEC_WORKER_CPU="$2"; shift 2 ;;
        --spec-draft-path) SPEC_DRAFT_PATH="$2"; shift 2 ;;
        --spec-output) SPEC_OUTPUT="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ "$SPEC_DECODE" != "1" && "${ACCEPT_RATE_SET:-0}" == "1" ]]; then
    echo "--accept-rate only applies to the speculative sweep; pass --spec-decode as well." >&2
    exit 2
fi

for value in "$WARMUP" "$REPEAT"; do
    [[ "$value" =~ ^[1-9][0-9]*$ ]] || {
        echo "warmup/repeat must be positive integers: $value" >&2
        exit 2
    }
done

cd "$REPO"
export PYTHONPATH=.
export OMP_NUM_THREADS="$OMP_THREADS"

if [[ "$PREPARE_MODEL" == "1" ]]; then
    [[ -d "$CHECKPOINT" ]] || { echo "vLLM checkpoint not found: $CHECKPOINT" >&2; exit 1; }
    echo "== prepare vLLM model =="
    rm -rf "$MODEL"
    PREPARE_ARGS=(--checkpoint "$CHECKPOINT" --output "$MODEL")
    if [[ "$COMPILE_DENOISE_STEP" == "0" ]]; then
        PREPARE_ARGS+=(--no-compile-denoise-step)
    fi
    python examples/offline_inference/lingbot_vla_v2/prepare_lingbot_vla_v2.py \
        "${PREPARE_ARGS[@]}"
else
    [[ -d "$MODEL" ]] || { echo "prepared vLLM model not found: $MODEL" >&2; exit 1; }
fi

COMPARE_ARGS=(
    spikes/lingbot_vla_v2/compare_openvino.py
    --vllm-only
    --model "$MODEL"
    --vllm-device "$VLLM_DEVICE"
    --vllm-dtype "$VLLM_DTYPE"
    --attention-backend "$ATTENTION_BACKEND"
    --attention-precision "$ATTENTION_PRECISION"
    --warmup "$WARMUP"
    --repeat "$REPEAT"
    --compile-max-relative-error "$COMPILE_MAX_RELATIVE_ERROR"
    --output "$OUTPUT"
)
if [[ "$COMPILE_DENOISE_STEP" == "1" ]]; then
    COMPARE_ARGS+=(--compile-denoise-step)
fi
if [[ "$COMPILE_PREFIX" == "1" ]]; then
    COMPARE_ARGS+=(--compile-prefix)
fi

python "${COMPARE_ARGS[@]}"

if [[ "$SPEC_DECODE" == "1" ]]; then
    echo
    echo "== speculative per-tick sweep (draft on the iGPU) =="
    # Drives the shipped SpecDecoder / IGpuDraftClient, not a spike copy, so the
    # numbers are the server's. ZE_AFFINITY_MASK pins this process to the dGPU
    # and leaves the iGPU to the worker it spawns.
    SPEC_ARGS=(
        spikes/lingbot_vla_v2/phase10_spec_accept_sweep.py
        --model "$MODEL"
        --device "$VLLM_DEVICE"
        --dtype "$VLLM_DTYPE"
        --attention-backend "$ATTENTION_BACKEND"
        --attention-precision "$ATTENTION_PRECISION"
        --compile-max-relative-error "$COMPILE_MAX_RELATIVE_ERROR"
        --warmup "$WARMUP"
        --ticks "$SPEC_TICKS"
        --accept-rate "$ACCEPT_RATE"
        --full-every "$SPEC_FULL_EVERY"
        --json-out "$SPEC_OUTPUT"
    )
    # shellcheck disable=SC2206  # word splitting is the point: "0.10 0.05" -> two args
    SPEC_ARGS+=(--t-list ${SPEC_T_LIST})
    if [[ "$COMPILE_DENOISE_STEP" == "1" ]]; then
        SPEC_ARGS+=(--compile-denoise-step)
    fi
    if [[ -n "$SPEC_WORKER_CPU" ]]; then
        SPEC_ARGS+=(--worker-cpu "$SPEC_WORKER_CPU")
    fi
    if [[ -n "$SPEC_DRAFT_PATH" ]]; then
        SPEC_ARGS+=(--draft-path "$SPEC_DRAFT_PATH")
    fi
    ZE_AFFINITY_MASK="${ZE_AFFINITY_MASK:-0}" python "${SPEC_ARGS[@]}"
fi
