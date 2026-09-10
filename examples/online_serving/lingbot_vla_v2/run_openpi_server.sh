#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd -- "$SCRIPT_DIR/../../.." && pwd)
CHECKPOINT="/llm/zhuyong/lingbovla/models/lingbot-vla-v2-6b"
OUTPUT="/tmp/lingbot-vla-v2-prepared-openpi"
PORT="8000"
# fp16, not bf16. This is the deployment entry point, so it has to agree with
# what the accuracy work was measured on: the checkpoint is fp32, bf16 was
# discarding three mantissa bits for nothing, and against a fp32 reference bf16
# scores worse than the OpenVINO path's int8. See
# `spikes/lingbot_vla_v2/PHASE7_NUMERICS.md`.
DTYPE="float16"
COMPILE_DENOISE_STEP="1"
# Intra-op threads. PyTorch defaults to one per logical CPU (12 on this host),
# and this request is host-dispatch-bound -- 282.6 of 286.5 ms is CPU, not
# device. On a hybrid CPU that pool spans P-cores, E-cores and a low-power
# island with no L3, and roughly half the time the one dispatch thread that
# matters loses a core to it: served latency goes bimodal, 328 ms or 502 ms.
# Capping the pool removes it entirely -- median 0.486 -> 0.326 s and the
# spread collapses from 181 ms to 6 ms. 1, 2 and 4 measure the same, so this is
# oversubscription rather than OpenMP itself; 4 keeps parallelism for CPU-side
# preprocessing without exceeding the P-core count. Measured in
# `spikes/lingbot_vla_v2/PHASE8_LATENCY_PARITY.md` under F6.
OMP_THREADS="${OMP_NUM_THREADS:-4}"

usage() {
    cat <<EOF
Usage: $0 [options]

Prepare LingBot-VLA 2.0 with OpenPI metadata and start its WebSocket server.
Run this script from the vLLM-Omni checkout inside the Docker container.

Options:
  --checkpoint PATH   Checkpoint path (default: $CHECKPOINT)
  --output PATH       Prepared model path (default: $OUTPUT)
  --port PORT         Server port (default: $PORT)
  --dtype DTYPE       Inference dtype (default: $DTYPE)
  --omp-threads N     Intra-op thread cap (default: $OMP_THREADS; see comment above)
    --no-compile-denoise-step
                                             Use eager denoising instead of the default Inductor path
  -h, --help          Show this help
EOF
}

while (($#)); do
    case "$1" in
        --checkpoint) CHECKPOINT="$2"; shift 2 ;;
        --output) OUTPUT="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --dtype) DTYPE="$2"; shift 2 ;;
        --omp-threads) OMP_THREADS="$2"; shift 2 ;;
        --no-compile-denoise-step) COMPILE_DENOISE_STEP="0"; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

cd "$REPO"
export PYTHONPATH=.
python -m pip install -e . --no-deps
PREPARE_ARGS=(--checkpoint "$CHECKPOINT" --output "$OUTPUT")
if [[ "$COMPILE_DENOISE_STEP" == "0" ]]; then
    PREPARE_ARGS+=(--no-compile-denoise-step)
fi
python examples/offline_inference/lingbot_vla_v2/prepare_lingbot_vla_v2.py \
    "${PREPARE_ARGS[@]}"
export OMP_NUM_THREADS="$OMP_THREADS"
echo "OMP_NUM_THREADS=$OMP_NUM_THREADS (uncapped costs ~160 ms of served median here; see F6)"
python -m vllm_omni.entrypoints.cli.main serve "$OUTPUT" \
    --omni --host 0.0.0.0 --port "$PORT" \
    --dtype "$DTYPE" --enforce-eager --disable-log-stats