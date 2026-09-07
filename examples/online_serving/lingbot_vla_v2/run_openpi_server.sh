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
COMPILE_DENOISE_STEP="0"

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
    --compile-denoise-step
                                             Enable experimental Inductor denoise compilation
  -h, --help          Show this help
EOF
}

while (($#)); do
    case "$1" in
        --checkpoint) CHECKPOINT="$2"; shift 2 ;;
        --output) OUTPUT="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --dtype) DTYPE="$2"; shift 2 ;;
        --compile-denoise-step) COMPILE_DENOISE_STEP="1"; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

cd "$REPO"
export PYTHONPATH=.
python -m pip install -e . --no-deps
PREPARE_ARGS=(--checkpoint "$CHECKPOINT" --output "$OUTPUT")
if [[ "$COMPILE_DENOISE_STEP" == "1" ]]; then
    PREPARE_ARGS+=(--compile-denoise-step)
fi
python examples/offline_inference/lingbot_vla_v2/prepare_lingbot_vla_v2.py \
    "${PREPARE_ARGS[@]}"
python -m vllm_omni.entrypoints.cli.main serve "$OUTPUT" \
    --omni --host 0.0.0.0 --port "$PORT" \
    --dtype "$DTYPE" --enforce-eager --disable-log-stats