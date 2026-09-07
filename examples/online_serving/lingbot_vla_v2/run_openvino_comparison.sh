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
WARMUP="5"
REPEAT="20"
OUTPUT="/tmp/lingbot-openvino-comparison.json"
COMPILE_DENOISE_STEP="1"
PREPARE_MODEL="1"

usage() {
    cat <<EOF
Usage: $0 [options]

Run vLLM latency and compare against the recorded OpenVINO reference timings.

Options:
  --checkpoint PATH     vLLM checkpoint directory (default: $CHECKPOINT)
  --model PATH          Prepared model output directory (default: $MODEL)
  --vllm-device D       vLLM device (default: $VLLM_DEVICE)
  --dtype DTYPE         vLLM dtype (default: $VLLM_DTYPE)
  --warmup N            vLLM warmup runs (default: $WARMUP)
  --repeat N            vLLM timed runs (default: $REPEAT)
  --output PATH         JSON report path (default: $OUTPUT)
  --eager               Use eager vLLM denoising instead of compiled denoising
  --no-prepare          Reuse --model instead of preparing it from --checkpoint
  -h, --help            Show this help
EOF
}

while (($#)); do
    case "$1" in
        --checkpoint) CHECKPOINT="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --vllm-device) VLLM_DEVICE="$2"; shift 2 ;;
        --dtype) VLLM_DTYPE="$2"; shift 2 ;;
        --warmup) WARMUP="$2"; shift 2 ;;
        --repeat) REPEAT="$2"; shift 2 ;;
        --output) OUTPUT="$2"; shift 2 ;;
        --eager) COMPILE_DENOISE_STEP="0"; shift ;;
        --no-prepare) PREPARE_MODEL="0"; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

for value in "$WARMUP" "$REPEAT"; do
    [[ "$value" =~ ^[1-9][0-9]*$ ]] || {
        echo "warmup/repeat must be positive integers: $value" >&2
        exit 2
    }
done

cd "$REPO"
export PYTHONPATH=.

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
    --warmup "$WARMUP"
    --repeat "$REPEAT"
    --output "$OUTPUT"
)
if [[ "$COMPILE_DENOISE_STEP" == "1" ]]; then
    COMPARE_ARGS+=(--compile-denoise-step)
fi

python "${COMPARE_ARGS[@]}"
