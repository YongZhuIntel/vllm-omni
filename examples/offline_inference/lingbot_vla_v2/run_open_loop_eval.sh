#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd -- "$SCRIPT_DIR/../../.." && pwd)
# The RoboTwin *fine-tune*, not the 6B foundation checkpoint one directory up.
# Both have identical architecture and tensor names, so pointing this at
# `lingbot-vla-v2-6b` loads and runs without complaint -- it just scores mae 0.62
# instead of 0.011, because the foundation model was pre-trained on 60k hours of
# general robot data and never saw RoboTwin's action space. Accuracy is the only
# signal that distinguishes them; use the foundation checkpoint for latency work
# only.
CHECKPOINT="/llm/zhuyong/lingbovla/models/lingbot-vla-v2-6b-robotwin/checkpoints/global_step_50000/hf_ckpt"
DATASET="/llm/zhuyong/lingbovla/datasets/open_loop/adjust_bottle_3ep_2chunks.npz"
OUTPUT_ROOT="/llm/zhuyong/lingbovla/datasets/open_loop/evaluation"
MODEL_ROOT="/tmp/lingbot-open-loop"
MODE="eager"
# fp16, not bf16. The checkpoint is stored in fp32, so bf16 was discarding three
# mantissa bits for no storage reason: against a fp32 reference it scores mae
# 6.2e-2 where fp16 scores 1.9e-2, and here it costs task accuracy too (mae
# 0.01125 vs 0.00785 on this bundle). fp16's 65504 ceiling is not a risk for this
# model -- peak activation is 11264 and peak weight 44.3. See
# `spikes/lingbot_vla_v2/PHASE7_NUMERICS.md`.
DTYPE="float16"
SEED="1234"
MAX_SAMPLES=""
EPISODES=""
PLOTS="1"

usage() {
    cat <<EOF
Usage: $0 [options]

Prepare LingBot-VLA 2.0 and run open-loop action accuracy evaluation.
Run this script inside the vLLM-Omni Docker container.

Options:
  --checkpoint PATH   Checkpoint directory (default: $CHECKPOINT)
  --dataset PATH      Portable open-loop NPZ bundle (default: $DATASET)
  --output-root PATH  Result root (default: $OUTPUT_ROOT)
  --model-root PATH   Prepared model prefix (default: $MODEL_ROOT)
  --mode MODE         eager, compiled, or both (default: $MODE)
  --dtype DTYPE       Inference dtype (default: $DTYPE)
  --seed INTEGER      Per-sample noise base seed (default: $SEED)
  --max-samples N     Evaluate only the first N selected samples
  --episodes CSV      Evaluate comma-separated episode IDs, e.g. 0,1,2
  --no-plots          Do not write episode PNGs
  -h, --help          Show this help
EOF
}

while (($#)); do
    case "$1" in
        --checkpoint) CHECKPOINT="$2"; shift 2 ;;
        --dataset) DATASET="$2"; shift 2 ;;
        --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
        --model-root) MODEL_ROOT="$2"; shift 2 ;;
        --mode) MODE="$2"; shift 2 ;;
        --dtype) DTYPE="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --max-samples) MAX_SAMPLES="$2"; shift 2 ;;
        --episodes) EPISODES="$2"; shift 2 ;;
        --no-plots) PLOTS="0"; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

case "$MODE" in
    eager|compiled|both) ;;
    *) echo "--mode must be eager, compiled, or both" >&2; exit 2 ;;
esac

[[ -d "$CHECKPOINT" ]] || { echo "checkpoint not found: $CHECKPOINT" >&2; exit 1; }
[[ -f "$DATASET" ]] || { echo "dataset bundle not found: $DATASET" >&2; exit 1; }

cd "$REPO"
export PYTHONPATH=.
mkdir -p "$OUTPUT_ROOT"

SELECTION_ARGS=()
if [[ -n "$MAX_SAMPLES" ]]; then
    SELECTION_ARGS+=(--max-samples "$MAX_SAMPLES")
fi
if [[ -n "$EPISODES" ]]; then
    IFS=',' read -r -a EPISODE_IDS <<<"$EPISODES"
    SELECTION_ARGS+=(--episodes "${EPISODE_IDS[@]}")
fi
PLOT_ARGS=()
if [[ "$PLOTS" == "1" ]]; then
    PLOT_ARGS+=(--plots)
fi

python examples/offline_inference/lingbot_vla_v2/open_loop_eval.py \
    --dataset "$DATASET" \
    --output-dir "$OUTPUT_ROOT/dry-run" \
    "${SELECTION_ARGS[@]}" \
    --dry-run

run_mode() {
    local mode=$1
    local model_dir="${MODEL_ROOT}-${mode}"
    local result_dir="${OUTPUT_ROOT}/${mode}"
    local prepare_args=(--checkpoint "$CHECKPOINT" --output "$model_dir")
    if [[ "$mode" == "compiled" ]]; then
        prepare_args+=(--compile-denoise-step)
    fi

    rm -rf "$model_dir" "$result_dir"
    python examples/offline_inference/lingbot_vla_v2/prepare_lingbot_vla_v2.py \
        "${prepare_args[@]}"
    python examples/offline_inference/lingbot_vla_v2/open_loop_eval.py \
        --model "$model_dir" \
        --dataset "$DATASET" \
        --output-dir "$result_dir" \
        --dtype "$DTYPE" \
        --seed "$SEED" \
        "${SELECTION_ARGS[@]}" \
        "${PLOT_ARGS[@]}"
}

if [[ "$MODE" == "eager" || "$MODE" == "both" ]]; then
    run_mode eager
fi
if [[ "$MODE" == "compiled" || "$MODE" == "both" ]]; then
    run_mode compiled
fi
if [[ "$MODE" == "both" ]]; then
    python examples/offline_inference/lingbot_vla_v2/compare_open_loop.py \
        --baseline "$OUTPUT_ROOT/eager" \
        --candidate "$OUTPUT_ROOT/compiled" \
        --output "$OUTPUT_ROOT/eager_vs_compiled.json"
fi

echo "open-loop results: $OUTPUT_ROOT"