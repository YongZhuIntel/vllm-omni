#!/usr/bin/env bash
set -euo pipefail

CONTAINER="test-image_zy_b8.3.2_lingbot_omni"
REPO="/llm/zhuyong/lingbovla/my/vllm-omni"
CHECKPOINT="/llm/zhuyong/lingbovla/models/lingbot-vla-v2-6b"
OUTPUT="/tmp/lingbot-vla-v2-prepared"
# fp16 over bf16: see `spikes/lingbot_vla_v2/PHASE7_NUMERICS.md`.
DTYPE="float16"
PROMPT="pick up the object"
SEED="0"

usage() {
    cat <<EOF
Usage: $0 [options]

Prepare LingBot-VLA 2.0 and run one XPU inference request in Docker.
All paths are interpreted inside the container.

Options:
  --container NAME    Docker container (default: $CONTAINER)
  --repo PATH         Repository path (default: $REPO)
  --checkpoint PATH   Checkpoint path (default: $CHECKPOINT)
  --output PATH       Prepared model path (default: $OUTPUT)
  --dtype DTYPE       Inference dtype (default: $DTYPE)
  --prompt TEXT       Robot instruction (default: $PROMPT)
  --seed INTEGER      Random seed (default: $SEED)
  -h, --help          Show this help
EOF
}

while (($#)); do
    case "$1" in
        --container) CONTAINER="$2"; shift 2 ;;
        --repo) REPO="$2"; shift 2 ;;
        --checkpoint) CHECKPOINT="$2"; shift 2 ;;
        --output) OUTPUT="$2"; shift 2 ;;
        --dtype) DTYPE="$2"; shift 2 ;;
        --prompt) PROMPT="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

docker inspect "$CONTAINER" >/dev/null
docker exec "$CONTAINER" sh -lc '
    set -eu
    repo=$1
    checkpoint=$2
    output=$3
    dtype=$4
    prompt=$5
    seed=$6

    cd "$repo"
    export PYTHONPATH=.
    python examples/offline_inference/lingbot_vla_v2/prepare_lingbot_vla_v2.py \
        --checkpoint "$checkpoint" \
        --output "$output"
    python examples/offline_inference/lingbot_vla_v2/lingbot_vla_v2.py \
        --model "$output" \
        --dtype "$dtype" \
        --prompt "$prompt" \
        --seed "$seed"
' sh "$REPO" "$CHECKPOINT" "$OUTPUT" "$DTYPE" "$PROMPT" "$SEED"