#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd -- "$SCRIPT_DIR/../../.." && pwd)
HOST="127.0.0.1"
PORT="8000"
PROMPT="pick up the object"
SEED="0"
NUM_STEPS="1"

usage() {
    cat <<EOF
Usage: $0 [options]

Send synthetic RobotWin observations to a running LingBot OpenPI server.
Run this script from the vLLM-Omni checkout inside the Docker container.

Options:
  --host HOST         Server host (default: $HOST)
  --port PORT         Server port (default: $PORT)
  --prompt TEXT       Robot instruction (default: $PROMPT)
  --seed INTEGER      Random seed (default: $SEED)
  --num-steps COUNT   Observations to send (default: $NUM_STEPS)
  -h, --help          Show this help
EOF
}

while (($#)); do
    case "$1" in
        --host) HOST="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --prompt) PROMPT="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --num-steps) NUM_STEPS="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

cd "$REPO"
export PYTHONPATH=.
python examples/online_serving/lingbot_vla_v2/openpi_client.py \
    --host "$HOST" --port "$PORT" --prompt "$PROMPT" --seed "$SEED" --num-steps "$NUM_STEPS"