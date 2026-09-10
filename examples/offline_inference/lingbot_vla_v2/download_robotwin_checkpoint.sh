#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
set -euo pipefail

REPO_ID="robbyant/lingbot-vla-v2-6b-robotwin"
REVISION="0451855729ec904f970600e0aec8b84661423afe"
OUTPUT="/llm/zhuyong/lingbovla/models/lingbot-vla-v2-6b-robotwin"
HF_ENDPOINT_VALUE="${HF_ENDPOINT:-https://hf-mirror.com}"
MAX_WORKERS="4"
VENV="/llm/zhuyong/lingbovla/frameworks.robotics.embodied-intelligence.lingbot-vla-v2/.venv-intel-dev"

usage() {
    cat <<EOF
Usage: $0 [options]

Download the official RobotWin post-trained LingBot-VLA 2.0 checkpoint.
Existing partial files are reused, so rerunning this command resumes downloads.

Options:
  --output PATH        Download directory (default: $OUTPUT)
  --endpoint URL       Hugging Face endpoint (default: $HF_ENDPOINT_VALUE)
  --revision REV       Model revision (default: $REVISION)
  --max-workers N      Parallel downloads (default: $MAX_WORKERS)
  --venv PATH          Python virtual environment (default: $VENV)
  -h, --help           Show this help
EOF
}

while (($#)); do
    case "$1" in
        --output) OUTPUT="$2"; shift 2 ;;
        --endpoint) HF_ENDPOINT_VALUE="$2"; shift 2 ;;
        --revision) REVISION="$2"; shift 2 ;;
        --max-workers) MAX_WORKERS="$2"; shift 2 ;;
        --venv) VENV="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ -f "$VENV/bin/activate" ]]; then
    # shellcheck disable=SC1090
    source "$VENV/bin/activate"
fi

export HF_ENDPOINT="$HF_ENDPOINT_VALUE"
export LINGBOT_ROBOTWIN_REPO_ID="$REPO_ID"
export LINGBOT_ROBOTWIN_REVISION="$REVISION"
export LINGBOT_ROBOTWIN_OUTPUT="$OUTPUT"
export LINGBOT_ROBOTWIN_MAX_WORKERS="$MAX_WORKERS"

python - <<'PY'
import os
from pathlib import Path

from huggingface_hub import snapshot_download

output = Path(os.environ["LINGBOT_ROBOTWIN_OUTPUT"]).expanduser().resolve()
path = snapshot_download(
    repo_id=os.environ["LINGBOT_ROBOTWIN_REPO_ID"],
    revision=os.environ["LINGBOT_ROBOTWIN_REVISION"],
    local_dir=output,
    allow_patterns=[
        "checkpoints/global_step_50000/hf_ckpt/**",
        "lingbotvla_cli.yaml",
        "README.md",
    ],
    max_workers=int(os.environ["LINGBOT_ROBOTWIN_MAX_WORKERS"]),
)
print(f"downloaded: {path}")
print(f"inference checkpoint: {output / 'checkpoints/global_step_50000/hf_ckpt'}")
PY

CHECKPOINT="$OUTPUT/checkpoints/global_step_50000/hf_ckpt"
test -f "$CHECKPOINT/model.safetensors.index.json"
for shard in "$CHECKPOINT"/model-*-of-00006.safetensors; do
    test -s "$shard"
done

echo "checkpoint verified: $CHECKPOINT"