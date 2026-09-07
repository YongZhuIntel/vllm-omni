#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
set -euo pipefail

# One command that answers "is LingBot-VLA 2.0 fast enough right now?".
#
# It regenerates the prepared directory (the config is serialised into it, so a
# stale one silently pins old settings), starts the OpenPI server, sends warm
# requests over one WebSocket connection, and checks the median against the two
# targets. It refuses to measure on a busy host, because that is exactly what
# once inflated a 1.9 ms stage to 178 ms and put a whole optimisation step on the
# plan. Run it from the vLLM-Omni checkout inside the container.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd -- "$SCRIPT_DIR/../../.." && pwd)
CHECKPOINT="/llm/zhuyong/lingbovla/models/lingbot-vla-v2-6b"
OUTPUT="/tmp/lingbot-vla-v2-perf"
PORT="8000"
# fp16, matching the accuracy path. Phase 5 measured the latency difference at
# 2.4% -- inside this harness's run-to-run noise -- so this costs nothing here
# and buys 3.2x on the numerical-equivalence metric. See
# `spikes/lingbot_vla_v2/PHASE7_NUMERICS.md`.
DTYPE="float16"
REQUESTS="8"
STARTUP_TIMEOUT="420"
TARGET_HZ="1.0"          # M4 acceptance
BASELINE_S="0.74"        # Phase 0 bare-kernel reference, for context
FORCE="0"
RUN_OFFLINE="1"
RUN_ATTRIBUTION="0"
COMPILE_DENOISE_STEP="0"

usage() {
    cat <<EOF
Usage: $0 [options]

Measure LingBot-VLA 2.0 end-to-end latency and check it against the targets.

Options:
  --checkpoint PATH   Checkpoint path (default: $CHECKPOINT)
  --output PATH       Prepared model path, regenerated (default: $OUTPUT)
  --port PORT         Server port (default: $PORT)
  --dtype DTYPE       Inference dtype (default: $DTYPE)
  --requests N        Measured warm requests, after one discarded (default: $REQUESTS)
  --no-offline        Skip the cold-start offline request
  --attribution       Also print the per-stage breakdown (adds a model load)
    --compile-denoise-step
                                             Compile predict_velocity with Inductor (experimental)
  --force             Measure even if the host looks busy
  -h, --help          Show this help
EOF
}

while (($#)); do
    case "$1" in
        --checkpoint) CHECKPOINT="$2"; shift 2 ;;
        --output) OUTPUT="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --dtype) DTYPE="$2"; shift 2 ;;
        --requests) REQUESTS="$2"; shift 2 ;;
        --no-offline) RUN_OFFLINE="0"; shift ;;
        --attribution) RUN_ATTRIBUTION="1"; shift ;;
        --compile-denoise-step) COMPILE_DENOISE_STEP="1"; shift ;;
        --force) FORCE="1"; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

cd "$REPO"
export PYTHONPATH=.
LOG_DIR=$(mktemp -d /tmp/lingbot-perf.XXXXXX)
SERVER_LOG="$LOG_DIR/server.log"
SERVER_PID=""

cleanup() {
    if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "-- stopping server (pid $SERVER_PID)"
        kill -TERM "$SERVER_PID" 2>/dev/null || true
        for _ in $(seq 30); do
            kill -0 "$SERVER_PID" 2>/dev/null || break
            sleep 1
        done
        kill -KILL "$SERVER_PID" 2>/dev/null || true
    fi
    # Orphaned workers survive their parent and hold both host RAM and the whole
    # XPU allocation, which corrupts every later measurement on this machine.
    #
    # Matched narrowly on purpose: PPID 1 (that is the failure mode — the parent
    # is gone), a python process, this run's own model directory, and never this
    # script or its parent. A bare `pgrep -f "$OUTPUT"` also matches any shell
    # whose command line happens to contain the path, including the one that
    # invoked this script.
    local leaked
    leaked=$(ps -eo pid,ppid,args --no-headers 2>/dev/null | awk \
        -v out="$OUTPUT" -v me="$$" -v parent="$PPID" \
        '$2==1 && $1!=me && $1!=parent && /python/ && index($0, out) {print $1}' || true)
    if [[ -n "$leaked" ]]; then
        echo "-- reaping leaked workers: $(echo "$leaked" | tr '\n' ' ')"
        # shellcheck disable=SC2086
        kill -TERM $leaked 2>/dev/null || true
        sleep 5
        # shellcheck disable=SC2086
        kill -KILL $leaked 2>/dev/null || true
    fi
}
trap cleanup EXIT

# -- host state ---------------------------------------------------------------
echo "== host =="
LOAD1=$(awk '{print $1}' /proc/loadavg)
ORPHANS=$(ps -eo pid,ppid,etimes,rss,args --no-headers 2>/dev/null \
    | awk '$2==1 && /python/ && !/vscode|claude/ {printf "  pid=%s age=%ss rss=%sKiB\n", $1, $3, $4}' || true)
printf "load1=%s  free=%s\n" "$LOAD1" "$(free -h | awk '/^Mem:/ {print $7}')"
if [[ -n "$ORPHANS" ]]; then
    echo "orphaned python processes (these poison timings):"
    echo "$ORPHANS"
fi
BUSY=$(awk -v l="$LOAD1" 'BEGIN {print (l > 2.0) ? 1 : 0}')
if [[ "$BUSY" == "1" || -n "$ORPHANS" ]] && [[ "$FORCE" != "1" ]]; then
    echo
    echo "REFUSING to measure: the host is not idle." >&2
    echo "Absolute latencies measured here are not comparable to any recorded number." >&2
    echo "Kill the leftovers (or wait for the load to drop) and re-run, or pass --force." >&2
    exit 3
fi

# -- prepare ------------------------------------------------------------------
echo
echo "== prepare =="
rm -rf "$OUTPUT"
PREPARE_ARGS=(--checkpoint "$CHECKPOINT" --output "$OUTPUT")
if [[ "$COMPILE_DENOISE_STEP" == "1" ]]; then
    PREPARE_ARGS+=(--compile-denoise-step)
fi
python examples/offline_inference/lingbot_vla_v2/prepare_lingbot_vla_v2.py \
    "${PREPARE_ARGS[@]}" >"$LOG_DIR/prepare.log" 2>&1 \
    || { echo "prepare failed:" >&2; tail -20 "$LOG_DIR/prepare.log" >&2; exit 1; }
MOE=$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['moe_implementation'])" \
    "$OUTPUT/transformer/config.json")
STEPS=$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['num_steps'])" \
    "$OUTPUT/transformer/config.json")
COMPILED=$(python -c "import json,sys; print(json.load(open(sys.argv[1])).get('compile_denoise_step', False))" \
    "$OUTPUT/transformer/config.json")
echo "prepared $OUTPUT  (moe_implementation=$MOE, num_steps=$STEPS, dtype=$DTYPE, compiled=$COMPILED)"
[[ "$MOE" == "dense" ]] || echo "NOTE: 'gather' measured 3.7x slower than 'dense' on the B60."

# -- serve --------------------------------------------------------------------
echo
echo "== serve =="
python -m vllm_omni.entrypoints.cli.main serve "$OUTPUT" \
    --omni --host 127.0.0.1 --port "$PORT" \
    --dtype "$DTYPE" --enforce-eager --disable-log-stats >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "server pid $SERVER_PID, log $SERVER_LOG"
for i in $(seq "$STARTUP_TIMEOUT"); do
    if grep -q "Application startup complete" "$SERVER_LOG" 2>/dev/null; then
        echo "ready after ${i}s"
        break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "server exited during startup:" >&2
        tail -30 "$SERVER_LOG" >&2
        exit 1
    fi
    if [[ "$i" == "$STARTUP_TIMEOUT" ]]; then
        echo "server did not become ready within ${STARTUP_TIMEOUT}s" >&2
        tail -30 "$SERVER_LOG" >&2
        exit 1
    fi
    sleep 1
done

# -- measure ------------------------------------------------------------------
echo
echo "== warm requests over one connection =="
# One extra request, discarded: the first one through a fresh connection pays any
# lazy kernel setup, and a robot's steady state is what this number is for.
python examples/online_serving/lingbot_vla_v2/openpi_client.py \
    --host 127.0.0.1 --port "$PORT" --num-steps "$((REQUESTS + 1))" \
    >"$LOG_DIR/client.log" 2>&1 \
    || { echo "client failed:" >&2; tail -20 "$LOG_DIR/client.log" >&2; exit 1; }
grep -E "^step=" "$LOG_DIR/client.log" | sed 's/^/  /'
mapfile -t SAMPLES < <(grep -oE "elapsed=[0-9.]+" "$LOG_DIR/client.log" | cut -d= -f2 | tail -n +2)
if ((${#SAMPLES[@]} == 0)); then
    echo "no timings parsed from the client output" >&2
    exit 1
fi
read -r MEDIAN MIN MAX < <(printf '%s\n' "${SAMPLES[@]}" | sort -n | awk '
    {v[NR]=$1}
    END {printf "%.3f %.3f %.3f\n", (NR%2) ? v[(NR+1)/2] : (v[NR/2]+v[NR/2+1])/2, v[1], v[NR]}')

cleanup
SERVER_PID=""

# -- offline cold -------------------------------------------------------------
OFFLINE="skipped"
if [[ "$RUN_OFFLINE" == "1" ]]; then
    echo
    echo "== offline, one cold request =="
    if python examples/offline_inference/lingbot_vla_v2/lingbot_vla_v2.py \
        --model "$OUTPUT" --dtype "$DTYPE" >"$LOG_DIR/offline.log" 2>&1; then
        OFFLINE=$(grep -oE "elapsed=[0-9.]+" "$LOG_DIR/offline.log" | cut -d= -f2 | tail -1)
        echo "elapsed=${OFFLINE}s (includes first-request setup; not comparable to warm)"
    else
        echo "offline run failed:" >&2
        tail -20 "$LOG_DIR/offline.log" >&2
        OFFLINE="failed"
    fi
fi

# -- attribution --------------------------------------------------------------
if [[ "$RUN_ATTRIBUTION" == "1" ]]; then
    echo
    echo "== per-stage attribution =="
    if [[ -f spikes/lingbot_vla_v2/phase5_latency.py ]]; then
        ATTRIBUTION_ARGS=(--model "$OUTPUT" --dtype "$DTYPE" --iters 5)
        if [[ "$COMPILE_DENOISE_STEP" == "1" ]]; then
            ATTRIBUTION_ARGS+=(
                --compile-denoise-step
                --compile-max-relative-error 0.05
            )
        fi
        python spikes/lingbot_vla_v2/phase5_latency.py \
            "${ATTRIBUTION_ARGS[@]}" 2>"$LOG_DIR/attribution.err" \
            | sed -n '/stage/,$p' | sed 's/^/  /' \
            || { echo "  attribution failed:"; tail -10 "$LOG_DIR/attribution.err"; }
    else
        echo "  spikes/lingbot_vla_v2/phase5_latency.py is not present; skipped"
    fi
fi

# -- verdict ------------------------------------------------------------------
HZ=$(awk -v m="$MEDIAN" 'BEGIN {printf "%.2f", 1/m}')
PASS=$(awk -v h="$HZ" -v t="$TARGET_HZ" 'BEGIN {print (h >= t) ? 1 : 0}')
echo
echo "======================================================================"
printf "warm WebSocket, %d requests   median %.3fs  (%.2f Hz)   min %.3fs  max %.3fs\n" \
    "${#SAMPLES[@]}" "$MEDIAN" "$HZ" "$MIN" "$MAX"
[[ "$OFFLINE" == "skipped" || "$OFFLINE" == "failed" ]] \
    || printf "offline cold request        %ss\n" "$OFFLINE"
printf "dtype                       %s\n" "$DTYPE"
printf "moe_implementation          %s\n" "$MOE"
printf "compile_denoise_step       %s\n" "$COMPILED"
printf "Phase 0 kernel reference    %ss/chunk (kernel only, no processor)\n" "$BASELINE_S"
echo "----------------------------------------------------------------------"
if [[ "$PASS" == "1" ]]; then
    echo "PASS  ${HZ} Hz >= ${TARGET_HZ} Hz (M4 acceptance)"
else
    echo "FAIL  ${HZ} Hz <  ${TARGET_HZ} Hz (M4 acceptance)"
fi
echo "======================================================================"
echo "logs: $LOG_DIR"
[[ "$PASS" == "1" ]] || exit 1
