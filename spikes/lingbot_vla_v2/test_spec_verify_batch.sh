#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Price the K-timestep verify, batched vs sequential, on the dGPU.
#
# Unlike test_spec_oneccl.sh this needs **no iGPU and no oneCCL**: the draft's
# content cannot change verify cost, so the probe stubs it out. Hence no
# CCL_PLUGIN and no --spec-worker-cpu here.
#
# ZE_AFFINITY_MASK=0 pins the dGPU. Without it the sweep may land on the iGPU and
# every number is off by the 13x device factor from §K.
set -euo pipefail

MODEL=${MODEL:-/tmp/lingbot-vla-v2-perf}
KS=${KS:-1,2,4,8,10}
OUT=${OUT:-/tmp/phase10_verify_batch.json}

cd /llm/zhuyong/lingbovla/my/vllm-omni

ZE_AFFINITY_MASK=0 PYTHONPATH=. python spikes/lingbot_vla_v2/phase10_verify_batch_sweep.py \
    --model "$MODEL" \
    --k "$KS" \
    --compile-denoise-step \
    --iters 20 \
    --repeats 3 \
    --json "$OUT"
