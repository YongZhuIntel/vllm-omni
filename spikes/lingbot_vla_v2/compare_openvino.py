#!/usr/bin/env python3
"""Compare vLLM LingBot latency with an OpenVINO reference.

By default this module can run both implementations. With ``--vllm-only`` it
runs only the vLLM probe and compares it with the recorded OpenVINO timings;
this is the mode used by the vLLM-Omni shell wrapper. OpenVINO's
``action-mode loop`` executes all ten denoise steps in one IR call, while the
current vLLM probe calls the compiled ``predict_velocity`` graph once per step.
The report keeps that distinction explicit.

Run this from the vLLM-Omni checkout, normally inside the vLLM container::

    python spikes/lingbot_vla_v2/compare_openvino.py \
        --vllm-only --model /tmp/lingbot-vla-v2-perf \
        --compile-denoise-step \
        --warmup 5 --repeat 20 \
        --output /tmp/lingbot-openvino-comparison.json
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

OV_STAGE_PATTERNS = {
    "vit": re.compile(r"^\[vit\].*?avg=(?P<ms>[0-9.]+) ms$", re.MULTILINE),
    "text": re.compile(r"^\[text\].*?avg=(?P<ms>[0-9.]+) ms", re.MULTILINE),
    "denoise": re.compile(r"^\[loop\].*?avg=(?P<ms>[0-9.]+) ms$", re.MULTILINE),
    "total": re.compile(r"^\[total\].*?avg=(?P<ms>[0-9.]+) ms$", re.MULTILINE),
}
VLLM_STAGE_PATTERNS = {
    "vit": re.compile(r"^model\.embed_prefix\s+(?P<ms>[0-9.]+)\s+", re.MULTILINE),
    "text": re.compile(r"^model\.prefix_fill\s+(?P<ms>[0-9.]+)\s+", re.MULTILINE),
    "denoise": re.compile(r"^model\.denoise\s+(?P<ms>[0-9.]+)\s+", re.MULTILINE),
    "total": re.compile(r"^total \(synced\)\s+(?P<ms>[0-9.]+)\s+", re.MULTILINE),
}
OPENVINO_REFERENCE_MS = {"vit": 16.0, "text": 42.0, "denoise": 188.0, "total": 246.0}


def parse_stages(output: str, patterns: dict[str, re.Pattern[str]], label: str) -> dict[str, float]:
    values: dict[str, float] = {}
    for stage, pattern in patterns.items():
        match = pattern.search(output)
        if match is None:
            raise RuntimeError(f"could not parse {label} stage {stage!r} from output")
        values[stage] = float(match.group("ms"))
    return values


def run(command: list[str], *, cwd: Path | None = None) -> str:
    print(f"$ {' '.join(shlex.quote(part) for part in command)}", flush=True)
    result = subprocess.run(command, cwd=cwd, text=True, capture_output=True)
    combined = result.stdout + result.stderr
    print(combined, end="")
    if result.returncode != 0:
        raise RuntimeError(f"command failed with exit code {result.returncode}")
    return combined


def run_openvino(args: argparse.Namespace, repo: Path) -> tuple[dict[str, float], str]:
    demo = repo / "demo" / "infer_demo.py"
    venv_python = repo / ".venv-intel-dev" / "bin" / "python"
    if not demo.is_file():
        raise FileNotFoundError(f"OpenVINO demo not found: {demo}")
    if not venv_python.is_file():
        raise FileNotFoundError(f"OpenVINO virtualenv not found: {venv_python}")

    command = [
        str(venv_python),
        str(demo),
        "--ir",
        str(repo / args.openvino_ir),
        "--device",
        args.openvino_device,
        "--suffix",
        args.openvino_suffix,
        "--cache-dir",
        args.openvino_cache,
        "--warmup",
        str(args.warmup),
        "--repeat",
        str(args.repeat),
        "--action-mode",
        "loop",
    ]
    output = run(command, cwd=repo)
    return parse_stages(output, OV_STAGE_PATTERNS, "OpenVINO"), output


def run_vllm(args: argparse.Namespace, repo: Path) -> tuple[dict[str, float], str]:
    probe = repo / "spikes" / "lingbot_vla_v2" / "phase5_latency.py"
    command = [
        sys.executable,
        str(probe),
        "--model",
        args.model,
        "--device",
        args.vllm_device,
        "--dtype",
        args.vllm_dtype,
        "--warmup",
        str(args.warmup),
        "--iters",
        str(args.repeat),
    ]
    if args.compile_denoise_step:
        command.extend(
            [
                "--compile-denoise-step",
                "--compile-max-relative-error",
                str(args.compile_max_relative_error),
            ]
        )
    output = run(command, cwd=repo)
    return parse_stages(output, VLLM_STAGE_PATTERNS, "vLLM"), output


def build_report(args: argparse.Namespace, ov: dict[str, float], vllm: dict[str, float]) -> dict[str, Any]:
    ratios = {stage: vllm[stage] / ov[stage] for stage in ov}
    return {
        "settings": {
            "warmup": args.warmup,
            "repeat": args.repeat,
            "openvino_device": args.openvino_device,
            "openvino_suffix": args.openvino_suffix,
            "vllm_device": args.vllm_device,
            "vllm_dtype": args.vllm_dtype,
            "vllm_compiled": args.compile_denoise_step,
        },
        "timing_boundaries": {
            "openvino": "vit_prefix + text_prefix + 10-step denoise in one action IR call",
            "vllm": "embed_prefix + prefix_fill + 10 predict_velocity calls, with device syncs",
            "excluded_from_both": "WebSocket, MessagePack, vLLM engine scheduling, and server IPC",
        },
        "milliseconds": {"openvino": ov, "vllm": vllm},
        "vllm_over_openvino_ratio": ratios,
        "notes": [
            "The denoise and total ratios are directional because the graph boundaries differ.",
            "Use run_perf_check.sh separately for end-to-end WebSocket latency.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--openvino-repo", default=None, help="OpenVINO LingBot repository")
    parser.add_argument("--openvino-ir", default="converter")
    parser.add_argument("--openvino-device", default="GPU.1")
    parser.add_argument("--openvino-suffix", default="int8")
    parser.add_argument("--openvino-cache", default="~/.ov_cache")
    parser.add_argument("--model", required=True, help="Prepared vLLM model directory")
    parser.add_argument("--vllm-device", default="xpu")
    parser.add_argument("--vllm-dtype", default="float16")
    parser.add_argument("--compile-denoise-step", action="store_true")
    parser.add_argument(
        "--vllm-only",
        action="store_true",
        help="Run only vLLM and compare with the recorded OpenVINO reference timings",
    )
    parser.add_argument("--compile-max-relative-error", type=float, default=0.05)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--output", type=Path, default=None, help="Write machine-readable JSON here")
    args = parser.parse_args()
    args.openvino_cache = str(Path(args.openvino_cache).expanduser())

    repo = Path(__file__).resolve().parents[2]
    if not args.vllm_only and args.openvino_repo is None:
        parser.error("--openvino-repo is required unless --vllm-only is used")
    vllm_stages, _ = run_vllm(args, repo)
    if args.vllm_only:
        ov_stages = OPENVINO_REFERENCE_MS
    else:
        ov_stages, _ = run_openvino(args, Path(args.openvino_repo).resolve())
    report = build_report(args, ov_stages, vllm_stages)

    print("\n== comparable stage summary (ms) ==")
    print(f"{'stage':<10} {'OpenVINO':>12} {'vLLM':>12} {'vLLM/OV':>12}")
    for stage in ("vit", "text", "denoise", "total"):
        ratio = report["vllm_over_openvino_ratio"][stage]
        print(f"{stage:<10} {ov_stages[stage]:12.1f} {vllm_stages[stage]:12.1f} {ratio:12.2f}x")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"\nJSON report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
