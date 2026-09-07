from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).parents[4] / "spikes/lingbot_vla_v2/compare_openvino.py"
SPEC = importlib.util.spec_from_file_location("compare_openvino", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
compare_openvino = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(compare_openvino)


OPENVINO_OUTPUT = """
[vit] min=15 ms max=17 ms avg=16 ms
[text] min=40 ms max=45 ms avg=42 ms | kv (36, 1, 286, 8, 128) float16
[loop] 10 denoise steps (1 IR call): min=180 ms max=195 ms avg=188 ms
[total] min=240 ms max=250 ms avg=246 ms
"""

VLLM_OUTPUT = """
model.embed_prefix             18.0     2.6%
model.prefix_fill              44.0     6.3%
model.denoise                  216.0    30.9%
total (synced)                 312.0   100.0%
"""


def test_parse_openvino_stage_output():
    stages = compare_openvino.parse_stages(OPENVINO_OUTPUT, compare_openvino.OV_STAGE_PATTERNS, "OpenVINO")

    assert stages == {"vit": 16.0, "text": 42.0, "denoise": 188.0, "total": 246.0}


def test_parse_vllm_stage_output():
    stages = compare_openvino.parse_stages(VLLM_OUTPUT, compare_openvino.VLLM_STAGE_PATTERNS, "vLLM")

    assert stages == {"vit": 18.0, "text": 44.0, "denoise": 216.0, "total": 312.0}


def test_parse_stages_rejects_missing_stage():
    with pytest.raises(RuntimeError, match="stage 'total'"):
        compare_openvino.parse_stages(
            "\n".join(
                (
                    "[vit] min=15 ms max=17 ms avg=16 ms",
                    "[text] min=40 ms max=45 ms avg=42 ms",
                    "[loop] 10 denoise steps (1 IR call): min=180 ms max=195 ms avg=188 ms",
                )
            ),
            compare_openvino.OV_STAGE_PATTERNS,
            "OpenVINO",
        )


def test_build_report_records_boundaries_and_ratios():
    args = type(
        "Args",
        (),
        {
            "warmup": 5,
            "repeat": 20,
            "openvino_device": "GPU.1",
            "openvino_suffix": "int8",
            "vllm_device": "xpu",
            "vllm_dtype": "float16",
            "compile_denoise_step": True,
        },
    )()
    report = compare_openvino.build_report(
        args,
        {"vit": 16.0, "text": 42.0, "denoise": 188.0, "total": 246.0},
        {"vit": 18.0, "text": 44.0, "denoise": 216.0, "total": 312.0},
    )

    assert report["settings"]["vllm_compiled"] is True
    assert report["milliseconds"]["openvino"]["total"] == 246.0
    assert report["vllm_over_openvino_ratio"]["denoise"] == pytest.approx(216 / 188)
    assert "one action IR call" in report["timing_boundaries"]["openvino"]
    assert "WebSocket" in report["timing_boundaries"]["excluded_from_both"]


def test_recorded_openvino_reference_is_stable():
    assert compare_openvino.OPENVINO_REFERENCE_MS == {
        "vit": 16.0,
        "text": 42.0,
        "denoise": 188.0,
        "total": 246.0,
    }
