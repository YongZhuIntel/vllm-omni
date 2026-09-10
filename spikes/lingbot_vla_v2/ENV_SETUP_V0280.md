# Environment setup for `branch_lingbot-vla-v2_v0280_test` (vLLM 0.28.0 XPU)

Record of installing vLLM-Omni's dependencies into the vLLM XPU container on
2026-09-10, while porting the LingBot-VLA 2.0 work onto vLLM `v0.28.0`.

Raw logs are in [`install_logs_v0280/`](install_logs_v0280/).

## Starting point

The container ships a complete vLLM XPU runtime but **none** of vLLM-Omni's own
dependencies (`aenum`, `diffusers`, `omegaconf`, ... were all absent), so no test
could import `vllm_omni`.

| | version |
|---|---|
| vLLM | `0.28.0+xpu` |
| torch | `2.13.0+xpu` |
| transformers | `5.15.0` |
| accelerate | `1.13.0` |
| Python | 3.12 (`/opt/venv`) |

`pip` is already configured with `--extra-index-url https://download.pytorch.org/whl/xpu`,
which is what keeps torch resolving to the XPU build.

## The risk, and how it was checked

`requirements/common.txt` pins `transformers >=5.10.1, <5.15` and `accelerate ==1.12.0`,
both of which the container *violates* — so installing was guaranteed to downgrade them.
Worse, many of the transitive packages (`openai-whisper`, `x-transformers`, `torchsde`,
`cache-dit`) depend on torch, and a bad resolve would have replaced `2.13.0+xpu` with a
CUDA wheel and destroyed the runtime.

So every install was **dry-run first** and the plan inspected before executing:

```bash
pip install --dry-run -r requirements/xpu.txt   # -> install_logs_v0280/dryrun.txt
```

The plan came back as 30 new packages, 2 downgrades, 0 upgrades, and — critically —
**no torch / torchvision / torchaudio entry at all**. Safe to proceed.

## What was installed

```bash
pip install -r requirements/xpu.txt   # exit 0 -> install_logs_v0280/install.txt
pip install matplotlib                # exit 0 -> install_logs_v0280/install-matplotlib.txt
```

**Downgraded (2)** — both required by `requirements/common.txt`:

| package | from | to | note |
|---|---|---|---|
| transformers | 5.15.0 | 5.14.1 | upstream pins `<5.15`; 5.15 has a known construction regression |
| accelerate | 1.13.0 | 1.12.0 | upstream pins `==1.12.0` |

vLLM 0.28.0 itself only requires `transformers>=5.5.3`, so the downgrade does not
conflict with it — verified with `importlib.metadata.requires("vllm")`.

**Added (30)** by `requirements/xpu.txt`: `aenum`, `antlr4-python3-runtime`, `av`,
`better-profanity`, `cache-dit`, `cosmos-guardrail`, `diffusers`, `einx`, `flatbuffers`,
`frozendict`, `gguf`, `imageio-ffmpeg`, `importlib_metadata`, `janus`, `kernels`,
`kernels-data`, `omegaconf`, `onnxruntime`, `openai-whisper`, `opencv-python`, `peft`,
`prettytable`, `retinaface-py`, `tomlkit`, `torch-einops-utils`, `torchsde`, `trampoline`,
`wcwidth`, `x-transformers`, `zipp`.

**Added (6)** by matplotlib: `contourpy`, `cycler`, `fonttools`, `kiwisolver`,
`matplotlib`, `pyparsing`. matplotlib is *not* a declared dependency — it backs the opt-in
`--plots` flag of `examples/offline_inference/lingbot_vla_v2/open_loop_eval.py` (and two
other example/benchmark scripts). It is installed here only so the plotting test runs
instead of skipping.

## Post-install verification

```
torch          2.13.0+xpu     <- unchanged
vllm           0.28.0
torch.xpu.is_available()  ->  True
```

`pip check` reports 4 warnings (`nixl`, `grpcio-tools`, `vcs-versioning`, `httpx2`). None
of those packages were touched by this install — compare `pip-freeze-BEFORE.txt` against
`pip-freeze-FINAL.txt`; they are pre-existing in the base image.

## Gotcha: running pytest

`pyproject.toml` sets `--dist=loadgroup` in addopts, but `pytest-xdist` is not installed in
this image, so plain `pytest` aborts with `unrecognized arguments: --dist=loadgroup`.
Clear addopts:

```bash
python -m pytest tests/diffusion/models/lingbot_vla_v2/ -o addopts="" -q
```

Result after setup: **48 passed, 1 skipped**. The skip is
`test_config.py:143`, which needs a real checkpoint via `LINGBOT_CKPT`.
