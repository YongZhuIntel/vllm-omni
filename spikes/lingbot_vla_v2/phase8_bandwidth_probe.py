# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""B60 achievable read bandwidth, and the MoE weight-traffic floor for one request."""
import time, torch
d = torch.device("xpu")
# 2.72 GB fp16 = exactly the per-step MoE weight footprint (36*3*32*768*512).
n = 36*3*32*768*512
x = torch.empty(n, device=d, dtype=torch.float16).normal_()
gb = x.numel()*2/1e9
for _ in range(3):
    x.sum()
torch.xpu.synchronize()
best = 1e9
for _ in range(10):
    torch.xpu.synchronize(); t = time.perf_counter()
    x.sum()
    torch.xpu.synchronize(); best = min(best, time.perf_counter()-t)
bw = gb/best
print(f"MoE weight footprint (fp16):        {gb:.2f} GB")
print(f"pure read (sum) best of 10:         {best*1e3:.1f} ms  -> {bw:.0f} GB/s achievable")
print()
print(f"per denoise step, all 36 MoE layers: {gb:.2f} GB -> {gb/bw*1e3:.1f} ms")
print(f"x10 steps, one request:              {gb*10:.1f} GB -> {gb*10/bw*1e3:.1f} ms  <-- DRAM floor for the loop")
print(f"measured loop (F1):                  201.4 ms")
print(f"OV fmha loop:                        188.0 ms")
print()
print("top-4 routing at 30.6/32 experts hit: weight traffic falls to "
      f"{gb*10*30.6/32/bw*1e3:.1f} ms (-{(1-30.6/32)*100:.0f}%)")
