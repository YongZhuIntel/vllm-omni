# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Is the dense MoE FLOP-bound or byte-bound? Roofline from measured peaks."""
import time, torch
d, dt = torch.device("xpu"), torch.float16
# achievable compute peak: large square GEMM
a = torch.empty(4096, 4096, device=d, dtype=dt).normal_()
for _ in range(5): a @ a
torch.xpu.synchronize()
best = 1e9
for _ in range(20):
    torch.xpu.synchronize(); t = time.perf_counter(); a @ a
    torch.xpu.synchronize(); best = min(best, time.perf_counter()-t)
peak_tflops = 2*4096**3/best/1e12
peak_bw = 449.0   # measured in phase8_bandwidth_probe.py

E,H,I,T = 32,768,512,51
bytes_per = E*3*H*I*2          # weights read per MoE layer-step
flops_per = 2*E*T*H*I*3        # dense arithmetic per MoE layer-step
INV = 360

print(f"B60 achievable: {peak_tflops:.1f} TFLOPS fp16 ({best*1e3:.2f} ms for 4096^3), {peak_bw:.0f} GB/s")
print(f"machine balance: {peak_tflops*1e12/(peak_bw*1e9):.0f} FLOP/byte\n")
print(f"dense MoE layer-step: {flops_per/1e9:.2f} GFLOP over {bytes_per/1e6:.1f} MB")
print(f"  arithmetic intensity: {flops_per/bytes_per:.0f} FLOP/byte")
print(f"  -> {'MEMORY-bound' if flops_per/bytes_per < peak_tflops*1e12/(peak_bw*1e9) else 'COMPUTE-bound'}\n")
t_mem = bytes_per/(peak_bw*1e9); t_cmp = flops_per/(peak_tflops*1e12)
print(f"  roofline time = max(mem {t_mem*1e3:.3f}, compute {t_cmp*1e3:.3f}) = {max(t_mem,t_cmp)*1e3:.3f} ms")
# Warm steady state from phase8_moe_gemm_probe.py. NOT 0.290: that is the value
# a cold first bench() call reports, and it inflated this whole chain by 14 ms
# until 2026-09-08.
MEASURED_MS = 0.251
print(f"  measured (M=51 bmm, warm)                                 = {MEASURED_MS:.3f} ms")
print(f"  -> at {max(t_mem,t_cmp)/(MEASURED_MS*1e-3)*100:.0f}% of roofline\n")
print(f"x{INV} invocations, MoE GEMMs only:")
print(f"  measured                  90.4 ms   <- phase8_moe_gemm_probe.py, warm")
print(f"  roofline (fp16 weights)  {max(t_mem,t_cmp)*INV*1e3:6.1f} ms   <- headroom {90.4-max(t_mem,t_cmp)*INV*1e3:.0f} ms, no FLOP change")
for bits,name in ((8,"int8"),(4,"int4")):
    tm = bytes_per*bits/16/(peak_bw*1e9)
    print(f"  roofline ({name} weights)  {max(tm,t_cmp)*INV*1e3:6.1f} ms")
