#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Paper-repro E4a: kernel-level flush + decode-step throughput on A3.

The paper's fig7 (kernel throughput, batch 512, 2.4-4.3x on GPUs) measures
the int8-payload readout kernel. Our M3.4b ring-readout is not built yet,
so today's A3 numbers cover what IS shipped:

  * flush (window-boundary checkpoint): fused triton vs PR1 torch path,
    roundtrip + payload modes, batch 16..512 — the measured win today;
  * decode step: triton recurrent_gdn_leapquant vs the fla_npu AscendC
    recurrent op (multi-request works for GDN) at batch 64/256/512 —
    currently a parity replacement (both read fp32 state), the bandwidth
    win lands with M3.4b;
  * triton KDA decode step (per-K decay) as an info point.

Run on die 15.
"""

import json
import time

import torch

DEVICE = torch.device("npu:0")


def bench(fn, iters=50, warmup=5):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3  # ms


def flush_bench(res):
    from vllm_ascend.ops.triton.leapquant import leapquant_flush_triton
    from vllm_ascend.ops.triton.mamba.state_index import (
        gather_ssm_states, scatter_ssm_states_)
    from vllm_ascend.quantization.methods.mamba_state.leapquant.payload import (
        LeapQuantPayloadPool)
    from vllm_ascend.quantization.methods.mamba_state.leapquant.state import (
        leap_window_flush)

    torch.manual_seed(0)
    shapes = [
        ("0.8B H16 b16", 64, 16, 16),
        ("9B   H32 b16", 64, 32, 16),
        ("9B   H32 b64", 128, 32, 64),
        ("9B   H32 b256", 512, 32, 256),
        ("9B   H32 b512", 1024, 32, 512),
        ("kimi H32 b64", 128, 32, 64),
    ]
    res["flush"] = {}
    for name, slots, heads, due in shapes:
        pool = torch.randn(slots, heads, 128, 128, device=DEVICE, dtype=torch.float32)
        slot_ids = torch.randperm(slots, device=DEVICE)[:due].to(torch.int64).contiguous()
        present = torch.ones(due, dtype=torch.bool, device=DEVICE)

        def torch_path():
            gathered = gather_ssm_states(pool, slot_ids, present)
            flushed = leap_window_flush(gathered, compensator_tokens=4)
            scatter_ssm_states_(pool, slot_ids, flushed.to(torch.float32).contiguous())

        def triton_path():
            leapquant_flush_triton(pool, slot_ids, compensator_tokens=4)

        t_torch = bench(torch_path, iters=20)
        t_tri = bench(triton_path, iters=50)

        lq_pool = LeapQuantPayloadPool.allocate(due, heads, 128, 128, 4, device=DEVICE)

        def payload_path():
            leapquant_flush_triton(pool, slot_ids, compensator_tokens=4, payload=lq_pool)

        t_pay = bench(payload_path, iters=50)
        res["flush"][name] = {"torch_ms": round(t_torch, 3), "triton_ms": round(t_tri, 3),
                              "payload_ms": round(t_pay, 3),
                              "speedup_roundtrip": round(t_torch / t_tri, 2),
                              "speedup_payload_vs_torch": round(t_torch / t_pay, 2)}
        print(f"[flush] {name:14s} torch {t_torch:8.2f} | triton {t_tri:7.2f} "
              f"({t_torch/t_tri:4.2f}x) | payload {t_pay:7.2f} ({t_torch/t_pay:4.2f}x) ms",
              flush=True)
        del pool, lq_pool
        torch.npu.empty_cache()


def decode_bench(res):
    from fla_npu.ops.ascendc import recurrent_gated_delta_rule
    from vllm_ascend.ops.triton.leapquant import (
        recurrent_gdn_leapquant, recurrent_kda_leapquant)

    torch.manual_seed(1)
    H, DK, DV = 32, 128, 128  # Qwen3.5-9B GDN geometry
    res["decode"] = {}
    for B in (64, 256, 512):
        slots = max(1024, 2 * B)
        q = torch.nn.functional.normalize(torch.randn(B, H, DK, device=DEVICE), dim=-1).to(torch.bfloat16)
        k = torch.nn.functional.normalize(torch.randn(B, H, DK, device=DEVICE), dim=-1).to(torch.bfloat16)
        v = torch.randn(B, H, DV, device=DEVICE).to(torch.bfloat16)
        g = -torch.rand(B, H, device=DEVICE) * 0.05
        beta = (torch.rand(B, H, device=DEVICE) * 0.5 + 0.5).to(torch.bfloat16)
        state = torch.zeros(slots, H, DV, DK, device=DEVICE, dtype=torch.float32)
        rows = torch.randperm(slots, device=DEVICE)[:B].to(torch.int32).contiguous()
        cu = torch.tensor([0] + [1] * B, device=DEVICE, dtype=torch.int32)

        def ascendc():
            recurrent_gated_delta_rule(
                q.contiguous(), k.contiguous(), v.contiguous(), state,
                g=g.contiguous(), beta=beta.contiguous(), scale=DK ** -0.5,
                actual_seq_lengths=cu, ssm_state_indices=rows)

        w = torch.zeros(slots, H, DV, DK, device=DEVICE, dtype=torch.float32)
        rows64 = rows.to(torch.int64)

        def triton():
            recurrent_gdn_leapquant(
                q[:, None].contiguous(), k[:, None].contiguous(), v[:, None].contiguous(),
                g[:, None].contiguous(), beta[:, None].contiguous(), scale=DK ** -0.5,
                w_in=w, w_out=w, w_rows=rows64)

        g_kda = (-torch.rand(B, 1, H, DK, device=DEVICE) * 0.05).contiguous()

        def triton_kda():
            recurrent_kda_leapquant(
                q[:, None].contiguous(), k[:, None].contiguous(), v[:, None].contiguous(),
                g_kda,
                beta[:, None].contiguous(), scale=DK ** -0.5,
                w_in=w, w_out=w, w_rows=rows64)

        t_asc = bench(ascendc, iters=30)
        t_tri = bench(triton, iters=30)
        t_kda = bench(triton_kda, iters=30)
        res["decode"][f"B{B}"] = {
            "ascendc_ms": round(t_asc, 3), "triton_gdn_ms": round(t_tri, 3),
            "triton_kda_ms": round(t_kda, 3),
            "ascendc_toks": round(B / t_asc * 1e3), "triton_gdn_toks": round(B / t_tri * 1e3),
            "parity_ratio": round(t_asc / t_tri, 3),
        }
        print(f"[decode] B={B:4d} AscendC {t_asc:7.2f} ms ({B/t_asc*1e3:7.0f} tok/s) | "
              f"triton GDN {t_tri:7.2f} ms ({B/t_tri*1e3:7.0f} tok/s, {t_asc/t_tri:.2f}x) | "
              f"triton KDA {t_kda:6.2f} ms", flush=True)
        del state, w
        torch.npu.empty_cache()


def main():
    torch.npu.set_device(DEVICE)
    res = {}
    flush_bench(res)
    decode_bench(res)
    res["note"] = ("fig7-style int8-payload decode readout = M3.4b (pending); "
                   "decode numbers above are the fp32-state parity replacement. "
                   "flush payload mode is the shipped bandwidth win (int8 write, "
                   "no fp32 rewrite).")
    with open("e4_kernel_bench.json", "w") as f:
        json.dump(res, f, indent=1)
    print("WROTE e4_kernel_bench.json", flush=True)


if __name__ == "__main__":
    main()
