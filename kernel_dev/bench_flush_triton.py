# SPDX-License-Identifier: Apache-2.0
"""Bench: fused triton LeapQuant flush vs PR1 torch path (gather/flush/scatter).

Realistic decode-time flush: state pool of a GDN layer, all running
sequences crossing a window boundary at once (worst case).
"""

import time

import torch

from vllm_ascend.ops.triton.leapquant import leapquant_flush_triton
from vllm_ascend.ops.triton.mamba.state_index import gather_ssm_states, scatter_ssm_states_
from vllm_ascend.quantization.methods.mamba_state.leapquant.state import leap_window_flush

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


def main():
    torch.manual_seed(0)
    torch.npu.set_device(DEVICE)
    shapes = [
        # (name, slots, heads, due, V, K)
        ("qwen3.5-0.8B H=16 batch16", 64, 16, 16, 128, 128),
        ("qwen3.5-9B   H=32 batch16", 64, 32, 16, 128, 128),
        ("qwen3.5-9B   H=32 batch64", 128, 32, 64, 128, 128),
    ]
    for name, slots, heads, due_n, v_dim, k_dim in shapes:
        pool = torch.randn(slots, heads, v_dim, k_dim, device=DEVICE, dtype=torch.float32)
        slot_ids = torch.randperm(slots, device=DEVICE)[:due_n].to(torch.int64).contiguous()
        present = torch.ones(due_n, dtype=torch.bool, device=DEVICE)

        def torch_path():
            gathered = gather_ssm_states(pool, slot_ids, present)
            flushed = leap_window_flush(gathered, compensator_tokens=4)
            scatter_ssm_states_(pool, slot_ids, flushed.to(torch.float32).contiguous())

        def triton_path():
            leapquant_flush_triton(pool, slot_ids, compensator_tokens=4)

        # keep inputs identical between paths: re-randomize is skipped;
        # the flush is (nearly) idempotent in timing, not in value, so
        # timing only — correctness is pinned by the unit test.
        t_torch = bench(torch_path)
        t_triton = bench(triton_path)
        print(f"{name:32s} torch {t_torch:8.3f} ms | triton {t_triton:8.3f} ms | speedup {t_torch / t_triton:5.2f}x", flush=True)


if __name__ == "__main__":
    main()
