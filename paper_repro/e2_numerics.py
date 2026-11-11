#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Paper-repro E2: numerics laws on A3 (fig3 / fig5 / fig6 / table3-numerics).

Reproduces the paper's numerical experiments with the production LeapQuant
quantizer (vllm_ascend state.py), on synthetic GDN decode trajectories with
realistic per-head decay/beta statistics:

  E2a (fig3): state MSE vs decode length over 64K steps — per-token bf16 /
      per-token naive int8 diverge, window-16 variants stay flat. Paper:
      bf16-w16 100x lower, int8-w16 39x lower at 64K.
  E2b (fig5): top-k singular value energy of boundary states (paper: top-12
      covers >80% median on Qwen3.5-9B).
  E2c (fig6): outlier ratio max/median through the quantizer pipeline —
      raw -> after compensators -> after smoothing (paper: 176x -> 47x -> 5.3x).
  E2d (table3): window p sweep and compensator r sweep terminal error.

Honest-scope note: the paper rolls real model states on PG-19; we roll
statistically-matched synthetic trajectories (same per-head [V,K]=[128,128]
geometry as Qwen3.5) — the laws being verified are quantizer properties.

Usage: ASCEND_RT_VISIBLE_DEVICES=14 PYTHONPATH=/workspace/y00924007/vllm-ascend \
       python3 e2_numerics.py [--steps 65536]
"""

import argparse
import json
import math
import time

import torch

from vllm_ascend.quantization.methods.mamba_state.leapquant.state import (
    fit_compensators,
    leap_window_flush,
    quantize_residual,
    reconstruct_state,
    start_basis,
)

H, V, K = 16, 128, 128
DEV = None


def draw_inputs(t):
    """Deterministic per-step draws, identical across variants (reseed)."""
    torch.manual_seed(7000 + t)
    k = torch.nn.functional.normalize(torch.randn(H, K, device=DEV), dim=-1)
    v = torch.randn(H, V, device=DEV)
    beta = (torch.rand(H, 1, device=DEV) * 0.5 + 0.5)
    return k, v, beta


def gate():
    torch.manual_seed(42)
    return -(0.003 + 0.05 * torch.rand(H, 1, 1, device=DEV))  # decay 0.95..0.997


def gdn_step(S, k, v, beta, g):
    sk = torch.einsum("hvk,hk->hv", S, k)
    return S * torch.exp(g) + torch.einsum("hv,hk->hvk", (v - sk) * beta, k)


def quant_int8_naive(S):
    """Per-token naive symmetric int8 (per-head-tensor scale) — paper's
    'per-token INT8' baseline."""
    scale = S.abs().amax(dim=(1, 2), keepdim=True).clamp_min(1e-30) / 127.0
    codes = torch.round(S / scale).clamp(-127, 127)
    return codes * scale


def quant_smooth_int8_r0(S):
    """Windowed quantizer with compensators disabled (r=0): smoothing +
    int8 only — the '+window' ablation leg."""
    s_k = S.abs().mean(-2).sqrt().clamp_min(1e-30)
    s_v = (S.abs() / s_k[..., None, :]).amax(-1).clamp_min(1e-30)
    codes = torch.round(S / (s_k[..., None, :] * s_v[..., :, None]) * 127.0)
    codes = codes.clamp(-127, 127)
    return codes / 127.0 * (s_k[..., None, :] * s_v[..., :, None])


def run_variant(name, steps, window, ckpt_every, quant_fn, g, ref_ckpts):
    """Roll `steps` decode steps; quantize per window (or per token when
    window=1); record MSE vs the fp32 reference checkpoints."""
    S = torch.zeros(H, V, K, device=DEV)
    mses, t0 = [], time.time()
    for t in range(steps):
        k, v, beta = draw_inputs(t)
        S = gdn_step(S, k, v, beta, g)
        if (t + 1) % window == 0:
            S = quant_fn(S)
        if (t + 1) % ckpt_every == 0:
            ref = ref_ckpts[(t + 1) // ckpt_every - 1]
            mses.append(torch.mean((S - ref) ** 2).item())
    dt = time.time() - t0
    print(f"  {name:24s} {steps} steps in {dt:.0f}s  final MSE {mses[-1]:.3e}", flush=True)
    return mses


def main():
    global DEV
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=65536)
    ap.add_argument("--out", default="e2_numerics.json")
    args = ap.parse_args()

    torch.npu.set_device(0)
    DEV = torch.device("npu:0")
    steps, ckpt = args.steps, 2048
    g = gate()
    res = {"H": H, "V": V, "K": K, "steps": steps, "ckpt_every": ckpt}

    # ---- fp32 reference trajectory + checkpoint cache -------------------
    print("[E2a] rolling fp32 reference ...", flush=True)
    S = torch.zeros(H, V, K, device=DEV)
    ref_ckpts, boundary_states = [], []
    for t in range(steps):
        k, v, beta = draw_inputs(t)
        S = gdn_step(S, k, v, beta, g)
        if (t + 1) % ckpt == 0:
            ref_ckpts.append(S.detach().clone())
        if (t + 1) % 8192 == 0:  # boundary states for fig5/fig6
            boundary_states.append(S.detach().clone())

    # ---- E2a: MSE vs length (fig3) --------------------------------------
    print("[E2a] variants:", flush=True)
    variants = {
        "pt_bf16": (1, lambda x: x.to(torch.bfloat16).float()),
        "pt_int8": (1, quant_int8_naive),
        "w16_bf16": (16, lambda x: x.to(torch.bfloat16).float()),
        "w16_int8_lq": (16, lambda x: leap_window_flush(x, compensator_tokens=4)),
    }
    series = {}
    for name, (w, fn) in variants.items():
        series[name] = run_variant(name, steps, w, ckpt, fn, g, ref_ckpts)
    res["mse_series"] = series
    res["mse_steps"] = [(i + 1) * ckpt for i in range(len(ref_ckpts))]
    f_bf = series["pt_bf16"][-1] / series["w16_bf16"][-1]
    f_i8 = series["pt_int8"][-1] / series["w16_int8_lq"][-1]
    res["improvement_bf16_w16"] = f_bf
    res["improvement_int8_w16"] = f_i8
    print(f"[E2a] 64K improvement: bf16 w16 {f_bf:.0f}x (paper ~100x), "
          f"int8 w16 {f_i8:.0f}x (paper ~39x)", flush=True)

    # ---- E2b: top-k singular value energy (fig5) -------------------------
    print("[E2b] singular value energy ...", flush=True)
    ks = [1, 2, 4, 8, 12, 16]
    fracs = {k: [] for k in ks}
    for Sb in boundary_states:
        for h in range(H):
            s = torch.linalg.svdvals(Sb[h].cpu())
            e = (s ** 2).cumsum(0) / (s ** 2).sum()
            for k in ks:
                fracs[k].append(e[k - 1].item())
    med = {k: sorted(v)[len(v) // 2] for k, v in fracs.items()}
    k80 = next((k for k in ks if med[k] >= 0.8), None)
    res["sv_energy_median"] = med
    res["sv_k_reaching_80pct"] = k80
    print(f"[E2b] median top-k energy: {med}  -> >=80% at k={k80} (paper: 12)", flush=True)

    # ---- E2c: outlier ratio through the pipeline (fig6) ------------------
    print("[E2c] outlier ratios ...", flush=True)
    q0 = start_basis(V, 4, device=DEV)
    ratios = {"raw": [], "after_comp": [], "after_smooth": []}
    for Sb in boundary_states:
        blocks = Sb.reshape(-1, V, K)
        u, q = fit_compensators(blocks, q0, 4)
        E = blocks - torch.einsum("nrv,nrk->nvk", q.float(), u.float())
        codes, s_k, s_v = quantize_residual(blocks, u, q)
        M = E / (s_k[..., None, :] * s_v[..., :, None])
        for key, T_ in (("raw", blocks), ("after_comp", E), ("after_smooth", M)):
            r = (T_.abs().amax(dim=(-2, -1)) / T_.abs().median(dim=-2).values.median(dim=-1).values)
            ratios[key].extend(r.tolist())
    med_r = {k2: sorted(v)[len(v) // 2] for k2, v in ratios.items()}
    res["outlier_ratio_median"] = med_r
    print(f"[E2c] outliers max/median: {med_r} (paper 176x -> 47x -> 5.3x)", flush=True)

    # ---- E2d: window p and compensator r sweeps (table3 numerics) --------
    print("[E2d] sweeps ...", flush=True)
    sw_steps = min(16384, steps)
    S = torch.zeros(H, V, K, device=DEV)
    for t in range(sw_steps):
        k, v, beta = draw_inputs(t)
        S = gdn_step(S, k, v, beta, g)
    ref_terminal = S
    p_res, r_res = {}, {}
    for p in (1, 2, 4, 8, 16, 32, 64, 128):
        mses = run_variant(f"p={p}", sw_steps, p, sw_steps,
                           lambda x: leap_window_flush(x, compensator_tokens=4), g,
                           [ref_terminal])
        p_res[p] = mses[0]
    for r in (0, 1, 2, 4, 8, 16):
        fn = (quant_smooth_int8_r0 if r == 0
              else (lambda x, r=r: leap_window_flush(x, compensator_tokens=r)))
        mses = run_variant(f"r={r}", sw_steps, 16, sw_steps, fn, g, [ref_terminal])
        r_res[r] = mses[0]
    res["p_sweep_mse"] = p_res
    res["r_sweep_mse"] = r_res
    print(f"[E2d] p sweep: { {k: f'{v:.2e}' for k, v in p_res.items()} }", flush=True)
    print(f"[E2d] r sweep: { {k: f'{v:.2e}' for k, v in r_res.items()} }", flush=True)

    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    print("WROTE", args.out, flush=True)

    # ---- plot (fig3 analog, English labels: no CJK font in env) ---------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        xs = res["mse_steps"]
        fig, ax = plt.subplots(figsize=(7.0, 4.4))
        style = {"pt_bf16": ("#9498a0", ":"), "pt_int8": ("#ff725c", "-"),
                 "w16_bf16": ("#4269d0", "--"), "w16_int8_lq": ("#6cc5b0", "-")}
        for name, (c, ls) in style.items():
            ax.plot(xs, series[name], color=c, ls=ls, lw=2, label=name)
        ax.set_yscale("log")
        ax.set_xlabel("decode steps")
        ax.set_ylabel("state MSE vs fp32")
        ax.set_title(f"state error accumulation ({steps} steps, window=16)", loc="left", fontsize=10)
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)
        plt.savefig("e2_fig3_mse.png", dpi=150, bbox_inches="tight")
        print("WROTE e2_fig3_mse.png", flush=True)
    except Exception as e:  # noqa: BLE001
        print("plot skipped:", e, flush=True)


if __name__ == "__main__":
    main()
