#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Paper-repro E2-real: fig5/fig6/table3-r on REAL Qwen3.5-9B states.

The synthetic-trajectory suite (e2_numerics.py) reproduces the accumulation
law (int8-w16 35x vs paper 39x) but not the spectral structure: random-walk
states have a flat spectrum and no outliers. The paper's fig5 (top-12 SV
energy > 80%) and fig6 (outliers 176x -> 47x -> 5.3x) are properties of real
model states — so here we roll the actual 9B over ~8K tokens of real English
(HF forward, cache exposes per-layer recurrent states) and run the paper's
measurements on them:

  E2b' (fig5): top-k singular-value energy, median over layers x heads;
               paper reports top-12 >= 80% median on Qwen3.5-9B.
  E2c' (fig6): outlier ratio max/median through the quantizer pipeline
               (raw -> after r=4 compensators -> after smoothing), with the
               paper's exact probe (layer 13, head 16, 8K ctx) called out.
  E2d' (table3 r-axis): checkpoint rel-error vs compensator rank r on real
               states (the synthetic r-sweep showed almost no r value; on
               the real spectrum it should matter).

Model forward on NPU (bf16, ~18 GB); analysis on CPU fp32.
"""

import glob
import json
import time

import torch

MODEL = "/workspace/y00924007/mnt-weight/Qwen3.5-9B"
ENWIKI = sorted(glob.glob("/workspace/y00924007/mnt-weight/datasets/enwiki/**/*",
                          recursive=True))
CTX = 8192
V, K = 128, 128


def load_text_tokens(tok):
    """First CTX tokens of real English wiki text."""
    text = []
    with open("/workspace/y00924007/mnt-weight/datasets/enwiki/merged-article-chunked.jsonl",
              encoding="utf-8") as f:
        for line in f:
            import json as _json
            d = _json.loads(line)
            text.append(next(v for k, v in d.items() if isinstance(v, str) and len(v) > 200))
            if sum(len(t) for t in text) > 200_000:
                break
    ids = tok("\n\n".join(text), return_tensors="pt").input_ids
    if ids.shape[1] < CTX:
        raise RuntimeError(f"only {ids.shape[1]} tokens")
    return ids[:, :CTX]


def sv_energy(state_vk, ks):
    s = torch.linalg.svdvals(state_vk.double())
    e = (s ** 2).cumsum(0) / (s ** 2).sum()
    return [e[k - 1].item() for k in ks]


def main():
    import os

    if os.path.exists("e2_real_states_cache.pt"):
        states = torch.load("e2_real_states_cache.pt", map_location="cpu")
        print(f"[cache] {len(states)} layers from e2_real_states_cache.pt", flush=True)
        run_analysis(states, ctx=CTX)
        return

    torch.npu.set_device(0)
    dev = torch.device("npu:0")

    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    ids = load_text_tokens(tok)
    print(f"[load] text tokens: {ids.shape[1]}", flush=True)

    t0 = time.time()
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to(dev).eval()
    print(f"[load] model on npu in {time.time()-t0:.0f}s", flush=True)

    t0 = time.time()
    with torch.inference_mode():
        out = model(ids.to(dev), use_cache=True)
    print(f"[fwd] {ids.shape[1]} tokens in {time.time()-t0:.0f}s", flush=True)

    cache = out.past_key_values
    states = {}  # layer_idx -> [H, K, V] fp32 cpu (HF layout [B, H, K, V])
    for li, layer_cache in enumerate(cache.layers):
        rs = getattr(layer_cache, "recurrent_states", None)
        if rs and rs[0] is not None:
            states[li] = rs[0][0].float().cpu()  # [H, K, V]
    print(f"[grab] linear layers with recurrent state: {sorted(states)}", flush=True)
    torch.npu.empty_cache()
    torch.save(states, "e2_real_states_cache.pt")
    run_analysis(states, ctx=ids.shape[1])


def run_analysis(states, ctx):
    ks = [1, 2, 4, 8, 12, 16]
    res = {"ctx": ctx, "layers": sorted(states), "k_list": ks}

    # ---- E2b': fig5 on real states ---------------------------------------
    fracs = {k: [] for k in ks}
    for li, S in states.items():
        for h in range(S.shape[0]):
            for k, f in zip(ks, sv_energy(S[h].t(), ks)):  # -> [V, K]
                fracs[k].append(f)
    med = {k: sorted(v)[len(v) // 2] for k, v in fracs.items()}
    k80 = next((k for k in ks if med[k] >= 0.8), None)
    res["sv_energy_median"] = med
    res["sv_k_reaching_80pct"] = k80
    print(f"[E2b'] median top-k energy {med} -> >=80% at k={k80} (paper: 12)", flush=True)

    # ---- E2c': fig6 outlier pipeline on real states ----------------------
    import sys
    sys.path.insert(0, "/workspace/y00924007/vllm-ascend")
    from vllm_ascend.quantization.methods.mamba_state.leapquant.state import (
        fit_compensators, quantize_residual, start_basis)

    q0 = start_basis(V, 4, device=torch.device("cpu"))
    probe = {"layer": 13, "head": 16}
    rows_ratios = {"raw": [], "after_comp": [], "after_smooth": []}
    probe_ratios = {}
    for li, S in states.items():
        blocks = torch.stack([S[h].t() for h in range(S.shape[0])])  # [H, V, K]
        u, q = fit_compensators(blocks, q0, 4)
        E = blocks - torch.einsum("nrv,nrk->nvk", q.float(), u.float())
        codes, s_k, s_v = quantize_residual(blocks, u, q)
        M = E / (s_k[..., None, :] * s_v[..., :, None])
        for key, T_ in (("raw", blocks), ("after_comp", E), ("after_smooth", M)):
            r = (T_.abs().amax(dim=(-2, -1)) / T_.abs().median(dim=-2).values.median(dim=-1).values)
            rows_ratios[key].extend(r.tolist())
            if li == probe["layer"]:
                probe_ratios[key] = round(r[probe["head"]].item(), 1)
    med_r = {k2: round(sorted(v)[len(v) // 2], 1) for k2, v in rows_ratios.items()}
    res["outlier_ratio_median"] = med_r
    res["outlier_ratio_paper_probe_L13H16"] = probe_ratios
    print(f"[E2c'] outliers max/median median: {med_r} (paper 176 -> 47 -> 5.3)", flush=True)
    print(f"[E2c'] paper probe layer13 head16: {probe_ratios}", flush=True)

    # ---- E2d': r sweep on real states (checkpoint rel error) -------------
    from vllm_ascend.quantization.methods.mamba_state.leapquant.state import leap_window_flush

    def r0_quant(blocks):
        s_k = blocks.abs().mean(-2).sqrt().clamp_min(1e-30)
        s_v = (blocks.abs() / s_k[..., None, :]).amax(-1).clamp_min(1e-30)
        codes = torch.round(blocks / (s_k[..., None, :] * s_v[..., :, None]) * 127.0)
        codes = codes.clamp(-127, 127)
        return codes / 127.0 * (s_k[..., None, :] * s_v[..., :, None])

    r_res = {}
    for r in (0, 1, 2, 4, 8, 16):
        rels = []
        for li, S in states.items():
            blocks = torch.stack([S[h].t() for h in range(S.shape[0])])
            rec = r0_quant(blocks) if r == 0 else leap_window_flush(blocks, compensator_tokens=r)
            rels.append(((rec - blocks).norm() / blocks.norm()).item())
        r_res[r] = round(sum(rels) / len(rels), 6)
    res["r_sweep_relerr_real"] = r_res
    print(f"[E2d'] r sweep rel err on real states: {r_res}", flush=True)

    with open("e2_real_states.json", "w") as f:
        json.dump(res, f, indent=1)
    print("WROTE e2_real_states.json", flush=True)


if __name__ == "__main__":
    main()
