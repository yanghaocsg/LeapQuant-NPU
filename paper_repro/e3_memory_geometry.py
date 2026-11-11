#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Paper-repro E3: memory geometry ledger across our four e2e models.

Extends leapquant_memory_proof.ipynb (0.8B) to every model we run e2e:
per-model linear-attn geometry, per-slot state bytes fp32 vs LeapQuant
payload, and the slots count where e2e memory reduction hits the paper's
56% (weights + linear-attn state ledger, KV cache excluded = conservative).

E3c: prefix-cache concurrency claim (+1.4x single-card requests, paper
5.2): the state-pool share of a fixed HBM budget scales by 3.368x, so the
number of cacheable prefixes at fixed total memory grows accordingly; we
solve the ledger for slots at fixed budget on the smallest realistic card
budget and report the multiplier.
"""

import glob
import json
import os

MODELS = {
    "Qwen3.5-0.8B": "/workspace/y00924007/mnt-weight/Qwen3.5-0.8B",
    "Qwen3.5-9B": "/workspace/y00924007/mnt-weight/Qwen3.5-9B",
    "Qwen3.5-35B-A3B": "/workspace/y00924007/mnt-weight/Qwen3.5-35B-A3B",
    "Kimi-Linear-48B-A3B": "/workspace/y00924007/mnt-weight/Kimi-Linear-48B-A3B-Instruct",
}

HBB = 19460  # payload head-block bytes @ V=K=128, R=4 (codes+s_k+s_v+u+q+hcnt)
RATIO = 4 * 128 * 128 / HBB  # fp32 bytes / payload bytes per element


def model_geometry(path):
    cfg = json.load(open(f"{path}/config.json"))
    tc = cfg.get("text_config", cfg)
    if "layer_types" in tc:  # Qwen3.5 family
        ltypes = tc["layer_types"]
        n_linear = sum(t in ("linear_attention", "linear") for t in ltypes)
        H = tc.get("linear_num_value_heads") or tc.get("linear_num_key_heads")
        DK = tc.get("linear_key_head_dim")
        DV = tc.get("linear_value_head_dim")
        return n_linear, H, DK, DV
    if "linear_attn_config" in tc:  # Kimi-Linear: kda_layers list
        lac = tc["linear_attn_config"]
        return len(lac["kda_layers"]), lac["num_heads"], lac["head_dim"], lac["head_dim"]
    raise KeyError(f"no linear-attn geometry in {path}/config.json")


def main():
    rows = {}
    for name, path in MODELS.items():
        if not os.path.isdir(path):
            # Kimi dir name fallback: search
            cands = glob.glob("/workspace/y00924007/mnt-weight/*Kimi*Linear*48B*")
            if not cands:
                print(f"[skip] {name}: no dir")
                continue
            path = cands[0]
        n_lin, H, DK, DV = model_geometry(path)
        if not (DK == 128 and DV == 128):
            print(f"[skip] {name}: head dims {DK}x{DV} != 128x128 (payload layout pinned to 128)")
            continue
        w_b = sum(os.path.getsize(f) for f in glob.glob(f"{path}/*.safetensors"))
        per_slot_fp32 = n_lin * H * DK * DV * 4
        per_slot_lq = n_lin * H * HBB
        # slots where (W+S) reduction crosses 56%: S*(1-r)/(W+S)=0.56
        r = per_slot_lq / per_slot_fp32
        s56 = 0.56 * w_b / (1 - r - 0.56) / per_slot_fp32
        rows[name] = {
            "linear_layers": n_lin,
            "heads": H,
            "weights_gib": round(w_b / 2**30, 3),
            "per_slot_fp32_mib": round(per_slot_fp32 / 2**20, 2),
            "per_slot_lq_mib": round(per_slot_lq / 2**20, 2),
            "state_ratio": round(1 / r, 3),
            "slots_at_56pct": round(s56),
            "asymptote_pct": round((1 - r) * 100, 1),
        }
        print(f"{name}: {n_lin}L x {H}h, weights {w_b/2**30:.2f} GiB, "
              f"slot fp32 {per_slot_fp32/2**20:.1f} MiB vs LQ {per_slot_lq/2**20:.2f} MiB "
              f"({1/r:.3f}x), 56% @ ~{s56:.0f} slots, asymptote {(1-r)*100:.1f}%")

    # E3c: fixed-budget concurrency (paper: +1.4x requests / single card).
    # Solve: with a 64GB die and Qwen3.5-9B TP1, how many cached-prefix slots
    # fit at fixed budget, fp32 vs payload-only (post M3.4a-2)?
    budget = 64 * 2**30
    name = "Qwen3.5-9B"
    if name in rows:
        row = rows[name]
        free = budget - row["weights_gib"] * 2**30
        slots_fp32 = free / (row["per_slot_fp32_mib"] * 2**20)
        slots_lq = free / (row["per_slot_lq_mib"] * 2**20)
        rows["fixed_budget_64g_slots"] = {
            "model": name,
            "slots_fp32": int(slots_fp32),
            "slots_leapquant": int(slots_lq),
            "prefix_capacity_ratio": round(slots_lq / slots_fp32, 2),
        }
        print(f"[E3c] 64GB die, 9B TP1: prefix slots fp32 {int(slots_fp32)} vs "
              f"LeapQuant {int(slots_lq)} -> {slots_lq/slots_fp32:.2f}x capacity "
              f"(paper: up to 1.4x requests with full KV+state mix)")

    with open("e3_memory_geometry.json", "w") as f:
        json.dump(rows, f, indent=1)
    print("WROTE e3_memory_geometry.json")


if __name__ == "__main__":
    main()
