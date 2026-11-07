# SPDX-License-Identifier: Apache-2.0
"""Compare two run_eval.py result files (baseline vs leapquant)."""

import json
import sys

base = json.load(open(sys.argv[1]))
lq = json.load(open(sys.argv[2]))


def common_prefix_tokens(a, b):
    # cheap token-free proxy: characters
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


print("=" * 64)
print(f"model        : {base['model']}  ({base['gsm8k']['n']} gsm / {base['mcq']['n']} mcq)")
print(f"leapquant    : window={lq['window']} rank={lq['rank']}")
print("=" * 64)

g0, g1 = base["gsm8k"], lq["gsm8k"]
print(f"GSM8K  acc   : {g0['acc']:.3f} -> {g1['acc']:.3f}   (delta {g1['acc']-g0['acc']:+.3f}, {g1['seconds']:.0f}s vs {g0['seconds']:.0f}s)")
agree = sum(1 for p, q in zip(g0["sample_preds"], g1["sample_preds"]) if p == q)
print(f"GSM8K  preds agree (first {len(g0['sample_preds'])}): {agree}/{len(g0['sample_preds'])}")

m0, m1 = base["mcq"], lq["mcq"]
print(f"MCQ    acc   : {m0['acc']:.3f} -> {m1['acc']:.3f}   (delta {m1['acc']-m0['acc']:+.3f})")
agree = sum(1 for d0, d1 in zip(m0["detail"], m1["detail"]) if d0["pred"] == d1["pred"])
print(f"MCQ    pred agreement: {agree}/{m0['n']}")
flip_gain = sum(
    1
    for d0, d1 in zip(m0["detail"], m1["detail"])
    if d0["pred"] != d0["gold"] and d1["pred"] == d0["gold"]
)
flip_loss = sum(
    1
    for d0, d1 in zip(m0["detail"], m1["detail"])
    if d0["pred"] == d0["gold"] and d1["pred"] != d0["gold"]
)
print(f"MCQ    flips: base-wrong->lq-right {flip_gain}, base-right->lq-wrong {flip_loss}")

t0, t1 = base["drift"]["texts"], lq["drift"]["texts"]
exact = sum(1 for a, b in zip(t0, t1) if a == b)
cps = [common_prefix_tokens(a, b) for a, b in zip(t0, t1)]
lens = [max(len(a), len(b)) for a, b in zip(t0, t1)]
print(f"DRIFT  exact match: {exact}/{len(t0)}")
for i, (cp, ln) in enumerate(zip(cps, lens)):
    print(f"  probe {i}: common-prefix {cp:4d} chars / {ln:4d}  ({100*cp/max(ln,1):.0f}%)")
