# SPDX-License-Identifier: Apache-2.0
"""LeapQuant e2e A/B eval: GSM8K 5-shot + cmmlu letter-argmax MCQ + drift probes.

One process per leg. Greedy everywhere. Writes a JSON result file.
"""

import argparse
import json
import random
import re
import time

import polars as pl

DATA = "/workspace/y00924007/mnt-weight/datasets"
GSM_TEST = f"{DATA}/gsm8k/main/test-00000-of-00001.parquet"
GSM_TRAIN = f"{DATA}/gsm8k/main/train-00000-of-00001.parquet"
CMMLU_TEST = f"{DATA}/cmmlu/test"
CMMLU_DEV = f"{DATA}/cmmlu/dev"

LETTERS = ["A", "B", "C", "D"]

DRIFT_PROMPTS = [
    "用一段话解释量子计算与经典计算的本质区别。",
    "Write a Python function that computes the n-th Fibonacci number with memoization.",
    "A train travels 120 km in 1.5 hours, then 80 km in 45 minutes. What is its average speed for the whole trip? Show your reasoning.",
    "Summarize the plot of Romeo and Juliet in exactly three sentences.",
    "列出五个提高代码可读性的具体实践，并各用一句话说明原因。",
    "Explain the difference between TCP and UDP to a junior engineer, using an analogy involving mail.",
    "如果地球自转速度突然加快一倍，日常生活中会出现哪些现象？请列举三条。",
    "Write a haiku about debugging production incidents at 3am, then explain its meaning.",
]


def gsm_shots(k=5):
    train = pl.read_parquet(GSM_TRAIN)
    return [(train[i, "question"], train[i, "answer"]) for i in range(k)]


def gsm_prompt(question, shots):
    parts = [
        "Solve the following grade school math problems step by step. "
        "End your final answer with '#### <number>'.\n"
    ]
    for q, a in shots:
        parts.append(f"Question: {q}\nAnswer: {a}\n")
    parts.append(f"Question: {question}\nAnswer:")
    return "".join(parts)


NUM_RE = re.compile(r"-?\$?[\d,]*\.?\d+")


def extract_gsm(text):
    if "####" in text:
        tail = text.split("####")[-1]
        m = NUM_RE.search(tail.replace(",", ""))
        if m:
            return m.group(0).replace("$", "").replace(",", "")
    nums = NUM_RE.findall(text.replace(",", ""))
    return nums[-1] if nums else None


def gsm_gold(answer):
    return answer.split("####")[-1].strip().replace(",", "")


def load_cmmlu(n, seed=0):
    rng = random.Random(seed)
    rows = []
    for subject_file in sorted(__import__("os").listdir(CMMLU_TEST)):
        if not subject_file.endswith(".csv"):
            continue
        subject = subject_file[:-4]
        with open(f"{CMMLU_TEST}/{subject_file}", encoding="utf-8") as f:
            import csv

            for r in csv.DictReader(f):
                r["subject"] = subject
                rows.append(r)
    rng.shuffle(rows)
    return rows[:n]


MCQ_TMPL = (
    "以下是一道关于{subject}的单项选择题，请直接给出正确答案的选项字母。\n\n"
    "{question}\nA. {a}\nB. {b}\nC. {c}\nD. {d}\n\n答案是："
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--leapquant", type=int, default=0)
    ap.add_argument("--payload-pool", type=int, default=0)
    ap.add_argument("--window", type=int, default=16)
    ap.add_argument("--rank", type=int, default=4)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-gsm", type=int, default=200)
    ap.add_argument("--n-mcq", type=int, default=200)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--backend", default="torch", choices=["torch", "triton"])
    ap.add_argument("--max-num-seqs", type=int, default=16)
    ap.add_argument("--gpu-mem-util", type=float, default=0.85,
                    help="0.85 default keeps historical legs reproducible; lower it to "
                         "shrink the KV/mamba capacity plan (e.g. M3.4a-1 dual-fp32 leg at 0.48)")
    ap.add_argument("--gsm-seed", type=int, default=0,
                    help="shuffle GSM8K question order with this seed (0 = fixed 0..n-1 order)")
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    add_cfg = {}
    if args.leapquant:
        add_cfg = {
            "leapquant_config": {
                "enabled": True,
                "window": args.window,
                "compensator_tokens": args.rank,
                "kernel_backend": args.backend,
                "payload_pool": bool(args.payload_pool),
            }
        }

    llm = LLM(
        model=args.model,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_mem_util,
        enable_prefix_caching=False,
        enforce_eager=True,
        tensor_parallel_size=args.tp,
        additional_config=add_cfg,
        trust_remote_code=True,
    )
    tok = llm.get_tokenizer()
    ctk = {"enable_thinking": False} if "enable_thinking" in (getattr(tok, "chat_template", "") or "") else {}

    result = {
        "model": args.model,
        "leapquant": bool(args.leapquant),
        "payload_pool": bool(args.payload_pool),
        "window": args.window,
        "rank": args.rank,
    }

    # ---- GSM8K 5-shot ----
    shots = gsm_shots(5)
    gsm = pl.read_parquet(GSM_TEST)
    idx = list(range(min(args.n_gsm, gsm.height)))
    if args.gsm_seed:
        random.Random(args.gsm_seed).shuffle(idx)
    prompts = [
        [{"role": "user", "content": gsm_prompt(gsm[i, "question"], shots)}] for i in idx
    ]
    golds = [gsm_gold(gsm[i, "answer"]) for i in idx]
    t0 = time.time()
    outs = llm.chat(
        prompts,
        SamplingParams(temperature=0.0, max_tokens=400),
        chat_template_kwargs=ctk,
    )
    gsm_time = time.time() - t0
    correct = 0
    preds = []
    for o, g in zip(outs, golds):
        p = extract_gsm(o.outputs[0].text)
        preds.append(p)
        if p is not None and p == g:
            correct += 1
    result["gsm8k"] = {
        "n": len(idx),
        "correct": correct,
        "acc": correct / len(idx),
        "seconds": round(gsm_time, 1),
        "sample_preds": preds[:15],
        "sample_golds": golds[:15],
    }
    print(f"[GSM8K] {correct}/{len(idx)} = {correct/len(idx):.3f} in {gsm_time:.0f}s", flush=True)

    # ---- cmmlu letter-argmax ----
    rows = load_cmmlu(args.n_mcq)
    letter_ids = []
    for l in LETTERS:
        for v in (l, " " + l):
            ids = tok.encode(v, add_special_tokens=False)
            if len(ids) == 1:
                letter_ids.append(ids[0])
    mcq_prompts = [
        [
            {
                "role": "user",
                "content": MCQ_TMPL.format(
                    subject=r["subject"],
                    question=r["Question"],
                    a=r["A"],
                    b=r["B"],
                    c=r["C"],
                    d=r["D"],
                ),
            }
        ]
        for r in rows
    ]
    outs = llm.chat(
        mcq_prompts,
        SamplingParams(temperature=0.0, max_tokens=1, allowed_token_ids=letter_ids),
        chat_template_kwargs=ctk,
    )
    correct = 0
    detail = []
    for o, r in zip(outs, rows):
        pred = o.outputs[0].text.strip()[:1]
        ok = pred == r["Answer"]
        correct += ok
        detail.append({"subject": r["subject"], "gold": r["Answer"], "pred": pred})
    result["mcq"] = {
        "n": len(rows),
        "correct": correct,
        "acc": correct / len(rows),
        "detail": detail,
    }
    print(f"[MCQ ] {correct}/{len(rows)} = {correct/len(rows):.3f}", flush=True)

    # ---- drift probes ----
    outs = llm.chat(
        [[{"role": "user", "content": p}] for p in DRIFT_PROMPTS],
        SamplingParams(temperature=0.0, max_tokens=256),
        chat_template_kwargs=ctk,
    )
    result["drift"] = {
        "prompts": DRIFT_PROMPTS,
        "texts": [o.outputs[0].text for o in outs],
    }

    with open(args.out, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    print("WROTE", args.out, flush=True)


if __name__ == "__main__":
    main()
