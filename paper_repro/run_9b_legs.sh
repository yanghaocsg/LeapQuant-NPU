#!/bin/bash
# Paper-repro E1 legs: Qwen3.5-9B accuracy ladder on AIME 2026 + GSM8K + MCQ.
# Pinned to die 15 (card 7 = our test dies 14/15): external TP job holds
# cards 0-6, so the historical card-scan pattern picks dies that are actually
# full (npu-smi card usage != die-level free memory). Each leg verifies die
# free memory via a real mem_get_info probe before starting.
#   leg 1 base  : leapquant off                     (FP32 state baseline)
#   leg 2 naive : window=1  rank=0 triton roundtrip (per-token smoothed int8)
#   leg 3 w16r0 : window=16 rank=0 triton roundtrip ("+window" ablation)
#   leg 4 full  : window=16 rank=4 triton payload   (production LeapQuant 8-bit)
# All: TP1, gpu-mem-util 0.48, max-model-len 12288, greedy; AIME thinking ON,
# max_tokens 8192 (paper allows ~80K; truncation hits all legs equally).
cd /mnt/share_space/y00924007/LeapQuant-NPU/paper_repro

DIE=15
MODEL=/workspace/y00924007/mnt-weight/Qwen3.5-9B
COMMON="--model $MODEL --n-gsm 200 --n-mcq 200 --n-aime 30 --aime-max-tokens 8192 --gpu-mem-util 0.48 --max-model-len 12288"

wait_die_free() {
  # need at least 30 GiB actually free on the die
  for i in $(seq 1 720); do
    free=$(ASCEND_RT_VISIBLE_DEVICES=$DIE python3 -c "
import torch
free, total = torch.npu.mem_get_info(0)
print(int(free / 2**30))" 2>/dev/null || echo 0)
    if [ "${free:-0}" -ge 30 ]; then
      echo "die $DIE free ${free}GiB"; return 0
    fi
    sleep 30
  done
  echo "TIMEOUT waiting for die $DIE"; return 1
}

run_leg() {
  local name="$1"; shift
  local out="e1_9b_${name}.json"
  local log="e1_9b_${name}.log"
  if [ -s "$out" ] && python3 -c "import json;d=json.load(open('$out'));exit(0 if 'aime' in d else 1)" 2>/dev/null; then
    echo "[skip] $name already complete"; return 0
  fi
  wait_die_free || return 1
  echo "[leg $name] start $(date)"
  VLLM_WORKER_MULTIPROC_METHOD=spawn ASCEND_RT_VISIBLE_DEVICES=$DIE \
    PYTHONPATH=/vllm-workspace/vllm-ascend:$PYTHONPATH \
    python3 ../e2e/run_eval.py $COMMON "$@" --out "$out" > "$log" 2>&1
  local rc=$?
  echo "[leg $name] exit=$rc $(date)"
  [ "$rc" -ne 0 ] && tail -5 "$log"
  return 0   # keep the ladder going even if one leg fails
}

run_leg base   --leapquant 0
run_leg naive  --leapquant 1 --window 1  --rank 0 --backend triton
run_leg w16r0  --leapquant 1 --window 16 --rank 0 --backend triton
run_leg full   --leapquant 1 --window 16 --rank 4 --backend triton --payload-pool 1
echo "ALL LEGS DONE $(date)"
