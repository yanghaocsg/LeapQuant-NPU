#!/bin/bash
# M3.3 NPU validation queue (cards 4..7, same waiter pattern as
# wait_and_run_payload.sh): first the a3_2 leapquant UTs (flush kernel +
# payload incl. the new per-row hcnt test), then the 0.8B payload e2e leg
# re-run under M3.3 crossing semantics (diff vs f_08b_lq_triton_payload.json
# isolates the semantic change; noise-floor gates apply).
#
# Race hardening identical to wait_and_run_payload.sh: 45s stability window,
# free-memory races retry up to 3 attempts with cooldown.
cd /mnt/share_space/y00924007/LeapQuant-NPU/e2e

hbm_used() {
  npu-smi info -t usages -i "$1" 2>/dev/null | grep "HBM Usage Rate(%)" \
    | awk '{print $NF}' | sort -rn | head -1
}

wait_free_card() {
  for i in $(seq 1 240); do
    for d in 4 5 6 7; do
      used=$(hbm_used "$d"); used=${used:-100}
      if [ "$used" -lt 40 ]; then
        sleep 45                       # stability window
        used2=$(hbm_used "$d"); used2=${used2:-100}
        if [ "$used2" -lt 40 ]; then
          echo "$d"; return 0
        fi
      fi
    done
    sleep 45
  done
  return 1
}

DEV=/workspace/y00924007/vllm-ascend   # dev repo first on PYTHONPATH for UTs

# ---- stage 1: a3_2 unit tests on the first free card --------------------
dev=$(wait_free_card) || { echo "TIMEOUT: no NPU freed for UTs"; exit 1; }
echo "UT device $dev at $(date)"
ASCEND_RT_VISIBLE_DEVICES=$dev \
  PYTHONPATH=$DEV:$PYTHONPATH \
  python3 /workspace/y00924007/vllm-ascend/tests/ut/quantization/methods/a3_2/test_leapquant_flush_triton.py \
  > ut_m33_flush.log 2>&1
rc1=$?
ASCEND_RT_VISIBLE_DEVICES=$dev \
  PYTHONPATH=$DEV:$PYTHONPATH \
  python3 /workspace/y00924007/vllm-ascend/tests/ut/quantization/methods/a3_2/test_leapquant_payload.py \
  > ut_m33_payload.log 2>&1
rc2=$?
echo "UT_EXIT flush=$rc1 payload=$rc2 (device=$dev)"
if [ "$rc1" -ne 0 ] || [ "$rc2" -ne 0 ]; then
  echo "UNIT TESTS FAILED — skipping the e2e leg"
  exit 1
fi

# ---- stage 2: e2e payload leg under M3.3 semantics ----------------------
attempt=0
while [ "$attempt" -lt 3 ]; do
  attempt=$((attempt + 1))
  dev=$(wait_free_card) || { echo "TIMEOUT: no NPU freed (attempt $attempt)"; exit 1; }
  echo "EVAL DEVICE $dev (attempt $attempt) at $(date)"
  VLLM_WORKER_MULTIPROC_METHOD=spawn ASCEND_RT_VISIBLE_DEVICES=$dev \
    PYTHONPATH=/vllm-workspace/vllm-ascend:$PYTHONPATH \
    python3 run_eval.py --model /workspace/y00924007/mnt-weight/Qwen3.5-0.8B \
    --leapquant 1 --window 16 --rank 4 --backend triton --payload-pool 1 \
    --out f_08b_lq_triton_payload_m33.json > leg_08b_lq_triton_payload_m33.log 2>&1
  rc=$?
  echo "EVAL_EXIT=$rc device=$dev attempt=$attempt"
  if [ "$rc" -eq 0 ]; then
    exit 0
  fi
  if grep -q "Free memory on device" leg_08b_lq_triton_payload_m33.log; then
    echo "Race with external job (free-memory), retrying after cooldown"
    sleep 120
    continue
  fi
  exit "$rc"
done
echo "All 3 attempts lost the race"
exit 1
