#!/bin/bash
# Wait for an NPU in 4..7 to free up (external VLLM::Worker_DP job cycles cards;
# test legs stay on cards 4,5,6,7), then run the LeapQuant payload_pool 0.8B
# e2e leg (M3.2b allocation-path smoke).
#
# Robustness against the 2026-11-08 race (card read 4% -> external job re-grabbed
# it within 40s -> engine died with "Free memory on device (5.07/61.27 GiB)"):
#   1. a card only counts as free after two consecutive checks 45s apart;
#   2. if the eval still dies on the free-memory error, go back to polling
#      (up to 3 attempts); other failures are real and abort.
cd /mnt/share_space/y00924007/LeapQuant-NPU/e2e

hbm_used() {
  npu-smi info -t usages -i "$1" 2>/dev/null | grep "HBM Usage Rate(%)" \
    | awk '{print $NF}' | sort -rn | head -1
}

attempt=0
while [ "$attempt" -lt 3 ]; do
  attempt=$((attempt + 1))
  dev=""
  # poll up to ~4h per attempt
  for i in $(seq 1 240); do
    for d in 4 5 6 7; do
      used=$(hbm_used "$d"); used=${used:-100}
      if [ "$used" -lt 40 ]; then
        sleep 45                       # stability window
        used2=$(hbm_used "$d"); used2=${used2:-100}
        if [ "$used2" -lt 40 ]; then
          dev=$d; break 2
        fi
      fi
    done
    sleep 45
  done
  if [ -z "$dev" ]; then
    echo "TIMEOUT: no NPU freed in 4h (attempt $attempt)"
    exit 1
  fi

  echo "DEVICE $dev stable-free (attempt $attempt), launching at $(date)"
  VLLM_WORKER_MULTIPROC_METHOD=spawn ASCEND_RT_VISIBLE_DEVICES=$dev \
    PYTHONPATH=/vllm-workspace/vllm-ascend:$PYTHONPATH \
    python3 run_eval.py --model /workspace/y00924007/mnt-weight/Qwen3.5-0.8B \
    --leapquant 1 --window 16 --rank 4 --backend triton --payload-pool 1 \
    --out f_08b_lq_triton_payload.json > leg_08b_lq_triton_payload.log 2>&1
  rc=$?
  echo "EVAL_EXIT=$rc device=$dev attempt=$attempt"
  if [ "$rc" -eq 0 ]; then
    exit 0
  fi
  if grep -q "Free memory on device" leg_08b_lq_triton_payload.log; then
    echo "Race with external job (free-memory), retrying after cooldown"
    sleep 120
    continue
  fi
  exit "$rc"
done
echo "All 3 attempts lost the race"
exit 1
