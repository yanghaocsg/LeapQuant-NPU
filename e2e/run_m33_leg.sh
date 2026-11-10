#!/bin/bash
# M3.3 e2e leg only (UTs already green on this tree, 2026-11-10).
# Same hardening as wait_and_run_payload.sh: 45s stability window, 3 attempts,
# free-memory race retry with cooldown.
cd /mnt/share_space/y00924007/LeapQuant-NPU/e2e

hbm_used() {
  npu-smi info -t usages -i "$1" 2>/dev/null | grep "HBM Usage Rate(%)" \
    | awk '{print $NF}' | sort -rn | head -1
}

attempt=0
while [ "$attempt" -lt 3 ]; do
  attempt=$((attempt + 1))
  dev=""
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
    echo "TIMEOUT: no NPU freed (attempt $attempt)"
    exit 1
  fi

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
