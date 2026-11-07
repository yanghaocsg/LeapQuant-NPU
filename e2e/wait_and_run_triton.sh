#!/bin/bash
# Wait for an NPU in 2..7 to free up (external job currently occupies all),
# then run the LeapQuant triton-backend 0.8B e2e leg on it.
cd /mnt/share_space/y00924007/LeapQuant-NPU/e2e
for i in $(seq 1 120); do  # up to ~2h of waiting
  for d in 2 3 4 5 6 7; do
    used=$(npu-smi info -t usages -i "$d" 2>/dev/null | grep "HBM Usage Rate(%)" | awk '{print $NF}' | sort -rn | head -1)
    used=${used:-100}
    if [ "$used" -lt 40 ]; then
      echo "DEVICE $d free (HBM usage ${used}%), launching at $(date)"
      VLLM_WORKER_MULTIPROC_METHOD=spawn ASCEND_RT_VISIBLE_DEVICES=$d \
        PYTHONPATH=/vllm-workspace/vllm-ascend:$PYTHONPATH \
        python3 run_eval.py --model /workspace/y00924007/mnt-weight/Qwen3.5-0.8B \
        --leapquant 1 --window 16 --rank 4 --backend triton \
        --out f_08b_lq_triton.json > leg_08b_lq_triton.log 2>&1
      echo "EVAL_EXIT=$? device=$d"
      exit 0
    fi
  done
  sleep 60
done
echo "TIMEOUT: no NPU freed in 2h"
exit 1
