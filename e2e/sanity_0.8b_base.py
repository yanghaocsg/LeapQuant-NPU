import os, sys
from vllm import LLM, SamplingParams
llm = LLM(model="/workspace/y00924007/mnt-weight/Qwen3.5-0.8B",
          max_model_len=4096, max_num_seqs=8, gpu_memory_utilization=0.85, enable_prefix_caching=False, enforce_eager=True,
          additional_config={})
sp = SamplingParams(temperature=0.0, max_tokens=64)
outs = llm.chat([{"role": "user", "content": "What is 13 * 17? Answer with just the number."}],
                sp)
print("REPLY:", outs[0].outputs[0].text[:100])
print("SANITY_OK")
