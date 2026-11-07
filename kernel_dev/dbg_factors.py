import torch

torch.npu.set_device(0)
torch.manual_seed(7)

from vllm_ascend.ops.triton.leapquant.flush_kernel import (
    _leapquant_flush_scales_kernel,
    _get_q0,
)
from vllm_ascend.quantization.methods.mamba_state.leapquant.state import (
    fit_compensators,
    quantize_residual,
    start_basis,
)

V = K = 128
R = 4
H = 1
pool = torch.randn(2, H, V, K, device="npu:0", dtype=torch.float32)
due = torch.tensor([0], device="npu:0", dtype=torch.int64)
q0 = _get_q0(V, R, pool.device)
blocks = 1
scratch = torch.zeros(blocks, K, V + K, dtype=torch.float32, device=pool.device)
sk = torch.zeros(blocks, K, dtype=torch.float32, device=pool.device)
sv = torch.zeros(blocks, V, dtype=torch.float32, device=pool.device)
_leapquant_flush_scales_kernel[(blocks,)](
    pool, due, q0, scratch, sk, sv,
    pool.stride(0), pool.stride(1), pool.stride(2), pool.stride(3),
    H=H, V=V, K=K, R=R, BLOCK_V=32, BLOCK_K=32, num_warps=8,
)
torch.npu.synchronize()

# torch reference on the same block
rows = pool[due].reshape(-1, V, K).float()
q0_ref = start_basis(V, R, device=rows.device, dtype=torch.float32)
print("q0 match:", torch.equal(q0, q0_ref))
u_ref, q_ref = fit_compensators(rows, q0_ref, R)   # u [N,R,K] fp16, q [N,R,V] fp16
codes_ref, sk_ref, sv_ref = quantize_residual(rows, u_ref, q_ref)

# kernel factors from scratch: qv_f rows 0..15 at cols [K, K+V) -> [R, V]
qv_f = scratch[0, :16, K:K + V]        # [16, V]; rows = rank
uk_f = scratch[0, :, :16].t()          # [16, K] -> rows = rank? stored [K,16]
print("q_ref [R,V] vs kernel qv_f[:R]:")
for r in range(R):
    d_direct = (qv_f[r].float() - q_ref[0, r].float()).abs().max().item()
    d_flip = (qv_f[r].float() + q_ref[0, r].float()).abs().max().item()
    print(f"  r={r}: direct={d_direct:.3e} flipped={d_flip:.3e}")
print("u_ref [R,K] vs kernel uk_f rows:")
uk_rows = scratch[0, :, :16]  # [K,16] element (k, r)
for r in range(R):
    d_direct = (uk_rows[:, r].float() - u_ref[0, r].float()).abs().max().item()
    d_flip = (uk_rows[:, r].float() + u_ref[0, r].float()).abs().max().item()
    print(f"  r={r}: direct={d_direct:.3e} flipped={d_flip:.3e}")
print("sk  maxdiff:", (sk[0] - sk_ref[0]).abs().max().item())
print("sv  maxdiff:", (sv[0] - sv_ref[0]).abs().max().item())
print("padding cols zero:", scratch[0, :, R:16].abs().max().item())
