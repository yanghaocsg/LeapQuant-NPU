# vllm-ascend LeapQuant 落地计划(两阶段 PR)

> RFC: vllm-project/vllm-ascend#21097 · 论文: arXiv 2609.38166 · 目标硬件: Atlas A3(Ascend910_9382, soc 253, 8×64GB)
> 目标模型: Qwen3.5-0.8B / Qwen3.5-9B(GDN)、Kimi-3.1 / Kimi-Linear(KDA)
> 阶段: **PR1 纯 torch-npu 功能版(RFC M1)→ 先合入;PR2 triton-ascend 融合算子版(RFC M3)→ 再合入**

---

## 0. 背景与算法要点(实现依据)

LeapQuant 把线性注意力循环状态的量化从"每 token 一次"改为"**每窗口 p=16 token 一次**":

| 论文机制 | 作用 | NPU 移植要点 |
|---|---|---|
| 窗口内 fp32 递推(eq.4, `S_ℓ = Γ_{1:ℓ}·Ŝ_0 + Σ Γ k̃_j u_jᵀ`) | 误差不在每步注入,窗口内**不累积** | 窗口内 fp32 顺序递推与论文展开式**代数等价**(实现依据:窗口边界 round-trip 语义正确的理论基础) |
| Compensator Tokens(eq.5–6, top-r 奇异对, r=4, fp16) | 保留状态主成分,量化后近无损 | 2×MGS 子空间迭代,Q̃ v [B,Nv,R,V] fp16 与 Ũ k [B,Nv,R,K] fp16 分开存(HF kernel 布局) |
| 残差平滑(eq.7, `s_k=√mean_col|E|`, `s_v=absmax(|E|/s_k)`, 对称 INT8) | outlier 只被本行/本列 scale 吸收,不拉爆全局 | 双逐通道 int8,优于 absmax |
| Readout(eq.10, `S_ℓᵀx = c·(Z/127)·B_q + Ũ(K̃ᵀΓx) + Σ u_j(k̃_jᵀΓx)`) | 低带宽读取重建态 | PR2 融合 kernel 的核心 |

**关键语义(经 HF kernel `humanfia-lab/leapquant` 交叉验证)**:decode 时的窗口重放是"**checkpoint + 本窗口已消耗 token 的缓冲更新 → 重建 → 推进本 token**"——即量化只发生在 flush 边界,窗口内永远从 checkpoint+缓冲重建 fp32 态。PR1 的"边界 round-trip"在数学上等价(误差注入时刻一致),因此**精度可直接对齐论文**,只是不做流量/显存优化。

### 0.1 三个参考实现(全部 Apache-2.0,可放心参考)

| 来源 | 内容 | 用途 |
|---|---|---|
| **论文** arXiv 2609.38166 | 公式与消融(窗口 p=16/r=4 最优;1.19B/elem;流量÷3.4;Qwen3.5-9B AIME INT8 per-window 87.9 = FP32 基线;GLM-5.3-Flash(KDA)64.8 vs 65.8 FP32) | 算法权威依据 |
| **`yanghaocsg/awesome_quant_algo` → `turboquant_exp/colab_01..05`** | **torch GPU 实现+验证**(用户指定参考):`rebuild_state / subspace_iteration(MGS) / quantize(sk,sv,codes) / compress_state`;Qwen3.5-0.8B/9B/35B-A3B MMLU/GSM8K 小样本注入实测(MMLU 综合 0.533 / GSM8K 0.625 bf16 基线,AIME 待跑) | **PR1 state.py 的直接蓝本**(已含 MMLU 评测防坑:只在 A/B/C/D 内 argmax、GSM8K 正则提数+max_new 256) |
| **HF kernels `humanfia-lab/leapquant`** | GDN+KDA 窗口边界重量化(Flush)kernel(CuTe DSL/B300,GDN 2.55×/KDA 3.0×,Apache-2.0) | **PR2 编码侧 kernel 的权威参考**:张量契约 `codes int8[B,H,128,128] / s_k,s_v fp32[B,H,128] / u fp16[B,H,4,128](K 侧) / q fp16[B,H,4,128](V 侧) / kbuf,ubuf bf16[B,H,16,128] / w fp32[B,H,16] / p∈(0,1] fp32[B,H] / q0 fp32[128,4]`;KDA 附加 `gbuf fp16[B,H,16,128]∈(0,1] + pcum fp32[B,H,128]`;`Pool+hcnt(==16 触发)+flush_inplace` 窗口状态机与 CUDA-graph 兼容设计 |

### 0.2 代码事实基础(两轮代码探索结论)

**上游 vllm(0.30)关键结论**:
- `MambaDType = Literal["auto","float32","float16","bfloat16"]`(vllm/config/cache.py:49),**无 int8 字面量** → PR1 不动 pool dtype;PR2 走 vllm-ascend 侧覆写(见 §3.2)。
- pool 分配:`GPUModelRunner._allocate_kv_cache_tensors` 按 `page_size_bytes` 分配 int8 原始字节 → `MambaBase.bind_kv_cache`(vllm/model_executor/layers/mamba/abstract.py:29-43)按 `get_state_shape()/get_state_dtype()` 切片+reinterpret。**bind_kv_cache 可被子类覆写**(PR2 的 payload pool 注入点)。
- `GDNAttentionMetadata` **没有 num_computed_tokens 字段**;builder 内 `context_lens_tensor = seq_lens − query_lens`(gdn_attn.py:401)。每序列位置可从 CommonAttentionMetadata 的 seq_lens/query_start_loc 推出——这正是 vllm-ascend `AscendGDNAttentionMetadataBuilder` 手里有的数据。
- FLA chunk 固定 `FLA_CHUNK_SIZE=64`,**按每序列 cu_seqlens 起点对齐**(prepare_chunk_indices),非绝对位置。⚠️ **修正 RFC 的一个说法**:prefill 结束位置=seq_len,只有 `seq_len % 16 == 0` 时才恰在窗口边界;PR1 必须按 `(context_len+query_len) % window` 逐序列判断,不能依赖"chunk 对齐 ⇒ 必在边界"。
- KDA 循环状态上游**硬编码 fp32**(mamba_utils.py:131-137 `kda_state_dtype`),忽略 `--mamba-ssm-cache-dtype`。
- additional_config 在 attention 层内有直接消费先例:`gdn_prefill_backend`/`kda_prefill_backend`(qwen_gdn_linear_attn.py:109-112、kda.py:441-446),且 CLI flag → additional_config 注入先例在 arg_utils.py:2510-2512。

**vllm-ascend(0.19.1rc2.dev)关键结论**:
- **额外 config 的权威解析在 `ascend_config.py` 的 `AscendConfig`(pydantic, extra="forbid")**,不是 platform.py:`init_ascend_config`(:1154-1254)先剔除 `_NON_USER_INPUT_KEYS` 再构造,**未知 key 直接被拒**。→ PR 必须加 typed 字段(模式:EplbConfig dataclass, ascend_config.py:92-146 + field default_factory)。跨配置互斥校验放 `AscendConfig.derive_and_validate()`(:332)或 platform.py `_check_ascend_config`(:798)。
- 设备门控:`get_ascend_device_type() == AscendDeviceType.A3`(soc 253=ascend910_9382→A3, device/hardware.py:39,:64-75)。
- decode 钩子点 `ops/gdn.py`:spec :457 / 混批 :481 / 纯 decode :565 三处调 `torch.ops._C_ascend.npu_recurrent_gated_delta_rule(state=ssm_state, ssm_state_indices=...)`;AscendC op **支持 fp32 state**(op_api DT_FLOAT)。`ssm_state = self.kv_cache[1]`,形状 [slots, Nv, Dv, Dk]。
- prefill 钩子点 `ops/gdn.py:534`(CANN fused)与 :551-553(triton)两处 scatter 写回。
- **KDA 现状**:vllm-ascend 已把 KDA triton kernel 移植到 NPU(`ops/triton/kda/`, #9035),但**只有测试在用,尚无生产接线**。→ Kimi-3.1 的 LeapQuant 需先把 KDA 路径接线(patch 上游 `KimiGatedDeltaNetAttention` → Ascend 版),这是 PR1 的一部分(或独立前置小 PR,见 §2.6)。
- CPU UT 放 `tests/ut/<module>/` 直跑(default_cpu_ut, 每 PR 必跑);NPU UT 按目录路由 `a2/ a2_2/ a3_2/ a3_4/`;e2e 在 `tests/e2e/pull_request/` 与 nightly ops 套件。
- triton 先例:`from vllm.triton_utils import tl, triton`;int8 饱和转换先例 swiglu_quant.py:55;循环 kernel 先例 kda/fused_recurrent_kda.py、mamba/lightning_attn.py;triton-ascend==3.2.2。
- 文档:supported_features.md / feature_guide/quantization.md / additional_config.md 选项表。

---

## 1. PR1 — 纯 torch-npu 功能版(RFC M1)

**目标**:默认关闭的 standalone PR,证明"每窗口量化 + Compensator + 平滑"在 NPU 上的端到端精度与论文对齐,**不改任何 kernel、不改 pool 分配、零 CI 影响**。GDN(Qwen3.5-0.8B/9B)与 KDA(Kimi-3.1/Kimi-Linear)都接。

### 1.1 配置面(AscendConfig typed 字段)

```
vllm_ascend/ascend_config.py:
  @dataclass class LeapQuantConfig:            # 模式:EplbConfig
      enabled: bool = False                     # 默认关
      window: int = 16                          # 校验: ∈{8,16,32,64}, 且必须整除 FLA_CHUNK_SIZE(64)
      compensator_tokens: int = 4               # 校验: 1..8
      bits: int = 8                             # M1 仅 8(写死,字段留作前向兼容)
      dtype: str = "float32"                    # 重建态 dtype(fp32 / bfloat16)
      # 派生: window_mask = window - 1
  AscendConfig.leapquant: LeapQuantConfig = field(default_factory=LeapQuantConfig)
```
- 用户入口:`--additional-config '{"leapquant": {"enabled": true, "window": 16, "compensator_tokens": 4}}'`。
- `derive_and_validate()` / `_check_ascend_config` 中硬校验(启动即 ValueError):
  - `get_ascend_device_type() != A3` → 暂不支持(A2/310P/A5 未验证);
  - `speculative_config is not None` → 排除(num_accepted_tokens 会跨窗);
  - `mamba_cache_mode == "align"` → 排除(mamba prefix caching 会回滚到窗口中间);
  - PD 分离/KV transfer(`kv_transfer_config` 非空)→ 排除(假设 fp32 布局);
  - `kv_cache_dtype != "auto"` 或 modelslim C8 量化开启 → 互斥;
  - `window` 整除 64。
- PR2 再加 SoC 细分与 `kernel_backend: "triton"|"torch"`。

### 1.2 核心模块 `vllm_ascend/quantization/methods/mamba_state/leapquant/`

```
__init__.py    导出 LeapQuantConfig(配置), leap_quantize_state, leap_rebuild_state, leap_window_flush
config.py      参数 dataclass + from_additional_config + validate(纯 CPU 可测)
state.py       纯 torch 算子(CPU/NPU 双跑):
  - fit_compensators(S, q0, R, iters=2) -> (Q̃, Ũ)     # 2×MGS 子空间迭代;每 (b,h) 一个 [V,K] 块
  - quantize_residual(S, Q̃, Ũ) -> (codes int8, s_k, s_v)
  - leap_quantize_state(S_fp32) -> LeapQuantPayload(codes, s_k, s_v, Q̃v, Ũk)
  - leap_rebuild_state(payload) -> S̃                    # S̃ = Q̃ᵀ·Ũ + codes·(s_k/127)·s_v
  - leap_window_flush(S, ...) -> S̃                      # quantize→rebuild,原地语义,入口统一
  - window 层帮助函数:boundary_mask(num_computed_tokens, window), apply_window_flush(ssm_state, rows)
```
- 算法蓝本 = colab_02 的 `subspace_iteration/quantize/rebuild_state`(逐行注释标注论文公式号),**batch/head 维度化为 [B*Nv, V, K]** 一次 einsum 批处理(NPU 上小矩阵批算友好)。
- q0 初始基:每层固定随机(层 id 播种)+ **option: 复用上一窗口 Q̃ 作 warm start**(HF kernel 明确不这么做——固定 q0,避免依赖顺序;跟随 HF 契约,默认固定)。
- 每 (b,h) 独立量化,与 pool 布局 [slots, Nv, Dv, Dk] 直接对齐。

### 1.3 GDN 集成(`vllm_ascend/ops/gdn.py`,全部 gated on leapquant.enabled)

**Decode**(三处调用点 :457/:481/:565 前统一插一个 helper):
```
# 位置 = 该序列已算 token 数 = seq_lens - query_lens(与上游 builder 同口径)
mask = (positions > 0) & (positions % window == 0)
if mask.any():
    rows = state_indices[mask]                      # spec/non_spec_state_indices_tensor[mask]
    ssm_state[rows] = leap_window_flush(ssm_state[rows])
# 之后照常调 npu_recurrent_gated_delta_rule(fp32 state, 原地更新)
```
- 语义 = 论文窗口语义:flush 只在"本 token 恰为窗口第 p 个 token"时发生,窗口内 fp32 原地推进 → **精度等价论文**;纯 torch 开销 = 边界处 gather+MGS+int8+scatter(每 16 步一次,ms 级)。
- vllm-ascend builder 里已有 seq_lens/query_start_loc(CommonAttentionMetadata),不依赖上游暴露新字段。

**Prefill**(两处 scatter 写回后 :534/:551-553):
```
end_pos = context_lens + query_lens                 # 逐序列
mask = (end_pos % window == 0) & (end_pos > 0)
ssm_state[prefill_state_indices[mask]] = leap_window_flush(...)
```
- ⚠️ 不依赖"chunk_size 整除 ⇒ 必在边界"(已证伪),按 end_pos 逐序列判断;附 UT 钉死。
- 校验器已保证 `window | 64`,chunk 计算不受影响(量化在 scatter 之后、不影响 chunk kernel 本身)。

### 1.4 KDA/Kimi-3.1 集成

现状:`ops/triton/kda/` 已移植未接线。方案(推荐 A):
- **A(并入 PR1)**:`vllm_ascend/patch/worker/patch_kimi_k3.py`(或扩展现有 patch 机制)把上游 `KimiGatedDeltaNetAttention` 的 kernel 调用替换为 `vllm_ascend.ops.triton.kda.*`,并在其 decode(`fused_recurrent_kda`, kda.py:113)与 prefill(`chunk_kda`, :1181)状态写回处加与 §1.3 相同的窗口 flush(KDA 状态 fp32, [slots, Nv, V, K],同布局直接复用 state.py)。
- **B(拆前置 PR)**:先提"KDA 接线"独立小 PR(仅通路,无量化),PR1 只做 GDN,Kimi 支持放 PR1.5/PR2。若 reviewer 认为 PR1 过大则降级到 B。
- Kimi-3.1 若 checkpoint 的 `linear_attn_config.use_full_rank_gate=True` 走 KDA(KimiK3DeltaAttention);复用同一 flush(state 布局相同)。gated 参数(gbuf/pcum)在 KDA 重建公式中的处理按论文附录 A 的 KDA 形式扩展(公式同构,readout 门控逐 K 维)。

### 1.5 测试与验证

**CPU UT(`tests/ut/quantization/methods/test_leapquant_*.py`,CI 每 PR 必跑)**:
1. `test_leapquant_config.py`:合法/非法 additional_config、互斥矩阵(spec/align/PD/C8/非 A3→ValueError)。
2. `test_leapquant_state.py`:
   - flush 幂等性:payload 相同 ⇒ rebuild 确定性;
   - codes ∈ [−127,127]、s_k/s_v > 0;
   - 重建误差界:随机 S ‖S̃−S‖/‖S‖ ≤ 0.02(8bit+R4);
   - **收敛性复现论文 Fig.3**:64 步 GDN 递推,fp64 基线 vs per-step-int8 vs per-window-LQ,断言 LQ 误差平坦且 ≥5× 低于 per-step(colab_02 已有该实验代码,直接移植);
   - 主成分保留:rank-1 主导 S 的重建近 exact。
3. `test_leapquant_window.py`:boundary_mask 边界条件(0/p/2p/非整除位置/prefill end_pos 各种余数)。

**NPU UT(`tests/ut/quantization/methods/a3_2/test_leapquant_state_npu.py`)**:state.py 在 NPU 上的数值一致性(CPU vs NPU allclose)+ 边界 flush 与 `npu_recurrent_gated_delta_rule` 的组合冒烟(3 序列×128 步,窗口 16,断言不 NaN、误差有界)。

**E2E(本机 8×A3,编号进 PR 描述;不放 CI)**:
- Qwen3.5-0.8B 单卡:开/关 leapquant 贪心生成对比(同 prompt token 一致率 + 长文本 sane);MMLU 3 子集×5 题 + GSM8K 8 题(评测代码移植 colab_03/04,已含 A/B/C/D argmax 与 GSM8K 正则两个坑的修复)。
- Qwen3.5-9B 单卡 bf16:MMLU 子集 + GSM8K,A/B(off vs on)准确率差 < 1pt(小样本口径,声明不与论文绝对值比;论文 Table 1 里 9B 8bit MMLU 83.8)。
- Kimi(-3.1/Linear):TP8 起服 + GSM8K 小样本 + decode 吞吐记录(基线数据,PR2 对比用)。

### 1.6 文档
- `docs/source/user_guide/feature_guide/quantization.md` 增 LeapQuant 小节;
- `docs/source/user_guide/configuration/additional_config.md` 选项表加 `leapquant`;
- `docs/source/user_guide/support_matrix/supported_features.md` 标 A3-only、Qwen3.5/Kimi-Linear。

### 1.7 PR1 验收标准
- 默认 `enabled=False` 行为零变化(所有现有测试不动也能过);
- 开启后:0.8B/9B MMLU+GSM8K 与 bf16 基线差 < 1pt(小样本);64 步误差曲线复现论文趋势;
- 互斥矩阵全部启动即报错;PR 描述附 RFC 链接与实验数据表。

**工作量估计**:代码 ~800 行(模块 300 + 集成 200 + 测试 300),日历时间约 1.5 周。

---

## 2. PR2 — triton-ascend 融合算子版(RFC M3)

**目标**:把 flush 编码侧 + decode 读取侧融合进 triton-ascend kernel,拿到**显存(状态池 4×↓)与带宽(有效 1.19B/elem, 流量 ÷3.4)**收益;单 A3 架构优先。以 HF CuTe kernel 的张量契约与 Pool/flush_inplace 语义为权威参考。

### 2.1 存储设计(payload pool,plugin 侧完成,不动上游 MambaDType)

| 张量 | dtype/形状(per slot) | 对应 HF 契约 |
|---|---|---|
| codes | int8 [Nv, Dv, Dk] | codes |
| s_k / s_v | fp32 [Nv, Dk] / [Nv, Dv] | s_k / s_v |
| Q̃v | fp16 [Nv, R, Dv] | q |
| Ũk | fp16 [Nv, R, Dk] | u |
| hcnt | int32 [Nv](per-head fill counter) | pool.hcnt(==16 触发) |
| 工作区 kbuf/ubuf/w(+KDA gbuf/pcum) | bf16 [Nv,16,Dk]/[Nv,16,Dv]/fp32 [Nv,16] | kbuf/ubuf/w(/gbuf/pcum) |

- **内存收益**:池 [slots,Nv,128,128] fp32 → int8 codes(+scales/补偿 ≈1.19B/elem)≈ **3.4× 缩减**;Qwen3.5-9B 单卡 state 池典型 2-3GB → ~0.7GB,并发上限显著提升(实测数字进 PR)。
- 分配路径:Ascend 注意力类覆写 `get_state_shape()/get_state_dtype()/bind_kv_cache`(上游 bind 按 shape/dtype 切片 reinterpret,plugin 可把同一 int8 字节池 unpack 成上述 payload 视图)——**PR1 不动,PR2 才做**,风险隔离。
- transient fp32 重建态仅 per-active-request(论文:8KiB/头/层@128×128)。

### 2.2 triton kernel 清单(`vllm_ascend/ops/triton/leapquant/`)

1. `flush_kernel.py` — **编码侧融合**(HF `_gdn.py`/`_kda.py` 的 triton-ascend 移植):
   - 每 CTA 处理一个 (slot, head):重建 S(fp32)→ MGS 子空间迭代取 R=4 → 残差平滑 int8 → 写回 payload;
   - counter==window 触发、reset w/p(HF 语义:flush 后 counter 停在 16,由 decode 步推进);
   - KDA 变体:门控逐 K 维(gbuf/pcum 路径),按论文附录 A。
2. `recurrent_leapquant.py` — **decode 读取侧融合**(改造 `kda/fused_recurrent_kda.py` 循环骨架):
   - 每步读 payload(1.19B/elem)+ 缓冲 → 按论文 eq.10 readout,不物化 fp32 全量态;
   - 窗口边界触发 flush(调用 1 或内联);
   - 输出 attn out + 更新缓冲/counter。
3. `chunk_boundary_quantize.py` — prefill 边界编码(chunk 输出 fp32 state → payload,替代 scatter 后的 torch flush)。
- 全部 gated:`get_ascend_device_type()==A3` 且 `kernel_backend=="triton"`;否则回落 PR1 torch 路径(保底可用)。
- triton-ascend 注意点:int8 饱和 cast(swiglu_quant.py:55 先例 `overflow_mode="saturate"`)、`from vllm.triton_utils import tl, triton`、小 MGS 循环在 kernel 内的寄存器压力(128×128 fp32 = 64KB,需分块或借 local buffer;HF kernel 用 TMEM,CANN 无对应物 → 分块 MGS/two-pass 方案,精确度对比进 PR)。

### 2.3 集成
- decode:`ops/gdn.py` / KDA 接线处,`leapquant.enabled && backend==triton` 时改调新 kernel(替换 `npu_recurrent_gated_delta_rule` 调用与 KDA `fused_recurrent_kda`);
- prefill:chunk 后接 `chunk_boundary_quantize`;
- CUDA-graph/图捕获兼容:参考 HF "pool 先建+warmup 再捕获" 语义,NPU 侧对应 graph 模式验证(tests/e2e nightly)。

### 2.4 测试与基准
- UT:kernel vs PR1 torch 参考实现的数值对齐(error ratio ≤1.01× 口径,借 HF 的 fp64 重建基准方法);counter/flush 状态机单测。
- E2E(A3):0.8B/9B 精度回归(与 PR1 相同口径,差 <1pt);Kimi TP8。
- 性能(进 PR 描述):
  - kernel 级:flush vs torch flush;readout 融合 vs fp32 recurrent;
  - 端到端:decode 吞吐(长上下文 4K/16K/32K)、TTFT、并发上限提升(状态池缩小后);
  - 显存:状态池 footprint 前后对比。

### 2.5 PR2 验收标准
- 精度:与 PR1 相同口径不回退(<1pt);
- 性能:长上下文 decode 吞吐提升可测(目标对齐论文方向,不承诺 1.47×);
- 默认关闭;A3-only;非 A3 或 backend=torch 回落 PR1 路径。

**工作量估计**:kernel 3 个+集成 ~1500 行,日历时间约 3-4 周。

---

## 3. 执行顺序与里程碑

| 步骤 | 内容 | 产出 |
|---|---|---|
| 0 | RFC #21097 下留言计划摘要(feedback period 一周) | 社区对齐 |
| 1 | PR1:模块+配置+GDN 集成+CPU UT | branch `leapquant-torch-m1` |
| 2 | PR1:0.8B/9B NPU 实测数据 | PR 描述数据表 |
| 3 | PR1 提交;KDA 接线按 §1.4 A/B 决策 | PR #1(含或不含 Kimi) |
| 4 | PR2:payload pool + flush kernel | branch `leapquant-triton-m3` |
| 5 | PR2:readout 融合 decode kernel + 基准 | PR #2 |
| 6 | 文档/支持矩阵/RFC 更新(M2 typed entry 决议留言) | 收尾 |

风险与备选:
- triton-ascend 上 MGS 寄存器压力超限 → 降级 two-pass(先重建残差再拟合)或 flush 拆两 kernel;
- Kimi-3.1 checkpoint 未公开/结构变化 → 先用 Kimi-Linear-48B-A3B-Instruct 验证 KDA 路径,3.1 就绪后回归;
- PR1 体积过大 → KDA 接线拆前置小 PR(§1.4 B);
- A3 之外需求出现 → PR2 的 SoC gate 放宽为逐架构验证清单。
