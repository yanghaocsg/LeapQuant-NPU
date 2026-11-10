# LeapQuant torch-npu 实现：逐文件逐行讲解（PR1 / PR #21120）

配套文档：`ARCH_TORCH_NPU.md`（公式↔代码总表、窗口语义演进）——本文是其"逐行版"，按 PR1
（`leapquant-torch-m1` 分支，提交 `d20cc8553` + 评审修复 `4d603dae6`）的文件清单展开，行号取自当前
`leapquant-triton-m3` 开发树（与 PR1 tip 的差异逐处标注，便于两边导航）。

PR1 文件清单（git show --stat d20cc8553）：

| 类别 | 文件 | PR1 增量 |
|---|---|---|
| 数值核心 | `quantization/methods/mamba_state/leapquant/state.py` | +201 |
| 运行时胶水 | `quantization/methods/mamba_state/leapquant/runtime.py` | +148 |
| 包导出 | `mamba_state/__init__.py` / `leapquant/__init__.py` | +3 / +35 |
| 配置 | `ascend_config.py` | +100 |
| 集成点 | `ops/gdn_attn_builder.py` | +29 |
| 集成点 | `ops/gdn.py` | +26 |
| 集成点 | `ops/kimi_kda.py` | +26 |
| 测试 | `test_leapquant_{state,runtime,config,gdn_integration}.py` | 383 行 / 21 用例 |
| 文档 | `additional_config.md` / `quantization.md` / `supported_features.md` | +44 |

一个 step 的数据流（先建立全局图，再逐文件展开）：

```
scheduler（decode 请求在前、prefill 在后）
 └─ AscendGDNAttentionMetadataBuilder.build
     └─ _attach_leapquant_metadata:
          attn_metadata.leapquant_seq_lens   = seq_lens_cpu_upper_bound[:D+P]  (CPU int64)
          attn_metadata.leapquant_query_lens = query_start_loc 差分            (CPU, M3.3)
 └─ AscendGatedDeltaNetAttention.forward_core_attn (ops/gdn.py)
     ├─ 1. causal conv1d + fused_gdn_gating
     ├─ 2. 状态更新（AscendC recurrent / chunked prefill → scatter 进 ssm_state）
     └─ 2.4 flush 站点: leapquant_flush_state_rows(ssm_state, indices, seq_lens, ...)
          ├─ due 判定（CPU）: crossed_boundary_mask / legacy %window==0
          ├─ slot_ids = state_indices[due_rows]
          └─ 路径: KDA编译plan / CPU索引 / triton gather+scatter
               → leap_window_flush(rows)
                    ├─ start_basis        固定种子 q0
                    ├─ fit_compensators   子空间迭代 + MGS → q/u (fp16)
                    ├─ quantize_residual  双尺度 int8 codes + s_k/s_v
                    └─ reconstruct_state  Ŝ = qᵀu + codes·(s_k/127)·s_v
 └─ 下一步（边界行状态已带量化误差 = 论文窗口语义）
```

---

## 1. `vllm_ascend/ascend_config.py` — 配置面（PR1 +100 行）

### 1.1 `class LeapQuantConfig`（:283-349，PR1 tip :45-88）

- **:284-298 docstring**：声明配置入口 `additional_config["leapquant_config"]`，一句话算法摘要
  （窗口边界 int8 量化、compensator_tokens 低秩补偿、双尺度残差量化），并预告互斥检查在
  `AscendConfig.derive_and_validate`。
- **:300-310 字段**（pydantic `@config` dataclass）：
  - `enabled: bool = False` — 默认关。所有运行时入口都经 `leapquant_config_or_none()`，禁用时零开销。
  - `window: int = 16` — 每窗口 token 数；论文 8-bit 最优为 16。
  - `compensator_tokens: int = 4` — 补偿子空间秩 r（留在 int8 之外的 top-r 奇异对）；论文 8-bit 用 4。
  - `bits: int = 8` — checkpoint 位宽，PR1 只实现 int8。
- **:311-320 `_validate`（model_validator mode="after"）**：
  - `window ∈ (8,16,32,64)` — 2 的幂窗口，跨界判定用整数除法才精确（见 §2.9）。
  - `1 ≤ compensator_tokens ≤ 8` — torch 路径上限（M3.1 triton 路径放宽到 16，见 :332-336）。
  - `bits == 8` — 里程碑约束，显式报错而非静默降级。
- **:344-349 `summary` property** — 启动日志一行摘要（`_check_leapquant_config` 末尾打印）。

### 1.2 `AscendConfig` 上的挂载（:704, :789）

- `:704 leapquant_config: LeapQuantConfig = dataclasses.field(default_factory=...)` —
  组合进 AscendConfig 单例，`get_ascend_config().leapquant_config` 即全局唯一实例。
- `:789` `derive_and_validate()` 里调用 `self._check_leapquant_config(vc)` — 跨选项互斥在启动时
  （EngineArgs 解析后）执行，而非首步。

### 1.3 `_check_leapquant_config`（:1080-1130）

- **:1087-1088** `not enabled → return`：默认关时不做任何检查、不打印日志。
- **:1093-1097 NOTE（评审修复引入）**：A3-only 设备门**故意不在这里**——解析设备族会初始化 NPU
  runtime，在 `fork` 出 EngineCore 子进程之前初始化会得到 "Cannot re-initialize NPU in forked
  subprocess"。设备门移到首次 flush 的 worker 进程里（§3.1）。
- **:1098-1103 拒 speculative decoding**：每步接受 token 数可变 → decode 一步可能跨过窗口倍数，
  "每步 1 token" 的边界假设破坏。
- **:1104-1110 拒 `mamba_cache_mode="align"`（prefix caching）**：回滚可能把状态恢复到量化窗口
  中间，窗口相位丢失。
- **:1111-1116 拒 PD disaggregation（kv_transfer）**：KV/state 传输契约假设 fp32 池布局。
- **:1117-1122 拒 C8 KV 量化**：与状态量化叠加未验证。
- **:1123-1129 拒 FULL cudagraph**：flush 是 attention custom op 内的**数据依赖 host 逻辑**
  （due 行数每步不同），FULL 捕获无法表达；piecewise/eager 允许。
- **:1130** `logger.info(summary)`。

五项共同设计原则：**启动时 fail-fast，绝不静默走错路径**——每一项报错文本都写明"为什么破坏假设 +
怎么关"。

---

## 2. `leapquant/state.py` — 数值核心（设备无关，PR1 +201；当前 223 行）

模块 docstring（:3-34）给出三件事：(a) 为什么要窗口化——`S_t = Diag(decay)·S_{t-1} + k̃_t u_tᵀ`
每步量化会**递归累积**舍入误差，LeapQuant 改为窗口内保持 fp32、只在边界 round-trip，且从反量化
Ŝ₀ 出发的窗口内 fp32 递推与论文窗口展开**代数等价**；(b) flush 三步 = 论文 eq.5-7（补偿子空间 /
平滑双尺度 int8 / 重建 eq.8-9）；(c) checkpoint 布局契约与参考 GPU kernel
（huggingface.co/kernels/humanfia-lab/leapquant）对齐：`[V,K]` 块、`q∈[R,V]`/`u∈[R,K]` fp16、
codes int8、scales fp32。本模块设备无关（CPU UT / NPU 运行）。

### 2.1 常量（:54-61）

- `:56 QMAX = 127.0` — 对称 int8 用 `[-127,127]`，**不用 -128**（参考内核契约），保证 0 两侧对称、
  `·127`/`/127` 精确互逆。
- `:57 EPS = 1e-12` — scale 下钳，防除零。
- `:61 _START_BASIS_SEED = 26093816600 & 0x7FFFFFFF` — arXiv 号 2609.38166 派生的确定性种子；
  `& 0x7FFFFFFF` 钳成 31-bit 正整数。

### 2.2 `start_basis`（:64-75）— 固定正交起始基 q0

```python
g = torch.Generator(device="cpu")          # :71  CPU 生成器
g.manual_seed(_START_BASIS_SEED + 17*v_dim + rank)   # :72  种子 = 基种子 + 17·V + R
q0 = torch.randn(v_dim, rank, generator=g).to(dtype) # :73
q0, _ = torch.linalg.qr(q0)                # :74  reduced QR → 列正交
return q0.to(device=device, dtype=dtype).contiguous()  # :75
```

- **为什么固定随机基而不是上一窗口的因子**：flush 结果与执行历史无关（确定性、可测试、对捕获顺序
  不敏感）——docstring :66-70 明说。
- **为什么 CPU 生成**：任何设备上 q0 逐位相同（NPU 上 randn 无法保证跨设备一致）。
- 种子偏移 `17·v_dim + rank`：每个 (V,R) 组合一个唯一确定基；全窗口、全层共用同一 q0。

### 2.3 `mgs_orthonormal_columns`（:78-93）— 批处理两遍 MGS（eq.5-6 的正交化）

```python
for j in range(m.shape[-1]):               # :87  逐列 j = 0..R-1
    v = m[..., j]                          # :88  当前列
    for _ in range(passes):                # :89  passes=2
        for c in cols:                     # :90  对每个已正交化列
            v = v - c*(c·v).sum(-1)        # :91  修改型投影
    cols.append(v / v.norm(-1).clamp_min(1e-30))  # :92  归一化 + 防零
return torch.stack(cols, -1)               # :93
```

- 输入 `[..., D, R]`：前导维任意（批 N 个块并行处理），最后一维的 R 列被正交化。
- **修改型** Gram-Schmidt：投影用**最新** v（经典 GS 用原始向量，误差 O(κ·ε) 累积）。
- `passes=2`：第二遍重复投影，列近平行时一遍后正交性不足。
- `clamp_min(1e-30)`：全零列（死头）不产生 NaN。

### 2.4 `fit_compensators`（:96-119）— 一轮子空间迭代 → top-r 补偿对（eq.5-6）

```python
x  = state.transpose(-1, -2)        # :112  x = Sᵀ  [N,K,V]
qk = mgs_orthonormal_columns(x @ q0) # :113  K-侧正交基
qv = mgs_orthonormal_columns(x.transpose(-1,-2) @ qk)  # :114  V-侧正交基 = mgs(S·qk) [N,V,R]
uk = x @ qv                          # :115  系数 = Sᵀ·qv [N,K,R]
u  = uk.transpose(-1,-2).to(torch.float16)  # :116  [N,R,K] fp16
q  = qv.transpose(-1,-2).to(torch.float16)  # :117  [N,R,V] fp16
del rank                             # :118
```

- 一步子空间迭代：`qk = orth(Sᵀq0)` → `qv = orth(S·qk)` → `uk = Sᵀ·qv`（最小二乘系数）。
  这是近似截尾 SVD 的标准一阶方法，R 小（4）时 top-r 能量已捕获大半。
- **q/u 落 fp16 而非 int8**：补偿因子承载 top-r 奇异能量（最大头），量化它们会伤精度最大的分量；
  fp16 是参考内核 checkpoint 契约的存储精度。
- `:118 del rank`：rank 隐含在 q0/qv 形状里，参数保留只为 API 自文档。

### 2.5 `quantize_residual`（:122-144）— 平滑双尺度 int8（eq.7）

```python
residual = state - torch.einsum("nrv,nrk->nvk", q.float(), u.float())  # :139  E = S − qᵀu
s_k = residual.abs().mean(-2).sqrt().clamp_min(EPS)   # :140  每 key 列 RMS scale [N,K]
s_v = (residual.abs()/s_k[...,None,:]).amax(-1).clamp_min(EPS)  # :141  每 value 行 scale [N,V]
codes = torch.round(residual / (s_k·s_v) * QMAX)      # :142  round-half-to-even
codes = codes.clamp(-QMAX, QMAX).to(torch.int8)       # :143
```

- `:139` **先 `.float()` 再乘**：q/u 是 fp16 存储，残差必须回 fp32 算（舍入顺序与参考一致）。
- `:140` `s_k = sqrt(mean_V |E|)`：对 V 取**均值**再开方 → 列 RMS（几何型，比 absmax 抗离群）。
- `:141` `s_v = absmax_K(|E|/s_k)`：先除列 scale 再取行 absmax → 联合分解 `|E[v,k]| ≈ s_v[v]·s_k[k]`。
- **"平滑"的含义**：离群元素只抬高自己的一行/一列 scale，不污染全矩阵单 scale（这是 LeapQuant
  相对朴素 per-tensor int8 的精度来源）。
- `:142` `torch.round` = **round-half-to-even**（银行家舍入，无正向偏置）；`·127` 与重建侧 `/127`
  精确互逆。

### 2.6 `reconstruct_state`（:147-157）— 重建（eq.8-9）

```python
state = torch.einsum("nrv,nrk->nvk", q.float(), u.float())       # :155  低秩项 qᵀu
state = state + codes.float() * (s_k[...,None,:]/QMAX) * s_v[...,,:,None]  # :156  int8 项
```

这就是论文的 S₀ 重建式，也是 payload 读out（M3.2/M3.4）复用的同一公式——量化侧 `·127` 与
重建侧 `/127` 互逆，codes/scale 一起构成本窗口的 checkpoint。

### 2.7 `leap_window_flush`（:160-181）— 编排入口

- `:172-174` 记住形状/dtype，`reshape(-1,V,K).float()` 把 `[B,H]` 折进 N，统一 fp32 计算。
- `:177-180` q0 → fit → quantize → reconstruct 四连。
- `:181` `.to(orig_dtype).reshape(orig_shape)` — 对调用方 dtype 透明（池是 fp32；UT 可喂 fp16）。
- docstring :170 点题：**这是量化误差进入数据流的唯一点**。

### 2.8 `decode_flush_mask` / `prefill_flush_mask`（:184-202）— M1 期的两个 due 判定

- `decode`: `ends = context+1`（每 decode 步贡献 1 token），`ends%window==0`；`ends>0` 过滤
  padding 行。
- `prefill`: `ends = context + query_lens`（chunk 末 token 数）。
- 生产路径 M3.3 起改用 §2.9；这两个保留给 UT 与语义对照。

### 2.9 `crossed_boundary_mask`（:205-222，M3.3 增量，非 PR1）

```python
ends   = seq_lens.to(torch.long)
starts = (ends - query_lens).clamp_min(0)
return (ends // window) > (starts // window)   # :222
```

- 语义：本步跨度 `[starts, ends)` 跨过的窗口倍数 ≥1 即 due。decode（query=1）时严格等价于
  `ends%window==0`；**chunked prefill 跨内部边界而不落在边界上**的 chunk 也 due（在 chunk 末
  flush，配 `hcnt=seq_len%window` 记录窗口相位）——修复了 PR1 "只 flush 边界对齐 chunk 末尾" 的
  缺口（PR1 runtime docstring :96-104 承认的已知限制）。
- 整数除法判定要求 window 是 2 的幂 → 对应 §1.1 的 `(8,16,32,64)` 校验。

---

## 3. `leapquant/runtime.py` — 运行时胶水（PR1 +148；当前 206 行）

模块职责（:3-9）：state.py 管数值，本模块管集成——读 AscendConfig 单例、从 seq_lens 推导 due 行、
把 flush 应用到状态池的行上。

### 3.1 `_device_gate_checked` + `_check_device_once`（:30-63）— A3 设备门

- **:41-42** flag 已置 → 直接返回（每层每步调用的热路径，只查一次）。
- **:43-47** 惰性 import 设备配置；解析失败（CPU-only CI）→ **return 且不置 flag**（:52 注释）：
  runtime 来得晚的 worker 下次 flush 重试，不会永远无门。
- **:53-59** 非 A3 → `RuntimeError` 且**不置 flag**：上层若 catch，下次 flush 仍会 raise——
  绝不静默放行。
- **为什么在首次 flush 而非 config 校验**（:33-38 docstring）：解析设备族会初始化 NPU runtime，
  `fork` 出的 EngineCore 会被毒化；首次 flush 时 worker 的 NPU 上下文已存在，安全。
  这是与 §1.3 NOTE 呼应的"设备门放 worker 侧"决策（评审修复 `4d603dae6` 的另一半）。

### 3.2 `leapquant_config_or_none`（:66-79）— 统一配置入口

- `:71-74` `get_ascend_config()` 抛 RuntimeError（CPU UT 无单例）→ None = disabled。
- `:75-76` 只有 `config.leapquant_config.enabled` 才返回配置对象。
- 所有注意力层/ builder 都从这里取配置：**禁用时每个调用点一次函数调用 + None 判断，零额外开销**。

### 3.3 `leapquant_flush_state_rows`（:82-206）— flush 主入口

PR1 签名只有 `state/state_indices/seq_lens/*/config/state_copy`；当前树多了
`query_lens`（M3.3）与 `payload`（M3.2）。docstring（:92-127）详述 due 语义：给 `query_lens`
用跨界判定，否则 legacy `%window==0`；payload 模式还持久化窗口相位。

**due 判定与行选择（:128-142）**：

- `:128-131` `total==0` 早退；设备门（每 worker 一次）。
- `:132-136` `query_lens` 给了 → `crossed_boundary_mask`（§2.9）；否则 `seq_lens%window==0`。
  两者都是 **CPU 张量运算**——builder 附带的是 CPU seq_lens（§5），flush 端判定零 D2H 同步。
- `:137-139` `nonzero` → due_rows；空 → return（绝大多数 decode 步只有少数行 due）。
- `:141-142` `rows.to(indices.device)`；`slot_ids = state_indices.index_select(0, rows)` ——
  注意区分：`due_rows` 是**请求行号**，`slot_ids` 才是**池行号**。

**路径选择（四条，按序试探）**：

1. **triton 融合路径（:148-175，M3.1+ 增量，非 PR1）**——门槛：`kernel_backend=="triton"` 且
   无 KDA plan 且非 CPU 且 V/K 均为 2 的幂。一次内核启动替代 gather+15 个 torch 算子+scatter。
   `:158-168` payload 模式（M3.2/M3.3）：`rewrite_state=True`（fp32 池行同时重写 = 双写迁移门，
   decode 侧仍读 fp32）+ `hcnt_values=seq_len%window`（窗口相位；标量默认值会丢相位，故必须逐行）。
   PR1 时期这条不存在——torch 路径就是下面的 2/3/4。
2. **KDA 编译 plan（:185-191）**——`state_copy` 非 None 且非 CPU：`present` 全 true（due 行必在
   池内）→ `gather` → `leap_window_flush` → `scatter`。为 Kimi KDA 的专属分页池布局开的口
   （§7）。
3. **CPU 路径（:192-198）**——`index_select`/`index_copy_`：覆盖 CPU UT 与任意 batch stride 的
   padded 池。
4. **NPU 通用路径（:200-205）**——`gather_ssm_states`/`scatter_ssm_states_`（triton 编译的
   gather/scatter）。`:949-951`（gdn.py 注释）解释为什么不用 `ssm_state[ids]`：hybrid KV-cache
   下状态池是 padded 物理页上的 strided view，advanced indexing 会先物化**整个 cache**。
   `flushed.to(state.dtype).contiguous()` 保证 scatter 的 dtype/布局契约。

**:177-183 防御**（M3.2 增量）：payload 非 None 但走不了 triton → `RuntimeError`——payload 消费方
读到 stale 行是正确性 bug，绝不静默丢 checkpoint。

---

## 4. 包导出（两个 `__init__.py`）

- `mamba_state/__init__.py`（+3）：仅包 docstring —— `quantization/methods/` 下新建
  mamba/线性注意力状态子包的占位声明。
- `leapquant/__init__.py`（+35）：re-export `runtime` 的 2 个符号 + `state` 的 8 个符号，
  `__all__` 字母序。集成点（gdn.py/kimi_kda.py/builder）只 import 这个包级入口，不触内部模块
  路径。当前树追加了 `crossed_boundary_mask`（M3.3）与 payload 池相关导出（M3.2）。

---

## 5. `ops/gdn_attn_builder.py` — metadata 附加（PR1 +29，当前 :1080-1117）

- **:1059** 调用点：`build()` 尾部、`_attach_non_spec_decode_metadata` 之后、`_finalize_metadata`
  之前——保证 spec/non-spec 索引已就绪。
- **:1097-1099** 双早退：配置未启用 → return（禁用时 builder 零开销）；`seq_lens_cpu_upper_bound`
  为 None → return（防御非标准路径）。
- **:1108-1117** 核心：取 `m.seq_lens_cpu_upper_bound`（CommonAttentionMetadata 的 CPU 上界张量），
  **裁掉 FULL-graph padding 行**：`[:num_decodes + num_prefills]` 正好是
  `non_spec_state_indices_tensor` 寻址的 decodes-then-prefills 真实请求行（调度器保证 decode 在
  前）；`.to(torch.int64)` 让 §2.9 的 `//` 运算安全。
- 产出 `attn_metadata.leapquant_seq_lens` —— **CPU 张量**，下游 due 判定全程 host 侧，无 D2H。
  （M3.3 在同函数追加 `leapquant_query_lens` 的差分计算，供跨界判定。）

## 6. `ops/gdn.py` — GDN flush 站点（PR1 +26；当前 :1048-1076）

- **:54-57** import 包级入口两符号。
- **:1048-1058 注释块（原 2.4 节，PR1 位于 :940 起）**：明确时序契约——flush 在本步 recurrent
  更新 / prefill scatter **全部完成之后**（量化对象是"步末状态"）；窗口内状态保持 fp32，量化误差
  只在边界进入 = 论文窗口形式的代数等价实现。
- **:1059-1063 三条件 gate**：配置启用 ∧ `non_spec_state_indices_tensor` 非 None ∧ metadata 带
  `leapquant_seq_lens`（walrus 取出）。任一不满足整段跳过——对非 leapquant 部署是纯三判断开销。
- **:1069-1076 调用**：`leapquant_flush_state_rows(ssm_state, indices, seq_lens, config=...)`。
  当前树追加了 `query_lens`（M3.3）与 `payload`（M3.4 起：`leapquant_payload_pool_for_layer`
  取层池，仅 `payload_pool` 开启时）——PR1 时刻只有前四个参数。
- **位置不变性**：M3.1-M3.4 的所有演进（triton flush、payload、triton decode）都只改 flush
  **内部**路径与 2.0 节 decode gate，这个 2.4 站点的位置和时序语义从未动过。

## 7. `ops/kimi_kda.py` — KDA flush 站点（PR1 +26，当前 :761-789）

- 与 §6 相同的三条件 gate → `leapquant_flush_state_rows(recurrent_state, ...)`。
- **KDA 差异点**：`state_copy=getattr(self, "_ascend_kda_state_copy", None) if _kda_state_copy_ready
  else None` —— Kimi KDA 的状态池是 KDA 专属分页布局，通用 gather 不适用；层持有 torch.compile
  编出的 `KDAStateCopyPlan`（`kda_state_copy_plan.py:207/224` 定义 `gather(state, indices, flags)`
  / `scatter(state, packed, indices)` 契约）。plan 未就绪（如捕获前）传 None 落到通用路径兜底。
  这就是 runtime.py :185-191 那条路径存在的原因——同一套数值，三套 gather/scatter 传输层。

---

## 8. 测试与文档（PR1 收尾面）

- `test_leapquant_state.py`（121 行）：数值单元——flush 幂等性（二过 flush 不再变）、双尺度分解
  的还原精度、补偿子空间对 top-r 能量的捕获、mask 边界条件。
- `test_leapquant_runtime.py`（109 行）：due 行选择、三条 gather/scatter 路径等价性、设备门行为。
- `test_leapquant_config.py`（55 行）：schema 校验 + 五项互斥。
- `test_leapquant_gdn_integration.py`（98 行，A3 实卡）：跑**真 AscendC recurrent 算子** 64 token
  + 4 次边界 flush——输出漂移 0.37%、状态漂移 0.53%、未触及 slot 精确为零（量化误差量级与论文
  一致的实证锚点）。
- 文档：`additional_config.md` +22（leapquant_config 全参数参考）、`quantization.md` +21（特性
  指南章节）、`supported_features.md` +1（支持矩阵条目）。

---

## 9. PR1 → 当前树（leapquant-triton-m3）的演进对照

| 文件 | PR1 之后的新增 | 里程碑 |
|---|---|---|
| `state.py` | `+crossed_boundary_mask` | M3.3 |
| `runtime.py` | `+query_lens` 跨界判定；`+payload`/phase + 拒绝静默丢池；`+triton` 融合快速路径 | M3.3 / M3.2 / M3.1 |
| `ascend_config.py` | `+kernel_backend("torch"\|"triton")`；`+payload_pool`（须 triton、r≤16） | M3.1 / M3.2 |
| `gdn.py` | `+2.0` triton decode gate（`recurrent_gdn_leapquant` 替 AscendC 算子）；flush 传 payload/query_lens | M3.4 / M3.3-3.4 |
| `kimi_kda.py` / `gdn_attn_builder.py` | 基本未动（KDA triton 化是 M3.4 后续） | — |
| 新增（PR2 线） | `ops/triton/leapquant/{flush_kernel,recurrent_leapquant}.py`、`payload.py` 池、`patch_qwen3_5.py` graft | M3.1-M3.4 |

**torch 路径至今仍是默认后端与永久 fallback**（`kernel_backend="torch"`）：PR2 的全部演进都保持
"数值与 PR1 逐位一致"，triton 内核的 UT 都以本文件的 torch 参考为 golden。
