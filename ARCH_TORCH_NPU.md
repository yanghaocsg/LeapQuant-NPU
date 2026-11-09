# LeapQuant torch-npu 实现架构（PR1 / M1 里程碑）

> 对应 PR #21120（branch `leapquant-torch-m1`，fork `yanghaocsg/vllm-ascend`）。
> 论文：LeapQuant（arXiv 2609.38166）；RFC：vllm-ascend#21097。
> 数值参考实现：`yanghaocsg/awesome_quant_algo` `turboquant_exp/colab_02_leapquant`；
> 存储契约：huggingface.co/kernels/humanfia-lab/leapquant。
> triton-ascend 融合 kernel 路线见姊妹文档 `ARCH_TRITON_NPU.md`（PR2）。

本文回答一个问题：**论文里的每条公式，落在哪个文件、哪个函数、哪几行代码上**。

---

## 1. 算法要点

GDN/KDA 一类线性注意力的递归状态按

$$S_t = \mathrm{Diag}(\mathrm{decay}_t)\cdot S_{t-1} + \tilde{k}_t\,u_t^{\mathsf{T}}$$

演化（`state.py` 模块 docstring，`vllm_ascend/quantization/methods/mamba_state/leapquant/state.py:7-13`）。
每个 decode step 都对 `S` 做量化会让舍入误差递归累积；LeapQuant 改为**窗口内保持 fp32，仅在窗口边界（每 `window` 个 token）把状态往返通过一次 int8 checkpoint**。这与论文的 window unroll 代数等价：从反量化边界状态 $\hat{S}_0$ 出发的窗口内 fp32 递推，精确复现窗口展开形式——量化误差只在边界注入一次。

边界 flush 分三步（论文式 (5)–(7)，`state.py:15-24`）：

**(1) 补偿因子（式 5–6）**：一轮子空间迭代（2-pass MGS）提取 `S` 的 top-`r` 奇异方向，低秩因子保留 fp16、永不进 int8：

$$Q_K=\mathrm{MGS}(S^{\mathsf{T}}q_0),\qquad Q_V=\mathrm{MGS}(S\,Q_K),\qquad U_K=S^{\mathsf{T}}Q_V$$
$$u=\mathrm{fp16}(U_K^{\mathsf{T}})\in\mathbb{R}^{r\times K},\qquad q=\mathrm{fp16}(Q_V^{\mathsf{T}})\in\mathbb{R}^{r\times V}$$

**(2) 平滑双尺度残差量化（式 7）**：残差 $E=S-q^{\mathsf{T}}u$ 用「每 key 列 × 每 value 行」一对尺度做 int8 量化，离群值只粗化自己所在的行/列而非全矩阵：

$$s_k=\sqrt{\mathrm{mean}_V|E|}\in\mathbb{R}^{K},\qquad s_v=\mathrm{absmax}_K\!\left(|E|/s_k\right)\in\mathbb{R}^{V}$$
$$\mathrm{codes}=\mathrm{clamp}\!\left(\mathrm{round}_{\text{half-even}}\!\left(\frac{E}{s_k\,s_v}\cdot 127\right),\,-127,\,127\right)\to \text{int8}$$

**(3) 往返（式 8–9）**：存储状态被重构值替换，这是量化误差进入数据流的**唯一入口**：

$$\hat{S}=q^{\mathsf{T}}u+\mathrm{codes}\cdot\frac{s_k}{127}\cdot s_v$$

---

## 2. 公式 ↔ 代码总表

数值核心全部在 `vllm_ascend/quantization/methods/mamba_state/leapquant/state.py`（设备无关：CPU 单测 / NPU 运行时同一份代码）：

| 论文公式 | 数学定义 | 代码（文件:行） | 关键实现细节 |
|---|---|---|---|
| 固定起始基 $q_0$ | 随机但**固定**的正交基 $[V,r]$ | `state.py:64` `start_basis` | 种子 `26093816600 & 0x7FFFFFFF + 17·V + r`（`state.py:61`），CPU Generator + `torch.linalg.qr`；与执行历史/捕获顺序无关 |
| 式 (5) MGS | $v_j \leftarrow v_j-\sum_{i<j}c_i\langle c_i,v_j\rangle$，归一化 | `state.py:78` `mgs_orthonormal_columns` | **2-pass** MGS（近平行列数值稳定）；范数 clamp `1e-30`；对最后一轴批量处理 `[..., D, R]` |
| 式 (6) 补偿因子 | $Q_K,Q_V,U_K$ 子空间迭代 | `state.py:96` `fit_compensators` | 三次矩阵乘交替：`x=Sᵀ`（转 `[N,K,V]`）→ `qk=MGS(x@q0)` → `qv=MGS(xᵀ@qk)` → `uk=x@qv`；**只在输出处转 fp16**（迭代本身 fp32），`u∈[N,r,K]`、`q∈[N,r,V]` |
| 式 (7) 残差与尺度 | $E,\ s_k,\ s_v$ | `state.py:122` `quantize_residual` | `residual = state − einsum("nrv,nrk->nvk", q.float(), u.float())`（fp16 因子升 fp32 参与运算）；`s_k=|E|.mean(-2).sqrt().clamp_min(1e-12)`，`s_v=(|E|/s_k).amax(-1).clamp_min(1e-12)`；`torch.round`（half-to-even）+ `clamp(±127)` + `.to(int8)` |
| 式 (8)–(9) 重构 | $\hat{S}=q^{\mathsf{T}}u+\mathrm{codes}\cdot(s_k/127)\cdot s_v$ | `state.py:147` `reconstruct_state` | 低秩项 einsum + codes 反量化项；全 fp32 |
| 边界 flush 组合 | 上述 (1)(2)(3) 串联 | `state.py:160` `leap_window_flush` | `rows[B,H,V,K] → reshape [N,V,K] → float()`；按 head 展平后逐块 flush，`.to(orig_dtype).reshape(orig_shape)` 返回 |
| decode due 判定 | $(\mathrm{ctx}+1)\bmod w=0$ | `state.py:184` `decode_flush_mask` | 每 decode step 一个 token，后验 token 数 = `context+1` |
| prefill due 判定 | $(\mathrm{ctx}+q)\bmod w=0$ | `state.py:195` `prefill_flush_mask` | M1 语义：仅 chunk **末端**恰落在边界时 flush |
| 跨界 due 判定（M3.3） | $\lfloor \mathrm{seq}/w\rfloor>\lfloor(\mathrm{seq}-q)/w\rfloor$ | `state.py:205` `crossed_boundary_mask` | 步区间 $[\mathrm{seq}-q,\ \mathrm{seq}]$ 跨过任一窗口倍数即 due；$q{=}1$ 时精确退化为 $\mathrm{seq}\bmod w=0$。见 §7 |

---

## 3. 配置面：`vllm_ascend/ascend_config.py`

`LeapQuantConfig`（`ascend_config.py:283-349`），挂在 `additional_config["leapquant_config"]`：

| 字段 | 默认 | 语义（论文对应） |
|---|---|---|
| `enabled` | `False` | 总开关，默认关 |
| `window` | `16` | 量化窗口 token 数；论文 8-bit 最优为 16；校验 `(8,16,32,64)` |
| `compensator_tokens` | `4` | 补偿因子秩 $r$（式 5–6）；论文 8-bit 用 $r=4$；校验 `1..8`（triton 后端放宽到 16） |
| `bits` | `8` | checkpoint 位宽；本里程碑仅 8 |
| `kernel_backend` | `"torch"` | `"torch"`=本文档路径；`"triton"`=PR2 融合 kernel（见姊妹文档） |
| `payload_pool` | `False` | 持久化 checkpoint（PR2 / M3.2），要求 `kernel_backend="triton"` |

校验器 `_validate`（`ascend_config.py:322-342`）；与 spec decoding、mamba prefix caching、PD 分离、C8 KV 量化、FULL cudagraph 的互斥检查在 `AscendConfig.derive_and_validate`。

启用示例：

```json
"additional_config": {
  "leapquant_config": {
    "enabled": true, "window": 16, "compensator_tokens": 4,
    "bits": 8, "kernel_backend": "torch"
  }
}
```

---

## 4. 运行时胶水：`vllm_ascend/quantization/methods/mamba_state/leapquant/runtime.py`

### 4.1 三个入口

| 函数 | 行 | 职责 |
|---|---|---|
| `leapquant_config_or_none` | `runtime.py:66` | 读 `get_ascend_config().leapquant_config`；单例未初始化（CPU 单测）→ None（视为关闭）。注意力层与 metadata builder 构造期调用 |
| `_check_device_once` | `runtime.py:33` | A3 设备门。**不能放在配置校验期**：解析设备族会初始化 NPU 运行时，毒化默认 `fork` 出的 EngineCore 进程（"Cannot re-initialize NPU in forked subprocess"）；首个 flush 时 worker 的 NPU 上下文已存在，安全。解析失败（CPU CI）不置位、下次重试；非 A3 每次都抛 |
| `leapquant_flush_state_rows` | `runtime.py:82` | 主入口，见下 |

### 4.2 `leapquant_flush_state_rows` 执行流（`runtime.py:82-194`）

```
入参: state[slots,H,V,K] fp32 池（可带 padded batch stride）,
      state_indices（与 seq_lens 对齐的槽位号）, seq_lens（CPU, 本步后 token 数）,
      config, query_lens=None（M3.3 可选）, state_copy=None（KDA 计划）, payload=None
  │
  ├─ total==0 / 无 due 行 → 直接返回
  ├─ _check_device_once()                          # A3 门 (runtime.py:131)
  ├─ due 判定 (runtime.py:132-137)
  │    query_lens 给定 → crossed_boundary_mask(seq, query, w)     # M3.3 跨界语义
  │    否则           → (seq % w) == 0                            # PR1 语义 (decode 等价)
  ├─ slot_ids = state_indices.index_select(0, due_rows)
  │
  ├─ 路径① 融合 triton (runtime.py:148-164)   kernel_backend=="triton" 且无 KDA 计划
  │    → leapquant_flush_triton(state, slot_ids, ...)   # PR2, 见姊妹文档
  ├─ payload 给定但未走① → RuntimeError (runtime.py:166-172)   # 绝不静默丢 checkpoint
  │
  ├─ 路径② KDA state_copy 计划 (runtime.py:174-180)   NPU + state_copy
  │    plan.gather → leap_window_flush → plan.scatter
  ├─ 路径③ CPU 索引 (runtime.py:181-187)              单测 / 任意 batch stride
  │    index_select → leap_window_flush → index_copy_
  └─ 路径④ NPU gather/scatter (runtime.py:189-194)    GDN 默认
       gather_ssm_states / scatter_ssm_states_          # vllm-ascend 现有 mamba
                                                        # state_index 算子, 非 LeapQuant 专属
```

四条路径**共享同一份数值核心** `leap_window_flush`（或 triton kernel 对它的逐行移植），差别只在 gather/scatter 方式。

---

## 5. 集成点（谁在什么时候调用 flush）

### 5.1 metadata 附加 — `vllm_ascend/ops/gdn_attn_builder.py:1096-1106`

`AscendGDNHostMetadataBackend` 构建 metadata 时把 CPU 侧 per-request 序列长附加为
`attn_metadata.leapquant_seq_lens = seq_lens_cpu[:num_decodes+num_prefills].to(int64)`，
flush 站点经 `getattr` 防御式读取（builder 与执行不在同一进程时为 None → 跳过）。
（M3.3 计划在此追加 `leapquant_query_lens = torch.diff(query_start_loc_cpu)[...]`，见 §7。）

### 5.2 GDN flush 站点 — `vllm_ascend/ops/gdn.py:980-997`

`core_attention_forward` 末尾（本步状态已写回池之后）：

```python
leapquant_config = leapquant_config_or_none()
if (leapquant_config is not None
        and (leapquant_seq_lens := getattr(attn_metadata, "leapquant_seq_lens", None)) is not None):
    payload = leapquant_payload_pool_for_layer(self, self_kv_cache, ssm_state, leapquant_config)  # PR2
    leapquant_flush_state_rows(ssm_state, non_spec_state_indices_tensor,
                               leapquant_seq_lens, config=leapquant_config, payload=...)
```

### 5.3 KDA flush 站点 — `vllm_ascend/ops/kimi_kda.py:761-789`

同构，差异有二：KDA 层持有 gather/scatter 计划（`state_copy`），payload 模式下 `use_state_copy = not payload_pool` **绕过计划直写池**（计划按 fp32 状态几何分配，不吃 payload 扩展）；传 `attn_metadata.non_spec_state_indices_tensor`。

### 5.4 模型类 graft — `vllm_ascend/patch/worker/patch_qwen3_5.py:225-230`、`vllm_ascend/models/kimi_k3.py:93-94,1309-1310`

`leapquant_extend_model_state_classmethods(...)` 包装 4 个模型类的配置期类方法（仅 payload_pool 开启时改变返回值，关闭时逐字节不变）——属 PR2 分配路径，机理见姊妹文档 §4。

---

## 6. 数值契约（与参考实现的 bit 级对齐）

- **int8 对称量化用 $[-127,127]$，不用 $-128$**：`QMAX = 127.0`（`state.py:54-56`），HF kernel 契约 "codes must remain in [-127, 127]"。
- **round-half-even**：`torch.round` 的平局取偶语义；triton 侧用 `_round_half_even`（`flush_kernel.py:79-86`）逐位复刻。
- **fp16 只作用于补偿因子** $u,q$（式 6 输出处一次 cast）；残差、尺度、codes 全程 fp32。`fit_compensators` 的子空间迭代用**未舍入**的 fp32 `qv` 计算 `uk`，仅存储面舍入——triton 移植为此保留了独立的 fp32 `qvr` 平面（姊妹文档 §2）。
- **固定 $q_0$**：跨窗口、跨层固定，flush 结果与执行历史无关（graph capture 友好）。
- **clamp 链**：$s_k,s_v\ge 10^{-12}$（`state.py:57`）；MGS 范数 $\ge 10^{-30}$。
- **误差唯一入口**：`leap_window_flush` 的往返替换；窗口内递推无量化。

---

## 7. 窗口语义：PR1 → M3.3 演进

| | PR1（已上线 #21120） | M3.3（本次升级，`state.py:205` + `runtime.py:88-136`） |
|---|---|---|
| decode | $(\mathrm{ctx}+1)\bmod w=0$ | $\lfloor\mathrm{seq}/w\rfloor>\lfloor(\mathrm{seq}-1)/w\rfloor$，**精确等价** |
| prefill chunk | 仅当 chunk 末端恰好 $\equiv 0 \pmod w$ 才 flush | 步区间跨过任一窗口倍数即 flush（chunk 末端执行） |
| 已知偏差 | chunk 跨内部边界而末端不对齐时，该边界当步不 flush，推迟到下一个对齐边界（中间窗口多积累若干 token，只可能**减小**单边界量化误差，且误差不跳变） | 消除：跨界即 due。chunked delta-rule kernel 只物化 chunk 末端状态，故跨 $k\ge1$ 个边界的 chunk 在 chunk 末端获得**恰好一次** flush；跨步看没有任何边界被跳过（每个边界在它之后第一个步末得到 flush） |
| payload `hcnt` | 恒 0 | 存 `seq % window`（窗口相位），供 readout 知道 checkpoint 折入了多少 token（见姊妹文档 §5） |

`runtime.py` 的 `query_lens` 参数向后兼容：不传即 PR1 语义（decode-only 调用方、单测）。
M3.3 剩余接线（builder 附加 `leapquant_query_lens`、gdn/kda 站点传参）进行中。

---

## 8. 测试

| 文件 | 覆盖 |
|---|---|
| `tests/ut/quantization/methods/test_leapquant_state.py` | 数值核心：MGS 正交性、往返误差界、mask 三种、固定种子确定性 |
| `tests/ut/quantization/methods/test_leapquant_runtime.py` | dispatch 四路径（CPU）、设备门、config-or-none 降级 |
| `tests/ut/quantization/methods/test_leapquant_config.py` | 配置校验（window/range/mutex） |
| `tests/ut/quantization/methods/a3_2/test_leapquant_gdn_integration.py` | NPU 上 GDN 站点端到端 |

运行（dev 仓）：

```bash
PYTHONPATH=/workspace/y00924007/vllm-ascend:$PYTHONPATH \
  python3 -m pytest tests/ut/quantization/methods/test_leapquant_state.py -q
```

（`PYTHONPATH` 前置是必须的：环境里有 playground 安装包，否则 import 解析到旧副本。）

## 9. 已知限制

- 仅 Atlas A3（`_check_device_once` 硬门）；`bits` 仅 8。
- 与 spec decoding / mamba prefix caching / PD 分离 / C8 KV 量化 / FULL cudagraph 互斥（`derive_and_validate`）。
- payload_pool 需 `kernel_backend="triton"`（PR2）。
- torch 路径每 due 行 ~15 个 torch op + 两次池拷贝；融合版本即 PR2 的动机。
