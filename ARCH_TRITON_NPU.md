# LeapQuant triton-ascend 实现架构（PR2 / M3 里程碑）

> 对应 branch `leapquant-triton-m3`（fork `yanghaocsg/vllm-ascend`）。
> 论文：LeapQuant（arXiv 2609.38166）；RFC：vllm-ascend#21097；开发计划：`TRITON_M3_DEV_PLAN.md`。
> torch-npu 功能路径（PR1）见姊妹文档 `ARCH_TORCH_NPU.md`——两份文档共享同一数值核心
> `state.py`；PR2 = **融合 kernel**（把 PR1 的 gather → ~15 op flush → scatter 折进 2–3 次启动）
> \+ **payload 持久存储**（int8 checkpoint 落池，~1.19 B/elem vs 4 B fp32，即论文的显存收益）。

里程碑状态：M3.1 融合 flush ✅ · M3.2 payload 池 + 引擎分配 ✅（commit `e2c9c5a3b`）·
M3.3 chunk 边界语义 🔨（数值与 kernel 侧已落地，站点接线进行中）· M3.4 readout 融合（式 10）· M3.5 PR2 装配。

---

## 1. 总体数据流：三种 flush 模式

入口 `leapquant_flush_triton`（`vllm_ascend/ops/triton/leapquant/flush_kernel.py:360`），
由 `runtime.leapquant_flush_state_rows` 的 triton 分支（`runtime.py:148-164`）调用
（`kernel_backend=="triton"`、无 KDA 计划、NPU 池、V/K 为 2 的幂）。

```
leapquant_flush_triton(state[slots,H,V,K] fp32, slot_ids, R, payload?, rewrite_state?, hcnt_values?)
  │
  ├─ launch 1  _leapquant_flush_scales_kernel      [式5-6 全部 + 式7 的尺度]
  │    scratch: qk fp32 [D·H,K,16]   qvr fp32 [D·H,V,16](raw)   qv fp16   uk fp16
  │              sk fp32 [D·H,K]     sv fp32 [D·H,V]
  │
  ├─ 模式 A roundtrip (M3.1, payload=None):
  │    launch 2  _leapquant_reconstruct_kernel      [式7 codes + 式8-9 原地重写 fp32 池]
  │             → 精度基线 & 永久回退路径
  │
  ├─ 模式 B payload (M3.2, payload=pool, rewrite_state=False):
  │    launch 2  _leapquant_payload_encode_kernel   [codes+尺度+fp16因子+hcnt → payload 池]
  │             → 真正省显存的模式；fp32 池不动（readout 切换前的过渡态）
  │
  └─ 模式 B' dual-write (payload=pool, rewrite_state=True):
       launch 2  _leapquant_payload_encode_kernel
       launch 3  _leapquant_reconstruct_kernel      [同一 scratch 输入 → 与模式 A bit 相等]
                → 迁移安全门：decode 侧仍读 fp32，池必须保持正确
```

**分裂启动的第二个动机**：launch 2/3 复用 launch 1 存储的 `s_k`/`s_v`，把两条 pass
钉在 bit 级相同的尺度上（重算会有浮点求和顺序差）。

每 due (slot, head) 一个 triton program（grid = `D·H`），块大小 `BLOCK_V=BLOCK_K=32`，`num_warps=8`。

---

## 2. 公式 ↔ kernel 对照表

三个 kernel 位于 `flush_kernel.py`；torch 参考（`state.py`）逐函数对应：

| 公式阶段 | torch 参考 | triton 代码 | 实现差异（为何不同） |
|---|---|---|---|
| $q_0$ 固定基 | `state.py:64 start_basis` | host 侧 `_get_q0`（`flush_kernel.py:351`）按 `(V,R,device)` 缓存，kernel 只读 | 基只依赖 `(V,R)`，缓存免重生成 |
| 式 (5) 2-pass MGS | `mgs_orthonormal_columns`（`state.py:78`） | `_mgs_cols`（`flush_kernel.py:90-107`） | 列抽取用 axis-1 掩码求和（见 §6-C）；范数 clamp `1e-30` 与参考一致；padding 列（`r..16`）恒零，后续 `tl.dot` 不受污染 |
| 式 (6) $Q_K=\mathrm{MGS}(S^{\mathsf{T}}q_0)$ | `fit_compensators`（`state.py:112-113`） | scales kernel 144-155 行：按 `BLOCK_V` 流式累加 `qk16 += dot(trans(s_t), q0_t)` | `[V,K]` 块放不进 UB（§6-A），流式 + `tl.dot`（N≥16 → rank 补到 16） |
| 式 (6) $Q_V=\mathrm{MGS}(S\,Q_K)$ | `state.py:114` | 157-168 行：按 `BLOCK_K` 流式 `qv16 += dot(s_t, qk_t)` | 同上 |
| 式 (6) $U_K=S^{\mathsf{T}}Q_V$ | `state.py:115` | 170-179 行：**读 fp32 raw 平面 `qvr`**（未舍入因子），流式 `uk16 += dot(trans(s_t), qv_t)` | 参考实现里 `uk` 用未舍入的 fp32 `qv` 计算；本 build 上寄存器内 fp16 cast 是 no-op（§6-B），舍入只能发生在 fp16 **store**——故 `qv` 存两份：fp16 平面（checkpoint 值）+ fp32 `qvr` 平面（此 dot 的输入） |
| 式 (6) fp16 因子 $u,q$ | `state.py:116-117` 的 `.to(fp16)` | fp16 store（168、178 行）→ `qv_s`/`uk_s` fp16 scratch | 同上：store 是唯一可靠舍入点 |
| 式 (7) $E=S-q^{\mathsf{T}}u$ | `quantize_residual` einsum（`state.py:139`） | scales kernel 181-200 / reconstruct 233-247 / encode 312-327 的 `lr_t` | 低秩项不做 dot：把 rank-R 因子列经 axis-1 抽取成 `[V]`/`[BK]` 向量后**手工展开 R 次外积累加**（§6-C/§6-E） |
| 式 (7) $s_k=\sqrt{\mathrm{mean}_V\|E\|}$ | `state.py:140` | scales kernel 198 行 `s_k_t = max(sqrt(sum(a_t,axis=0)/V), 1e-12)`，逐 K tile 存出 | clamp 语义与参考一致 |
| 式 (7) $s_v=\mathrm{absmax}_K(\|E\|/s_k)$ | `state.py:141` | scales kernel 189-201 行：`s_v` 跨 K tile running max，最后 clamp `1e-12` | 流式化后 absmax 变 loop-carried 累加器 |
| 式 (7) codes round/clamp | `state.py:142-143` | `_round_half_even`（79-86）+ `clamp(±127)`（reconstruct 250-252 / encode 330-332） | `_round_half_even` 逐位复刻 `torch.round` 平局取偶 |
| int8 落盘 | `.to(torch.int8)` | `codes.cast(tl.int8, overflow_mode="saturate")`（encode 335） | 值已 clamp，saturate 是 no-op 保险（swiglu_quant.py 先例） |
| 式 (8)–(9) $\hat{S}=q^{\mathsf{T}}u+\mathrm{codes}\cdot\frac{s_k}{127}\cdot s_v$ | `reconstruct_state`（`state.py:147`） | reconstruct 253-254：`out = lr_t + codes*(s_k_t/127)*s_v[:,None]` 原地写回 | 与参考同序同精度 |
| hcnt 窗口相位（M3.3） | — | encode 348：`tl.store(hcnt_ptr + slot*pl_sb_f32 + h*pl_hb_f32, tl.load(hcnt_vals_ptr + d))` | 逐 due 行的 `seq % window`，非标量（§5） |

正确性钉子：`tests/ut/quantization/methods/a3_2/test_leapquant_flush_triton.py` 把 kernel 输出
钉在 torch 参考上（误差比 ≤ 1.01 + 逐元素 ≤ 2 个 int8 code 步）。

---

## 3. Payload 存储：`vllm_ascend/quantization/methods/mamba_state/leapquant/payload.py`

### 3.1 布局（论文 checkpoint 契约 ↔ 字节表）

`PayloadLayout`（`payload.py:63`）描述每个 (slot, head) head-block 的静态布局，
字段顺序保证一切偏移 4 字节对齐（`field_offsets`，`payload.py:129`）。
以 GDN/KDA 几何 `V=K=128, R=4`：

| 偏移 | 字段 | dtype/形状 | 论文/公式量 | 字节 |
|---:|---|---|---|---:|
| 0 | `codes` | int8 `[V,K]` | 式 (7) codes | 16384 |
| 16384 | `s_k` | fp32 `[K]` | 式 (7) $s_k$ | 512 |
| 16896 | `s_v` | fp32 `[V]` | 式 (7) $s_v$ | 512 |
| 17408 | `u` | fp16 `[K,R]`（行主序） | 式 (6) $u$ | 1024 |
| 18432 | `q` | fp16 `[V,R]` | 式 (6) $q$ | 1024 |
| 19456 | `hcnt` | int32 | 窗口相位 `seq%w` | 4 |
| | **合计** | | | **19460 B** |

- `bytes_per_element = 19460/(128·128) = 1.1877` B/elem vs fp32 的 4 B → **3.37×**（论文 ~1.19 B/elem）。
- `u/q` 物理按 `[K,16]`/`[V,16]` 平面行主序存储（镜像 kernel scratch 取向，规避转置读；见 §6-C），属性视图再转置成 `[R,K]`/`[R,V]` 暴露。
- 校验（`payload.py:73-82`）：V/K 必须 2 的幂、`rank ≤ 16`。

### 3.2 `LeapQuantPayloadPool`（`payload.py:146`）

一个 uint8 缓冲解包成各字段视图，**池不拥有任何东西，只有缓冲**：

- 独立分配：`allocate`（204）→ 连续 `[slots, heads, head_block_bytes]`。
- **引擎分配（M3.2b）**：缓冲是 mamba 页里切出的**跨步视图**——slot stride = 整页
  （conv + ssm + payload + padding），head 在页内密排。两类缓冲走同一构造器；
  `head_stride_bytes`/`slot_stride_bytes`（194-202）把实际字节步长交给 flush kernel。
- `_view`（222）：`buffer.view(dtype)` 重释 + 显式 `as_strided`（slot/head 步长换算成目标 dtype 元素数，字段在 head block 内密排）；构造期校验 4 字节对齐（fp32/fp16/int32 字段）。
- `reconstruct_rows`（284）：torch 侧读回（式 8-9 的 `state.reconstruct_state` 逐 head 调用），单测与后续宿主工具用。

### 3.3 配置要求

`payload_pool: true` 强制 `kernel_backend="triton"`（`ascend_config.py:337-341`）——
融合 kernel 是唯一 encoder；torch 路径遇到 payload 直接 `RuntimeError`（`runtime.py:166-172`），
绝不静默丢 checkpoint。

---

## 4. 引擎分配路径（M3.2b，commit `e2c9c5a3b`）

payload 池不是旁路分配，而是**作为 mamba 页的第 3 个状态分量**进入引擎常规分配——
与 conv/ssm 同页、同生命周期、同 slot 几何：

```
进程启动
  └─ check_and_update_config → init_ascend_config        # AscendConfig(含 leapquant_config)就位
  └─ uniproc_executor._init_executor
       ├─ load_model                                        # worker 补丁生效:
       │      patch_qwen3_5.py / kimi_k3.py 的 graft 已应用
       │        ├─ 实例级: 注意力层 get_state_shape/get_state_dtype 追加
       │        │    (heads, head_block_bytes) uint8       # leapquant_payload_state_extension
       │        │    (payload.py:304; 取 base 形状最后一项 (heads,V,K) 推几何)
       │        └─ 类级:   leapquant_extend_model_state_classmethods
       │                (payload.py:346) 包装模型类配置期类方法   # 见下
       └─ update_block_size_for_backend
            └─ site-packages vllm/platforms/interface.py
                 _align_hybrid_block_size (871-880)         # 用【类方法】算 mamba_page_size
                 (950-957) 钉 cache_config.mamba_page_size_padded
  └─ allocate_kv_cache_main → _reshape_mamba_kv_cache → _adjust_dsv4_kv_layout
       └─ MambaSpec 3 分量页 → 每层 bound kv_cache 的 caches[2] = payload uint8 视图
  └─ 运行期 flush 站点: leapquant_payload_pool_for_layer
       (payload.py:390; 以 view.data_ptr()+形状 为键缓存在 layer 上, rebind 才重建)
```

**为什么要类级包装（e2c9c5a3b 修的 bug）**：平台在 worker 存在前、用**模型类类方法**
`get_mamba_state_{shape,dtype}_from_config` 计算混合页尺寸并钉死
`mamba_page_size_padded`。只做实例级扩展时，真实（实例级）MambaSpec 长过了钉死的 pad，
worker 启动即触发 `MambaSpec.page_size_bytes` 断言 `page_size_padded >= page_size`。
`leapquant_extend_model_state_classmethods` 把同样的
`((heads, head_block_bytes), uint8)` 追加进类方法返回值，让配置期算术看到真实页。
关闭 LeapQuant（或 AscendConfig 未建）时类方法逐字节透传，行为不变。
幂等（类 `__dict__` 标记，`_leapquant_state_extended`）；定义了自己类方法的子类不被跳过。

应用点（4 个类）：

| 文件:行 | 类 | 备注 |
|---|---|---|
| `vllm_ascend/patch/worker/patch_qwen3_5.py:229-230` | `Qwen3_5ForCausalLMBase`, `Qwen3_5ForConditionalGeneration` | GDN graft 之后 |
| `vllm_ascend/models/kimi_k3.py:1309-1310` | `AscendKimiLinearForCausalLM`, `AscendKimiK3ForConditionalGeneration` | **陷阱**：upstream K3 的类方法直接委托给 upstream `KimiLinearForCausalLM`（vllm/models/kimi_k3/amd/model.py:232-241），只包 Ascend Linear 类盖不住 K3，两个都包 |

---

## 5. 窗口语义 M3.3（进行中）

PR1 语义（`seq % w == 0`）对 decode 精确，但 chunk 跨内部边界而末端不对齐时该边界当步不 flush
（PR #21120 评审讨论点）。M3.3 换成**跨界判定**（数学与代码见姊妹文档 §7）：

$$\text{due} \iff \lfloor \mathrm{seq}/w\rfloor > \lfloor(\mathrm{seq}-q)/w\rfloor$$

- 已落地：
  - `state.py:205 crossed_boundary_mask`；
  - `runtime.py:88-136` `query_lens` 参数（不传 = PR1 语义，向后兼容）；
  - `flush_kernel.py` 逐行 hcnt：kernel 参量 `hcnt_vals_ptr`（267）+ store（348），
    wrapper `hcnt_values`（368、441-450，int32/同设备/形状校验，标量 `hcnt_value` 仍可用）。
- payload 侧 `hcnt` 存 `seq % window`（窗口相位）：边界对齐 flush 存 0（新窗口），
  chunk 末端落在窗口中间的 flush 存其偏移——readout（M3.4）据此知道 checkpoint 折入了多少 token。
- 待接线：builder `_attach_leapquant_metadata`（`gdn_attn_builder.py:1096-1106`）追加
  `leapquant_query_lens = torch.diff(query_start_loc_cpu)[:num_decodes+num_prefills]`；
  `gdn.py:991` / `kimi_kda.py:777` 站点传 `query_lens=`，payload 路径把
  `seq % window` 经 `hcnt_values` 传入融合 flush。

---

## 6. triton-ascend 硬件/编译器约束清单

本 build（Ascend AIV / CANN BiShengHIR 前端）上"交学费"换来的约束，每条附规避代码位置：

**(a) UB 192 KB——整块放不下**：`[V,K]=128×128` fp32 全块多缓冲 ~700 KB。
全部 pass 按 `BLOCK_V×K` / `V×BLOCK_K`（32）流式化（scales kernel 三个循环体）。
融合 dual-write（encode+reconstruct 合一、两套 store 同时活）同样溢出 UB——
所以 dual-write 是**两次独立启动**（`flush_kernel.py:542-563`）。

**(b) 寄存器 fp16 cast 是 no-op**：`x.to(tl.float16).to(tl.float32)` 不发生舍入（fp16 疑似 fp32 寄存器仿真）。补偿因子的 fp16 舍入只能发生在 **fp16 store**：
`qv`/`uk` scratch 是 fp16 内存（每个消费者都 load 到真正舍入过的值，与参考一致），
另设 fp32 `qvr` raw 平面给 `uk = Sᵀqv` 这个必须用**未舍入**因子的 dot（164-166 行注释）。

**(c) axis-0 提取静默清零**：`tl.where(cond[:,None]==r, block, 0)` 及其 `trans` 组合会错低化为零；
只有 axis-1 模式 `tl.sum(tl.where(offs_r[None,:]==r, block, 0.0), axis=1)` 在 `[D,16]` 块上可靠。
因此每个子空间向量都住进连续 `[D,16]` 平面、只以该取向读取（`_mgs_cols` 98/102 行、
scales 185-187 行、reconstruct 235-237 行、encode 314-316 行），低秩乘积是手工展开的
R 次外积而非 dot。

**(d) 两个结构相同的 `tl.dot` 循环 → BiShengIR SIGABRT**（ConvertLinalgIRToBinary 崩溃）。
scales kernel 三个 dot 循环体刻意结构差异化（累加量/掩码/trans 用法各不相同）。

**(e) `static_range(R)` 全展开 × 大块 → 前端编译 ~1 小时**：runtime 循环 + 小 loop-carried
累加器保持 IR 规模（这正是 R 次 unrolled 外积小到可接受的原因）。

**(f) int8 cast 用 `overflow_mode="saturate"`**（encode 335）：值已 clamp 进 `[-127,127]`，
saturate 是防御性 no-op（swiglu_quant.py 先例）。

**(g) `tl.dot` N ≥ 16**：rank 补到 16（`_LEAPQUANT_RANK_PAD`，73 行），padding 列恒零。

---

## 7. 实测数字（Qwen3.5-0.8B, TP1, bf16, A3, M3.2 冒烟）

| 量 | payload 关 | payload 开 | 变化 |
|---|---|---|---|
| mamba 内容页（conv+ssm） | 561,152 B | 872,512 B | +311,360 B = 16 head × 19,460 B（**+55.5%**） |
| attn block | 640 tok | 768 tok | 混合页对齐连带 |
| mamba pad 浪费 | 20.75% | 12.60% | payload 搭了既有 padding 的车 |
| 净 KV 容量 | 2,709,094 tok | 2,508,117 tok | **−7.4%** |
| flush 墙钟（dual-write） | 基线 | +16.6% | readout 切掉 fp32 读后回收 |

精度：gsm8k 15 样本 10/15 与基线一致、8 段 drift 文本 2-5 段一致——恰为跨腿噪声底
（任意两条 0.8B 腿的逐对一致率同分布；批重排敏感所致）；mcq 200/200 逐位相同 ⇒ 无回归。

## 8. 测试

| 文件 | 覆盖 |
|---|---|
| `tests/ut/quantization/methods/a3_2/test_leapquant_flush_triton.py` | kernel ↔ torch 参考 bit 级钉子（误差比 ≤1.01 + ≤2 int8 步）；roundtrip/payload/dual-write 三模式 |
| `tests/ut/quantization/methods/a3_2/test_leapquant_payload.py` | 布局字节表、跨步池视图、状态扩展、模型类方法包装（含子类/幂等）——16 用例 |
| `tests/ut/quantization/methods/a3_2/test_leapquant_gdn_integration.py` | NPU GDN 站点端到端 |
| e2e | `e2e/wait_and_run_payload.sh`（卡 4-7 空闲窗口 + spawn + 45 s 稳定期 + 抢卡重试） |

## 9. 路线图衔接

- **M3.4** `recurrent_leapquant` readout（论文式 10）：decode 侧直接从 payload 读
  （codes·(s_k/127)·s_v + qᵀu + hcnt 相位递推），替代 fp32 双写——GDN 先行再复制到 KDA；
  届时回收 §7 的 −7.4% 容量与 dual-write 墙钟。
- **M3.5** 集成加固 + PR2 装配（含 playground 全部探针清理：`kimi_k3.py` `_KV_PROBE`/
  `_page_nan`/`FIA2-ROWLOOP`/`NAN-WATCH`/`NAN-DUMP`/`LQ-FLUSH-DIAG`、`mla_v1.py`、
  `model_runner_v1.py` `POOL-DUMP`、`kimi_linear_compat` 探针）。
- chunk 边界原生编码（`chunk_boundary_quantize`，chunk 循环内逐窗口出 payload）列于
  `LEAPQUANT_PR_PLAN.md` §2，M3.3 的跨界 flush 语义是其前置。
