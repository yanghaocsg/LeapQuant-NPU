# PR2 (M3) triton-ascend 融合开发规划

**日期**: 2026-11-07 · **分支**: `leapquant-triton-m3`(已推 fork,commits `5d6fdf014` + `0b5bad1a5`)
**上游 PR #21120**(torch M1)评审中;本规划覆盖 RFC #21097 M3 剩余全部工作。

---

## 0. 基线(已完成,2026-11-06/07)

| 项 | 状态 | 证据 |
|---|---|---|
| flush 融合核(GDN,双 launch:scales→reconstruct) | ✅ | `ops/triton/leapquant/flush_kernel.py`,UT 5 绿(error-ratio ≤1.01× + 逐元素 ≤1 码步) |
| `kernel_backend` 配置面(torch 默认/triton 选配)+ 回落 | ✅ | `leapquant_config.kernel_backend`,非 A3 / triton 不可用→自动回落 torch |
| kernel 级 bench | ✅ | 0.8B 形状 5.9×,9B decode 3.8×,64 due-rows 0.77×(回退 torch) |
| e2e(0.8B/9B,triton 腿) | ✅ | 9B GSM8K 93.5%(base 93.0),MCQ 200/200 零翻转,flush 417→31s(13.4×);0.8B 噪声带内 |
| **KDA 几何验证(2026-11-07 新增)** | ✅ | `[slots,H,V,K]`(H=8/4,V=K=128)triton vs torch rel≈1.1e-3 — **Kimi/35B 可直接用 triton backend**,两腿已启动 |
| torch 腿开销(动机数据) | ✅ | 35B +105%(268→551s),Kimi-48B +9.1%(578→631s),9B flush 417s(已由 triton 解决) |

**已知 triton-ascend 约束**(flush_kernel docstring 已记录,后续 kernel 直接沿用):
AIV UB 192KB→分块流式;两个同构 `tl.dot` 循环→BiShengIR fusion SIGABRT;axis-0 where 抽取/trans(where) 静默错零(只用 axis-1 `[D,16]` 平面);`list.append` jit 内不可用;static_range R 展开×规约→编译 ~1h(用 runtime 循环+小累加器);int8 cast 用 `overflow_mode="saturate"`(swiglu_quant.py:55 先例);**寄存器内 `x.to(fp16).to(fp32)` 编译为 no-op(fp16 以 fp32 寄存器仿真)— fp16 舍入只能靠 fp16 显存 store 完成**(M3.2a 发现:scratch 改 fp16 显存 + 独立 raw fp32 平面供 `uk=S^T qv` 用,详见 flush_kernel 模块注释);REWRITE 双写分支(编码+重建两组 store 同核)在 128×128/R=4 超 UB(≈206KB>192KB)→ dual 模式用第三 launch 复用 reconstruct 核(bit 级等价)。

---

## 1. 目标与验收(RFC §2.5)

1. **精度**: 与 PR1 相同口径不回退(GSM8K <1pt,MCQ 翻转数对称);
2. **性能**: 长上下文 decode 吞吐可测提升(flush 开销归零化 + 状态池读取带宽 3.4×);
3. **显存**: 状态池 fp32→payload ≈3.4× 缩减(9B 单卡 2-3GB→~0.7GB,并发上限提升);
4. **安全**: 默认关闭;A3-only;任何新 kernel 失败→回落 torch(M1)路径。

---

## 2. 里程碑分解(按依赖排序,每个独立可交付/可 gated)

### M3.0 快赢(今天,~0.5 天)— 纯配置复测,零新代码
- [x] KDA 几何 UT(已过,rel 1.1e-3)
- [x] 35B `--backend triton` 腿(2026-11-07):**性能大胜,精度待判** — GSM8K 305.9s(torch 550.9s,base 268.3s)→ flush 开销 +105%→+14%;但 GSM8K 95.0%(base 97.5,torch LQ 96.5)且 MCQ 3 翻转(torch LQ 0)— **open question**,排查中:
  - [x] 核完整性 UT(64 slots×H=32,子集 flush):non-due 行 bit 级不变,due 行 rel 1.1e-3 → 无 OOB
  - [x] 串行 B=1 MCQ A/B(f_35b_mcqser_*.json):base vs triton-LQ **0/200 翻转**(逐题一致,均 165/200)→ triton 路径无串扰;**且 B=1 base vs B=16 base = 3/200 翻转** → 原生引擎本身就有 ±3 题的批组成数值敏感(MCQ max_tokens=1 纯 prefill,与 flush 无关)→ 35B triton 腿的 3 翻转 = flush 时速差改变调度批组成的同款良性效应,**非 kernel bug**
  - [x] 35B triton GSM8K 复跑(f_35b_lq_triton_rerun.json):**95.0% 精确复现**(rerun-vs-first:MCQ 逐题 0 翻转、GSM8K 同 190/200、vs base 同 3 MCQ 翻转)→ 不是跑间噪声
  - [x] 35B triton **换题序腿**(--gsm-seed 7,f_35b_lq_triton_seed7.json):**96.0%**(192/200,308s)→ 离开 95.0 → −1.5pp(vs torch 96.5)主要是题序/边缘抽取敏感,非硬性核回退;35B triton 区间 [95.0, 96.0],vs base −1.5~−2.5pp 仍略超 <1pt 门 → 数据表标注「order-sensitive,非系统性核缺陷;待 M3.4 payload readout 后复核」
- [x] Kimi-48B `--backend triton` 腿(f_kimi48b_lq_w16_triton.json):**KDA 路径完美** — GSM8K 181/200=90.5%(与 torch 腿逐分一致,base 182);MCQ 68.0%,7 翻转与 torch 腿**完全同题**(= 量化+批组成的共同抽取,与 backend 无关);GSM8K sample 预测 torch-vs-triton 0/15 差异;墙钟 554.5s(torch 631.0,base 578.3)→ **LQ 腿比 base 还快 4%,flush 开销归零**
- 产出:PR2 描述的跨模型 triton 数据表;35B 的 +105%→+14% 是最戏剧性的前后对比

**M3.0 数据表(GSM8K acc / 墙钟 / MCQ acc)**:

| 模型 | base | torch flush | triton flush |
|---|---|---|---|
| Qwen3.5-0.8B | 40.5% | 45.0%(噪声带),flush 384s | 40.5%(噪声带),flush 23s |
| Qwen3.5-9B | 93.0% | 92.5%,flush 417s | 93.5%,flush 31s(13.4×) |
| Qwen3.5-35B-A3B | 97.5%,268s | 96.5%,551s(**+105%**) | 95.0%,306s(**+14%**)精度 open |
| Kimi-Linear-48B-A3B | 91.0%,578s | 90.5%,631s(+9.1%) | 90.5%,554s(**≈0%**,比 base 还快 4%) |

### M3.1 ✅ 已完成(见 §0 基线)

### M3.2 payload pool(存储层,~4-5 天)— PR2 的显存收益来源 ✅(2026-11-09 e2e 冒烟收口)
**文件**: `quantization/methods/mamba_state/leapquant/payload.py`(新)+ `runtime.py` 扩展
- **M3.2a ✅(2026-11-08,commit c50a99126)**:payload.py(PayloadLayout+Pool,uint8 单缓冲 head-major unpack,19460B/头块=1.1877B/elem=3.37×)+ flush 核 `_leapquant_payload_encode_kernel` + wrapper `payload=/rewrite_state=/hcnt_value=` 三模式(roundtrip/payload/dual);UT 8 绿(纯 payload:状态池 bit 不动+非 due 行 bit 零+对拍门同 roundtrip;dual:与 roundtrip 核 bit 一致)+ 旧 UT 5 绿无回退。**顺带修掉 M3.1 隐患**:fp16 因子舍入此前被编译器折叠为 no-op(scratch 一直存 raw),改 fp16 显存 scratch+raw qvr 平面后与 torch 参考严格一致。
- 布局(§2.1 表,per slot):codes int8 `[Nv,Dv,Dk]`;s_k/s_v fp32 `[Nv,Dk]/[Nv,Dv]`;Q̃v/Ũk fp16 `[Nv,R,Dv/Dk]`;hcnt int32 `[Nv]`;工作区 kbuf/ubuf/w fp16/fp16/fp32 `[Nv,16,·]`(KDA 变体加 gbuf/pcum)
- 分配路径:注意力类覆写 `get_state_shape()/get_state_dtype()/bind_kv_cache` — 上游 bind 按 shape/dtype 切片 reinterpret,同一 int8 字节池 unpack 成 payload 视图;**不动上游 MambaDType**(plugin 侧完成)
- **M3.2b ✅(2026-11-08)**:runtime 接线 + 引擎分配路径落地,两级:
  - step1(commit ec256a826):`LeapQuantConfig.payload_pool` typed 字段(validator: 必须 triton 后端;summary 可见)+ `leapquant_flush_state_rows(payload=)` 双写分支(payload 模式强制 dual-write,decode 侧 fp32 读者不受影响)+ never-silently-dropped guard(KDA gather/scatter plan 等非融合路径带 payload 直接 RuntimeError);UT 3 绿(runtime dual-write e2e / guard / config validator)。
  - step2(本次):引擎分配路径。**实测发现 Ascend 侧不走上游 `MambaPoolLayer.bind_kv_cache`**——`patch_bind_kv_cache` 直接 `layer.kv_cache = kv_cache` 裸赋值,而 `allocate_kv_cache_main → _reshape_mamba_kv_cache → _adjust_dsv4_kv_layout` 按 `MambaSpec.shapes/dtypes` 逐状态字节切视图返回 **list**(`_forward_core` 里 `self_kv_cache[0]/[1]` 即由此而来)。因此"分配路径"的正确挂点是 **给 `MambaSpec` 追加第 3 个 `(heads, head_block_bytes)` uint8 状态分量**:
    - `payload.py::leapquant_payload_state_extension(base_shapes)`:config off → None(页字节不变);on → `((heads,hbb), uint8)`,非 2 幂头维在 spec 构建期 fail-fast;
    - GDN(`ops/gdn.py::AscendGatedDeltaNetAttention`,A3):`get_state_shape` 追加 + 新增 `get_state_dtype`(显式 `GatedDeltaNetAttention.get_state_dtype(self)` 基类调用——该方法被嫁接到 Qwen3.5 模型 GDN 类,`super()` 会按嫁接目标 MRO 解析;patch_qwen3_5 A3 分支补 `get_state_dtype` 嫁接,310p 分支不动——LeapQuant 仅 A3);
    - KDA(`ops/kimi_kda.py::AscendKimiK3DeltaAttention`):同型覆写;`conv, ssm = self.kv_cache` 改索引(容忍第 3 元素);
    - `LeapQuantPayloadPool` 泛化为 **strided 池**:buffer 变 `[slots, heads, hbb]` uint8 视图(slot 步长=整页 conv+ssm+payload+pad),字段视图 `view(dtype)+as_strided` 构造(4B 对齐断言);flush wrapper 改用 `pool.head/slot_stride_bytes` 实际步长;`leapquant_payload_pool_for_layer` 在 flush 站点按 data_ptr 缓存包装(gdn.py:986 / kimi_kda.py:753)。
  - 引擎侧连锁反应:① KDA state_copy plan 的 `len(kv_cache)!=2` guard 实为 **worker 启动即 raise**(非静默跳过,初版判断有误)→ followup commit 6a51fe5ac 修复:guard 放宽为 `len<2`(plan 只消费 recurrent state),且 flush 站点在 payload_pool 开启时显式绕过 plan(payload 要求 fused triton flush,直接索引 strided `[slots,H,V,K]` 池 —— 与 GDN 同一核);**prefill 保留 plan,无 fallback 退化**;② mamba 页增大推高 hybrid 公共页(attention 页 pad 到 `common_page_size`,attn_utils.py:409)——dual-write 期的临时代价,终态(M3.4 丢 fp32 ssm 分量)页反而缩小到 conv+payload;footprint 数字 e2e 冒烟时实测进 PR 表;③ spec-decode/`mamba_cache_mode=all/align` 被 leapquant config validation 拒绝,无 slot 迁移,payload 与 state 生命周期天然一致。
  - UT 14 绿(新增 3:引擎 strided 池双写字节级检查[conv/padding/非 due 槽 bit 不动]+错位视图拒绝+state_extension helper off/on/非2幂)+ 旧 flush 5 绿 + GDN 集成 1 绿。已知环境问题:本机 site-packages 缺 `vllm.v1.core.kv_cache_planning`/`fusion.mm_input_norm`,无法直接 import `worker.v2.attn_utils` 做真分配器 UT——用 `_adjust_dsv4_kv_layout` 同构 emulation 覆盖,真实路径由 e2e 冒烟验证。
  - **step3 ✅(2026-11-09,commit e2c9c5a3b)e2e 冒烟踩出真 bug**:首跑 0.8B payload 腿在 spec 构建期 `MambaSpec.page_size_bytes` 断言 `page_size_padded >= page_size` 崩溃。根因:**平台在 worker 出现之前**就定 hybrid 公共页——`update_block_size_for_backend → _align_hybrid_block_size` 从 **模型类** classmethod(`get_mamba_state_{shape,dtype}_from_config`)算 `mamba_page_size` 并把 `cache_config.mamba_page_size_padded` 钉到覆盖它的 attn 页;step2 只扩了实例级 shape,页被 payload 撑过后钉死的 pad 反而小于内容 → 断言。修复:`payload.py::leapquant_extend_model_state_classmethods(model_cls)`(幂等,class-dict 标记使自定义 classmethod 的子类不被跳过)追加同一尾部 uint8 分量,应用到 `Qwen3_5ForCausalLMBase`+`Qwen3_5ForConditionalGeneration`(patch_qwen3_5 嫁接;0.8B 实际解析到后者)与 `AscendKimiLinearForCausalLM`+`AscendKimiK3ForConditionalGeneration`(kimi_k3.py;上游 K3 classmethod 直连上游 KimiLinear 类,单包 Linear 不够)。UT 16 绿(dev+playground 双路径)+ probe:off `((3,1536),(4,128,128))(bf16,bf16)` → on `+((4,19460),uint8)`。
  - **M3.2 footprint 冒烟 ✅(2026-11-09,f_08b_lq_triton_payload.json,TP1)**:0.8B 干净走完真分配器(attn block 640→768,mamba 页 pad 20.75%→12.60%,引擎日志 interface.py:933/957 直接可见修复生效)。**数字(TP1 bf16,per mamba slot)**:内容页 561,152B → +payload 311,360B(16 头×19,460B)= 872,512B(**+55.5% 内容**);但 payload 大部分骑进了原本浪费的 padding——实际池代价 = KV 容量 2,709,094→2,508,117 tok(**−7.4%**)/并发 661×→612×。dual-write 墙钟 157s→183s(+16.6%,含 payload 编码核+flush 双写)。**正确性**:mcq 0.365 与参考腿 200/200 pred 全同;gsm8k 0.415(腿间 band 0.405–0.450);逐对一致性 10/15 gsm8k、5/8 drift——与任意两腿间的噪声底完全一致(base vs lq_triton 也是 10/15、2/8;腿间布局差[本次 block 640→768]即重排 batch → 顺序敏感数值漂移,seed7 归因已证实该机制),非 payload 相关退化。wait-script 模式复用:外部作业两次抢卡,45s 稳定窗+冷却重试第 2 次尝试成功。
- 语义分层(关键设计决策):
  - `mode="roundtrip"`(M3.1 现状): flush 原地 fp32 重建,池仍是 fp32 — 精度基准与回退路径,永久保留;
  - `mode="payload"`(M3.2 引入): flush 只写 payload 不重建;decode 侧读 payload — 显存与带宽收益来自这里。
  - 切换由 `kernel_backend=="triton"` + `payload_pool=True`(新 typed 字段)双 gate;两模式共用同一 flush 核(scales 核写 payload,reconstruct 核仅在 roundtrip 模式 launch — 双 launch 结构天然支持)。
- 状态机:hcnt==window 触发 flush,flush 后停在 16 由 decode 步推进(HF Pool 语义);UT 单测状态机转移
- 测试门:payload 写读回环 vs torch `leap_window_flush` bit 级一致(codes/scales/补偿因子);池 footprint 前后实测进 PR

### M3.3 chunk_boundary_quantize(prefill 边界,~2-3 天)— 顺手关评审意见
**文件**: `ops/triton/leapquant/chunk_boundary_quantize.py`(新)
- 现状:prefill 跨边界的行不 flush(M1 语义,peterDengcx 在 #21120 指出);chunk 后 scatter fp32 state
- 新:chunk 输出 last_state → 直接编码进 payload(或 roundtrip 模式下原地重建),替代 scatter 后的 torch flush;复用 flush_kernel 的 scales 核,新增入口处理 per-request 边界偏移(query_len 内最后一个窗口边界)
- `prefill_flush_mask`(state.py:194 已有)提供 due 行计算
- 测试门:UT 对拍(chunk 边界行 vs 逐步 flush 的等价性);e2e MCQ 零翻转保持
- 交付后回复 #21120 评审(等用户信号)

### M3.4 recurrent_leapquant(decode 读取侧融合,~5-7 天,最大单块)
**文件**: `ops/triton/leapquant/recurrent_leapquant.py`(新)
- 改造 `kda/fused_recurrent_kda.py` 循环骨架(及 GDN `npu_recurrent_gated_delta_rule` 对应骨架):
  - 每步读 payload(1.19B/elem)+ kbuf/ubuf/w 工作区 → 论文 eq.10 readout,**不物化 fp32 全量态**;
  - 窗口边界在 kernel 内触发 flush(内联 scales 编码或调 M3.2 核);
  - 输出 attn out + 更新缓冲/counter。
- 顺序:先 GDN(Qwen3.5 全系,9B/35B 直接回归)→ 后 KDA(Kimi,gbuf/pcum 门控逐 K 维,论文附录 A;KDA flush 变体核在 M3.2 的 scales 核上加门控参数)
- 寄存器压力预案(§2 风险):128×128 fp32=64KB 超 UB → readout 分 K 块流式;若 MGS+readout 同核编译炸 → 拆两核边界处两次 launch(丢一次带宽换可编译)
- 测试门:kernel vs PR1 roundtrip 逐 token 对齐(误差比 ≤1.01×,fp64 重建基准);B=1/B=16/B=64 批形状;e2e 9B 精度 <1pt
- 基准:decode 吞吐 @ 4K/16K/32K 长上下文(payload 带宽收益主要在这里);TTFT;并发上限提升(池缩小后)

### M3.5 集成硬化 + PR2 组装(~3-4 天)
- 图捕获兼容:HF "pool 先建 + warmup 再捕获"语义 → NPU 侧 graph 模式验证(hcnt/payload 指针静态化);enforce_eager 路径保持默认
- 回落矩阵测试:A3/A3I/非 A3 × torch/triton × roundtrip/payload × GDN/KDA 全组合,非法组合拒绝或回落并 log 一次
- 文档:`docs/` 支持矩阵 + 配置说明;RFC #21097 更新(M2 typed entry 决议 + M3 完成留言)
- PR2 描述数据表:M3.0 跨模型 + M3.4 吞吐/显存 + UT 汇总

---

## 3. 排期(日历,串行关键路径 = M3.2→M3.4;M3.3 可与 M3.2 并行)

| 周 | 内容 |
|---|---|
| W1 | M3.0(当天)+ M3.2 payload pool ∥ M3.3 chunk boundary |
| W2 | M3.2 收尾(状态机 UT+footprint)→ M3.4 GDN readout 开工 |
| W3 | M3.4 GDN 完成(kernel UT+e2e 9B)→ KDA 变体 |
| W4 | M3.4 KDA + M3.5 集成硬化 + PR2 提交 |

与原计划(§2 估 3-4 周)一致;M3.1 已提前完成。

## 4. 风险与预案

| 风险 | 概率 | 预案 |
|---|---|---|
| MGS+readout 同核 UB 超限/编译失败 | 高(有先例) | 拆双核;two-pass(先重建残差再拟合);readout 分 K 块 |
| payload 布局与上游 bind_kv_cache 切片假设冲突 | 中 | plugin 侧 reinterpret 已验证思路;不行则退化为独立池+双池索引映射 |
| eq.10 readout 数值漂移超门 | 中 | fp64 基准定位;compensator_tokens 自适应升 R(4→8);最坏 mode=payload 仅用于显存受限场景并文档化精度代价 |
| KDA gbuf/pcum 门控无 HF 参考实现细节 | 中 | 论文附录 A 直接推导;HF `_kda.py` 若已发布则对拍 |
| 图捕获与 hcnt 动态推进互斥 | 低 | graph 模式暂不 payload(声明限制),eager 默认(现状) |
| 35B/Kimi triton 腿精度回退 | 低(几何 UT 已过) | 若回退:定位到 KDA 页跨步视图交互,payload 模式推迟到该模型 |

## 5. 立即行动(本规划落地后)
1. ✅ 35B/Kimi triton 腿(跑中,今天出数)
2. M3.2 payload.py 骨架:布局常量 + unpack/pack 视图 + roundtrip/payload 双模式 runtime 分支
3. M3.2 的 scales 核改造:scales 核加 payload 写出(与现有 qk_s/qv_s/uk_s 工作区结构同构 — 它们本来就是 payload 形状,只差持久化),reconstruct 核改 conditional launch
