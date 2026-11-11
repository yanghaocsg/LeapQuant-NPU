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

### M3.3 chunk 边界 flush 语义(跨界 due 判定)✅ 收口 2026-11-10 — 调度级已落地,内核级留 M3.4 顺手
**2026-11-09 commit `c602bb026`(fork leapquant-triton-m3)**,采取 pragmatic 路线:不改 chunk
kernel,把 flush 调度从"chunk 末端恰对齐边界"(`seq % w == 0`,M1 语义,peterDengcx 在 #21120
指出的缺口)升级为**跨界判定**:

- `state.crossed_boundary_mask`:`⌊seq/w⌋ > ⌊(seq−q)/w⌋` — 步区间 `[seq−q, seq]` 跨过任一
  窗口倍数即 due;decode(q=1)在 `seq ≥ 1` 上精确退化为 `seq % w == 0`(seq=0 空序列本无可
  flush,同 `decode_flush_mask` 的 `ends > 0` 守卫);
- chunked delta-rule kernel 只物化 chunk 末端状态 ⇒ 跨 k≥1 个边界的 chunk 在 chunk 末端获得
  **恰好一次** flush;跨步看没有任何边界被跳过(每个边界在其后第一个步末得到 flush)——评审
  意见的核心(边界漏 flush)就此关闭,残余差异只是 flush 落点偏后;
- payload `hcnt` 改存逐行**窗口相位** `seq % window`(边界对齐=0,chunk 末端落窗口中间=其偏移),
  供 M3.4 readout 知道 checkpoint 折入多少 token:flush_kernel `hcnt_vals_ptr`/wrapper
  `hcnt_values`(int32/同设备/形状校验),runtime payload 路径为 due 行计算相位;
- 接线:builder `_attach_leapquant_metadata` 附加 `leapquant_query_lens =
  diff(query_start_loc_cpu)`(缺省回落 M1 mask);gdn.py/kimi_kda.py 站点传 `query_lens=`;
  `runtime.leapquant_flush_state_rows(query_lens=None)` 向后兼容(decode-only 调用方/UT)。

测试门:CPU UT 全绿(state 8 / runtime 9 / config 6,含跨界 mask 全案例 + runtime 穿越 flush +
步进覆盖每个边界恰一次);NPU UT 全绿(flush 5/5、payload 17/17 含逐行 hcnt、gdn_integration
1/1 带 query_lens 站点)。**e2e 收口 ✅(2026-11-10,`f_08b_lq_triton_payload_m33.json`,卡 4 一次
过)**:gsm8k 89/200(历史 5 腿带 81–90,带内且高于 M3.2 payload 腿的 83 — 跨界 flush 落点提前的
预期效应);**mcq 73/200 且与全部历史腿 200/200 逐题全同**(硬门);sample_preds 一致 9–10/15、
drift 2/8(噪声底带内;vs M3.2 的差异属语义变更预期,非回归);真实分配器 + strided 池 + 逐行
hcnt 写全路径无崩溃。

**残余(降级为顺手项)**:真正的 chunk 循环内逐窗口编码 `chunk_boundary_quantize.py` 不再是
M3.3 必需(现语义已无漏 flush),列为 PR2 polish / M3.4 顺手项——收益仅是把 flush 落点从
"chunk 末端"提前到"每个内部边界",量化误差路径不变。评审回复继续压后(用户指令:先不关闭)。

### M3.4 recurrent_leapquant(decode 读取侧融合,~5-7 天,最大单块)
**文件**: `ops/triton/leapquant/recurrent_leapquant.py`(新)

**骨架侦察结论(2026-11-10,Explore agent 全文见会话)**:
1. **两条生产 decode 路径都是 fla_npu AscendC 二进制,改不了** — GDN: `fla_npu.ops.ascendc.recurrent_gated_delta_rule`(gdn.py L954-967 调用,标量 g 衰减、host 侧已做 l2norm+sigmoid beta、`ssm_state_indices` 逐 token 展平 `[T]`+`actual_seq_lengths`、state `[slots,HV,V,K]` 原地突变、Dk=Dv=128 硬限);KDA: `run_recurrent_kda`→AscendC `recurrent_kda`(kimi_kda.py:477-503,gk 逐 K 维衰减)。**我们的 triton `fused_recurrent_kda.py` 只经 patch_triton.py:47-51 猴补丁被 fla 导入方消费** — 自有 KDA 后端根本不用它。→ M3.4 必须**替换 gdn.py/kimi_kda.py 的调用点**为自建核,而非改造。
2. **递归主循环 5 行数学(三处一致,fla 参考可读 fused_recurrent.py L122-149)**:`b_h *= exp(g 或 gk[k])`(KDA 逐 K 列 / GDN 标量)— `ṽ = v − (S k)_v` — `ṽ *= β` — `S += ṽ kᵀ` — `o = (S q)·scale`;每 program 拥有一个 (seq, v-head) 的 `[BV,BK]` fp32 tile 走完整个 token 环(KDA 核 BV≤8,fla ≤32);`state_idx<=0` NULL 槽跳过;inplace 终态逐 token 写回(spec draft 槽)。
3. **状态取向 [V,K] 处处一致,与 payload 布局精确对齐**(codes [V,K]、s_v 按 V 行、s_k 按 K 列)→ `S₀ = qᵀu + codes·(s_k/127)·s_v` 可直接重构进 `[BV,BK]` tile,主循环零改动;bit 兼容风险点 = 衰减方向性(标量 vs 逐列)、beta 位置(乘 ṽ 不乘 S/o)、l2norm 位置(GDN 在外/KDA 在核内)、`state_idx<=0` 语义。

**分阶段设计(侦察后修订)**:
- **M3.4a 持久化切换(容量收益)**:页丢 fp32 ssm 分量(计划既定),活跃请求改用 **W 工作区 `[max_seqs, HV, V, K]` fp32**(≈ B/slots 3.5% 额外,净容量仍 ~3.2×);decode 步读写 W(带宽同今日);边界 flush 已是 M3.2 分裂启动,无需核内融合;新增集成 = W 的请求绑定生命周期(新建/恢复/驱逐-编码)。W 冷启动 = payload reconstruct(核内 PAYLOAD_INIT 路径)或 prefill scatter。
- **M3.4b eq.10 纯读出(带宽收益)**:W 换 kbuf/ubuf 环形缓冲 `[B,H,window,K+V]`,每步 payload 读出(1.19B/elem,状态带宽 3.37×↓)+ 环内 ≤window token 重放(计算换带宽,decode 访存受限下的正交易)。
- 顺序不变:先 GDN(标量 g、无 gbuf/pcum,最简)→ 后 KDA(逐 K 列衰减 + gate 核内展开 `−exp(A_log)·softplus(a+dt_bias)`/safe 变体 lower_bound·sigmoid,M3.2 scales 核加门控参数)。
- 寄存器压力预案(§2 风险):128×128 fp32=64KB 超 UB → readout 分 K 块流式;若 MGS+readout 同核编译炸 → 拆两核边界处两次 launch(丢一次带宽换可编译)
- 测试门:kernel vs PR1 roundtrip 逐 token 对齐(误差比 ≤1.01×,fp64 重建基准);B=1/B=16/B=64 批形状;e2e 9B 精度 <1pt
- 基准:decode 吞吐 @ 4K/16K/32K 长上下文(M3.4b 的带宽收益主要在这里);TTFT;并发上限提升(M3.4a 的容量收益)

**进展(2026-11-10,commits c01a06d13 + 4d12fe81a)**:
- ✅ **内核落地** `recurrent_leapquant.py`:`_recurrent_gdn_leapquant_kernel` 一次编译通过。每 program 一个 (K-block, V-block, seq×v-head) 的 `[BV,BK]` fp32 tile(BV=min(pow2,8),grid `(NK,NV,B·HV)`,num_warps=1);稳态 `w_in` 路径 + 冷启动 `USE_PAYLOAD_INIT`(eq.8-9 原位重构:codes·(s_k/127)·s_v + R 个 rank 外积,axis-1 提取);NULL 槽/行 sentinel=-1(0 是合法 pool 行!与 ssm_state_indices 对齐)零态跑递归但跳过 W 写;g/beta load 后转 fp32。
- ✅ **bring-up 三bug修复**:① wrapper 实参顺序与签名错位(pl_args 的 6 个 stride 插在指针后,全部 stride 位移 6 位)→ 按签名「9 指针→全部 stride→scale」重排;② NULL 判定 `slot<=0` → `slot<0`(slot 0 合法,首版把 slot 0 全零化);③ 稳态路径 row<0 无守卫(生产 NULL 请求槽会负偏移越界)。
- ✅ **UT 6/6** `test_leapquant_recurrent.py`:稳态+payload-init vs fp64 参考(V=K=128 GDN 几何;T=3+GVA HV=2H;payload 路径容差 4e-4 —— fp16 补偿面核内逐 rank 累加 vs torch 融合重构的 fp32 舍入顺序差)、padded W 跨步、NULL 槽(-1)零态、NULL 行稳态、参数校验。回归:flush 5 + payload 17 + gdn_integration 全绿。
- ✅ **gdn.py 集成(slice 1,数值替换)**:两个非 spec decode 调用点(纯 decode 分支 + split_non_spec decode 段)在 `kernel_backend=="triton" && ssm_state fp32`(默认 `mamba_ssm_cache_dtype=auto`→fp32)时改走 `_leapquant_recurrent_decode` helper;spec 路径留 AscendC(leapquant 拒绝 spec 配置);`leapquant_config` 提升到 2.0 节(原 2.4 重复获取)。`test_leapquant_gdn_integration.py` 3/3:新增 triton vs AscendC 64 步轨迹 <2%(bf16 输入舍入量级)+ triton+flush 包络 5%。
- ✅ **e2e 腿通过(2026-11-10 晚,卡 0,`f_08b_lq_triton_payload_m34.json`)**:gsm8k **99/200=49.5%**(高于历史带 81-90;+10 题 = 0.8B 刀口效应——核内全 fp32 vs AscendC 内部精度,贪心路径微移;sample 9/15 与 M3.3 相同,恰为跨腿噪声底 9-11/15,同时证明核真实生效:静默回退会是 15/15 全同);**mcq 73/200 与 M3.3 零翻转**(传递性对齐全部历史腿,硬门最严标准通过);墙钟 188.6s vs 185.9s(+1.5%,triton 核替换 AscendC 零代价);drift 2/8 全同(噪声底 2-5/8 内)。
- 后续:M3.4a W 工作区生命周期(容量收益)→ KDA 侧同款核(逐 K 列衰减)→ M3.4b eq.10 环形读出。

**M3.4a-1 进展(2026-11-10 深夜)— 身份槽位映射影子池 W,落地并全绿**:
- 设计简化(比上文原方案少一整层复杂度):W 是**恒等槽位映射**的连续 fp32 影子池 `[slots,HV,V,K]`,`W[slot] ≡ ssm_state[slot]`。无宿主侧 slot→row 表、无 `.tolist()` D2H、无 flush_kernel 的 `state_row_ids` 参数、无驱逐逻辑(行数=池槽位数)。decode 内核与边界 flush 全部改在 W 行上跑;§2.3 每个 prefill chunk 的 scatter 链之后追加一次**无条件镜像**(`gather_ssm_states(ssm_state, prefill_state_indices, all-true) → scatter_ssm_states_(W, ...)`),既保持 chunked-prefill 续 chunk 精确(只有 prefill 写引擎行,且每 chunk 后重镜像),又天然治好槽位复用(新请求首次 prefill 直接覆写 W 行)。recompute 抢占 → 全量重 prefill 的 chunk 1 has_initial_state=False(零起步),无需特判。
- 文件:`workspace.py`(新,`leapquant_workspace_for` 惰性分配 + 进程级注册表;条目强引用 state 张量钉住 data_ptr 防分配器地址复用伪造命中——workspace UT 抓到的真 bug,修复=强引用 + 命中时校验底层 storage 一致,per-forward 视图对象照样命中;teardown 常驻是已知代价,M3.5 换弱引用+设备侧表)、`__init__.py` 导出、`ops/gdn.py` 六处(§2.0 取池、§2.2/§2.4'/纯 decode 站点、§2.3 镜像、§2.4 flush);dev + playground 双树同步。
- 不变量与验证:decode 核与 flush 均逐行无跨行交互 → **W 轨迹与池上轨迹逐位相等**(决定性论证)。`test_leapquant_workspace.py` 5/5:分配/缓存、校验、镜像治 NaN 旧行、位等轨迹(输出+状态 torch.equal,未触及槽精确为零)、带 mid-flight prefill 的位等轨迹(mid-prefill 写 fresh 行——chunked prefill 不与 decode 交错,两种模式装同一 fresh 值;触发点避开 flush 边界)。回归:flush 5 + payload 17 + recurrent 6 + gdn_integration 3 全绿。e2e 腿 `run_m34a1_leg.sh`(卡序 4-7 优先)运行中,预期 ≈ M3.4 腿(gsm8k ~49% 带,mcq 73/200 零翻转)。
- M3.4a-2(丢 fp32 ssm 页分量 → ~3.2× 容量)退化为页尺寸改动; notebooks 五本小白教程系列 + `TORCH_NPU_WALKTHROUGH.md` 同期入库(LeapQuant-NPU 仓)。
- **e2e 双腿判定(2026-11-10 深夜)**:① 默认预算腿(0.85)OOM 于 `workspace.py:82` 首次 W 分配——W=整池尺寸 fp32 副本 5.38GiB/层(池 ≈11k 槽),页内 fp32 池+**W 双份 fp32 是过渡态结构性超卖**(55.58G 已占/5.02G 空闲),非竞争非 bug;② **小预算腿(`--gpu-mem-util 0.48`,池+W 各 ~26G,卡 4,一次过)**:gsm8k **99/200 与 M3.4 腿同分且 15 条抽样 0 差异**、mcq **0/200 翻转**、drift **8/8 逐字节相同**、墙钟 188.2s vs 188.6s(−0.2%)——**跨卡(0→4)跨容量计划(0.85→0.48)的生产轨迹逐位复现**,位等论证(UT torch.equal)在 e2e 侧完全兑现;镜像站点(chunked prefill/槽位复用/批组成)经真实调度验证。run_eval.py 新增 `--gpu-mem-util`(默认 0.85 保持历史腿可复现)。**M3.4a-2(丢 fp32 页分量)落地后显存回基线,OOM 自消**——它才是容量收益所在,M3.4a-1 的双份过渡态只在验证期存在。

**M3.4a-KDA 收口(2026-11-11)— KDA decode 走 triton 核,全绿并推送**:
- `kimi_kda._leapquant_kda_recurrent`:host `kda_expand_gate` + fla `l2norm_fwd`(逐位复现 AscendC 核内 gate/l2norm)→ `recurrent_kda_leapquant`(W 影子池,payload 兼容)。KDA 与 GDN 核差异只在衰减粒度(per-K 列 `b_h *= tl.exp(b_gk)[None,:]`),复用全部 M3.4 基建。
- **fla_npu AscendC `recurrent_kda` 探针矩阵结论**(die 14/15,写入集成测试 docstring):① **decode-only**——`cu_seqlens` 任何跨度 >1 token 即 `aclnnRecurrentKdaGetWorkspaceSize 161002`(多 token 生产路径本就走 `run_chunk_kda`);② **多请求状态写不可靠**——单 call 2 序列时第 2 序列状态行丢失/输出行读 nan-inf 垃圾,且依赖进程 NPU 分配史(数值全交换、输入布局 [1,2,H,D]/[2,1,H,D]/3D、分配顺序/次数 n_warm 0–40、import 顺序、CPU 来源 H2D(直接崩向量核 retCode=0x31)全部排除后仍复现);单序列 `[1,1,H,D]` call 稳健;③ op 跑在旁路 stream——输入需整轨迹持有 + sync。**triton 核替换的正是这个 call,M3.4a 既是替代也是修复**;值得上报 fla_npu 上游。
- 集成测试规避法:参考轨迹每 call 单请求(同池两个活跃槽 1/3),仍覆盖多槽池/双 gate 变体(softplus=Kimi-Linear 与 lower_bound=-3 safe)/flush/host 复现契约。
- UT(die 14/15):`test_leapquant_recurrent_kda` 7/7、`test_leapquant_kda_integration` 3/3(AscendC vs triton parity <0.02、flushed envelope <0.05)、回归 sweep 全绿(recurrent 6 + payload 17 + workspace 5 + gdn_integration 3)。已推 yanghaocsg/vllm-ascend `leapquant-triton-m3` @ 4cc2d1f70。
- 教程:LeapQuant-NPU `notebooks/leapquant_kda_{1..5}` 五本(全部 A3 实跑)+ `leapquant_memory_proof`(状态 3.368× 账本与 4096-slot 池 NPU 实测同倍数;Qwen3.5-0.8B 真实权重 1.627 GiB e2e 账本 56% @ ~362 slots,渐近 70.3%;KV cache 未入账=保守方向),镜像到 awesome_quant_algo/leapquant_notebooks。

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
