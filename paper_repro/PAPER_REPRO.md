# LeapQuant 论文实验复测 (A3 NPU)

论文: *LeapQuant: Efficient Linear Attention with Accurate Recurrent State
Quantization* (arXiv 2609.38166)。本文档在 Atlas A3 上逐条复测论文实验,
按"论文数字 → A3 复测口径 → 结果"组织。GPU 专属绝对数字(B200/RTX 的
2.05–3.70× 内核加速、1.47× e2e)在 A3 上不可复现,标注为 N/A 并给出 A3
现状;所有精度/数值/显存实验可复测。

运行环境: vllm-ascend `leapquant-triton-m3`(dev 996662883 + playground 同步),
A3 die 14/15, `paper_repro/` 脚本即复测入口。

## 实验矩阵与状态

| # | 论文实验 | 论文数字 | A3 复测 | 状态 |
|---|---|---|---|---|
| E1 | 表1/5 下游精度 (8bit) | 12 模型任务对平均 75.5% = FP32;per-token INT8 26.3% | 9B AIME2026+GSM8K+MCQ 四腿梯子(base/naive/w16r0/full) | 腿运行中 |
| E2a | 图3 状态 MSE vs 长度 (64K) | bf16-w16 ↓100×,int8-w16 ↓39× | 合成轨迹 64K,生产量化器 | ✅ int8 35×(论文 39×);bf16 13×(偏弱,见注) |
| E2b | 图5 奇异值能量 | top-12 ≥80% (中位) | 合成: 64% ✗ → 真实 9B 状态重测 | 真实状态运行中 |
| E2c | 图6 离群值 | 176× → 47× → 5.3× | 同上(层13头16 探针) | 真实状态运行中 |
| E2d | 表3 p/r 扫描 | p=16/r=4 饱和 | 16K 合成轨迹 p/r 扫描 | ✅ p≥16 饱和;r 在合成谱上价值小(待真实状态) |
| E3a | 状态内存 3.4× | 1.19 B/elem | payload 账本 + 4096-slot 池 NPU 实测 | ✅ 3.368×(memory notebook) |
| E3b | e2e 显存 56% | 前缀缓存场景最高 ↓56% | 0.8B/9B/35B/Kimi 几何账本 + slots 扫描 | ✅ 56% 落点 362/1501/4474/9169 slots,渐近 70.3% |
| E3c | 并发 +1.4× | 单卡并发请求数 | 纯状态池口径 3.37×(9B@64GB: 981→3306 slots) | ✅(口径注:不含 KV/权重混合稀释) |
| E4a | 图7 内核吞吐 B=512 | GDN 2.4–4.3× (GPU) | flush torch/triton/payload × B 扫描 + decode 步 | 部分(b16 2.86×;B≥64 0.6–0.9× — 见发现) |
| E4b | 图8/10 解码吞吐 | 1.22–1.57× (GPU) | e2e 腿墙钟 + M3.4b 后重测 | M3.4b 前仅持平(设计使然) |
| E4c | 图9 e2e 吞吐 1.47× | GPU | 同上 | N/A(GPU 口径) |
| E5 | 6/4 bit | 72.4%/60.4% | bits=8 内核(int8 codes);低比特 = 后续工作 | 未测(超出当前内核) |
| — | 表2 消融梯 | 7.1→82.4→86.6→87.9 | E1 四腿即消融梯(naive→+window→+comp+smooth) | 腿运行中 |

## 已落地结果

### E2a — 误差累积定律 (`e2_numerics.json`, `e2_fig3_mse.png`)

64K 步 GDN 轨迹(16 头 × [128,128],真实衰减统计),生产量化器:

- per-token INT8 发散,w16 INT8 **改善 35×**(论文 39×)— 主工作点吻合;
- per-token bf16 → w16 bf16 改善 13×(论文 100×)。注:合成随机游走状态
  的 bf16 舍入相对更小;论文轨迹为真实模型状态。int8 工作点不受影响;
- p 扫描(1→128)单调下降,p≥16 后边际收益 <2.5×/8×window — 与论文
  "p=16 精度饱和"一致;r 扫描在合成谱上价值小(top-12 能量仅 64%),
  待 E2b' 真实状态重测。

### E3 — 内存几何 (`e3_memory_geometry.json`)

| 模型 | 线性层×头 | 权重 | slot fp32/LQ | 3.368× 恒定 | 56% 落点 |
|---|---|---|---|---|---|
| Qwen3.5-0.8B | 18×16 | 1.63 GiB | 18.0/5.34 MiB | ✓ | ~362 slots |
| Qwen3.5-9B | 24×32 | 17.98 GiB | 48.0/14.25 MiB | ✓ | ~1501 slots |
| Qwen3.5-35B-A3B | 30×32 | 66.97 GiB | 60.0/17.82 MiB | ✓ | ~4474 slots |
| Kimi-Linear-48B | 20×32 | 91.50 GiB | 40.0/11.88 MiB | ✓ | ~9169 slots |

E3c: 9B TP1 @64GB die,纯状态池前缀容量 981 → 3306 slots(3.37×)。
论文 1.4× 是整请求(KV+状态+权重)混合口径,3.37× 是状态侧上界。

### E4a — 内核基准 (部分, `e4_kernel_bench.json` 补全中)

flush(窗口边界 checkpoint,due rows × H=32):

| 形状 | torch | triton | payload | triton/torch |
|---|---|---|---|---|
| b16 | 6.62 ms | 2.31 | 2.36 | **2.86×** |
| b64 | 7.98 | 9.25 | 9.39 | 0.86× |
| b256 | 23.57 | 37.16 | 37.72 | 0.63× |
| b512 | 44.44 | 73.65 | 74.86 | 0.60× |

**发现(待办进 M3.5)**: 融合 triton flush 在 B≥64 反超不过 torch 路径 —
大 batch 下 scales launch 的并行划分成为瓶颈,需要 batch 维重划分。
历史"5.9×/3.8×"数字出自小 due-row 形状,大 batch 场景需重设计后再引用。
decode 步(AscendC vs triton GDN/KDA)结果补全中 — 当前为 parity 替换
(都读 fp32),int8 读出收益在 M3.4b 环形读出落地后兑现。

## 复现入口

```bash
# E2 数值定律(合成轨迹)
ASCEND_RT_VISIBLE_DEVICES=14 PYTHONPATH=/workspace/y00924007/vllm-ascend \
  python3 e2_numerics.py
# E2 真实状态(9B HF 前向 8K enwiki → 谱/离群值/r 扫描)
ASCEND_RT_VISIBLE_DEVICES=14 python3 e2_real_states.py
# E3 内存几何(纯 CPU 账本)
python3 e3_memory_geometry.py
# E4a 内核基准
ASCEND_RT_VISIBLE_DEVICES=15 PYTHONPATH=/workspace/y00924007/vllm-ascend \
  python3 e4_kernel_bench.py
# E1 精度梯(9B, die 15, 数小时)
bash run_9b_legs.sh   # → e1_9b_{base,naive,w16r0,full}.json
```

## 口径差异声明(诚实账)

1. GPU 绝对加速数字(B200/PRO6000/5090)不可在 A3 复现;A3 给同构实验。
2. E2 合成轨迹复现"定律",谱结构(fig5/6)必须真实状态 → E2b'/E2c'。
3. AIME max_tokens 8192(论文 ~80K):截断对四腿同等,finish_reasons 记录。
4. naive 腿 = window=1 + r=0(平滑仍开,引擎最小档):比论文裸 per-token
   INT8 略强,对梯子结论保守(如果 collapse,论文的更裸版本更 collapse)。
5. E4b/图8 解码吞吐:M3.4a decode 仍读 fp32 W(影子池过渡态),设计上
   与 AscendC 持平;M3.4b 丢 fp32 读 int8 后才是论文口径的收益点。
