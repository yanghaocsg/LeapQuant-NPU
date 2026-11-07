# [Draft — not posted] Hybrid (KDA + MLA) pools alias one physical slab on vllm-ascend — KDA state writes corrupt MLA KV cache

**Environment**: vllm-ascend releases/v0.30.0 (playground tree with local Kimi-Linear
low-rank-gate support patches), Atlas A3 (910_9382) ×4 TP, torch 2.10 +
torch_npu 2.10.0.post7, model = Kimi-Linear-48B-A3B-Instruct (3 KDA : 1 MLA
hybrid, 45 layers), eager mode, prefix caching off.

**Symptom**
- Prefill is correct (first token right, MCQ single-token eval 4/4).
- From a request whose KDA state slot / MLA pages land in the aliased
  region: first decode step already returns NaN / garbage rows from
  `AscendKimiMLAAttention`; corruption is continuous (values change every
  step) and propagates to all downstream layers. In batched decode this
  surfaces as "first B≥2 step NaNs row 1"; in serial decode the second
  request is corrupted (first request's state row doesn't reach the aliased
  pages).
- With finite (non-NaN) state values the corruption is silent.

**Root cause (address-level proof)**
- Hybrid (KDA + MLA) pools intentionally overlay ONE shared backing — "a
  block ID belongs to only one group at a time" — with the common physical
  page = 589824 B (0x90000) = 4 MLA kernel slots × 147456 B
  (nope 0x20000 + rope 0x4000) = mamba conv 0x4800 + state 0x80000 + slack.
- `attn_utils.get_kv_cache_spec` aligns MLA into the mamba page geometry
  (`align_kv_cache_with_mamba` default True) but leaves
  `AscendMLAAttentionSpec.page_size_padded = None`, so the V1 runner's
  tuple branch cannot know the pages are padded. Verified pool dump
  (Kimi-Linear-48B-A3B, TP4, 7335 blocks / 29340 MLA kernel blocks):
  - KDA conv view `[7335,3,3072]` and state view `[7335,8,128,128]`
    fp32, strides (294912,…) / (147456,16384,128,1), state base = slab +
    0x4800 — **page-strided by 0x90000 correctly**.
  - MLA nope view `[29340,128,1,512]` bf16 stride[0] = 65536 elts
    (0x20000) and rope view stride[0] = 8192 elts (0x4000) — **DENSE
    FLAT views** (all nope rows packed, then all rope rows), because the
    trim branch splits the raw slab into two dense halves and reshapes.
- Collision math: KDA state row r spans slab bytes [0x4800 + 0x90000·r,
  0x84800 + 0x90000·r) while MLA kernel page p lives at dense offset
  0x20000·p. Row r's tail [0x90000·(r+1), 0x84800+0x90000·r) overlaps MLA
  kernel page 4(r+1)'s slots 0–17. Probe: r1 had MLA kernel page 16
  (page 4 = row 3 + 1) — `kn_nan_slots=[1,3,4,5,7,10,11,13,…]` all < 18,
  exactly matching row idx=3's tail; `KDA-DEC->MLA-NAN … OVERLAP=True`.
- Timeline: prefix clean at prefill write → `NAN-WATCH hit at L0
  AscendKimiLinearGatedDeltaNetAttention` (first KDA layer of the first
  decode step) → NaN counts in the watched prefix fluctuate (20→32→28→26)
  across steps as the KDA state keeps being rewritten. Same bug explains
  the "batched B≥2 row-1 NaN" and the serial second-request corruption —
  both are allocation-adjacency victims of the alias.

**Fix (implemented in worker/model_runner_v1.py, tuple branch of
`_reshape_kv_cache_tensors`, just before the page-padding trim)**
- When `hybrid_with_attn_and_mamba` + `use_hybrid_blocks` + single raw
  backing + `AscendMLAAttentionSpec`, skip the dense trim/split and build
  the same page-strided views the single-raw MLA path uses:
  `slot_bytes = page_size_bytes // kernel_blocks_per_manager` (= 147456),
  nope view at slot offset 0, rope view at offset `kernel_block × heads ×
  nope_dim × dtype_size` (= 131072), dim-0 stride = slot_bytes. This
  restores the shared-page contract stated in the MambaSpec branch
  ("both views must advance by the same padded page size"); nope+rope =
  147456 fits the slot exactly.
- Verified arithmetic: 29340 × 147456 = 4,326,359,040 = layer slab bytes
  exactly; same layout `_reshape_combined_mla_kv_cache` builds on V2
  (the V2 runner was never affected — its bare-tensor MLA branch already
  routes through that helper; only the V1 tuple branch diverged).

**Verification (serial, max_num_seqs=1, 3 fixed prompts, TP4)**
- Pool dump after fix: MLA nope view `[29340,128,1,512]` stride[0] =
  **73728** elts (= 147456 B kernel-slot stride), rope view stride[0] =
  73728 at slot offset +131072 B — page grid now identical to the KDA
  state view's.
- Second request r1 (the deterministic `"Imagine!!!"` victim) now
  produces a **coherent TCP/UDP answer**; all three requests coherent;
  `FIA2-NAN` (op-output NaN) = 0 and `PREFIX-NAN` (write-side) = 0 on
  all 4 ranks.
- Residual instrumentation alarms are benign and diagnostic: r1's MLA
  page contains ~100 NaN elements in its **unwritten tail** — stale
  bytes from the physical page's previous life as request r0's KDA
  state row (block id freed as mamba, reallocated as attention; legal
  in the shared-pool design, never zeroed). The V1 decode path's
  `_sanitize_unwritten_paged_tails` zeroes rows ≥ kvlen before
  `npu_fused_infer_attention_score_v2`, which ingests beyond-kvlen
  slots of block-table pages — so the garbage never reaches the op
  (hence FIA2-NAN = 0). Two separable upstream issues, both now
  evidenced: (1) the dense-view aliasing (fixed by page-strided
  views), (2) cross-type page reuse + FIA v2 beyond-kvlen ingestion
  make tail sanitization (or free-time zeroing) required.

**Earlier evidence chain (pre-root-cause, kept for context)**
1. op-level wrap of `npu_fused_infer_attention_score_v2`: first B=2 call
   has finite inputs, sane block_table/kvlen/layout, yet `outnan=[1]` on
   all 4 ranks.
2. Zeroing every paged slot ≥ kvlen (correct per-row indexing) does NOT
   fix it — not an unwritten-tail-garbage problem.
3. Splitting a B=2 call into per-row B=1 op calls does NOT fix it — not a
   batched-kernel numerics problem.
4. Pure-torch reference paged attention reproduces the NaN reading the
   same `k_nope` view — the cache contents themselves are NaN at read time.
5. The paged-KV writer (`npu_scatter_pa_kv_cache`) verified the slots
   clean immediately after writing them; they are NaN at the next read.


---

## Companion finding: LeapQuant KDA flush hook silently inert (solved)

While validating the LQ leg on Kimi-Linear-48B, the per-layer LeapQuant
window-boundary flush (`AscendKimiLinearGatedDeltaNetAttention._leapquant_kda_flush`
in the local kimi_k3.py) never fired: outputs were token-identical to base,
0 flush calls across 5 probe iterations.

Root cause chain (resolved 2026-11-06 via an all-guards one-line diagnostic
logging pre-resolved vs context metadata type, prefix hit, seq_lens/indices
presence and caches type on the first 12 calls + every 200th, BEFORE any
early return):

1. The old per-guard skip logs were capped at 2/rank and exhausted during
   warmup (where metadata legitimately is None), so real-step guard failures
   were invisible.
2. Real steps in fact pass every metadata guard: `pre=dict ctx=dict
   nkeys=27 prefix='model.layers.0.self_attn' hit=True
   mtype=GDNAttentionMetadata seq_lens=True idx=True`.
3. The failing guard was the last one: hybrid mamba-style layers hold their
   caches as a **list** `[conv_states, ssm_states]` — the same layout
   ops/gdn.py indexes as `self_kv_cache[1]` — but the guard required a
   2-**tuple**. Fix: `isinstance(caches, (list, tuple))`; the pool remains
   `caches[1]`.
4. Warmup vs real-step dispatch note: the overridden `_forward` runs during
   warmup AND real steps (3,400+ calls in a 3×64-token probe), but only the
   first 12 calls see `None` metadata; everything after sees the dict.
   Earlier probes misread this because the log cap hid real steps.

Diagnostic-hygiene notes: (a) a contiguous-field grep regex
(`pre=… ctx=… caches=…`) cannot match dict-branch lines (dict branch inserts
`nkeys/prefix` between `ctx=` and `caches=`), which briefly made it look like
no real-step lines existed; (b) the diagnostic must use
`getattr(self, 'kv_cache', None)` — during profile_run the attribute is not
yet bound and an unconditional read crashes engine init with AttributeError.

Verification (probe7, after the one-line fix): 20 `LQ-KDA-FLUSH #` lines
(5/rank; first at seq_len 25, pool `(7336,8,128,128)`), and the three probe
outputs diverge from base exactly where window-16 boundary flushes begin —
e.g. haiku last line "Dawn waits, bugs unslain." → "Dawn hides in the code.";
TCP analogy "…considered 'delivered.'" → "…delivers the next one." — while
remaining fully coherent. Quantization confirmed active end-to-end.
