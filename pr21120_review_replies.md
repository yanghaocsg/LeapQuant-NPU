# PR #21120 — prepared replies to peterDengcx (await user go-ahead)

## Reply to: prefill chunks crossing window boundaries (runtime.py)

Good catch — spelling out the intended semantics:

- **Current M1 behavior**: flushes fire when the *chunk-end* sequence length
  lands on a window boundary. Interior boundaries crossed by a chunked
  prefill chunk are not flushed, because the chunked delta-rule kernel only
  materializes the end-of-chunk state — the intermediate states needed for
  a mid-chunk flush are not produced by the stock kernel.
- **Error-direction argument**: a flush is the lossy step, so skipping an
  interior boundary can only *reduce* the quantization error injected during
  prefill for that request (the state accumulates a few extra tokens before
  the next boundary-aligned compression). The exact per-window cadence
  resumes from the first decode boundary onward, where every boundary is
  flushed.
- **Measured impact**: the e2e legs reported in this PR (Qwen3.5-0.8B/9B,
  GSM8K + CMMLU) run chunked prefill with this exact behavior and stay
  within noise of the unquantized baseline (MCQ 200/200 per-question
  identical across all legs).
- **Plan**: exact per-boundary flushes land with the fused-kernel milestone
  (PR2), whose chunk loop can emit per-window states natively
  (`chunk_boundary_quantize` in the RFC's M3 scope).

A docstring note documenting this explicitly is in commit 4d603dae6.

## Reply to: device-gate flag set before the check (runtime.py)

Right — fixed in 4d603dae6: `_device_gate_checked` is now set only after the
device resolves *and* is A3. Consequences:

- a caller catching the non-A3 `RuntimeError` now hits the same raise at the
  next flush instead of silently running ungated forever;
- a worker whose device detection is not yet resolvable (transient) retries
  at the next flush rather than disabling the gate permanently.

---
*(gemini-code-assist also suggested a retitled PR summary — harmless, can
apply or ignore on your call.)*
