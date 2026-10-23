# Recurrent-State Quantization (PACE)
*Depth — 4-bit post-hoc quantization of the recurrent state in linear-attention / hybrid language models.*

**TL;DR:** Linear-attention LMs (Mamba, hybrid-attention stacks) keep their context in a *fixed-size recurrent state* rather than a growing KV cache. Storing that state at 4 bits halves memory vs. 8-bit, doubling concurrent sequences — but naive per-token rounding destroys accuracy. **PACE** rounds the state **periodically** (once every few tokens, with intervening tokens in a small full-precision buffer), uses **channel-aligned scales** (one scale per state column, where big magnitudes cluster), and keeps the few rounding-sensitive heads at 8 bits. Matches 8-bit SOTA at ~⅔ memory on 4B–35B hybrids.

**Prereqs:** [_number-formats](_number-formats.md), [attention](../fundamentals/attention.md)
**Related:** [fp8](fp8.md), [mla](../architectures/mla.md)

---

## What it is

Transformers cache K/V per token, so inference memory grows with sequence length. Linear-attention and SSM-style models (Mamba) instead maintain a **fixed-size recurrent state** that is updated step-by-step and never grows — a different memory bottleneck.

That state is still large (depends on head count × hidden size), and it is accessed every token, so lower-precision storage directly multiplies serving throughput. The challenge: because the state is *updated recurrently*, quantization error compounds. A quantizer that re-rounds the state after every generated token accumulates error step-by-step and loses accuracy at 4 bits.

PACE is a post-hoc (no retraining) 4-bit quantizer specifically designed for this recurrent-state setting.

---

## How it works

Three design moves:

**1. Periodic rounding.** Instead of rounding after every generated token, PACE keeps the state in 4 bits only *every T tokens* (T is a small integer, e.g. 4–8). Between rounding events, the T intermediate state updates live in a small FP buffer. At rounding time, the whole buffer collapses into a single quantized state, and the next T steps start fresh.

   Rationale: compounded rounding error is the dominant failure mode. Rounding T× less often reduces that compounding by roughly a factor of T.

**2. Channel-aligned scales (per-column).** Analyzing recurrent-state magnitudes shows that large entries cluster in specific *columns* (channel dimensions), not uniformly. PACE assigns one quantization scale per column — so columns with outlier magnitudes get wide dynamic range, and quiet columns get precision.

   Rationale: analogous to per-channel weight quantization in GPTQ/AWQ, but applied to the recurrent state.

**3. Mixed-precision heads.** A small fraction of heads have state entries whose quantization error is disproportionately costly for downstream accuracy. PACE keeps those heads at 8 bits and quantizes the rest at 4.

   Rationale: head-level sensitivity analysis is cheap offline; selecting ~10–20% of heads for 8-bit storage recovers most of the quality gap.

---

## Why it matters

Linear-attention / hybrid models are serving-friendly precisely *because* their state is bounded. But that only pays off if the state fits — a 35B hybrid at 8-bit state is still huge. PACE halves state memory with minimal quality cost, which doubles the number of concurrent sequences on the same GPU. Across 4B–35B hybrids:

- PACE mixed 4/8-bit heads ≈ DAMP (8-bit baseline) at ~⅔ its memory on the three larger models.
- PACE all-4-bit heads > Q-Mamba at equal memory.
- Periodic rounding preserves the inference speedup of the buffer (the buffer is small, cheap to read/write).

---

## Gotchas & tricks

- **T is a tradeoff.** Larger T reduces error compounding but grows the FP buffer and the per-rounding compute spike. T=4 to T=8 is the typical sweet spot.
- **Column scales are offline-calibrated.** Scales are chosen on a short calibration set, not during inference. If the deployment distribution shifts a lot from calibration, re-calibrate.
- **Head sensitivity is model-specific.** Which 10–20% of heads to keep at 8 bits must be re-selected per model.
- **Only applies when the state is recurrent.** Standard KV-cache quantization is a *different* problem — this method is for Mamba-family / hybrid models.

---

## Sources

- Paper: *PACE: Periodic and Channel-Aligned Encoding of Four-Bit Recurrent States* — anonymous (academic), 2026 — [arXiv:2610.29545](https://arxiv.org/abs/2610.29545).
- Background: Mamba (Gu & Dao, 2023); DAMP, Q-Mamba (prior 8-bit baselines for recurrent-state quantization).
