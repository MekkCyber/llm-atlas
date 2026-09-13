# Hybrid-Precision Attention Quantization
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Attention scores collapse under aggressive uniform low-bit quantization because a few outlier tokens (the "vertical lines" in the attention map) and the local decode window carry a disproportionate share of the signal. Hybrid-precision routes those regions to a high-precision path (fp8/16) and quantizes everything else to very low bit-widths, using a cheap attention-pattern probe to pick the exceptions. Introduced by **HyQuant** (2026).

**Prereqs:** [_number-formats.md](./_number-formats.md), [fp8.md](./fp8.md), [../fundamentals/attention.md](../fundamentals/attention.md)
**Related:** [../architectures/multi-head-attention.md](../architectures/multi-head-attention.md)

---

## What it is

Post-training quantization for LLM attention. Prior approaches either (a) globally *smooth* activations before uniform low-bit quantization (SmoothQuant-family) or (b) leave attention in high precision while quantizing MLPs and KV cache. Neither is satisfactory: (a) still loses accuracy at very low bit-widths, (b) gives up a big fraction of the memory/bandwidth win because attention dominates long-context inference.

Hybrid-precision attention takes a third route: **route accuracy-critical positions to a high-precision path**, quantize the rest aggressively.

## How it works

At attention-compute time, the softmax pattern across tokens tends to concentrate on:

1. A small set of **vertical-line** tokens — positions that are attended to strongly by many query positions (sink tokens, some content-anchor tokens).
2. The **local window** around the query — the recent past that decoding attends to heavily.

HyQuant runs a lightweight probe on the query–key structure that predicts which tokens will land in either category, without materializing the full score matrix. Then:

- **High-precision lane** (fp8 or fp16): scores/values for vertical-line positions plus the local window.
- **Low-precision lane** (INT4 / MXFP4 or similar): the residual set — most of the sequence.

The final attention output sums contributions from both lanes. Because the low-precision lane covers the long tail where each individual score is small, its quantization error contributes minimally to the aggregate.

## Why it matters

- **Attention was the last stubborn piece.** KV cache is quantizable, MLPs are quantizable, but attention scores + softmax are numerically fragile; low-bit uniform attention has been where accuracy collapses.
- **Long-context serving payoff.** Attention memory and bandwidth grow with context length; a working attention-quantization scheme is one of the few remaining levers for making very long contexts cheap.
- **Architecture-agnostic.** The selector uses only attention-pattern statistics, so it plugs into MHA, GQA, MLA, and hybrid attention variants with modest engineering.

## Gotchas & tricks

- **The selector cost cannot exceed the savings.** A heavy classifier per attention op wipes out the quantization win. HyQuant's "vertical-line-aware" probe is deliberately cheap — a small linear on aggregate query statistics.
- **Not the same as sink-token attention.** Sink tokens (StreamingLLM) is *which tokens to keep in the KV cache*; hybrid-precision quantization is *which token positions to compute in high precision this step*.
- **Head-level tuning.** Different heads have different attention-pattern statistics (some are near-uniform, some highly-concentrated). The selector's threshold should be per-head.
- **Softmax is the ratchet.** Low-precision noise on very small pre-softmax scores can leak into non-trivial post-softmax mass; the local window keeps this bounded near the current query.
- **Complementary to KV-cache quantization.** Hybrid-precision attention quantizes *this step's compute*; KV-cache quantization changes *what is stored*. Both can be applied.

## Sources

- Paper: *HyQuant: Hybrid-Precision Quantization for LLM Attention* — Xing et al. (XMU, SJTU, XJTU, Tencent Penglai), 2026 — arXiv:2608.27875.
- Related: *SmoothQuant* — Xiao et al., 2022 — the "smooth the outliers" alternative that HyQuant contrasts with.
- Related: *StreamingLLM / attention sinks* — Xiao et al., 2023 — the "vertical line" attention pattern this scheme exploits.
