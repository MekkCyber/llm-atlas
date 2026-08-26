# TileMix mixed-precision attention
*Depth — per-tile precision routing inside fused dense attention for long-context prefill.*

**TL;DR:** Long-context prefill is dominated by dense self-attention's quadratic score computation. Existing accelerations either drop everything to a uniform low precision (quality loss) or sparsify token interactions (skip work). TileMix does neither: it partitions the score matrix into **hardware-aligned tiles** and routes each tile group to FP16 or INT8 score compute, while both paths feed a shared online-softmax accumulator.

**Prereqs:** [../architectures/multi-head-attention](../architectures/multi-head-attention.md), [../quantization/_number-formats](../quantization/_number-formats.md)
**Related:** [../quantization/fp8](../quantization/fp8.md)

---

## What it is

A fused attention kernel that keeps the *dense* score computation but varies precision per tile. Routing decisions are packed into compact bitmasks so the dispatcher costs almost nothing at runtime. Both FP16 and INT8 tiles update the same running softmax state, so there are no numerical seams between tile groups.

## How it works

- **Tile partition:** attention scores `QK^T` are cut into a grid of tiles aligned to the hardware GEMM shape (e.g. 128×128 on Hopper, block-scaled on Blackwell).
- **Precision routing:** for each tile group, a routing decision picks FP16 or INT8 score compute. Routing can be static (heuristic: near-diagonal → FP16, far-off-diagonal → INT8) or learned from calibration.
- **Bitmask dispatcher:** decisions pack into per-tile bits; the kernel branches once per tile group, not per element.
- **Shared online softmax:** each tile group produces a partial score, gets normalized into the running max/sum tracked by the online-softmax accumulator regardless of source precision. No re-scaling at seams.
- **Output projection** stays in higher precision as usual.

## Why it matters

The precision-vs-sparsity axis is *orthogonal* to token-sparsity. TileMix demonstrates you can run *dense* attention (no dropped interactions, no quality-vs-coverage tradeoff) at close-to-INT8 cost by routing precision spatially. Directly relevant to any long-context prefill deployment, and composes with sparse attention: sparsify some tiles, precision-route the rest.

## Gotchas & tricks

- Numerical stability of INT8 tiles hinges on the shared online-softmax accumulator preserving the running max. Care must be taken during the row-max update when mixing tile-precision within a row.
- Routing heuristics vary by model — near-diagonal tiles usually carry the most signal for long-context, so keep them in FP16; router calibration matters most for far-off-diagonal tiles.
- Reported gains are prefill-side; decode-time attention has small tile counts and benefits less.
- Bitmasks are tiny and can be precomputed at prompt-time; no runtime overhead.

## Sources

- Paper: *TileMix: Tile-Centric Mixed-Precision Attention for LLM Inference Acceleration* — Zhang et al., 2026 — [arXiv:2608.17336](https://arxiv.org/abs/2608.17336)
