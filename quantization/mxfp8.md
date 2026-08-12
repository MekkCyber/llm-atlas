# MXFP8
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** An 8-bit floating-point format pairing FP8 mantissa/exponent bits (typically E4M3 or E5M2) with a shared **per-block scale factor** (a small integer or FP8 scalar covering, e.g., 32 consecutive elements). Compared to classical per-tile FP8, the MX (Microscaling) format's fine-grained per-block scaling handles wider dynamic ranges more gracefully — especially at MoE routing boundaries where activation values can spike sharply. Adopted by Motif 3 (2026) in both compute *and* communication.

**Prereqs:** [_number-formats.md](_number-formats.md), [fp8.md](fp8.md), [../pre-training/fp8-training.md](../pre-training/fp8-training.md)
**Related:** [../case-studies/motif-3.md](../case-studies/motif-3.md), [../case-studies/deepseek-v3.md](../case-studies/deepseek-v3.md)

---

## What it is

Classical FP8 training (DeepSeek-V3 recipe) uses **per-tile scaling**: assign a single FP32 scale to a 128×128 weight block or a 1×128 activation tile, and quantize the values in the tile to E4M3 or E5M2 relative to that scale. Works well when values in the tile share a scale range.

The **Microscaling (MX) format**, standardized by OCP in 2023, uses a smaller **block size** (typically 32 elements) with a shared exponent per block. MXFP8 is the FP8-mantissa variant of this format. The smaller block size means a spike in a few outlier values doesn't force the whole tile's scale up (which would erase precision on the non-outliers). MXFP8 pays for the extra scale storage with tighter dynamic range coverage on activations.

## How it works

- **Element format.** 8 bits per element, split as E4M3 (higher precision, tighter range) or E5M2 (lower precision, wider range). Same element format as classical FP8.
- **Block scaling.** Every 32 consecutive elements share a scale (typically an 8-bit exponent, so `x_real = element × 2^scale`). Scale metadata is stored inline with the block.
- **Effective bit-width.** `8 + 8/32 = 8.25` bits per element on average.
- **Compute.** Matmul kernels operate directly on MXFP8: read a block's scale, cast elements to higher precision on the fly, accumulate in FP32. Modern GPU tensor cores (Blackwell generation) support MXFP8 natively.
- **Communication.** All-to-all and all-reduce operations can transmit MXFP8 directly, halving bandwidth vs BF16. Critical at MoE dispatch boundaries where per-step communication volume is large.

## Why it matters

- **Better activation handling than per-tile FP8.** At MoE routing boundaries, expert outputs can differ in scale by orders of magnitude across tokens. Per-tile scaling has to accommodate the largest, losing precision on the smallest; per-block MXFP8 keeps precision uniform.
- **Native tensor-core support.** Blackwell GPUs (2024–2025 generation) do MXFP8 matmul at full rate — no software emulation penalty.
- **Communication savings compose.** Compute-side MXFP8 is one win; sending MXFP8 across all-to-all is a second, independent win. Motif 3 uses both.
- **Sets the pattern for MXFP4.** The same block-scale idea at 4-bit mantissa/exponent is the next generation — MXFP8 is the intermediate step where dynamic range still comfortably covers gradient magnitudes.

## Gotchas & tricks

- **Block boundaries matter for memory alignment.** 32-element blocks need to align with matmul tiling for kernel efficiency. Non-aligned data pays a repacking cost.
- **Scale storage is not free.** 8-bit scale per 32 elements = 3.1% overhead. Small, but real — accounting matters when comparing MXFP8 vs classical FP8 memory footprint.
- **E4M3 vs E5M2 choice depends on the tensor.** Weights and forward activations typically use E4M3 (higher precision, tighter range); gradients use E5M2 (wider range needed). Motif 3 follows this convention.
- **Master weights stay high-precision.** Same rule as classical FP8: keep FP32 master weights for AdamW, cast to MXFP8 only for forward/backward compute.
- **Sensitive components stay in BF16.** Embeddings, LM head, MoE gating, normalization, and attention keep higher precision. The "protect the sensitive components" recipe from DeepSeek-V3 transfers directly.
- **Numerical accumulation must promote.** Even with MXFP8 inputs, matmul accumulation should be in FP32 with periodic promotion to prevent drift over long dot products.

## Sources

- Paper: *Microscaling Data Formats for Deep Learning* — OCP MX specification, 2023 — defines the MX-format family.
- Paper: *Motif 3 Technical Report* — Motif Technologies, 2026 — frontier-scale MXFP8 in compute + communication.
- Related: [../pre-training/fp8-training.md](../pre-training/fp8-training.md) for the classical FP8 recipe MXFP8 extends.
- Related: [fp8.md](fp8.md) for the underlying 8-bit floating-point format.
