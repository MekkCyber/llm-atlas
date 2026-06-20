# UFP4
*Depth — uniform 4-bit grids (E1M2 / INT4) instead of E2M1 for FP4 LLM pretraining.*

**TL;DR:** A 4-bit pretraining recipe that argues against the E2M1 element format used by Blackwell / Rubin / MI350. E2M1's asymmetric, non-uniform bins produce a systematic negative rounding error ("Shrinkage Bias") that compounds layer-by-layer and is amplified by the Random Hadamard Transform (RHT) preconditioner. Swapping to uniform 4-bit grids (E1M2 or INT4) cancels the bias, lets RHT actually deliver its SQNR gain, and restricts stochastic rounding to dY alone.

**Prereqs:** [_number-formats](_number-formats.md), [fp8](fp8.md)
**Related:** [fp8-training](../pre-training/fp8-training.md)

---

## What it is

E2M1 (FP4) packs 16 representable values with more density near zero. Round-to-nearest on a symmetric distribution is therefore biased *downward* in magnitude — every layer shrinks its outputs slightly. UFP4 reframes this as a **geometric** problem of the element format, not a scale-granularity problem, and prescribes uniform grids (E1M2 with equally-spaced exponent bins, or pure INT4) so the round-to-nearest expectation is unbiased.

## How it works

Three ingredients:

1. **Uniform 4-bit element format.** Either E1M2 (1 sign bit, 1 exponent bit, 2 mantissa bits → symmetric bins) or INT4. Both have RTNE expectation equal to zero on symmetric inputs.
2. **Random Hadamard Transform applied to all three training GEMMs** (forward Y, weight-gradient dW, input-gradient dX). RHT spreads outliers across channels so per-block scaling captures more of the dynamic range; uniform grids convert this "improved bucket utilization" into actual SQNR.
3. **Selective stochastic rounding.** Only the dY GEMM uses stochastic rounding; forward and dW stay on RTNE. Stochastic rounding everywhere costs SQNR; restricting it to dY recovers the unbiased-gradient property without paying the variance tax elsewhere.

The Shrinkage Bias derivation (paper-side) shows the bias is **multiplicative across depth** — a small per-layer shrinkage compounds to a large drift over 60+ layers, explaining the long-run instability of E2M1-only recipes that look fine in short runs.

## Why it matters

- The two largest near-term accelerators (Blackwell / Rubin) ship FP4 tensor cores built around **E2M1 only**. UFP4 is a concrete recommendation for the next silicon generation to add E1M2 / INT4 as a first-class training primitive.
- Most published FP4 training instability traces back to "we tried higher learning rates and it diverged" rather than "we tried the other element format." This paper isolates the format as the root cause.
- Cleanly separates **scale granularity** (which UFP4 keeps fixed) from **element geometry** (which UFP4 changes) — useful framing for the rest of the low-precision-training literature.

## Gotchas & tricks

- E1M2 is **not E2M1 with bits relabelled.** The exponent bit count changes the dynamic range (E1M2 has ~half the range of E2M1) so block-scale recalibration thresholds shift accordingly.
- The RHT-amplifies-bias result depends on the transform being applied *before* quantization. Pipelines that apply RHT post-quantization would see a different interaction.
- Stochastic rounding on the forward pass actively hurts UFP4 — it's adding variance to an already-uniform grid. Don't reach for SR as a universal fix.
- Scale-granularity tuning still matters: UFP4 holds tile shape fixed for the fair comparison, but production deployment will want to tune both axes jointly.

## Sources

- Paper: *Rethinking Shrinkage Bias in LLM FP4 Pretraining: Geometric Origin, Systemic Impact, and UFP4 Recipe* — Chen et al., Ant Group, 2026 — arXiv 2606.20381.
- Spec: *OCP Microscaling Formats (MX) v1.0* — 2023 — defines MXFP4 with E2M1 elements.
- Paper: *FP8 Formats for Deep Learning* — Micikevicius et al., 2022 — the FP8 spec UFP4 generalizes from.
