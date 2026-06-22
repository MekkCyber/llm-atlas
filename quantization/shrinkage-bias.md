# Shrinkage Bias
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A systematic **negative rounding bias** in low-precision training that occurs when the number format's representable bins are asymmetric around zero. The bias is geometric (each bin is wider above its midpoint than below) and accumulates **multiplicatively across layers**, dragging activation magnitudes toward zero. It's most visible in **FP4 E2M1** and the OCP MXFP4 spec, and is *amplified* — not suppressed — by Random Hadamard Transform (RHT) preprocessing. Diagnosed and named in the UFP4 paper (Ant Group, 2026).

**Prereqs:** [_number-formats](_number-formats.md), [fp8](fp8.md)
**Related:** [../pre-training/fp4-training.md](../pre-training/fp4-training.md), [../pre-training/_training-stability.md](../pre-training/_training-stability.md), [../pre-training/fp8-training.md](../pre-training/fp8-training.md)

---

## What it is

Floating-point formats with very few mantissa bits have a **non-uniform** spacing of representable values. **E2M1** (2 exponent bits, 1 mantissa bit — the 4-bit element format in OCP MXFP4) has only 16 representable magnitudes, exponentially spaced. The bin around each representable value is *asymmetric*: the distance to the next value above is generally larger than the distance to the next value below.

When you round a real activation to the nearest representable value, you're systematically rounding **toward smaller magnitude** more often than larger. Each layer's matmul output is then slightly biased downward. Stack 80 layers and the multiplicative shrinkage compounds — late layers see drastically attenuated activations, loss degrades, and frontier-scale FP4 training drifts.

The UFP4 paper calls this **Shrinkage Bias** and shows it has a **geometric origin** (the bin asymmetry) rather than being a calibration or clipping problem.

## How it works

For a positive value $x$ in bin $[v_i, v_{i+1}]$ with representable points $v_i, v_{i+1}$, round-to-nearest picks $v_i$ when $x < (v_i + v_{i+1})/2$ and $v_{i+1}$ otherwise. For a uniformly distributed $x$ in the bin, $E[\text{round}(x)] = (v_i + v_{i+1})/2$.

In an exponentially spaced format like E2M1, **bin widths grow with magnitude**. The bin centered at a representable value $v$ is wider above $v$ than below $v$ (since the next value up is $\sim 2v$ but the next value down is $\sim v/2$). For naturally distributed activations (Gaussian-ish), more mass lands in the *upper half* of each bin, but round-to-nearest pulls everything to the boundary — which is biased low because the midpoint sits closer to $v$ than to $v_{i+1}$ in *value*, not in *probability*.

Formally, $E[\text{round}(x)] / E[x] < 1$ for typical activation distributions on E2M1. The ratio gap is small per matmul ($\sim 10^{-3}$) but **multiplies across $L$ layers** to a measurable loss gap.

### Interaction with Random Hadamard Transform

RHT (multiplying activations by a random orthogonal Hadamard matrix before quantization) is a standard outlier-suppression trick — it makes the distribution closer to Gaussian and stabilizes scale. Counterintuitively, the UFP4 paper shows RHT **worsens** Shrinkage Bias under E2M1: the Gaussian-shaped distribution after RHT concentrates more mass near the asymmetric bin boundaries, increasing the bias's bite.

## Why it matters

- The current OCP MXFP4 spec is built on E2M1. If Shrinkage Bias is structural to E2M1, then "MXFP4 pretraining" inherits the bias **by design**, not by implementation accident.
- Explains a class of FP4 training failures previously attributed to outliers or learning-rate tuning.
- The remedy in UFP4 is to **switch to uniform 4-bit grids** (E1M2 or INT4 with per-block scale) — and apply RHT to **all three** training matmuls, which is now a net win because the underlying grid is symmetric.

## Gotchas & tricks

- **Diagnosis at small scale fails.** Single-layer FP4 forward looks fine. The bias only becomes visible after dozens of layers compound.
- **Don't blame loss scaling.** Loss scaling fixes underflow, not bin asymmetry.
- **Per-tile FP4 scaling does not save you** — it controls dynamic range, but bin asymmetry is per-element and survives rescaling.
- **RHT-helps-FP8 intuition does not transfer to FP4.** What's good for FP8 outlier suppression can hurt FP4 if the underlying element format is non-uniform.

## Sources

- Paper: *Rethinking Shrinkage Bias in LLM FP4 Pretraining: Geometric Origin, Systemic Impact, and UFP4 Recipe* — Chen, Tian, Jiang, Zhang, Yu, Jiang, Gong, Liu, Liu, Zhang, Zhou (Ling Team, Ant Group), 2026, arXiv 2606.20381.
- Spec: *OCP Microscaling Formats (MX) v1.0* — for the E2M1 definition Shrinkage Bias attacks.
- Related: NVIDIA / Hopper FP8 training literature (RHT use under E4M3, where the bias is mild).
