# NVFP4 (W4A4 microscaling FP4)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** NVFP4 is NVIDIA's 4-bit floating-point microscaling format (E2M1 elements with an E4M3 block scale, per 16-element block) targeting Blackwell-generation W4A4 inference. Elements and activations both fit in 4 bits; the E4M3 scale gives more precision than the OCP MXFP4 E8M0 scale at the cost of one extra bit per block. Kozyrev & Maiboroda (2026) show a 27B hybrid LLM survives full W4A4 NVFP4 across all 496 linear layers — 17.5 GiB checkpoint, near-full-precision quality — with an explanation of *why* linear-attention layers absorb the noise.

**Prereqs:** [_number-formats](_number-formats.md), [fp8](fp8.md)
**Related:** [../architectures/gated-deltanet.md](../architectures/gated-deltanet.md)

---

## What it is

A microscaling 4-bit format for weights and activations:

```
element         : FP4 E2M1  (1 sign / 2 exponent / 1 mantissa)  — 16 possible values, max 6.0
block scale     : FP8 E4M3  (per 16-element block)
block size      : 16
```

The E4M3 block scale is the differentiator vs. the OCP-standard **MXFP4**, which uses an 8-bit *power-of-two* (E8M0) scale per 32-element block. NVFP4 spends one more bit per block on scale precision — non-integer exponents *and* a mantissa — in exchange for tighter reconstruction of blocks whose values don't align with a power-of-two multiple.

W4A4 = weight in 4 bits *and* activation in 4 bits. That's aggressive: activation quantization is where most 4-bit-weight-only schemes break down under outliers.

## How it works

- **Per-block scale.** Each 16-element block gets its own E4M3 scale. Encoded value: `x ≈ FP4_element · E4M3_scale`.
- **Hardware path.** Blackwell tensor cores execute FP4 matmuls with block scales in-flight; software emulation on Hopper is possible but pays a bandwidth-plus-scale-management tax.
- **Calibration.** A short calibration pass sets per-block scales from activation statistics. Kozyrev & Maiboroda quantize *all* 496 linear layers in a 27B hybrid — no per-layer sensitivity search — indicating that the format is robust when combined with the right architecture.

## Why it matters

The Kozyrev & Maiboroda result identifies *four* mechanisms behind the hybrid model's survival at W4A4:

1. **Block scaling absorbs outliers.** 16-element blocks are fine-grained enough that a single outlier only widens *its* block's scale, not neighbors'.
2. **Gate projections are naturally robust** — they saturate anyway, so 4-bit quantization is close to a no-op.
3. **Recurrent state exponentially forgets impulses** — quantization noise in a linear-attention state decays out over subsequent tokens instead of accumulating (as it would in softmax attention over long context).
4. **Per-token quantization cost stabilizes at long context.** After warmup, quantization error settles into a stationary distribution instead of drifting.

Mechanism (3) is the crux: it argues linear-attention/hybrid architectures are *quantization-friendly by construction*, not merely inference-friendly.

## Gotchas & tricks

- **Not a drop-in replacement for MXFP4.** The scale layout is different (E4M3 vs. E8M0) and the block size is different (16 vs. 32). Runtimes have to know which they're dealing with.
- **Hardware dependency is real.** Full-throughput NVFP4 requires Blackwell; on Hopper you'll pay for scale management in software and lose most of the throughput win.
- **Activation quantization is where it breaks.** For softmax-attention-only architectures at W4A4, the same recipe underperforms — the exponential-forgetting argument doesn't apply.
- **Calibration data quality still matters.** A tiny calibration set from a distribution shift from deployment leaves block scales miscalibrated on outlier channels.

## Sources

- Paper: *Why Gated DeltaNet Survives 4-Bit Quantization: NVFP4 W4A4 for the Recurrent Half of a Hybrid 27B LLM* — Kozyrev & Maiboroda, 2026 — [arXiv:2609.04098](https://arxiv.org/abs/2609.04098).
- NVIDIA Blackwell architecture whitepaper — hardware support for FP4 with block scales.
- OCP Microscaling Formats v1.0 — for the MXFP4 (E8M0-scale) baseline this format contrasts with.
