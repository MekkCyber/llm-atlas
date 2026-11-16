# G-STAR
*Depth — group-scaled token capacity for deeply compressed autoregressive image generation.*

**TL;DR:** Discrete autoregressive image models eat latency because they emit thousands of tokens per image. G-STAR shows that under a fixed nominal information budget with product quantization, the group size should scale *quadratically* with the per-axis spatial compression factor — keeping the budget constant. Applied at 32× and effective 64× compression, G-STAR hits competitive text-to-image quality while generating a 1024×1024 image in 2.5 s on a single H200 (≈20× faster than prior discrete-AR).

**Prereqs:** [../fundamentals/bpe.md](../fundamentals/bpe.md)
**Related:** [../architectures/capacity-factor.md](../architectures/capacity-factor.md)

---

## What it is

An autoregressive image generator that tokenizes a spatial grid with *product quantization* (PQ) — each spatial location splits into several sub-codes — and decodes those tokens autoregressively. Deep spatial compression shrinks the token sequence (fast) but reduces representational capacity per token (quality collapses). G-STAR's contribution is the scaling rule that recovers quality under aggressive compression.

## How it works

- **Scaling rule.** Fix a nominal per-image information budget. With product quantization, group size *g* (number of sub-codes per spatial location) should scale as g ∝ s², where s is the per-axis compression factor. Doubling s requires quadrupling g to hold the budget constant.
- **Deep compression.** G-STAR uses 32× and effective 64× per-axis spatial compression, with group sizes chosen to satisfy the rule.
- **Semi-autoregressive decoder.** Within each group, dependencies among the g sub-codes are modeled by a small semi-autoregressive head; across groups the main model is purely autoregressive. This keeps global sequence length short while recovering intra-group structure.

## Why it matters

Reopens discrete autoregressive image generation as a credible path alongside diffusion. The latency gap is the main reason practitioners moved to diffusion; G-STAR closes most of it (2.5 s vs 20× slower) at competitive quality. The quadratic scaling rule also transfers — it is a clean budget-accounting handle for anyone shrinking token streams in audio, video, or 3D AR generation.

## Gotchas & tricks

- The quadratic rule holds under the *fixed-budget* assumption with PQ. Different quantizers or loss shapes may shift the exponent.
- "Effective 64×" means an additional compression on top of the backbone's spatial reduction; the exact factorization matters for reproducibility.
- Latency reporting is on H200 with one image at a time. Batched or streaming settings are not shown.
- The semi-autoregressive intra-group decoder is small but load-bearing — swapping it for a parallel head (predict all g sub-codes at once) collapses quality back to the naive-compression baseline.
- Group size growth is quadratic in s, so each additional 2× in compression quadruples codebook memory. There is a practical ceiling around s=8 before the codebook blows out of HBM.

## Sources

- Paper: *G-STAR: Group-Scaled Token Capacity for Deeply Compressed Autoregressive Generation* — Huang, Huang, Li, Yue, Zou, Jing, Li, Yan, Luo, Wang, Zhou, 2026 — [arXiv:2611.13004](https://arxiv.org/abs/2611.13004)
