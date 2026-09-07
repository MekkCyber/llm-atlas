# NVFP4
*Depth — NVIDIA's 4-bit floating-point tensor format with 16-element block scaling.*

**TL;DR:** NVFP4 packs each tensor as FP4 (E2M1) *elements* grouped into 16-element blocks, each block carrying an FP8 (E4M3) scale. Compared to MXFP4 (32-element blocks with an E8M0 power-of-two scale), NVFP4's finer blocks + higher-precision scale localize outliers better and preserve more dynamic range, making W4A4 (weights *and* activations in 4-bit) feasible for full-model inference — including the "fragile" linear-attention recurrences in hybrid LLMs, as demonstrated by Minima on Qwen3.8-27B.

**Prereqs:** [_number-formats](_number-formats.md), [fp8](fp8.md)
**Related:** [gated-deltanet](../architectures/gated-deltanet.md)

---

## What it is

A microscaling variant introduced by NVIDIA (Blackwell generation). Storage: each block of 16 elements shares one FP8 E4M3 scale (in contrast to MXFP4's block of 32 elements + E8M0 scale). The element format is FP4 E2M1 (sign, 2 exponent bits, 1 mantissa bit — 16 possible non-NaN values, max magnitude 6.0).

Rough footprint per element, amortized: 4 element bits + 8 scale bits / 16 = **4.5 bits/element** (vs. MXFP4's 4 + 8/32 = 4.25 bits/element).

Hardware path lives on Blackwell tensor cores; software emulation is possible on Hopper at a throughput cost.

## How it works

Encoding a tensor:

```
for each block of 16 contiguous elements:
    amax = max(|x_i|) for i in block
    scale = amax / 6.0            # 6.0 is FP4 E2M1's max magnitude
    scale = quantize(scale, FP8_E4M3)
    for each element in block:
        x_q = round_to_FP4_E2M1(x / scale)
```

Decoding: `x ≈ x_q * scale`, done inline in the matmul unit.

Two properties that matter for LLM tensors:

1. **Block size 16 vs 32.** A single outlier corrupts its block; halving the block size halves the number of siblings whose precision is destroyed by an outlier.
2. **FP8 E4M3 scale vs E8M0.** E8M0 stores a power-of-two only; E4M3 has 3 mantissa bits, so the scale itself is finer-grained. Two nearly-equal amax values in adjacent blocks are not forced to snap to different powers of two.

The composition ("finer blocks, finer scale") gives NVFP4 more headroom against activation outliers than MXFP4, which is what makes W4A4 for both weights *and* activations robust enough for competitive perplexity.

## Why it matters

- **Full W4A4** — with both weights and activations in 4-bit, the entire matmul runs on FP4 tensor cores. Vs. weight-only INT4 (INT4-W, BF16-A), throughput improves markedly and activation memory falls.
- **Hybrid-model quantization** — Minima (2026) shows NVFP4 W4A4 across all 496 linear layers of a Gated-DeltaNet + softmax-attention hybrid (Qwen3.8-27B) matches BF16 within seed noise on reasoning/eval suites, refuting the assumption that recurrent halves must be kept ≥8-bit.
- **Fits between MXFP4 and FP8** — for teams choosing a 4-bit path, NVFP4 is the safer choice on Blackwell when MXFP4 loses accuracy, at the cost of ~5% more storage.

## Gotchas & tricks

- **Global-scale mismatch when kernels fuse modules.** If per-module NVFP4 calibrations are served through a kernel that fuses those modules into a single GEMM, the module scales silently disagree. Minima documents this and rescales before fusion.
- **Ship FP8 KV cache scales.** Calibrating the KV cache in FP8 recovers most of the long-context KV penalty at essentially zero throughput cost.
- **Not free on Hopper.** Software emulation is available but you lose the tensor-core throughput advantage — the win is Blackwell-specific.
- **Outlier-heavy channels still need attention.** NVFP4's finer blocks help but do not eliminate the problem; SmoothQuant-style channel rescaling before quantization is compatible and often complementary.

## Sources

- Paper: *Why Gated DeltaNet Survives 4-Bit Quantization: NVFP4 W4A4 for the Recurrent Half of a Hybrid 27B LLM* — Kozyrev & Maiboroda, Minima-AI, 2026 — full W4A4 recipe and mechanism analysis — https://arxiv.org/abs/2609.04098
- Model card: `minima-ai/mnma_qwen3.8_27b_nvfp4` — released NVFP4 checkpoint.
- NVIDIA Blackwell architecture whitepaper — NVFP4 hardware spec.
- Spec: *OCP Microscaling Formats (MX) v1.0* — background on block-scaled 4-bit formats.
