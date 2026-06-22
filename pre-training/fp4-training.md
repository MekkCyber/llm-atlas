# FP4 Pretraining
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** End-to-end pretraining with **4-bit matmuls** for the heavy weight × activation operations, while keeping master weights, optimizer state, norms, attention, and routing in higher precision. The UFP4 recipe (Ant Group, 2026) makes this viable at frontier scale by switching off the non-uniform E2M1 element format (which suffers [Shrinkage Bias](../quantization/shrinkage-bias.md)) and using a **uniform 4-bit grid** (E1M2 or INT4 with per-block scale), with Random Hadamard Transform applied to **all three training matmuls** instead of just the forward.

**Prereqs:** [fp8-training](fp8-training.md), [../quantization/_number-formats.md](../quantization/_number-formats.md), [../quantization/shrinkage-bias.md](../quantization/shrinkage-bias.md)
**Related:** [_training-stability](_training-stability.md), [../quantization/fp8.md](../quantization/fp8.md)

---

## What it is

FP8 pretraining ([fp8-training](fp8-training.md)) is now standard at frontier scale; FP4 promises another 2× memory and matmul throughput on top, but the naïve OCP MXFP4 recipe (E2M1 elements + per-32 block E8M0 scale) drifts and underperforms BF16 reference loss for any nontrivial model size.

UFP4 is the first openly documented recipe that closes most of the gap to BF16 across **1.5B → 124B** dense pretraining. Its two changes:

1. **Uniform 4-bit element grid** — replace E2M1 with E1M2 (one exponent bit, two mantissa bits → near-uniform spacing) or with INT4 + appropriate per-block scaling. Eliminates [Shrinkage Bias](../quantization/shrinkage-bias.md).
2. **RHT on all three matmuls** — Random Hadamard Transform applied to forward (output = X·W), weight gradient (∂L/∂W = X·∂L/∂Y), and input gradient (∂L/∂X = ∂L/∂Y·W). Under the uniform grid, RHT cleanly suppresses outliers in all three without amplifying any geometric bias.

## How it works

### Which tensors live in FP4 and which don't

- **FP4 (uniform grid: E1M2 or INT4 + block scale):** forward matmul activations and weights; both backward matmul inputs.
- **BF16 / FP32:** embeddings, LM head, RMSNorm, attention softmax / scaling chain, MoE router, master weights, accumulated gradients.
- **Optimizer moments $m, v$:** BF16 (carrying the FP8-recipe convention forward).

Rule of thumb is unchanged from [fp8-training](fp8-training.md): FP-low-precision for big-FLOP matmuls, high-precision for everything numerically delicate.

### Uniform grid choice

| Format | Layout | Notes |
|---|---|---|
| INT4 + per-block FP32 scale | signed 4-bit integer, 16 uniform levels | best implementation match for current INT4 tensor cores |
| E1M2 | 1 exp / 2 mant, 16 uniform-ish magnitudes | retains a "float" interface so FP-pipeline code reuses |
| **E2M1 (OCP MXFP4)** | 1 exp / 1 mant, 16 magnitudes, non-uniform | what UFP4 explicitly **avoids** for training |

### RHT in all three matmuls

A Random Hadamard Transform $H_r = D \cdot H \cdot D'$ (with random sign diagonals $D, D'$ and a fixed Hadamard $H$) is orthogonal, cheap on-chip, and disperses outliers. For training:

```
Forward:           Y = (X·H) · (Hᵀ·W) → quantize each factor in uniform FP4
Backward (W):      ∂L/∂W = (Xᵀ·H) · (Hᵀ·∂L/∂Y)
Backward (X):      ∂L/∂X = (∂L/∂Y·H) · (Hᵀ·Wᵀ)
```

All three pairs of factors pass through `H/Hᵀ` before FP4 quantization. The transforms cancel mathematically (`Hᵀ·H = I`) but reshape the per-element distribution into something the uniform grid handles well.

UFP4's ablations isolate the contribution of each fix and show both are needed: **uniform grid alone** still leaves a residual gap from un-suppressed outliers; **RHT alone on E2M1** *worsens* loss vs no RHT (because it intensifies Shrinkage Bias).

### Higher-precision accumulation

Inherits the FP8-training trick from [fp8-training](fp8-training.md): periodically flush the tensor-core partial sum to FP32 on CUDA cores. With FP4 elements, the inner-dim rounding noise per MAC is larger, so the flush frequency is tighter (paper recommends every 64 inner-dim elements vs FP8's 128).

## Why it matters

- **First openly documented FP4 pretraining recipe stable at 100B+ scale.** Validated up to 124B parameters with loss curves tracking BF16.
- **~2× memory and ~2× matmul throughput** over FP8 for the same hardware that supports both.
- **Direct critique of OCP MXFP4 for training.** UFP4 argues future accelerators should expose **uniform 4-bit grids as primary training primitives** rather than treating non-uniform FP4 as the default.

## Gotchas & tricks

- **Don't use MXFP4 element format (E2M1) for training.** This is the headline failure mode the paper diagnoses.
- **RHT cost is real.** Three additional Hadamard ops per matmul aren't free; the paper bounces this against the 2× matmul speedup and shows net positive on Hopper-class and INT4-tensor-core hardware.
- **Validate at the biggest model you intend to use.** Multiplicative bias and rounding effects only surface at depth; small-scale ablations underestimate the failure mode.
- **Attention still in BF16.** Same reason as FP8 — softmax and Q·Kᵀ are too numerically brittle. FP4 attention is open research.
- **Optimizer in FP4 is open.** UFP4 keeps optimizer moments in BF16; FP4 moments are not yet shown to work.

## Sources

- Paper: *Rethinking Shrinkage Bias in LLM FP4 Pretraining: Geometric Origin, Systemic Impact, and UFP4 Recipe* — Ling Team, Ant Group, 2026, arXiv 2606.20381.
- Paper: *DeepSeek-V3 Technical Report* — 2024 — the FP8 template UFP4 extends.
- Spec: *OCP Microscaling Formats (MX) v1.0* — the MXFP4 spec UFP4 critiques.
- Background: NVIDIA Blackwell whitepapers (FP4 tensor cores, MXFP4 native path).
