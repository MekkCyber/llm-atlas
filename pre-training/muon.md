# Muon
*Depth — a momentum-orthogonalization optimizer designed for hidden-layer matrix parameters in deep networks.*

**TL;DR:** Muon (Momentum via UN-decomposed orthogonalization) replaces AdamW on hidden-weight matrices with an update that orthogonalizes the momentum via a short Newton-Schulz iteration before applying it. On transformers and DiTs it consistently matches or beats AdamW at the same wall-clock budget with lower memory (no second moment) and no per-tensor scaling hyperparameters. Adopted by LLaDA-Image (2026) for a 6B DiT trained from scratch alongside parameter-free RMSNorm.

**Prereqs:** [transformer-block](../architectures/transformer-block.md), [_training-stability](_training-stability.md)
**Related:** [fp8-training](fp8-training.md)

---

## What it is

An optimizer for matrix-shaped hidden parameters (attention QKV/O, MLP up/down/gate) that plugs into a training loop otherwise identical to AdamW's. Non-matrix parameters — embeddings, biases, normalization scales — remain on AdamW (or SGD). Muon's step is:

1. Momentum accumulation: $M \leftarrow \mu M + G$ (same as SGD-momentum on the gradient $G$).
2. Orthogonalization: $U \leftarrow \text{OrthNS}(M)$ via a few (typically 5) Newton-Schulz iterations to approximate $M(M^\top M)^{-1/2}$.
3. Update: $W \leftarrow W - \eta \, U$ with $\eta$ scaled by the parameter's shape.

The result is an approximately-orthogonal update matrix — the direction of the momentum, its scale renormalized so no singular-value direction dominates.

## How it works

The Newton-Schulz iteration:

```
X = M / ||M||_F
for _ in range(5):
    X = 1.5 X - 0.5 X (X^T X)
return X
```

Five iterations approximate the orthogonalization to sufficient precision without any SVD call. Cost per step: a handful of matrix-matrix multiplies of the same shape as the parameter — small relative to the forward/backward.

Why not just Adam? Adam scales each element by an EMA of its squared gradient; on matrix parameters this can be dominated by a few large-gradient directions that consume the effective step. Muon's orthogonalization redistributes the update over singular directions so all directions receive comparable per-step progress.

## Why it matters

- **Memory.** No second-moment tensor. Halves optimizer-state memory vs. AdamW — meaningful for large-DiT / large-LLM training runs.
- **Better wall-clock at scale.** LLaDA-Image (6B DiT) uses Muon end-to-end and achieves competitive image-generation results at open-recipe scale.
- **Fewer scale hyperparameters.** No per-tensor learning-rate scaling; a single LR schedule covers all matrix parameters.
- **Interacts well with RMSNorm.** Parameter-free RMSNorm removes affine scale parameters that Muon would treat as matrix-shaped; the two together are a common combination in recent DiT training.

## Gotchas & tricks

- **Matrix parameters only.** Embeddings, biases, and layer-norm scales stay on AdamW (or SGD-momentum). Muon on non-matrix parameters degrades or diverges.
- **Newton-Schulz stability.** Convergence relies on the Frobenius-normalization pre-step; skipping it makes early training explode.
- **Weight decay.** Apply decoupled weight decay (AdamW-style) — coupling it into the orthogonalized step warps the direction.
- **Initialization.** Standard transformer inits (scaled normal or truncated normal) work; small extra scale on output projections is not needed.
- **Not a distillation-only or fine-tuning-only trick.** Muon works from-scratch. Reports of it failing in fine-tuning usually trace to weight-decay coupling or applying it to non-matrix parameters.

## Sources

- Paper: *LLaDA-Image: Building Strong Image Generators with Fully Open Training Recipes* — Chen et al., 2026 — production use of Muon in a 6B DiT — https://arxiv.org/abs/2609.03796
- Blog: *Muon: An optimizer for hidden layers in neural networks* — Keller Jordan, 2024 — original write-up and Newton-Schulz derivation.
- Related: *nGPT / Symmetric Power Transformers* — associated normalization-friendly transformer designs.
