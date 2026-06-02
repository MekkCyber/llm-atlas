# Muon Optimizer

*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A matrix-aware optimizer for transformer pretraining. Standard optimizers (Adam, AdamW) treat each parameter as a scalar and apply a coordinate-wise second-moment correction. Muon treats each 2-D parameter (weight matrix) as a matrix and uses a **Newton-Schulz iteration to orthogonalize the momentum** before applying the update — effectively turning the raw gradient into a matrix-direction update with bounded singular values. Reported to match or beat AdamW at the same wall-clock cost for transformer pretraining, with better convergence on the largest matrices in the model.

**Prereqs:** *(assumes familiarity with Adam/AdamW and the standard pretraining loop)*
**Related:** [_training-stability](_training-stability.md) · [wsd-schedule](wsd-schedule.md) · [fp8-training](fp8-training.md)

---

## What it is

AdamW updates each parameter by scaling its gradient by a per-coordinate second-moment estimate. This treats the parameter as a flat vector and is blind to any matrix structure — but transformer weights are matrices, and their gradients have *coherent low-rank structure* that the coordinate-wise scaling ignores.

Muon's idea: for each 2-D parameter $W \in \mathbb{R}^{m \times n}$, maintain the standard momentum $M$, but before applying it as the update, **orthogonalize $M$** — replace it with the matrix $UV^\top$ where $M = U \Sigma V^\top$ is its SVD. The update direction now has unit singular values across all components, treating the matrix's principal directions equally rather than letting one dominant direction dwarf the rest.

Computing the SVD explicitly is expensive; Muon uses a **Newton-Schulz iteration** to approximate $UV^\top$ in 5 matrix-matrix multiplies (cheap relative to the gradient computation itself).

For 1-D parameters (biases, layer norms, embeddings) and the LM head, Muon falls back to AdamW.

---

## How it works

### The Newton-Schulz orthogonalization

Given momentum matrix $M$, compute the polynomial approximation to $UV^\top$:

```
X = M / ||M||_F          # normalize
for k = 1 to 5:
    X = a · X + b · (X X^T X) + c · (X X^T X X^T X)
```

with coefficients $(a, b, c)$ chosen to push the singular values toward 1 in fewer iterations. After 5 iterations, $X \approx UV^\top$ to acceptable precision for an optimizer update. The cost is 5 matmuls per parameter per step — a few percent of the forward+backward cost on a typical transformer.

The update is then:

```
W ← W - lr · X
```

with the same weight-decay term as AdamW.

### What stays AdamW

- **Embeddings and LM head.** Treated as vectors per row/column; the matrix structure is different from internal linear layers. AdamW handles them as before.
- **Biases, norms.** 1-D parameters where matrix orthogonalization is undefined.

Roughly 95%+ of parameter mass (the FFN and attention weight matrices) uses Muon; the rest uses AdamW. The "Muon optimizer" in practice is a hybrid.

### Why orthogonalization helps

The intuition is that gradient momentum naturally develops large singular values along a few directions — the model "learns" along the dominant gradient component first. Orthogonalizing the update spreads the step equally across all directions in the momentum subspace, preventing dominant directions from over-dominating and helping the smaller directions catch up. This is a kind of preconditioning that AdamW's per-coordinate second moment cannot capture.

---

## Why it matters

- **Matches or beats AdamW at frontier scale.** Reported to converge as fast or faster than AdamW on transformer pretraining, at comparable wall-clock cost (the orthogonalization is a few percent overhead).
- **Better for the largest matrices.** The advantage is largest on the wide FFN matrices where gradient anisotropy is greatest. Smaller matrices behave roughly like AdamW.
- **Stack with FP8 training and WSD.** Muon has been validated alongside [FP8 training](fp8-training.md) and [WSD schedules](wsd-schedule.md), notably in Mellum 2's recipe.

---

## Gotchas & tricks

- **Newton-Schulz coefficients matter.** The published values target 5 iterations to a specific accuracy. Tuning iterations down (3–4) for speed introduces bias in the update direction. The standard is 5.
- **Mixing with AdamW for 1-D / embedding params.** Don't try to extend Muon to vector parameters — the orthogonalization is matrix-specific. Hybrid optimizer state is a small but real bookkeeping cost.
- **Different LR than AdamW.** The orthogonalized update has different magnitude than a raw momentum step; effective LRs typically need to be retuned ~2–3× when porting from AdamW.
- **Numerical precision.** Newton-Schulz is numerically robust but does multiply matrices repeatedly. In low precision (BF16, FP8), the orthogonalization step is often kept in higher precision (BF16/FP32) even when the rest of the training is FP8.
- **Public recipes mostly at small/medium scale.** Muon's frontier-scale validations (Mellum 2, several recent open releases) are emerging but the optimizer is much newer than AdamW; hyperparameters at very large scale are less established.

---

## Sources

- Paper: Original Muon optimizer announcement (Keller Jordan, 2024) — the Newton-Schulz orthogonalized momentum idea.
- Paper: *Mellum 2* — JetBrains, 2026 — applies Muon under FP8 hybrid precision at the 12B-MoE / 10.6T-token scale, with WSD scheduling.
- Background: optimizer-state and matrix-structure literature on transformer pretraining (Shampoo, K-FAC) for related matrix-aware optimizers.
