# FOAM
*Depth — a lighter second-order optimizer in the Shampoo/SOAP line, with input-side-only preconditioning and a factored Adam second moment.*

**TL;DR:** FOAM (**F**actored **O**ne-sided **A**dam-**M**oment) is a cheaper variant of the SOAP optimizer. It keeps SOAP's preconditioning on the **input side** of each weight matrix, drops the output-side preconditioner entirely, and replaces Adam's dense second-moment statistic with a **factored** per-row estimate. The result matches SOAP and Muon on validation loss from 124M to 760M LLM pretraining, at near-Muon wall-clock, and halves optimizer memory relative to SOAP.

**Prereqs:** [_training-stability](_training-stability.md), [_lr-schedules](_lr-schedules.md)
**Related:** [_optimizers](_optimizers.md)

---

## What it is

Shampoo and its successors (SOAP, Muon) precondition the gradient by an inverse root of the left and right Gram matrices $L = G G^\top$, $R = G^\top G$. This gives a cheap approximation to a full second-order update and is competitive with AdamW on LLM pretraining.

Two costs dominate at scale:

1. **Memory.** You store $L$ and $R$, plus an Adam second moment, for every weight matrix.
2. **Compute.** You periodically eigendecompose $L$ and $R$ to recompute the inverse root.

FOAM is a design choice that drops roughly half of both while keeping the quality.

## How it works

### One-sided preconditioning

For a weight matrix $W \in \mathbb{R}^{d_\text{in} \times d_\text{out}}$ with gradient $G$, SOAP applies

$$
\tilde{G} = L^{-1/4} \cdot G \cdot R^{-1/4}
$$

before the Adam-style update. FOAM observes empirically that for LLM-shape matrices (tall or wide, not square), almost all the benefit comes from the input-side term:

$$
\tilde{G}_{\text{FOAM}} = L^{-1/4} \cdot G
$$

Drop $R$, drop the second eigendecomposition.

### Factored second moment

Adam stores a per-element second-moment matrix $V \in \mathbb{R}^{d_\text{in} \times d_\text{out}}$ of the same size as $W$. FOAM factors it as

$$
V_{ij} \approx u_i \cdot v_j
$$

so only two vectors ($u$, $v$) per weight matrix are stored. This echoes Adafactor's trick but is compatible with the Shampoo preconditioner because the preconditioner has already removed most of the inter-coordinate correlation.

The final update is

$$
W \leftarrow W - \eta \cdot \frac{\tilde{G}_{\text{FOAM}}}{\sqrt{\hat{V}} + \epsilon}
$$

with $\hat{V}$ reconstructed on-the-fly from $u, v$.

## Why it matters

- Shampoo/SOAP/Muon are the active frontier for "better than AdamW" on LLM pretraining. The main reason they're not the default yet is cost.
- FOAM halves the optimizer memory and removes one eigendecomposition. That's what it takes to seriously compete with AdamW on infrastructure, not only on validation loss.
- Robustness across a wide learning-rate range matters at scale because sweeping LR for a frontier run is expensive.

## Gotchas & tricks

- The one-sided choice ($L$, not $R$) assumes the LLM-matrix shape convention (`[in, out]`). If you implement with the opposite convention, swap sides.
- The factored second moment degrades when the gradient's second moment is sharply non-separable — rare in LLM pretraining, common in some fine-tuning regimes.
- Still periodic eigendecompositions on $L$. Amortize across many steps as SOAP does.

## Sources

- Paper: *Factored One-sided Adam-Moment for Practical and Scalable SOAP (FOAM)* — Yu, Han, 2026 — https://arxiv.org/abs/2611.16905
- Paper: *SOAP: Improving and Stabilizing Shampoo using Adam* — 2024.
- Paper: *Muon* — 2024.
