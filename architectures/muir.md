# Muir
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A gated linear-attention block that separates the three gates (write, erase, readout) asymmetrically: the **erase gate uses a lower-rank projection than the write gate** (deciding what to forget is simpler than what to store), and a **matched-filter readout gate** rebalances the row norms of the fixed KV state so retrieval doesn't get swamped by accumulated drift. Beats GDN-2 and a Transformer baseline on perplexity and commonsense tasks from 166M to 1.34B parameters.

**Prereqs:** [attention](../fundamentals/attention.md), [_linear-attention](_linear-attention.md)
**Related:** [multi-head-attention](multi-head-attention.md), [mla](mla.md)

---

## What it is

A specific gated-linear-attention variant that can replace the attention block in a Transformer block one-to-one. It inherits the fixed-size KV state of linear attention and adds two structural tweaks to the gating:

- A **low-rank erase gate** (fewer learnable directions than the write gate).
- A **matched-filter readout** that normalizes state rows before the query inner product.

## How it works

Starting from a gated linear-attention layer with state $S_t \in \mathbb{R}^{d_v \times d_k}$:

**Write gate:** full-rank, same as GLA — the model needs to encode a rich new memory per step.
$$S_t^{\text{write}} = v_t \otimes \phi(k_t)$$

**Erase gate:** low-rank projection — Muir's claim is that the structure of "what to forget" is simpler than "what to store," so a bottlenecked gate suffices and regularizes the network:
$$\alpha_t = \sigma\big(U_\alpha V_\alpha^\top h_t\big), \quad \text{rank}(U_\alpha V_\alpha^\top) \ll d$$

$$S_t = \alpha_t \odot S_{t-1} + S_t^{\text{write}}$$

**Matched-filter readout:** before the query inner product, each row of $S_t$ is normalized (matched-filter form) so that retrieval depends on *direction* rather than cumulative norm:
$$\tilde S_t[i, :] = \frac{S_t[i, :]}{\|S_t[i, :]\| + \epsilon}, \quad y_t = \tilde S_t \cdot \phi(q_t)$$

This fixes a known failure mode of linear attention where early tokens dominate the readout because their state rows grow without bound.

## Why it matters

- Clean **inductive bias**: gates in a linear-attention cell don't all deserve equal capacity. Erase is a simpler decision than write, which is a simpler decision than readout.
- Reported quality: lower perplexity and higher commonsense-accuracy average than GDN-2 and a Transformer baseline across 166M, 335M, 760M, and 1.34B parameter scales. Preserves long-context extrapolation.
- The asymmetric-gate principle plausibly transfers to any gated SSM/linear-attention family (DeltaNet, RWKV-7, Mamba-2) — a design-knob worth trying in future variants.

## Gotchas & tricks

- The low-rank erase rank is a hyperparameter; too low and the model can't forget enough, too high and you've reinvented GLA. The paper sweeps ranks per scale.
- Matched-filter readout adds a per-row normalization at every step — cheap on TPU/GPU, but kernel-fused implementations are needed to avoid a memory round trip.
- Long-context extrapolation requires the row-norm bookkeeping to be numerically stable; use double precision for the normalization or add a floor $\epsilon$.

## Sources

- Paper: Muir: Linear Attention with Sensitivity-Allocated Gating and Matched-Filter Readout — Tabak & Schaefer, 2026 — [arXiv:2611.08376](https://arxiv.org/abs/2611.08376)
