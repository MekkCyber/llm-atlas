# Attention Residuals
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Standard Transformers propagate information through a single additive residual stream — every sublayer reads only the most recent hidden state. **Attention residuals** relax this: each sublayer reads all previous layers' outputs via a learned softmax — attention *over depth*. **Multi-Head Attention Residuals (MHAR)** splits that routing query into per-subspace heads so different feature subspaces can attend to different depths. The advantage over single-query routing grows with model width.

**Prereqs:** [transformer-block](transformer-block.md), [multi-head-attention](multi-head-attention.md)
**Related:** [../pre-training/mid-training](../pre-training/mid-training.md)

---

## What it is

The residual stream carries information across depth, but its "read" is fixed: layer $\ell$ sees only layer $\ell{-}1$'s output. Attention residuals learn a distribution over all prior layer outputs and let each sublayer pull selectively from earlier ones. Single-query routing (one distribution shared across the whole hidden width) forces every feature subspace to agree on which layers matter — a compromise that gets worse as width grows.

MHAR reshapes the routing query into $H$ per-subspace heads. Each head has its own softmax over the depth history; the read is block-diagonal.

## How it works

Given hidden states $h_1, \ldots, h_{\ell-1}$ from prior layers, MHAR computes a routing distribution per head:

$$
\alpha^{(h)}_{\ell, k} = \mathrm{softmax}_k\!\left( q^{(h)}_\ell \cdot k^{(h)}_k \right), \quad x^{(h)}_\ell = \sum_{k < \ell} \alpha^{(h)}_{\ell, k} \cdot h^{(h)}_k
$$

Each $h^{(h)}$ is the subspace slice of the residual for head $h$. Concatenating $x^{(h)}_\ell$ across $H$ heads gives the input read by sublayer $\ell$. Zero added parameters, negligible compute over vanilla attention residuals; $H=1$ recovers them exactly.

A hyperparameter-free default: set the number of routing heads equal to the number of KV heads. An **identity-preserving conversion** via *delta* attention residuals lets an existing additively-residual model be retrofitted mid-training without a loss spike.

## Why it matters

- Cheap and scale-safe. Validation loss improvements over standard Transformer at **100M / 350M / 1B: -0.049 / -0.080 / -0.063**.
- The single-head routing baseline is actually *worse* than additive residuals at 1B (+0.105) — MHAR's win over it widens from 0.010 → 0.168 across that range.
- Fused Triton kernels lift throughput from 0.2–0.5× to 0.55–0.88× of baseline at near-baseline peak memory.
- Retrofit works: 8B mid-training via delta conversion gains **+3.2 GSM8K, +3.1 GPQA**.

## Gotchas & tricks

- Depth-attention memory scales linearly with layer count. Cache the projected keys of past layers; the read cost is $O(L)$ per token, not $O(L^2)$.
- Single-query routing gets *worse* with width — don't ship it. Always use MHAR (or plain additive residuals) at $\ge 1$B.
- Setting routing heads to KV-head count is a defensible default across model sizes; it also matches existing kernel-fusion patterns.
- For retrofit, use the delta variant — direct swap changes the residual algebra and produces a loss step.

## Sources

- Paper: *Multi-Head Attention Residuals* — Luo, Cai, Hu, 2026 — [arXiv:2607.27230](https://arxiv.org/abs/2607.27230).
- Precursor: attention-over-depth ideas from prior "highway" and DenseNet-style residual routing.
