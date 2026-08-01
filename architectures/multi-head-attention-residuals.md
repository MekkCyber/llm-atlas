# Multi-Head Attention Residuals (MHAR)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** In a standard Transformer, each sublayer reads only the *most recent* residual state. Attention residuals extend this by letting each sublayer softmax-attend over **all previous layer outputs** — an "attention over depth." MHAR fixes the width bottleneck in that scheme: instead of one shared routing query, each attention head (or feature subspace) gets its own softmax over the depth history. Zero extra parameters, monotone quality gains at 100M → 1B, and fused Triton kernels bring the throughput cost near baseline.

**Prereqs:** [multi-head-attention](multi-head-attention.md), [transformer-block](transformer-block.md), [attention](../fundamentals/attention.md)
**Related:** [mla](mla.md)

---

## What it is

The standard Transformer propagates information across depth through a single additive residual stream: sublayer $L$ reads state $h_{L-1}$ (only the most recent output). *Attention residuals* relax this — sublayer $L$ softmax-attends over the outputs of all previous layers $\{h_0, h_1, \ldots, h_{L-1}\}$:

$$
\tilde h_L \;=\; \sum_{k=0}^{L-1} \alpha_L^{(k)} \, h_k , \quad
\alpha_L = \mathrm{softmax}\bigl(\text{query}(h_{L-1}) \cdot \text{key}(h_k)\bigr)
$$

MHAR replaces the **single** routing query with $H$ per-subspace queries — one softmax over depth per feature subspace / head. The read becomes block-diagonal across the width dimension; $H=1$ recovers vanilla attention residuals exactly.

## How it works

- Split the routing query into $H$ heads (identical reshape trick to standard multi-head attention). Each head gets its own softmax over the depth history $\{h_0, \ldots, h_{L-1}\}$.
- **Zero added parameters** (just a reshape of the existing query) and negligible extra compute.
- Recommended default: **$H$ = number of KV heads** — hyperparameter-free and empirically near-optimal.
- Implemented with fused Triton routing kernels: attention-residual training goes from **0.2–0.5×** of plain-additive throughput to **0.55–0.88×**.
- An *identity-preserving conversion* (delta attention residuals) allows retrofitting MHAR into a pretrained model for mid-training.

## Why it matters

The residual stream is a hidden bottleneck: adding more depth doesn't help if every sublayer sees only the last output. Attention residuals widen the channel — but at scale, a single routing query forces disagreeing feature subspaces to compromise. MHAR closes that gap and, critically, the **advantage grows with scale**:

- Val loss vs standard Transformer: **-0.049 at 100M · -0.080 at 350M · -0.063 at 1B** (all on FineWeb-Edu).
- Single-head attention residuals *hurt* by 0.105 at 1B; MHAR's gap over single-head widens from 0.010 (100M) to 0.168 (1B).
- 8B mid-training via delta-attention conversion: **+3.2 GSM8K · +3.1 GPQA**.

The direct probe of trained queries confirms that heads learn *disagreeing* preferences over layers — the theoretical motivation shows up in the weights.

## Gotchas & tricks

- Single-head attention residuals are a **negative-scaling** technique — they help small models but hurt large ones. Don't ship them without multi-head routing.
- The fused Triton kernel is what makes MHAR practical; without it you pay ~2× on training throughput.
- Setting $H$ equal to KV heads is the "free lunch" default; tuning $H$ further gave marginal gains at 1B in the paper.

## Sources

- Paper: *Multi-Head Attention Residuals* — Luo, Cai, Hu, 2026 — [arXiv:2607.27230](https://arxiv.org/abs/2607.27230)
