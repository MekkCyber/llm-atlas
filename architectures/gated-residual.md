# Gated Residual (GR)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Residual-stream widening: instead of the standard single-stream residual $x \leftarrow x + \text{block}(x)$, the residual is broken into **four parallel branches**, each written into by a subset of blocks; readout mixes them through an **elementwise output gate**. Introduced in Qwen3.8-Next as a modest-cost architectural change that measurably improves loss and downstream benchmarks at fixed active-parameter budget.

**Prereqs:** [transformer-block.md](transformer-block.md), [_normalization.md](_normalization.md)
**Related:** [qwen-sparse-attention.md](qwen-sparse-attention.md) · [../case-studies/qwen3-8-next.md](../case-studies/qwen3-8-next.md)

---

## What it is

A standard transformer block has one residual stream — a $d$-dimensional vector at each position that every block reads from and writes to. Attention and FFN blocks add their outputs back into that single stream. This forces all block outputs to share representational bandwidth and to be read out uniformly at every subsequent layer.

Gated Residual splits the stream into **four parallel sub-streams** with per-position, per-branch **elementwise gates** controlling how they combine at readout. Blocks are assigned to write into specific branches; downstream reads pull from all branches through the gate.

## How it works

**Four-branch residual.** Instead of $x \in \mathbb{R}^d$, the residual is $\{x^{(1)}, x^{(2)}, x^{(3)}, x^{(4)}\}$, each in $\mathbb{R}^d$ (or shaped subspaces). Blocks (attention, FFN, MoE) write into specific branches.

**Elementwise output gate.** At the next block's read, the combined input is

$$
x_{\text{read}} = \sum_i g^{(i)} \odot x^{(i)}
$$

where $g^{(i)} \in \mathbb{R}^d$ is a learned elementwise gate (per position, per dimension) with $\odot$ elementwise multiplication. Gates are typically produced by a small projection from a shared summary vector.

**Parameter cost.** Branches share dimension $d$; gates add $\sim 4d$ parameters per gating site. Total added parameter count is small relative to the block MLP/attention weights it enables.

**Effect on representational capacity.** Each branch can specialize (e.g., one carrying local information, one carrying long-range attention outputs) while the gate decides at readout how much of each to consume. This unshares what the single-stream residual force-shares.

## Why it matters

- **Beats single-stream residual at fixed active parameters.** In the Qwen3.8-Next ablations, GR improves both training loss and downstream benchmarks with negligible increase in active parameters.
- **Cheap.** Adds $\sim 4d$ gate params per site — a fraction of what a wider FFN or an extra attention head would cost for equivalent capacity gain.
- **Composes with MoE routing.** Different experts can write into different branches, giving the gate a natural handle on cross-expert combination.
- **Complements sparse-attention layers.** By separating stream contributions, downstream layers can gate away noise that a sparse-attention pass added.

## Gotchas & tricks

- **Branch assignment is a design choice.** Round-robin, block-type-based (attention → branch A, FFN → branch B), or learned gating on the write side — the paper's specific assignment matters and isn't fully specified in the abstract.
- **Gate initialization affects early dynamics.** Initialize gates near 1/4 (uniform mix) or near a slight preference for the dominant branch depending on what your downstream benchmark most values.
- **Interacts with normalization placement.** Where LayerNorm/RMSNorm sits relative to the branch merge matters; norm inside each branch produces different dynamics than norm on the merged stream.
- **Compatibility with KV cache and pipeline parallelism.** Four residual streams mean four times the residual-stream state to carry per layer — negligible for compute but slightly larger activation memory. Pipeline splits must respect branch boundaries.

## Sources

- Paper: *On the Design of Qwen3.8-Next Architecture: Evaluation, Efficiency, and Training Stability* — Qiu, Wang, Li, et al. — Qwen team / Alibaba, 2026 — arxiv.org/abs/2608.30320.
