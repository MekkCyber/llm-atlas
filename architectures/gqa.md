# Grouped-Query Attention (GQA)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** The default modern attention variant. **Share one K/V head across a group of query heads**, reducing KV-cache size and inference bandwidth proportionally to the group size. GQA-1 is standard MHA; GQA-H (one KV head total) is MQA; practical sweet spots are GQA-2, GQA-4, or GQA-8. Nearly every open LLM since Llama 2-70B uses GQA.

**Prereqs:** [multi-head-attention.md](./multi-head-attention.md), [../fundamentals/attention.md](../fundamentals/attention.md)
**Related:** [mla.md](./mla.md), [transformer-block.md](./transformer-block.md)

---

## What it is

Standard multi-head attention has $H_Q = H_{KV}$ query, key, and value heads. The KV cache at inference stores one K and one V vector per head per token per layer — so KV cache size is $\propto 2 \cdot H_{KV} \cdot d_{\text{head}} \cdot \text{seqlen}$. For long context, KV cache dominates memory bandwidth and GPU VRAM.

**Multi-Query Attention (MQA)** collapses this by using one KV head for all queries ($H_{KV} = 1$). The KV cache shrinks $H_Q\times$, but quality drops noticeably — one KV head cannot support the diversity of attention patterns $H_Q$ heads learn.

**Grouped-Query Attention (GQA)** is the compromise: use $H_{KV}$ KV heads with $H_Q$ query heads where $H_Q = g \cdot H_{KV}$. Each group of $g$ query heads shares one KV head. GQA-$H_{KV}$ denotes the variant with $H_{KV}$ KV heads.

- GQA-$H_Q$ = standard MHA.
- GQA-1 = MQA.
- GQA-8 or GQA-4 — typical modern choices.

---

## How it works

### Forward pass

For input $X \in \mathbb{R}^{n \times d}$:

- Project $X$ to $H_Q$ query heads: $Q \in \mathbb{R}^{n \times H_Q \times d_h}$.
- Project $X$ to $H_{KV}$ key/value heads: $K, V \in \mathbb{R}^{n \times H_{KV} \times d_h}$.
- For each query head $i$, pick its KV group $g(i) = \lfloor i / g \rfloor$ and compute attention against $K_{g(i)}, V_{g(i)}$.
- Concatenate $H_Q$ outputs and project with $W_O$.

### KV cache at inference

Only $H_{KV}$ KV vectors are cached per token per layer, instead of $H_Q$. Memory saving: $H_Q / H_{KV}$ ratio. For Llama 2-70B (GQA-8, $H_Q = 64$) the cache is 8× smaller than MHA.

### Training

Llama 2's recipe trains from an MHA checkpoint by **mean-pooling** the KV heads within each group, then continuing pretraining — "uptraining." Modern recipes initialize with GQA directly.

---

## Why it matters

- **Standard across modern LLMs.** Llama 2, Llama 3, Mistral, Qwen, DeepSeek-coders, Gemma — all GQA. The question "MHA or GQA?" is settled; the only remaining variable is the group count.
- **Bandwidth-bound inference.** Decoding is memory-bandwidth-bound (one token at a time, must stream the full KV cache). Shrinking the cache almost linearly improves decode throughput.
- **Quality is nearly matched.** With $H_{KV} \geq 4$ at Llama-70B scale, benchmark deltas vs MHA are typically within noise. MQA (one head) is where you start to see real drops.
- **Enabled MLA.** [MLA](./mla.md) is the next step past GQA — share a low-rank *latent* across all heads and reconstruct K, V per-head on-the-fly, saving even more cache at comparable quality.

---

## Gotchas & tricks

- **Group count is a quality-vs-cache knob.** Common defaults: 70B models use GQA-8; 7B–14B models use GQA-4; some small models go GQA-2. Below GQA-2 (MQA), expect quality drops that outweigh cache savings for most workloads.
- **Combines freely with RoPE.** Query heads still get their own rotary rotation; the KV sharing happens on already-rotated keys/values.
- **Uptraining works.** Mean-pooling KV weights of an MHA checkpoint, then continuing training for ~5% of pretraining compute, recovers quality — cheaper than retraining from scratch.
- **Virtual Head Attention improves on it.** [virtual-head-attention](./virtual-head-attention.md) retains GQA's inference footprint while adding linear premix/postmix operations trained in and then folded into projection weights — strictly Pareto-better than GQA at matched inference compute.

---

## Sources

- Paper: *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints* — Ainslie et al., Google, 2023 — introduces GQA and the uptraining recipe.
- Paper: *Fast Transformer Decoding: One Write-Head is All You Need* — Shazeer, 2019 — the MQA precursor.
- Paper: *Llama 2* — Touvron et al., Meta, 2023 — first widely-used GQA LLM release.
