# MiniMax Sparse Attention (MSA)

*Depth — blockwise sparse attention layered on top of Grouped Query Attention, trained inside a 109B multimodal model.*

**TL;DR:** A two-branch attention layer that drops in for dense GQA in a frontier-scale transformer. An **Index Branch** scores key-value blocks per query group; a **Main Branch** then runs exact softmax attention over the *selected* blocks only. Trained end-to-end inside a 109B-parameter multimodal model, MSA matches dense GQA accuracy while cutting per-token attention compute by **28.4× at 1M context**.

**Prereqs:** [_sparse-attention.md](_sparse-attention.md), [multi-head-attention.md](multi-head-attention.md) (GQA), [../fundamentals/attention.md](../fundamentals/attention.md)
**Related:** [mla.md](mla.md)

---

## What it is

A long-context attention layer designed to be a **drop-in replacement** for GQA in a production-scale transformer, without quality regression. The design constraint is that it must be trainable end-to-end and serve efficiently with standard block-sparse kernels.

## How it works

### Block decomposition

KV cache is sliced into contiguous blocks of fixed length $b$ (e.g. $b = 64$). For sequence length $L$ this gives $M = L/b$ blocks.

### Index Branch — block scoring

For each query group (GQA-style amortization across heads in the group), a learned linear projection produces a score per KV block:

$$
s_{g,m} = \phi_{\text{idx}}(Q_g) \cdot \psi_{\text{idx}}(K_{\text{block } m})
$$

The Index Branch has its own small parameter set and runs in $O(M)$ — cheap because it scores blocks rather than individual tokens. Per-group amortization means the index cost doesn't multiply by the number of heads.

### Block selection — top-$K'$

For each query group, the top-$K'$ blocks are selected (typical $K' \ll M$). Selection is differentiable via a straight-through estimator on the block scores.

### Main Branch — exact block-sparse attention

Standard softmax attention is computed over the *selected blocks only*:

$$
\text{Attn}(Q, \text{select}_{K'}(KV)) \quad \text{with exact softmax over the chosen blocks}
$$

Because selection is block-aligned, this is implemented as efficient block-sparse matmul (compatible with FlashAttention-style kernels at block granularity).

## Why it matters

- **28.4× per-token attention compute reduction at 1M context** vs dense GQA, with parity on broad eval suites.
- **End-to-end trainable** alongside the rest of the model — not bolted on post-hoc and not requiring a re-pretraining schedule.
- **Production-scale demonstration.** Frontier multimodal model at 109B parameters with sparse attention as the default attention layer. Not a research-scale ablation.
- **Compatible with existing serving stacks.** Block granularity allows standard block-sparse attention kernels.

## Gotchas & tricks

- **Block size $b$ is the central knob.** Small $b$ = finer selection but higher index overhead; large $b$ = coarser selection but cheaper indexing. The paper uses $b = 64$ as the default at the 109B scale.
- **Index Branch overfitting.** Because the Index Branch is the only path that sees all blocks, it can over-attend to positional priors (recent tokens, sink tokens) early in training. The paper notes this is mitigated by warmup with a softer top-$K'$ relaxation.
- **Selection is per-group, not per-head.** This is the GQA amortization — without it, the index branch cost would dominate. As with GQA, this is a small accuracy↔cost tradeoff vs per-head selection.
- **Long-context training stability.** Sparse attention at frontier scale only stabilizes with proper KV-cache normalization upstream; check that QK-norm or equivalent is in the block before MSA.
- **Doesn't subsume KV compression.** MSA reduces *positions attended*; KV compression ([MLA](mla.md)) reduces *bytes per position*. The two compose; production stacks should use both.

## Sources

- Paper: *MiniMax Sparse Attention* — Xu, Yang, Chen et al., MiniMax / NVIDIA, 2026 — [arXiv:2606.13392](https://arxiv.org/abs/2606.13392).
- Related: [_sparse-attention.md](_sparse-attention.md), [multi-head-attention.md](multi-head-attention.md), [mla.md](mla.md).
