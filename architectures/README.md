# Architectures

*The shape of the model itself — how tokens flow through blocks, how attention mixes information, how experts specialize, how sequences are indexed.*

---

## What This Is

Everything in this folder is about the model's internal structure: the building blocks, their connections, and the variants that modern LLMs have converged on. Pretraining, post-training, quantization, and inference all operate on a specific architecture — this is where the architecture is defined.

---

## What Belongs Here

- **Transformer block** — attention + FFN + residual + norm, the repeating unit.
- **Attention variants** — multi-head, MQA, GQA, MLA, sliding window, sparse.
- **FFN variants** — dense, SwiGLU, mixture-of-experts (MoE).
- **Normalization** — LayerNorm, RMSNorm, pre-norm vs. post-norm placement.
- **State-space models** — Mamba, S4, and the post-Transformer line.
- **Design tradeoffs** — depth vs. width, head count, tied embeddings, biases.

## Reading Order

1. Transformer block
2. Multi-head attention (and variants: MQA, GQA, MLA)
3. FFN and SwiGLU
4. Mixture-of-experts
5. State-space models (Mamba, S4)

---

## Overview Pages (taxonomies)

- [Normalization](_normalization.md) — LayerNorm, RMSNorm, pre/post/reordered placement.
- [Mixture-of-Experts](_moe.md) — sparse FFN variants and their routing / balancing choices.

## Concept Pages (depth)

- [Transformer Block](transformer-block.md)
- [Multi-Head Attention](multi-head-attention.md)
- [Multi-head Latent Attention (MLA)](mla.md)
- [QK-norm](qk-norm.md)
- [Reordered norm](reordered-norm.md)
- [DeepSeekMoE](deepseek-moe.md)
- [Load-balancing auxiliary loss](load-balancing-loss.md)
- [Sequence-wise balance loss](sequence-wise-balance-loss.md)
- [Auxiliary-loss-free balancing](aux-loss-free-balancing.md)
- [Expert capacity factor](capacity-factor.md)

---

## Related

- [fundamentals/](../fundamentals/) — the primitives these architectures compose.
- [pre-training/](../pre-training/) — how architectural choices interact with scaling.
- [inference/](../inference/) — which architectural choices cost you at serving time.
