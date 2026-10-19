# Attention Variants

*Taxonomy — the design space of attention block variants that trade quality, compute, and KV-cache size.*

**TL;DR:** Every modern LLM attention block answers four questions: *how many query heads, how many KV heads, how big is each head, and what structure (if any) is imposed between heads?* MHA → MQA → GQA is one axis (shrinking the KV side); MLA and VHA are two different reformulations that give MHA-level expressiveness at GQA-or-better inference cost. The modern default is GQA (any model since Llama 2); MLA is the strongest known cache-side reformulation; VHA is the strongest known head-side reformulation.

**Related taxonomies:** [_moe](_moe.md) · [_normalization](_normalization.md)
**Depth files covered here:** [multi-head-attention](multi-head-attention.md) · [gqa](gqa.md) · [mla](mla.md) · [virtual-head-attention](virtual-head-attention.md) · [copy-paste-lm](copy-paste-lm.md)

---

## The problem

Attention's cost at inference is dominated by the **KV cache**: per token, per layer, store one K and one V per head. For long context this is both VRAM-heavy and bandwidth-bound at decode time. On the training side, large head counts multiply projection FLOPs.

Every variant below is a different way to decouple *attention expressiveness* (how many heads of distinct behavior) from *attention cost* (how much state is cached, how many FLOPs per step).

---

## The shared pattern

All variants factor the attention block as:

$$
Q = X W_Q, \quad K = X W_K, \quad V = X W_V, \quad \text{heads}: (Q_i, K_i, V_i), \quad \text{out} = \mathrm{Concat}(\mathrm{Attn}_i) W_O
$$

They differ in three choices:

- **Head counts.** $H_Q$ query heads vs $H_{KV}$ key/value heads.
- **Factorization.** Whether K/V are materialized per head (MHA, GQA) or reconstructed from a shared low-rank latent (MLA).
- **Linear augmentation.** Whether additional *linear* premix/postmix operations widen training-time expressiveness while folding into projection weights at inference (VHA).

---

## Variants

| Technique | Key idea | Main tradeoff | When it wins |
| --- | --- | --- | --- |
| [multi-head-attention](multi-head-attention.md) | $H_Q = H_{KV}$, one K/V per query head | Full expressiveness, full KV cost | Reference baseline; small models |
| MQA (no depth file yet) | $H_{KV} = 1$ (all queries share one K/V) | Smallest KV cache; quality drop | Only when cache is the hard constraint |
| [gqa](gqa.md) | $H_{KV} = H_Q / g$ for group size $g$ | KV cache / $g$; small quality cost | **Modern default** — Llama 2+, Qwen, Mistral, Gemma |
| [mla](mla.md) | Share one low-rank KV latent across all heads; reconstruct per head via up-projection; decoupled RoPE | ~57× smaller KV cache per layer vs full MHA | Long-context inference; DeepSeek-V2/V3 |
| [virtual-head-attention](virtual-head-attention.md) | GQA-2 at inference; linear Q Premix + Linear Postmix during training fold into projection weights | Strict Pareto over GQA at matched inference cost | Any time GQA is appropriate — drop-in replacement |
| [copy-paste-lm](copy-paste-lm.md) | Add a pointer head over context positions, mixed with the vocab head via a learned gate | Extra ~linear-in-context compute per step; big long-context retrieval win | Long-context exact-copy workloads; sliding-window stacks |

---

## How to choose

- **Default:** GQA with group count $g \in \{4, 8\}$. Anything else needs a reason.
- **Need the smallest KV cache you can get:** MLA. The 2024–2026 state of the art for cache-side reformulation.
- **Want strictly better than GQA at the same inference cost:** VHA. Serving stack doesn't change; weights swap in.
- **Long-context retrieval is the main failure mode:** add the CPLM pointer head on top of whichever base attention variant you use. It is complementary to GQA/MLA/VHA.
- **Avoid pure MQA.** The one-KV-head extreme usually loses more quality than it saves in cache.

Combining: GQA + CPLM is natural (both compose cleanly). MLA + VHA has no published combination; the design spaces are orthogonal in principle but need validation.

---

## Adjacent but distinct

- [_positional-encoding](../fundamentals/_positional-encoding.md) — how positions get mixed into attention (RoPE, ALiBi, decoupled RoPE for MLA). Orthogonal choice to head structure.
- [_moe](_moe.md) — scaling FFNs via experts. Separate axis of model scaling; combines freely with any attention variant.
- SSM / Mamba — not variants of attention; a different mixer. Belongs in a separate taxonomy once we add it.

---

## Sources

- Paper: *Attention Is All You Need* — Vaswani et al., 2017 — baseline MHA.
- Paper: *Fast Transformer Decoding: One Write-Head is All You Need* — Shazeer, 2019 — MQA.
- Paper: *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints* — Ainslie et al., 2023.
- Paper: *DeepSeek-V2* — DeepSeek, 2024 — introduces MLA.
- Paper: *Virtual Head Attention* — Baidu / Peking University, 2026 — arXiv [2610.17959](https://arxiv.org/abs/2610.17959).
- Paper: *Copy-Paste Language Models* — 2026 — arXiv [2610.19247](https://arxiv.org/abs/2610.19247).
