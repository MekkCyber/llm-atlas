# Linear attention
*Taxonomy — gated and non-gated attention variants with fixed-size state.*

**TL;DR:** Softmax attention has $O(L^2)$ cost and an $O(L)$ cache; linear attention replaces the softmax with a kernel feature map so attention becomes a cumulative sum into a *fixed-size* state $S \in \mathbb{R}^{d \times d}$. The modern variants are **gated** — write, erase, and readout gates control how the fixed state evolves — and the research space is organized around how those gates are parameterized. Introduced in the 2020 linear-transformer line, matured in Mamba-2 / GDN / DeltaNet / RWKV, and continues to produce new variants (Muir, 2611.08376).

**Related taxonomies:** [_normalization](_normalization.md)
**Depth files covered here:** [muir](muir.md)

---

## The problem

Softmax attention scales as $O(L^2)$ in FLOPs and $O(L)$ in KV cache, which dominates the cost of long-context training and inference. Linear attention reduces both to $O(L)$ FLOPs and $O(1)$ cache per layer — but at the cost of a bounded-capacity hidden state. Every design decision in the family is about getting as much representational power out of that fixed state as possible.

## The shared pattern

A linear-attention layer maintains a running state $S_t \in \mathbb{R}^{d_v \times d_k}$ updated per token:

```
S_t  =  α_t  ⊙  S_{t-1}  +  v_t  ⊗  φ(k_t)        (write)
y_t  =  readout(S_t, φ(q_t))                       (read)
```

where $\phi$ is a kernel feature map (identity, ELU+1, exponential, learned MLP), $\otimes$ is the outer product, and $\alpha_t$ is a **gate** (scalar / vector / matrix) that erases past state. Variants differ in:

- **Shape of $\alpha_t$** (scalar, diagonal, low-rank, full-rank).
- **Feature map $\phi$** (fixed vs learned).
- **Readout** (plain inner product, normalization, matched filter, delta rule).

## Variants

| Technique | Key idea | Main tradeoff | When it wins |
| --- | --- | --- | --- |
| Linear Transformer (Katharopoulos) | Drop softmax, use $\phi(q)^\top \phi(k)$ | Quality lags softmax | Original proof-of-concept |
| RWKV | Time-mixing + channel-mixing with scalar time-decay | Long training history | Resource-constrained open-source |
| Mamba / S4 | Diagonal SSM, input-dependent $\Delta t$ | Weak at associative recall | Long context, low memory |
| GLA / GDN-2 | Gated linear attention with data-dependent diagonal gates | Hard to reach softmax quality at same params | Strong linear baseline |
| DeltaNet | Delta-rule write (associative memory update) | Slightly slower than diagonal variants | Associative recall tasks |
| [muir](muir.md) | Low-rank erase + full-rank write + matched-filter readout | Added bookkeeping in readout | Strong perplexity at 166M–1.34B |

## How to choose

- **Softmax if you can afford it.** Modern GPUs and FlashAttention make softmax competitive at long contexts; linear attention wins only where cache or latency is the hard constraint (edge inference, extreme-context pretraining).
- **Mamba / state-space for very long context** with modest recall demands.
- **Gated linear (GLA family, Muir) for the strongest quality-at-fixed-state.** Treat the gates asymmetrically — erase and write want different capacity.
- **Hybrids** (Jamba, Zamba, Samba) mix linear-attention and softmax blocks — standard pattern for production models that need both.

## Adjacent but distinct

- [MLA](mla.md) — full softmax attention with a compressed KV cache; different mechanism (low-rank cache vs fixed-state recurrence).
- [GQA / MQA](multi-head-attention.md) — reduce cache by sharing heads, not by swapping attention form.
- Sparse / sliding-window attention — still softmax, just structured masks.

## Sources

- Paper: Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention — Katharopoulos et al., 2020 — https://arxiv.org/abs/2006.16236
- Paper: Mamba: Linear-Time Sequence Modeling with Selective State Spaces — Gu & Dao, 2023 — https://arxiv.org/abs/2312.00752
- Paper: Gated Linear Attention — Yang et al., 2024 — https://arxiv.org/abs/2312.06635
- Paper: Muir: Linear Attention with Sensitivity-Allocated Gating and Matched-Filter Readout — Tabak & Schaefer, 2026 — [arXiv:2611.08376](https://arxiv.org/abs/2611.08376)
