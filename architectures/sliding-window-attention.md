# Sliding-Window Attention (SWA)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Restrict each query to attend only to a fixed window of recent keys. KV cache, FLOPs, and memory grow $O(W)$ per token instead of $O(n)$, where $W$ is the window size. SWA is the "efficient" half of nearly every modern hybrid attention stack — Mistral, Gemma, GPT-OSS, RecurrentGemma, etc. The 2026 *Rethinking* paper shows SWA mostly shapes how *fast* long-context capability emerges, while *long-range retrieval* still requires the full-attention layers it's paired with.

**Prereqs:** [attention](../fundamentals/attention.md), [multi-head-attention](multi-head-attention.md)
**Related:** [_hybrid-attention](_hybrid-attention.md), [mla](mla.md)

---

## What it is

Standard self-attention has each query token attend to all keys, giving $O(n^2)$ time and $O(n)$ per-token KV cache. Sliding-window attention restricts the attention mask to a local window: query at position $i$ attends only to keys at positions $[i - W, i]$ for some fixed window $W$ (a few hundred to a few thousand tokens).

The effect on resource usage:

- **KV cache:** $O(W)$ scalars per layer per token — bounded, regardless of sequence length.
- **Attention FLOPs:** $O(n \cdot W)$ instead of $O(n^2)$.
- **Receptive field:** with $L$ stacked SWA layers, the effective receptive field is $L \cdot W$ — like a CNN's growing receptive field, information flows long-range through stacking.

---

## How it works

### The masked attention

For sequence length $n$ and window $W$:

$$
M_{ij} = \begin{cases} 0 & \text{if } i - W \le j \le i \\ -\infty & \text{otherwise} \end{cases}
$$

$$
\mathrm{SWA}(Q, K, V) = \mathrm{softmax}\!\left( \frac{Q K^\top}{\sqrt{d_k}} + M \right) V
$$

In efficient kernels (FlashAttention, etc.) the mask is implicit: blocks outside the window are never computed.

### KV cache eviction

At decode time, the oldest entries fall outside the window and can be evicted. The KV cache for an SWA layer is a ring buffer of size $W$, regardless of total generated length.

### Stacked receptive field

A single SWA layer sees only the last $W$ tokens. Two stacked layers see $2W$ (through the residual chain). $L$ layers see $L \cdot W$ — the receptive field grows with depth, similar to a stack of 1D convolutions.

This is the trade Mistral / Gemma make: $W = 4096$, $L \approx 32$ gives a stacked receptive field of $\sim 100$K tokens with $O(W)$ per-layer KV.

---

## Why it matters

- **The standard efficient-attention building block** in modern hybrid architectures. Pairs with full attention (Mistral 7B), MLA, or recurrent mixers (RecurrentGemma) to deliver long-context capability at bounded KV cost.
- **Hardware-friendly.** Fixed window size means kernel implementations can be predictable in tile sizes, prefetch patterns, and KV buffer geometry — much easier than dynamic sparsity.
- **Behavior under stacking.** The 2026 *Rethinking* paper shows SWA + full-attention hybrids ultimately match full-attention models on long context given enough training; SWA shapes the *speed* of long-context capability emergence rather than the asymptotic ceiling.

---

## Gotchas & tricks

- **SWA-only models can't retrieve long-range.** Without at least one full-attention layer, the stacked receptive field is technically large but information flow degrades with each hop. Retrieval-heavy tasks (needle-in-haystack at 100K+) need full-attention layers somewhere in the stack — see [_hybrid-attention](_hybrid-attention.md).
- **Window size is a real hyperparameter.** Too small ($W < $ typical "useful local context") and stacking can't recover global info. Too large and you lose the FLOP/KV savings. $W \in [512, 8192]$ is the practical range across modern models.
- **Causal mask interaction.** SWA is usually combined with the causal mask; the union is $\{j : i - W \le j \le i\}$.
- **Position-encoding interaction.** RoPE-style positional encodings extrapolate poorly outside their training window. If you train an SWA model with $W = 4096$ and want to serve with a larger window, retrofitting RoPE (NTK / YaRN) is the same problem as for full attention.
- **Combine, don't replace.** Frontier open models that ship SWA always pair it with at least one of: full attention slices (every $N$th layer), MLA (compress the full-attention KV), or recurrent mixers (Mamba). Pure-SWA stacks are vanishingly rare in 2025–2026 releases.

---

## Sources

- Paper: *Longformer: The Long-Document Transformer* — Beltagy et al., 2020 — early sliding-window + global attention recipe.
- Paper: *Mistral 7B* — Jiang et al., 2023 — SWA with $W = 4096$ as a frontier-quality decoder default.
- Paper: *Rethinking the Role of Efficient Attention in Hybrid Architectures* — Ziqing Qiao et al., Tsinghua / OpenBMB, 2026 — [arXiv:2606.15378](https://arxiv.org/abs/2606.15378) — mechanism analysis of SWA inside hybrid stacks.
