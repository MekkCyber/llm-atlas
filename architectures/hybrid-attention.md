# Hybrid Attention
*Depth — layer-wise mixing of full attention and linear attention within a single transformer stack.*

**TL;DR:** Full self-attention is $O(n^2)$; linear-attention variants (GatedDeltaNet, Mamba-style, DeltaNet, Linear Transformer) are $O(n)$ but typically lose quality on long-range / precise-recall tasks. **Hybrid attention** keeps some layers full and others linear, giving the quality of full attention where it matters and the cost of linear attention everywhere else. The pattern is becoming standard in unified multimodal models.

**Prereqs:** [multi-head-attention](multi-head-attention.md)
**Related:** [mla](mla.md), [_moe](_moe.md)

---

## What it is

A transformer stack in which the attention sub-layer *alternates* between two attention operators across depth:

- **Full attention:** standard quadratic self-attention — precise, position-exact recall; $O(n^2)$ time and KV cache.
- **Linear attention** (e.g., GatedDeltaNet, Mamba-2, DeltaNet, RetNet, Linear Transformer): state-space or feature-map-based attention that runs in $O(n)$ time with a bounded-size recurrent state.

Which layers are full vs. linear is a hyperparameter. Typical patterns: 1-in-K full (K = 4–8), interleaved blocks of M linear then one full, or heavy-at-the-start / heavy-at-the-end.

## How it works

For each layer $\ell$:

$$
y_\ell = x_\ell + \mathrm{Attn}_\ell(\mathrm{Norm}(x_\ell))
$$

$$
\mathrm{Attn}_\ell = \begin{cases} \text{Full softmax attention} & \text{if } \ell \in F \\ \text{Linear attention (e.g., GatedDeltaNet)} & \text{otherwise} \end{cases}
$$

The FFN half of the block is unchanged. During inference, only full-attention layers consume KV cache; linear layers hold a fixed-size recurrent state, so KV footprint scales with $|F|$ instead of with total depth.

The design trick is that **copy / recall / long-range in-context tricks** (induction heads, needle-in-haystack retrieval) are overwhelmingly done by a small number of full-attention layers. Models like UniSpark (hybrid full + GatedDeltaNet) and Jamba (hybrid attention + Mamba) ablate this: hiding more than ~25% of the layers as linear starts to cost on recall; below that threshold, the quality hit vanishes.

## Why it matters

- **Long-context inference.** Linear-attention layers' KV cost is constant in sequence length. In a 1M-token context, moving 75% of layers to linear attention is the difference between infeasible and routine.
- **Unified multimodal.** When the same stack has to handle text, image token sequences (which are long but have strong local structure), and video, linear attention absorbs the token explosion while full attention retains global understanding.
- **Training stability.** Mixing in full-attention layers keeps gradient flow through the attention pathway healthy; pure-linear stacks (early Mamba) had stability tradeoffs that hybrid avoids.

## Gotchas & tricks

- **Placement matters.** Full-attention layers should be *distributed through depth*, not clustered. Early full layers set up induction-like behavior; late full layers polish the output.
- **Head size and head count can differ** between full and linear layers — they are different operators; shared sizing is a cosmetic constraint, not a functional one.
- **KV cache implementations** need two code paths — easy to under-benchmark the hybrid if only the full-attention path is optimized.
- **Rotary positional encodings** apply to the full layers; linear layers typically use their own implicit position encoding (gated state updates). Don't try to share.

## Sources

- Paper: *UniSpark: Exploring Hybrid Attention for Multimodal Understanding and Generation* — 2026 — https://arxiv.org/abs/2611.16517
- Paper: *Jamba: A Hybrid Transformer-Mamba Language Model* — 2024.
- Paper: *Gated DeltaNet* — 2024.
