# Hybrid Attention
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Instead of picking one attention flavor per model, **hybrid attention** interleaves multiple attention variants within the same backbone so each handles the pattern it's best at. Chimera (Adobe, 2026) combines **Kimi Delta Attention** (linear-time, long-context), **Multi-head Latent Attention (MLA)** (global interaction, KV-compressed), and **modality-aware short convolutions** (local spatiotemporal context). Runs positionally-embedding-free on unified token sequences and reaches Chinchilla-optimal scaling with 7.3× fewer FLOPs than matched baselines.

**Prereqs:** [mla](mla.md), [multi-head-attention](multi-head-attention.md)
**Related:** [_moe](_moe.md), [../fundamentals/_positional-encoding](../fundamentals/_positional-encoding.md)

---

## What it is

Every attention variant is a compromise: full softmax attention is quadratic; linear/state-space variants trade recall for cost; local convolutions capture only nearby structure. Rather than pick one and eat the cost, hybrid attention **stacks different attention types in different layers (or interleaves them within a layer)** and lets each cover its strength. The design question moves from "which variant?" to "which interleaving pattern, and in what ratio?"

## How it works

Chimera's recipe uses three attention modules in a repeating pattern:

1. **Kimi Delta Attention** — a linear-time softmax alternative for long-context, high-token-count reads.
2. **Multi-head Latent Attention (MLA)** — periodic full-attention blocks (KV-compressed) providing global interaction and mixing.
3. **Modality-aware short convolutions** — 1D/2D/3D depthwise convolutions with modality-specific kernels, capturing local spatiotemporal patterns.

All three consume the same unified token stream (image, video, and text tokens concatenated), with no positional embeddings — position is carried implicitly by the convolution and by MLA's learned biases.

Sparsity happens at the parameter level: the 11B model activates only 2B params per token (MoE-style routing on top of the hybrid attention stack).

Compute-optimal scaling: **HeteroP** is a module-wise scaling scheme that lets each module type (Delta, MLA, conv) have its own hyperparameter transfer curve. Tune small, scale large without re-searching.

## Why it matters

- **7.3× fewer FLOPs** than matched-quality baselines at 11B (2B activated). Not a marginal win.
- **Zero-shot temporal extrapolation**: video generation extends from 5 → 30 s with minimal quality degradation — the linear-time attention paths hold up outside their training length.
- **Blueprint for token-intensive generation**: high-res images, long video, multimodal context are all quadratic-explosion regimes for full attention. Hybrid stacks make the compute tractable without paying the recall cost of pure-linear stacks.
- **HeteroP is reusable** — module-wise scaling transfer is applicable to any hybrid stack, not just Chimera's.

## Gotchas & tricks

- The interleaving *ratio* matters more than the individual choices — too-frequent full-attention layers recover the quadratic cost, too-sparse ones lose global recall.
- Positional-embedding-free only works when at least one module carries position implicitly (convolutions do). Pure linear + MLA stacks still need some form of position.
- Modality-aware short convs need per-modality kernels; sharing them across modalities hurts. Route by token type.
- Kernel fusion is essential — a naive implementation of three attention flavors in sequence pays serialization overhead. Chimera fuses adjacent modules.

## Sources

- Paper: *Chimera: Designing and Chinchilla-Scaling Hybrid Visual Diffusion Transformers* — Ge et al., Adobe Research, 2026 — [arXiv:2607.28611](https://arxiv.org/abs/2607.28611).
- Precursor: hybrid Mamba/attention stacks (Jamba, Zamba) established the interleaving pattern for language models.
