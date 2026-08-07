# SWA-Only RoPE

*Depth — apply RoPE only to sliding-window-attention layers, omit it from global-attention layers, for cheap 256K-context training.*

**TL;DR:** RoPE anchors positional information into every attention layer, but at very long contexts (128K+) it becomes an obstacle: RoPE-based long-context training needs frequency rescaling, positional interpolation, or NTK-aware tricks to extend beyond the training window. K-EXAONE 2.0 sidesteps this in hybrid local/global architectures: **apply RoPE only to the sliding-window (local) layers, and omit it entirely from the global layers**. The global layers carry no positional encoding at all and rely on the local layers to inject position. Enables clean 8K → 64K → 256K staged extension.

**Prereqs:** [../fundamentals/rope.md](../fundamentals/rope.md), [multi-head-attention.md](multi-head-attention.md)
**Related:** [../fundamentals/_positional-encoding.md](../fundamentals/_positional-encoding.md) · [../fundamentals/sinusoidal-encoding.md](../fundamentals/sinusoidal-encoding.md) · [transformer-block.md](transformer-block.md) · [../case-studies/k-exaone-2.md](../case-studies/k-exaone-2.md)

---

## What it is

A design choice for hybrid architectures that interleave *local* sliding-window attention layers with *global* full-attention layers (Gemma 2, K-EXAONE, Mistral-style hybrid stacks). Standard practice applies RoPE to every layer's queries and keys. **SWA-only RoPE** removes RoPE from the global layers and keeps it only on the local (sliding-window) layers.

The global layers then have *no explicit positional encoding* — they see the token stream unordered from a positional-encoding standpoint. Order information reaches them only via the residual stream, populated by the local RoPE-carrying layers below.

## How it works

The K-EXAONE architecture uses an `LLLG` block: three local sliding-window attention layers `L`, one global attention layer `G`, repeated. With SWA-only RoPE:

- **Local `L` layers:** standard RoPE on Q and K, sliding-window mask.
- **Global `G` layer:** no RoPE, full-attention over the full context window.

At training time:

1. Local layers learn to represent position-sensitive patterns (local n-gram structure, syntactic dependencies) with the help of RoPE.
2. Global layers learn to integrate long-range signal *from the already-position-aware local representations* — they don't need their own RoPE because the residual stream already encodes position by the time the global layer runs.

Staged context-length extension (K-EXAONE 2.0's 8K → 64K → 256K) then applies only to the local RoPE frequencies; the global layers need no positional re-scaling because they have no positional encoding to re-scale.

## Why it matters

- **Cheap long-context extension.** RoPE-based long-context requires frequency rescaling / NTK-aware interpolation / YaRN-style tricks that all interact with every RoPE layer. Halving the number of RoPE layers halves the surface area for these adjustments.
- **Cleaner staged extension.** K-EXAONE 2.0 reports perfect Needle-in-a-Haystack retrieval at 256K after a two-stage extension (mid-Stage 1: 64K; mid-Stage 2: 256K) — the global layers require no reconfiguration between stages.
- **Compute saving.** RoPE per layer is a small but nonzero cost; skipping it on ~25% of layers (in an LLLG block) is a small pretty-much-free win.
- **Modular reasoning.** Separates the "who's next to whom" (local) from the "what depends on what across the doc" (global) concerns, matching what the two attention types are architecturally already for.

## Gotchas & tricks

- **Only makes sense in hybrid local/global architectures.** In a stack that's *all* full-attention (standard transformer), removing RoPE from any layer removes all positional info and breaks the model.
- **Ratio of local:global matters.** LLLG (3:1) works; heavier reliance on global layers (LG or all-G) leaves too few RoPE-carrying layers to inject position robustly. Ablations on this ratio are useful.
- **Not a plug-in replacement.** If you swap RoPE on all layers → SWA-only RoPE post-training, the model will need substantial adaptation — the global layers were relying on their own RoPE and won't degrade gracefully. Design in from step 0.
- **Verify on retrieval, not just perplexity.** Perplexity can look fine while long-range retrieval quietly degrades. NIAH-style tests at each staged extension are essential.
- **Interacts with rotation-noise MoE upcycling** in the K-EXAONE 2.0 recipe: both are norm-preserving / low-disruption changes designed to compose with existing pretraining dynamics.

## Sources

- Paper: *K-EXAONE 2.0 Technical Report* — LG AI Research, 2026 — [arXiv 2608.04505](https://arxiv.org/abs/2608.04505). Introduces SWA-only RoPE for the 256K-context staged extension.
- Related: Gemma 2 (hybrid local/global blocks with RoPE on both) — the baseline that SWA-only RoPE simplifies against.
