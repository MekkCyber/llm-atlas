# Dynamic Sparse Attention — Content-Adaptive Block Partitioning
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** For high-resolution joint video+audio diffusion, full attention is quadratic *and* increasingly wasteful as resolution scales (most tokens are redundant). **Prism** (2026) sparsifies attention at **training time** by partitioning tokens into spatiotemporal macro-zones and choosing each zone's block shape from local signal: video-feature variance per channel, and the audio-to-video cross-attention norm. A hybrid per-query block-selection strategy then decides sparsity. 2.5× faster training than full attention, with better generation quality.

**Prereqs:** [../fundamentals/attention.md](../fundamentals/attention.md), [multi-head-attention.md](multi-head-attention.md)
**Related:** [mla.md](mla.md) · [../fundamentals/dca.md](../fundamentals/dca.md) · [../inference/id-forcing.md](../inference/id-forcing.md)

---

## What it is

Most sparse-attention literature either (a) accelerates **inference** on a model trained with full attention, or (b) applies a modality-agnostic sparsity pattern (block-sparse, sliding window, LSH) that doesn't exploit modality structure.

Prism is **native sparse training** for a *joint video + audio* diffusion model: the sparsity structure is chosen per zone from signals that reflect how visual content varies and how strongly audio influences each region. The result is a sparse pattern that preserves exactly the attention paths that carry learning signal and prunes the rest.

## How it works

- **Spatiotemporal macro-zones.** The token sequence is partitioned into coarse zones along space (patch regions) and time (frame groups).
- **Per-zone signal.**
  - **Visual variance** — compute feature variance along channel axes within the zone. High variance ⇒ visually rich region deserving finer partitioning.
  - **Audio→video cross-attention norm** — the norm of the audio-to-video cross-attention weights into this zone. High norm ⇒ sound-producing region, deserving finer partitioning.
- **Dynamic block shape.** For each zone, Prism picks a block shape that is **finer along axes of high visual change and strong audio coupling**. Blocks within a zone therefore stay semantically coherent, so block-level features capture content and A/V interaction patterns together.
- **Hybrid block selection.** A per-query strategy decides which blocks each query attends to — mixing top-$k$-by-score and bucketed selection. This yields per-query sparsity that adapts to the query's content.

## Why it matters

- **2.5× training speedup over full attention** while surpassing it on generation quality.
- A principled recipe for making sparsity structure co-adapt with content, applicable beyond A/V (any multimodal diffusion where cross-modal interactions concentrate spatially).
- Resolves a tension: native sparse training usually trades quality for speed; Prism gains both because the pruned attention paths were uninformative.

## Gotchas & tricks

- **Zone granularity is a knob.** Too coarse → sparsity structure can't adapt to local content; too fine → zone bookkeeping becomes the bottleneck.
- **Signal choice matters.** Variance + cross-attention norm work for video+audio. For other modality pairs (e.g., text+image), use the analogous "visual change + cross-modal coupling" signals.
- **Training-time, not post-hoc.** Prism's gains come from training on the sparse pattern; applying it as a training-free inference accelerator loses most of the quality gain.
- **Composes with denoising-side tricks.** Pairs cleanly with ID-Forcing (long-horizon KV stability) and distillation on the full model.

## Sources

- Paper: *Prism: Dynamic Sparse Attention for Native 2K Joint Video-Audio Generation Model Training* — Tu et al., 2026 — [arXiv:2610.05416](https://arxiv.org/abs/2610.05416).
