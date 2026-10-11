# MC-Sparse
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A training-free sparse attention scheme for diffusion transformers that stays close to dense attention in output quality. For each query, MC-Sparse selects KV tokens using the **exact** attention probabilities (not an approximation); groups similar queries into GPU-friendly tiles so the per-query cost amortizes; and caches both the selected index sets and a dense–sparse output residual across adjacent denoising steps. Reports 1.80× speedup on video generation and 2.32× on 3D asset generation at higher fidelity than prior sparse DiT methods.

**Prereqs:** [../fundamentals/attention.md](../fundamentals/attention.md), [multi-head-attention.md](multi-head-attention.md)
**Related:** [../inference/README.md](../inference/README.md)

---

## What it is

DiT inference is attention-bound at long token counts, especially for video and 3D. Existing sparse-attention methods for DiTs trade dense-equivalent quality for throughput by using approximate selection (e.g. hashing, learned predictors). MC-Sparse argues that *selection quality is the entire gap* between dense and sparse: when selection is accurate, sparsity does not cost quality.

## How it works

Three ideas, stacked:

1. **Exact-probability selection.** For each query, compute real attention logits over the full KV set at low precision, then take the top-$k$ keys by exact probability mass. The cost of this scan is paid once per selection step, not per layer evaluation.
2. **Query tiling.** Group queries whose selected KV sets overlap heavily into GPU-friendly tiles; one block of a tile is served by one union KV set, so wall-clock time stays linear in tile count instead of query count.
3. **Inter-step reuse.** Because adjacent denoising steps have correlated attention patterns, cache both the selected indices and the dense–sparse *output residual*. On a subsequent step, if the residual is small, reuse the cached pattern and add the residual rather than re-selecting from scratch.

The method is training-free and drops into existing DiT backbones.

## Why it matters

- Addresses the real bottleneck in long-token DiT inference (video, 3D) without retraining.
- Decouples selection from execution — selection is cheap and exact, execution reuses patterns.
- Shows that "approximate selection" was the wrong lever for DiT sparse attention; the right lever is amortizing exact selection over tiles and steps.

## Gotchas & tricks

- The residual cache can mask quality regressions if the selection drifts slowly over many steps; a periodic "cold" reselection pass is prudent on long schedules.
- Tile grouping is more effective for high-resolution video where queries within a spatial neighborhood genuinely share KV sets; low-resolution batches get smaller speedups.
- Picking $k$ is dataset-dependent — paper reports stability across a reasonable range, but very small $k$ (sub-5% keep rate) does hurt fidelity.

## Sources

- Paper: *MC-Sparse: Deconstructing and Closing the Dense–Sparse Attention Gap in Diffusion Transformers* — Chen, Lai, Wang et al., Tencent HY / Fudan / CUHK MMLab / SII, 2026 — [arXiv:2610.06801](https://arxiv.org/abs/2610.06801).
