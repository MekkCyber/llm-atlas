# Sparse Attention in Diffusion Transformers (MC-Sparse)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Sparse attention for diffusion transformers (video, 3D) typically trades fidelity for speed. MC-Sparse diagnoses the loss as three distinct problems — grouped token selection, inaccurate interaction picks, and missing attention mass from dropped tokens — and fixes each: individual KV-token selection, tile-aligned query grouping, and cached metadata/residuals reused across denoising steps. Reports 1.80× speedup on Minimax-H3-Base and 2.32× on 3D asset generation at closer-to-dense fidelity.

**Prereqs:** [multi-head-attention.md](./multi-head-attention.md), [../fundamentals/attention.md](../fundamentals/attention.md)
**Related:** [../inference/README.md](../inference/README.md)

---

## What it is

A training-free sparse-attention replacement for the long-sequence attention layers in DiT-family generators. It targets the regime where previous sparse-attention methods lose noticeable quality at high sparsity ratios.

## How it works

Three mechanisms stacked:

1. **Individual KV-token selection.** Instead of selecting at the block/group level (which pulls in nearby irrelevant tokens), score each KV token individually against the query distribution.
2. **Tile-aligned query grouping.** Scoring per-token is cheap; dispatching per-token on a GPU is not. Group *queries* into tile-aligned sets so the kernel can issue one coalesced load per tile, keeping the per-token KV picks.
3. **Cross-step metadata & residual caching.** Across denoising steps, the KV distribution changes slowly. Cache the selection metadata and residual contributions from dropped tokens, and reuse them when the step-to-step delta is small, recovering most of the "lost attention mass" that naive sparse attention throws away.

## Why it matters

Dense attention is the latency wall for interactive video and 3D asset generation. Previous sparse variants cut compute by 2–4× but surrendered fidelity, which breaks the use case. MC-Sparse's oracle-driven decomposition — attributing quality loss to specific mechanisms rather than treating sparsity as a monolithic knob — is as useful as the method itself; the three fixes can be ported individually into other sparse-attention stacks.

## Gotchas & tricks

- Residual caching assumes denoising-step continuity; sampler schedules with large step jumps (early steps of some SDE samplers) see smaller gains.
- Tile-aligned grouping forces a tile size compatible with the KV cache layout — mix with FlashAttention-style tiling carefully.
- The method is training-free *when the model was trained with dense attention*; models trained with structured sparsity need recalibration of the KV-selection scores.
- Reported 1.80× / 2.32× speedups are on video / 3D; pure-image DiTs with shorter sequences see smaller wins because the dense baseline wasn't that bad.

## Sources

- Paper: *Deconstructing and Closing the Dense–Sparse Attention Gap in Diffusion Transformers* — Chen et al., Tencent HY / Fudan / CUHK, 2026 — [arXiv:2610.06801](https://arxiv.org/abs/2610.06801)
