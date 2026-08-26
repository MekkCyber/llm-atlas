# Block-wise autoregressive diffusion
*Depth — autoregressive at the block level, diffusion inside each block; Block3D's text→3D recipe.*

**TL;DR:** Autoregressive token-level generation is slow but preserves causal dependencies; parallel diffusion is fast but loses ordering. Block-wise AR diffusion partitions the sequence into contiguous **blocks** and generates them autoregressively — but denoises *all* tokens inside a block in a single (few-step) diffusion pass. A confidence-guided correction step revisits low-confidence tokens inside a block before it is finalized. Introduced by Block3D for text→3D generation with a 5.15× end-to-end speedup and no quality loss.

**Prereqs:** [multi-head-attention](multi-head-attention.md), [transformer-block](transformer-block.md)
**Related:** [../fundamentals/attention](../fundamentals/attention.md)

---

## What it is

A hybrid AR/diffusion generative recipe operating over a sequence of latent tokens. Causal dependency is preserved between blocks; inside a block, tokens are treated as a set and denoised jointly.

## How it works

- **Block partitioning.** Split the target sequence into contiguous blocks (e.g. 64 or 128 tokens each) along the AR order.
- **Autoregressive over blocks.** Block `k` is generated conditional on the finalized blocks `0..k-1`. Attention inside the model is block-causal.
- **Diffusion within a block.** Each block starts from noise; a small number of diffusion steps (few-shot, typically 4–10) refine all tokens in the block jointly. This is where the speedup vs pure token-AR comes from.
- **Confidence-guided correction.** After the last diffusion step, tokens with low confidence (measured by e.g. entropy of the predicted distribution) are re-denoised for one more step before the block is committed. Errors inside a block would otherwise propagate to the next block via the AR conditioning.
- **Emit and advance.** Commit the block, cache its representation, condition the next block on it.

## Why it matters

Between pure token-AR (slow but exact ordering) and pure parallel diffusion (fast but ordering-lossy), block-wise AR diffusion picks the sweet spot: coarse-grained causal ordering with fine-grained parallel refinement. Reported on text→3D with 5.15× end-to-end speedup and no quality drop; the recipe is generic and should transfer to image/video/audio token-based generation.

## Gotchas & tricks

- Block size is a knob: too small collapses to token-AR (no speedup); too large loses causal structure inside the block. 64–128 tokens is the reported sweet spot on 3D.
- Confidence-guided correction is critical — without it, low-confidence tokens near block boundaries poison the next block's conditioning.
- Diffusion steps per block set the compute floor; 4–10 is enough for reported quality on 3D but tasks with harder in-block distributions may need more.
- Composes with speculative decoding along the block axis (draft a whole block, verify at commit).

## Sources

- Paper: *Block3D: Efficient Text-to-3D Generation via Block-Wise Diffusion* — Cui et al., 2026 (ZipLab, Zhejiang University) — [arXiv:2608.19567](https://arxiv.org/abs/2608.19567)
