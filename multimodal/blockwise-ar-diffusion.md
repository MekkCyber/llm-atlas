# Blockwise Autoregressive Diffusion
*Depth — high-resolution image synthesis via tile-AR diffusion with thumbnail conditioning.*

**TL;DR:** A full-resolution diffusion denoiser at 4K is prohibitive to train. Blockwise AR diffusion sidesteps this: train the model on 1K tiles, and at inference compose a 4K image tile-by-tile, conditioning each tile on its already-generated upper and left neighbors plus a compressed thumbnail of the whole scene. A Wavefront-AR schedule processes all tiles on the same anti-diagonal in parallel across GPUs, so the AR dependency is amortized spatially. Introduced in Qwen-Image-HD.

**Prereqs:** [case-studies/qwen2-5.md](../case-studies/qwen2-5.md)
**Related:** [dualpipe.md](../systems/dualpipe.md) · [multimodal/README.md](README.md)

---

## What it is

A compositional recipe for arbitrary-resolution image generation on top of a fixed-resolution base model. The base is a standard DiT trained at 1K. The HD generator wraps it: break the target canvas into a grid of 1K tiles, generate the tiles one at a time in reading order, and give each tile three conditions — the upper-neighbor tile, the left-neighbor tile, and a low-resolution thumbnail of the whole scene generated so far. Tiles on the same anti-diagonal can run concurrently on separate GPUs (none depends on the other), giving a parallel schedule that hides most of the AR latency.

## How it works

1. **Thumbnail pass.** Generate a low-resolution preview of the full scene (e.g., 1024 × 1024) with the base DiT and the text prompt.
2. **Tile grid.** Divide the target canvas (e.g., 4K) into a grid of 1K tiles.
3. **Tile pass, in reading order.** For each tile `(i, j)`:
   - Inputs: upper-neighbor tile `(i-1, j)`, left-neighbor tile `(i, j-1)`, thumbnail, prompt.
   - Denoise this tile with the base DiT, conditioned on the above.
4. **Wavefront-AR schedule.** Tiles on the same anti-diagonal (`i + j = k`) have no dependencies on each other and can run on separate GPUs in parallel. Walk the wavefront from top-left to bottom-right; per-step latency is one tile denoise, not one tile per `(i, j)` pair.
5. **Boundary care.** Each tile is denoised with a small overlap with its neighbors to prevent seams; final tiles are blended.

Training uses only the 1K base model plus a small adapter that accepts the three conditions. Local training is cheap — no 4K forward pass is ever done.

## Why it matters

- **Decouples train and inference resolution.** Same base model serves 1K, 2K, 4K, arbitrary resolutions. No per-resolution fine-tune.
- **Better quality than naive 4K.** Reported to outperform direct 4K generation with FLUX-3-Image-Klein and full-image 4K fine-tuning on Qwen-Image-Bench at 4K — at lower training compute.
- **Wavefront schedule is a general spatial-AR trick.** The same idea applies to any spatial AR model where dependencies form a DAG.
- **Thumbnail conditioning fixes long-range consistency.** Without the thumbnail, far-apart tiles can disagree on global style/composition. With it, every tile sees the whole scene at low resolution.

## Gotchas & tricks

- **Thumbnail quality is load-bearing.** A bad thumbnail propagates into every tile. Spend compute on the thumbnail pass.
- **Overlap vs. seam tradeoff.** Too little overlap → visible tile seams; too much overlap → wasted compute and texture smearing. 10–15% overlap is a working default.
- **Wavefront parallelism is not free.** You still need enough GPUs to cover the longest anti-diagonal. On a 4×4 grid the max anti-diagonal is 4 tiles — exploiting it needs 4 GPUs.
- **Scene thumbnail evolves.** Some implementations regenerate the thumbnail from partially-completed tiles mid-walk; others fix it. The paper fixes the thumbnail once; regenerating helps only marginally.
- **Text alignment can drift at the edges.** A prompt like "the cat in the top-left corner" needs the model to know global position; the thumbnail helps but isn't a full fix. Expect to add explicit position tokens for strict alignment.

## Sources

- Paper: *Qwen-Image-HD: Autoregressive Diffusion for High Definition Image Generation* — Qwen team (Alibaba), 2026 — [arXiv:2611.32736](https://arxiv.org/abs/2611.32736).
