# Latent-to-4D
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Skip RGB when going from a pretrained video diffusion transformer to dynamic 3D. Take the model's *final denoised latents*, align them with the token grid of a small 4D decoder that shares the same VAE, refine with frame-wise + global spatiotemporal attention, and predict explicit 4D directly. Trained on ~1K reconstruction clips, a single checkpoint transfers unchanged across multiple video DiTs in the same VAE family — the shared VAE latent space becomes the reusable interface between video and 4D. Introduced in *Beyond Pixels: From Video Priors to 4D Worlds* (Zhejiang University, 2026).

**Prereqs:** [README.md](README.md)
**Related:** [../architectures/multi-head-attention.md](../architectures/multi-head-attention.md)

---

## What it is

"4D generation" = producing a *dynamic 3D scene* from a text or image condition. Two dominant approaches to date:

1. **RGB-then-reconstruct.** Generate an RGB video with any off-the-shelf video model, then feed frames to a separate 4D reconstruction model. Suffers from distribution mismatch (the 4D model wasn't trained on generated frames) and error propagation.
2. **Bolt geometry onto the video model.** Adapt one specific video generator to predict geometry directly. Ties 4D prediction to a single generator; changing the generator or conditioning regime forces retraining.

Latent-to-4D proposes a third route: use the **final denoised latents** of the video model — not RGB, not attention features — as the reusable interface. Any video DiT that shares a VAE with the 4D decoder can plug in without retraining.

## How it works

1. **Denoise as usual.** Run the pretrained video DiT to completion; keep the final latent tensor $Z \in \mathbb{R}^{T \times H_l \times W_l \times C}$ instead of decoding to RGB.
2. **Align to the 4D decoder's token grid.** The 4D decoder expects tokens in a specific $(T', H', W', C')$ layout; a lightweight adapter reshapes/projects $Z$ into that grid.
3. **Refine with spatiotemporal attention.** A small transformer stack alternates **frame-wise attention** (fixes per-frame geometry) and **global spatiotemporal attention** (enforces temporal consistency across frames).
4. **Decode to explicit 4D.** The 4D decoder emits the dynamic 3D representation (Gaussian splats, occupancy, or the reconstruction target the paper uses).

Training data: **~1K reconstruction clips** — small because the video prior already carries most of the burden.

## Why it matters

- **Composability contract.** A single 4D decoder now works across every video DiT that shares its VAE. No retraining per (video model, task) pair — the pretrained video model becomes a plug-in feature extractor.
- **Beats matched cascades.** On Text4D-200 / I4D-200, Latent-to-4D beats matched same-latent Wan+4RC cascades in projection-based DINO-F1 by **+2.88–3.45** / **+5.81** points, and wins human preference on geometry, temporal stability, and overall quality.
- **Data-efficient.** ~1K clips is orders of magnitude below what a from-scratch 4D generator needs — the video prior is doing the heavy lifting.

## Gotchas & tricks

- **VAE family is the boundary.** Cross-family transfer (different VAE) is not addressed and probably requires the small adapter to be retrained.
- **Final latents only.** The paper picks the *fully denoised* latent as the interface; intermediate denoising latents carry more noise and were not the target of the alignment adapter.
- **DINO-F1 is a projection-based proxy** for 4D quality — human preference correlates but not perfectly. Report both.

## Sources

- Paper: *Beyond Pixels: From Video Priors to 4D Worlds* — Liu, Shen, Zhou, Quan, Yang (Zhejiang University), arXiv 2608.10744, 2026.
