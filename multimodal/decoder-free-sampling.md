# Decoder-Free Sampling (DWD)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Latent diffusion models sample in a learned latent space, then run a VAE decoder to reach pixels. The decoder adds inference cost and visibly degrades video quality (warping). **Decoding Without a Decoder (DWD)** attaches a lightweight *linear pixel head* to a finetuned diffusion transformer so it outputs pixels directly; the original denoising head stays frozen as an anchor to preserve pretrained knowledge. Reduces inference cost, cuts video warping error by 23%, and reaches ImageNet FID 1.43 among pixel-flow models.

**Prereqs:** [_flow-matching.md](_flow-matching.md)
**Related:** [anisotropic-flow.md](anisotropic-flow.md), [README.md](README.md)

---

## What it is

Latent diffusion decomposes generation into (1) denoise in latent, (2) VAE-decode to pixels. The decoder is often the quality ceiling for video (temporal warping, upscaling blur) and a nontrivial fraction of inference cost.

DWD replaces step (2) with a *linear* pixel head attached to the same DiT, trained with pixel-space losses. The original denoising head is **anchored** (kept frozen) so the pretrained diffusion knowledge is preserved.

---

## How it works

Let $f_\theta$ be a pretrained DiT producing latent-space denoised output. Add two heads:

- $\mathrm{Head}_\text{anchor}$: the original denoising head, **frozen**. Its loss term keeps $f_\theta$ anchored to the pretrained behavior.
- $\mathrm{Head}_\text{pixel}$: a new linear map from DiT features to pixel residuals, trained with pixel-space losses (L1 / LPIPS) during finetuning.

During sampling, run only $\mathrm{Head}_\text{pixel}$: the model emits pixels end-to-end, no VAE decode. During training, both heads run; the anchor loss prevents representation drift that would hurt the pixel-space objective's long-term usefulness.

---

## Why it matters

- **Removes the VAE decoder cost** at inference. Large for high-resolution video.
- **Less video warping.** The VAE decoder's temporal independence produces frame-to-frame inconsistency; direct pixel prediction improves temporal coherence by 23%.
- **Anchor trick is reusable.** The pattern of "add a new head with its own objective, keep the original head frozen as an anchor" generalizes to any specialization task on top of a frozen diffusion model.

---

## Gotchas & tricks

- **Linear head only works because the DiT features carry pixel-level structure.** Earlier latent DiTs had features at lower spatial resolution; DWD-friendly DiTs need feature resolution close to pixel resolution.
- **Anchor loss weight is sensitive.** Too high and the pixel head can't learn; too low and the pretrained prior drifts away.
- **Quality on text-to-image still slightly below a well-tuned VAE** at the highest fidelity tier; the gain is cost-adjusted, not Pareto.
- **No free lunch for extreme compression.** If your VAE is doing 16× spatial compression, going decoder-free requires that DiT features be at pixel resolution, which blows up memory elsewhere.

---

## Sources

- Paper: *Decoding Without a Decoder: Direct Pixel Prediction from Diffusion Transformers* — Shengqu Cai, Jan Ackermann, Rayhan Zirvi, Gordon Wetzstein, Xinchen Yan, Leonidas Guibas (Stanford / Caltech / Luma AI) — 2026 — [arXiv:2611.22287](https://arxiv.org/abs/2611.22287).
- Background: latent diffusion (Rombach 2022) — the VAE-decoded setup DWD shortcuts.
