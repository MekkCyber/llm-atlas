# Latent tokenizers for multimodal generation (KVAE)
*Depth — VAE-family tokenizers for latent-diffusion generation across audio, image, and video.*

**TL;DR:** Latent diffusion models (LDMs) map raw signal to a compact latent, run diffusion in that latent space, and decode back. The tokenizer choice is load-bearing: its compression ratio bounds the diffusion model's effective sequence length, and its reconstruction fidelity bounds sample quality. KVAE ships a coherent family — KVAE-Audio (48 kHz → 50 Hz, 64 channels), KVAE-2D (image, 8× spatial, 32 channels), KVAE-3D (video, causal, 4×16×16 and 4×8×8 variants).

**Prereqs:** [../fundamentals/_tokenization.md](../fundamentals/_tokenization.md)
**Related:** [../multimodal/README.md](../multimodal/README.md)

---

## What it is

A family of continuous-latent tokenizers designed to feed text-conditioned latent diffusion models. Not quantized codebooks (VQ) — continuous latents, so the downstream diffusion is a continuous LDM. The family covers three modalities under one design philosophy so a multimodal LDM stack can share tokenizer shape and training recipe.

## How it works

**Per-modality variants.**

| Variant | Compression | Latent shape |
| --- | --- | --- |
| KVAE-Audio | 48 kHz → 50 Hz | 64 channels, continuous |
| KVAE-2D (image) | 8× spatial | 32 channels |
| KVAE-3D (video, small) | 4× temporal × 16 × 16 spatial | continuous |
| KVAE-3D (video, large) | 4× temporal × 8 × 8 spatial | continuous |

**Encoder/decoder.** Standard VAE architecture per modality (convolutional for image/video, temporal-convolutional for audio). Trained with reconstruction + KL regularization + a perceptual/adversarial loss depending on modality.

**Causal video.** The 3D variants are *causal* over time — the latent at time `t` depends only on frames ≤ `t`. This makes streaming decode possible and matches the causal decoding assumed by most video-LDMs.

**Family design.** Shared philosophy means a multimodal LDM (e.g. text→video with audio) can stack the modality-specific tokenizers and share a text encoder above them, rather than fighting three unrelated latent geometries.

## Why it matters

- **Tokenizer is the LDM bottleneck.** Latent budget determines the diffusion model's compute and quality — a coherent family that hits sensible compression/quality points saves teams from re-inventing tokenizers per modality.
- **Causal video LDMs need causal tokenizers.** The 3D variants respect the causal constraint that streaming decode requires — non-trivial engineering choice, useful to have open.
- **Coverage.** Audio + image + video with matched design lowers the barrier to building multi-modal LDMs from open components.

## Gotchas & tricks

- Continuous latents mean the diffusion model is trained in continuous space; if you want a discrete-token flow (autoregressive transformer over video), you need a separate VQ step downstream.
- 4× temporal video compression is aggressive — long-video decode quality degrades faster than image quality; use the higher-spatial variant when temporal detail matters.
- Reconstruction ≠ downstream sample quality; measure both when picking a variant.

## Sources

- Paper: *KVAE: Family of Tokenizers for Multimodal Generative Models* — Shutkin et al., 2026 — [arXiv:2608.05798](https://arxiv.org/abs/2608.05798) — Kandinsky Lab
