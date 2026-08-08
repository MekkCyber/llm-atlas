# KVAE
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Open family of latent tokenizers spanning audio, image, and video for text-conditioned diffusion. **KVAE-Audio** compresses 48 kHz to 64 latent channels @ 50 Hz; **KVAE-3D** provides two causal video tokenizers (4×16×16 and 4×8×8 compression); **KVAE-2D** handles images at 8× spatial compression with 32 channels. Consistent design language across modalities so a single latent-diffusion stack can plug into any of them.

**Prereqs:** [../fundamentals/_tokenization.md](../fundamentals/_tokenization.md) (tokenizer taxonomy).
**Related:** [README.md](./README.md)

---

## What it is

Latent diffusion needs a **tokenizer** that compresses input signals into a low-dimensional latent grid where the diffusion transformer operates. KVAE ships a *family* of such tokenizers so that image, video, and audio diffusion systems can share design decisions (channel counts, compression ratios, decoder architecture) rather than reinventing them per modality.

## How it works

**KVAE-2D (images).**
- Continuous VAE (not VQ).
- 8× spatial compression, 32 latent channels.
- Trained with reconstruction (L1 + LPIPS) + adversarial loss.

**KVAE-3D (video).** Two variants for different compute budgets:
- 4×16×16 (temporal × spatial × spatial): heavy compression, fastest downstream diffusion.
- 4×8×8: milder compression, higher fidelity.
- Both use **causal** temporal convolutions — output frame $t$ depends only on input frames $\leq t$. Enables autoregressive generation of long videos without re-encoding.

**KVAE-Audio.**
- Continuous full-band (48 kHz) compression.
- 64 latent channels @ 50 Hz (960× compression from raw waveform).
- Perceptual loss (PESQ, PSNR) + adversarial loss.

**Shared design language.** Same VAE backbone shape across modalities: an encoder–decoder pair with GroupNorm, SiLU, ResBlocks, and attention at low resolutions. Only the input/output patchifier and dimensionality change per modality.

## Why it matters

- **Latent tokenizers are the unsung bottleneck** of multimodal diffusion. Every downstream FID gain compounds against a fixed VAE. An open, well-benchmarked cross-modality family lowers fixed cost for teams building diffusion systems.
- **Causal video tokenizer unblocks streaming/autoregressive video generation** — you can encode frames as they arrive without waiting for the whole clip.
- **Cross-modality consistency** means a single text-to-latent diffusion transformer can be swapped between modalities with minimal architectural changes.
- Matches or surpasses frontier open-source tokenizers on reconstruction (PSNR, LPIPS, PESQ) and on generation FID/CLIP/CLAP metrics.

## Gotchas & tricks

- **Continuous vs discrete tradeoff.** Continuous VAEs are more compressible and easier for diffusion, but discrete VQ tokenizers are needed for autoregressive/next-token generation. KVAE is continuous throughout; pick VQ variants for AR pipelines.
- **Channel count vs KL regularization.** More latent channels = higher fidelity but harder for the diffusion transformer to model; KVAE picks channels per modality to balance both.
- **Causal video tokenizers cost latency at inference.** Non-causal alternatives are faster for offline generation.
- **Adversarial training is unstable.** Reproducers should mirror the paper's loss schedule (reconstruction-only warmup, then GAN turn-on).
- **PESQ / PSNR / LPIPS don't align with perceptual quality on all content types.** Human eval remains the tiebreaker; the paper reports side-by-side wins.

## Sources

- Paper: *KVAE: Family of Tokenizers for Multimodal Generative Models* — Shutkin et al., Kandinsky Lab, 2026 — [arXiv:2608.05798](https://arxiv.org/abs/2608.05798).
- Code and weights released publicly.
