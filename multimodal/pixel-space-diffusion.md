# Pixel-Space Text-to-Image Diffusion (Latent-to-Pixel Recipe)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Modern high-quality T2I diffusion is almost always **latent** — a VAE encodes to a low-dim latent, diffusion runs there, a decoder maps back. Pixel-space diffusion converges much slower under naive large-scale pretraining, so it stayed a research curiosity. Alibaba's 2026 empirical study fixes that with a **latent-to-pixel** recipe: pretrain the generative prior in latent space, transition to pixel space during post-training with careful choices of weight init, data composition, prediction target, decoder architecture, and noise schedule. Matches or exceeds latent counterparts while delivering **3.18–4.75× end-to-end inference speedups** (no VAE decode at inference).

**Prereqs:** [../multimodal/README.md](README.md), [../pre-training/mid-training.md](../pre-training/mid-training.md)
**Related:** [../pre-training/_lr-schedules.md](../pre-training/_lr-schedules.md)

---

## What it is

Latent diffusion (Stable Diffusion, SDXL, DALL·E 3, FLUX) wins on training compute — the generative prior is learned in a small latent space and quality scales cleanly with compute. Pixel-space diffusion (Imagen-style) wins on inference — no VAE decoder at test time, no VAE artifacts — but pretraining converges slowly enough that most modern systems just eat the inference cost.

The paper's contribution is a *transition recipe* that gets both: use latent diffusion for the expensive prior-acquisition phase, then move to pixel-space for post-training and inference.

## How it works

The recipe is deliberately empirical — the paper is an ablation study of the transition-time choices:

1. **Latent-space pretraining.** Standard large-scale T2I diffusion in the compressed latent space. This is where the generative prior is cheaply learned.
2. **Weight initialization for the transition.** How to seed the pixel-space model's weights from the latent-space checkpoint — direct copy of parameters is not viable across the resolution / channel gap; the paper works out a translation.
3. **Data composition at transition.** The data mix shifts (higher-resolution images, different aesthetic filtering) — mismatched mixes stall the transition.
4. **Prediction target.** ε-prediction, v-prediction, and x₀-prediction all behave differently as the model moves out of latent space; the paper reports which wins here.
5. **Decoder architecture.** Pixel-space needs a large decoder that isn't a VAE — its architecture choice (channel count, upsampling policy) is a first-order factor for final quality.
6. **Noise schedule.** The noise schedule tuned for latent-space diffusion is *not* the right schedule after transition; the paper re-tunes.

Trained end-to-end, the pixel-space model matches or beats its latent counterpart on the reported benchmarks and skips the VAE decoder entirely at inference — worth 3.18–4.75× wall-clock.

## Why it matters

- Latent diffusion won on training compute; pixel-space won on inference. This recipe stops making that a hard choice.
- **3.18–4.75× inference speedup** matters most for interactive image editing and for video, where VAE decoder cost dominates per-frame latency.
- Extends nicely to video — the same VAE-decoder cost that limits image throughput limits video far more, so the recipe's payoff scales with modality complexity.

## Gotchas & tricks

- **Transition is not a hyperparameter search.** Each of the five design choices interacts with the others; ablating one at a time can miss the recipe. Follow the paper's joint recipe, then vary.
- **Data mix at transition is under-appreciated.** A mix optimized for latent-space training over-samples patterns the VAE happens to preserve well; those biases have to be unwound.
- **Doesn't fix everything about pixel-space.** Sample memory footprint at high resolution is still worse than latent, and the transition adds engineering complexity. If your bottleneck is training cost more than inference cost, latent is still the right default.
- **Decoder ≠ VAE decoder.** The pixel-space model *has* a large decoder as part of the U-Net/DiT; it just isn't a VAE-shaped one. Don't confuse "no VAE" with "no decoder."

## Sources

- Paper: *An Empirical Study of Training Pixel-Space Text-to-Image Diffusion Models* — Zanyi Wang, Mingzhe Zheng, Xiangpeng Yang, Huanqia Cai, Aiming Hao, Yuming Jiang, Peng Gao, Harry Yang, Steven Hoi — arXiv:2608.16887 — 2026 (Alibaba).
- Contrast: *Latent Diffusion Models* — Rombach et al., 2022 — the latent-space ancestor.
- Contrast: *Imagen* — Saharia et al., 2022 — a pixel-space precursor that trained at scale from scratch.
