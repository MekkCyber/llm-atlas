# MrFlow — Multi-Resolution Flow Matching

*Depth — training-free acceleration of pretrained flow-matching text-to-image models via a staged low→high-resolution pipeline.*

**TL;DR:** Prior training-free acceleration of text-to-image diffusion / flow matching does upsampling in **latent space** and selectively modifies partial regions, producing blurring / artifacts. MrFlow does the whole pipeline in **stages**: fast low-resolution structure generation, **pixel-space** super-resolution with a lightweight GAN, small noise injection to re-enter the flow-matching sampler for high-frequency resampling, then a short high-resolution refine. Achieves **10× end-to-end speedup** on FLUX.1-dev and Qwen-Image within 1% OneIG of the un-accelerated model — and **25×** stacked with timestep distillation. No training, no custom kernels.

**Prereqs:** [README.md](README.md)
**Related:** [../pre-training/README.md](../pre-training/README.md)

---

## What it is

A staged sampling recipe on top of any pretrained flow-matching text-to-image model. All the compute wins come from doing the bulk of the flow-matching steps at reduced resolution, where the quadratic-in-tokens cost is proportionally cheaper, then reserving a handful of high-resolution steps for detail. The "resolution pyramid" idea is not new; MrFlow's contribution is *how* to bridge the resolutions without introducing artifacts.

## How it works

### Stage 1 — low-resolution structure

Run the pretrained flow-matching sampler at low resolution (e.g. 512×512) for most of the trajectory. The main scene structure, layout, and colors form here; token count is $O(N^2)$ in resolution and the low-res tokens are cheap.

### Stage 2 — pixel-space super-resolution

Upsample the low-resolution result to the target resolution **in pixel space** using a lightweight pretrained GAN-based super-resolver. Doing this in pixel space is the key move: prior work upsamples in latent space, and latent upsampling doesn't preserve the flow-matching model's noise-schedule invariants — you get blur and artifacts.

### Stage 3 — noise injection and high-frequency resampling

Inject a **small-strength noise** into the upsampled pixel image and re-encode into the flow-matching model's latent. This gives the flow-matching sampler something to denoise — but only a small $\Delta$ from the current state, so a short high-resolution trajectory suffices to add high-frequency detail. The magnitude of the injected noise controls how much the flow-matching model can restore vs how much of the GAN's structure it preserves.

### Stage 4 — high-resolution refinement

Run a handful of high-resolution flow-matching steps to lock in details. Total number of high-resolution steps is much smaller than a from-scratch high-resolution generation would be.

### Composes with timestep distillation

Because MrFlow is a *sampling pipeline* rather than a modified sampler, it stacks orthogonally with timestep distillation (fewer sampling steps at each stage). The paper reports 25× end-to-end speedup with distillation on top.

## Why it matters

- **Training-free is the deploy-side lever.** Any acceleration technique that requires retraining runs into weight-management friction. MrFlow drops into an existing pretrained flow-matching model and works.
- **Fixes the latent-upsample failure mode.** Moving the upsample to pixel space is a small change with a big quality difference. Prior multi-resolution acceleration recipes hit blurring exactly here.
- **Multiplicative with distillation.** 10× alone, 25× stacked. That kind of headroom matters for consumer-side deployment.

## Gotchas & tricks

- **Noise strength is the knob.** Too little and stage 4 can't remove the GAN's fingerprints; too much and stage 4 rewrites the composition. The paper reports stable choices for FLUX.1-dev and Qwen-Image; expect to re-tune for a new base model.
- **Super-resolver identity matters.** A weak GAN adds artifacts that stages 3–4 can't fully fix. Use a GAN pretrained on natural images at the target resolution.
- **Not all flow-matching models transfer.** Very compressed latent spaces (extreme VAE downsampling) may need latent-space bridging even with MrFlow's pipeline. Sanity-check on your base model.
- **Metric choice matters.** OneIG is used as the headline quality metric; FID and CLIP alignment often move differently. Report the metric that matches your downstream use.

## Sources

- Paper: *Multi-Resolution Flow Matching: Training-Free Diffusion Acceleration via Staged Sampling* — Zheng, Liu, Ding, Feng, Lin, Guo, Qin (Beihang / ETH Zürich / USTC / CAS), 2026 — [arXiv:2607.01642](https://arxiv.org/abs/2607.01642).
- Base models: FLUX.1-dev, Qwen-Image (flow-matching text-to-image).
