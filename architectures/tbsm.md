# Three-Body Scattering Modeling (TBSM)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A training objective for **one-step** generative models that side-steps adversarial critics, prescribed noise-to-data paths, and autoregressive factorization. Each fake sample ("projectile") is attracted to *one* real sample and repelled from *one* independently generated fake — a per-sample O(B) interaction whose expected direction equals the 2-Wasserstein gradient-flow velocity of half the squared energy distance. On ImageNet-256 it reaches **FID 2.23** (pixel PixelDiT-XL) and **FID 1.63** (latent DiT-XL) at NFE=1.

**Prereqs:** [../fundamentals/attention.md](../fundamentals/attention.md)
**Related:** none yet

---

## What it is

A non-adversarial, non-multi-step training recipe for one-step generators. Energy-distance-based objectives typically require an all-pairs O(B²) minibatch field. TBSM shows that a **three-body** (projectile ↔ one real ↔ one generated) interaction has the same expected direction, at O(B) cost per batch.

## How it works

- **Per projectile.** For each generated sample \(x\) (the "projectile") with condition \(c\), sample one real reference \(y_r \sim Q(\cdot \mid c)\) and one independently generated reference \(y_g \sim P_\theta(\cdot \mid c)\).
- **Force.** Take a positive pull toward \(y_r\) and an equal-magnitude push away from \(y_g\), yielding a per-sample regression target.
- **Expectation identity.** Conditioned on the projectile and its context, the expectation of this force equals the 2-Wasserstein gradient-flow velocity of \(\tfrac{1}{2}D_E^2(P_\theta, Q)\) — the same quantity the all-pairs energy-distance field targets.
- **Variance reduction.** Tracking the conditional expectation online (running average per projectile) further reduces field noise.
- **Design map.** The paper places TBSM alongside diffusion-style supervision, drift-like dynamics, and GAN-like objectives on a single design map — TBSM occupies the non-adversarial, one-step, O(B)-per-batch corner.

## Why it matters

Fast (one-step) generators today mostly come from **distillation** of multi-step diffusion or GANs. TBSM is trained end-to-end from scratch with a simple regression signal, at linear cost, and reaches FID numbers competitive with distilled multi-step baselines on ImageNet-256. If the recipe holds on text-conditioned latent diffusion or video, it displaces both consistency-distillation and GAN-distillation as the default for fast generators.

## Gotchas & tricks

- **Sample-level variance.** With only two references per projectile, the per-step gradient is noisy; the online running-average trick matters more than it sounds.
- **Latent vs. pixel space.** FID 1.63 at NFE=1 is in latent space (DiT-XL); pixel-space PixelDiT-XL is 2.23 — a real gap.
- **Not obviously text-conditioned.** The paper demonstrates on ImageNet-256; text-to-image / video scaling is future work.

## Sources

- Paper: *Three-Body Scattering for Generative Modeling* — Sun, Cheng, Liu, Xie, Shang, Lin (Westlake University / Zhejiang University / UCL), 2026 — [arXiv:2607.18198](https://arxiv.org/abs/2607.18198).
