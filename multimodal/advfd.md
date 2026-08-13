# Adversarial Fréchet Distance (AdvFD)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A post-training loss for visual generators that complements the fixed pretrained feature space in a standard Fréchet-distance loss with a **calibrated, adversarially learned representation**. Fixes "Fréchet hacking" — the failure mode where a generator's FD-in-one-feature-space keeps improving while visual quality and FD in other feature spaces stagnate or degrade. Introduced in *AdvFD: Boosting Visual Generation via Adversarial Fréchet Distance Loss* (Peking University / KlingAI, 2026).

**Prereqs:** [README.md](README.md)
**Related:** [../post-training/_rewards.md](../post-training/_rewards.md)

---

## What it is

Modern generator post-training layers a **distribution-level objective** — some flavor of Fréchet distance between real and generated feature distributions — on top of the base sample-level diffusion / flow-matching loss. The Fréchet feature space is usually **pretrained and fixed** (Inception, DINO, CLIP).

Two problems with a fixed feature space:

1. **Incomplete view.** Any one pretrained encoder captures only some invariances. A generator can improve the target FD without improving the aspects that encoder is blind to.
2. **Fréchet hacking.** The generator overfits to *this specific view* of "distribution mismatch." Target FD keeps dropping, FD in other spaces stagnates or worsens, and human-perceived quality follows the other spaces.

AdvFD keeps the useful signal from the static space and adds a **learned adversarial space** that keeps moving — the target the generator is fighting is no longer a fixed distribution proxy.

## How it works

- Retain the existing FD-loss on the pretrained feature space (Inception / DINO / CLIP, whatever the paper defaults to).
- Add an **adversarial encoder** $\phi_\text{adv}$ trained jointly to maximize the Fréchet distance between real and generated feature distributions in its own representation.
- The generator loss becomes a weighted sum of:
  - the sample-level score / flow-matching loss;
  - the static-space FD loss;
  - the adversarial-space FD loss (with $\phi_\text{adv}$ updated adversarially).
- **Calibration** keeps the adversarial encoder from diverging into a discriminator that only measures noise; details of the calibration term differ by variant but the core is a regularizer that ties $\phi_\text{adv}$'s features to real-data statistics.

The result is a Fréchet objective whose "feature space" is a moving target — the generator can't overfit to a single pretrained view because the adversarial view keeps evolving to expose whatever mismatch remains.

## Why it matters

- **Closes the Fréchet-hacking loophole.** Static-space FD losses have been quietly damaging results by directing optimization at one narrow projection of the distribution.
- **Works on top of standard training.** AdvFD is a *post-training* loss — it slots into existing diffusion / flow-matching pipelines without touching the base loss or architecture.
- **Adversarial revival for generative post-training.** GAN-style adversarial signals had largely lost to score-based methods; AdvFD shows the adversarial signal has a clean job as a *feature-space regularizer* rather than as the main generator loss.

## Gotchas & tricks

- **Balance the two FDs.** Too much weight on the adversarial term and the generator chases the discriminator's noise; too little and you're back to Fréchet hacking. The paper's default weighting is a starting point, not universal.
- **Calibration is not optional.** Without it, the adversarial encoder collapses into a plain discriminator with all the usual instability.
- **Reporting.** Always report FD in *multiple* pretrained feature spaces plus human preference. Reporting only the target FD hides exactly the failure AdvFD is designed to fix.

## Sources

- Paper: *AdvFD: Boosting Visual Generation via Adversarial Fréchet Distance Loss* — Gao, Zhou, Gai, Yu, Tang (Peking University / KlingAI Research), arXiv 2608.11205, 2026.
