# Masked Boundary Modeling (MBM)

*Depth — self-supervised vision pretraining that samples masks around discovered boundaries.*

**TL;DR:** A masked-image-modeling objective for vision encoders that concentrates its masked targets on *shape discontinuities* rather than uniformly random patches. A learned sub-pixel boundary detector runs alongside the encoder; the tokens it flags as boundary-bearing become the masked reconstruction targets. Fixes the "semantic-invariance over geometric detail" tradeoff that CLIP- and DINO-style objectives smear over — the encoder is forced to encode the cues that dense spatial perception (depth, normals, correspondence) needs. Introduced in *Vision Pretraining for Dense Spatial Perception* (2026).

**Prereqs:** [README.md](README.md), [pre-training/README.md](../pre-training/README.md)
**Related:** [latent-foresight.md](latent-foresight.md)

---

## What it is

Modern vision encoders come in two flavors:

- **Semantic-invariance heavy** — CLIP, DINO, MAE with random masks. Great at classification, weak at depth/normals/correspondence.
- **Geometry-heavy but supervised** — depth/normal pretraining. Needs labels, doesn't transfer.

MBM is an unsupervised objective that recovers geometric fidelity by biasing the *mask sampling* toward boundaries. Boundaries are where information about shape lives; predicting them from context forces the encoder to encode geometry.

## How it works

Two co-trained modules on top of a ViT-style encoder:

1. **Sub-pixel boundary head.** A lightweight decoder that predicts a per-pixel boundary probability from the encoder features. Bootstrap: start with a soft signal (image-gradient magnitude as a coarse proxy) and refine on-the-fly as the encoder improves.
2. **Masked reconstruction head.** Standard MIM: mask a fraction of the tokens, reconstruct their features (or pixels) from the visible tokens.

The link between the two:

```
boundary_prob = boundary_head(features)
mask ~ Bernoulli( sample_probability(boundary_prob) )   # boundary-biased
recon_loss  = || feature_target[mask] - features[mask] ||^2
boundary_loss = self-supervised discovery loss (e.g., consistency across augmentations)
loss = recon_loss + λ · boundary_loss
```

Sample probability is higher on patches that overlap discovered boundaries. Non-boundary patches are still masked sometimes (to avoid trivially always-mask-boundaries), but the distribution is skewed.

Because the boundary head trains alongside the encoder, MBM does not need a hand-engineered edge detector — the model discovers what its own encoder finds informative to reconstruct.

## Why it matters

- **Restores geometric fidelity in unsupervised pretraining.** Depth/normal/correspondence downstream tasks benefit; semantic transfer is roughly preserved.
- **No labels required.** Fits the same "pretrain on web-scale images" pipeline as CLIP/DINO/MAE.
- **Composable.** Can layer on top of existing objectives (MIM + MBM sampling schedule).
- **Feeds VLM/VLA stacks.** Any downstream system that needs geometry (robotics, dense prediction, 3D-aware VLMs) benefits directly from a better encoder without changing the multimodal stack.

## Gotchas & tricks

- **Bootstrap the boundary head carefully.** If the coarse proxy is too noisy at init, the mask distribution is random and MBM collapses to vanilla MIM.
- **Don't oversample boundaries.** 100% boundary masks make the objective too hard — reconstruction has no context. 50–70% boundary + rest uniform is typical.
- **Sub-pixel matters.** Coarse edge maps (Canny at 16× downsampling) lose the geometry cue. The boundary head must operate at sub-token resolution.
- **Feature-space targets vs pixel targets.** Feature-space reconstruction (à la data2vec, iBOT) mixes better with MBM than pixel-space MAE reconstruction, which can privilege texture over shape.
- **Watch the semantic tax.** Very heavy boundary bias can hurt classification transfer. Ablate the mixing coefficient on both a semantic and a geometric downstream to confirm.

## Sources

- Paper: *Vision Pretraining for Dense Spatial Perception* — Fu, Tan, Sun, Liu, Zheng, Xu, Zhu, Shen, Xue, 2026 — introduces MBM.
- Related: *Masked Autoencoders Are Scalable Vision Learners (MAE)* — He et al., 2021 — MIM baseline MBM extends.
- Related: *DINOv2: Learning Robust Visual Features without Supervision* — Oquab et al., 2023 — semantic-invariance-heavy baseline MBM contrasts with.
