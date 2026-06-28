# Likelihood Score Alignment (LISA)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A regularizer for dual-branch controllable diffusion/flow-matching models (ControlNet-style: frozen main net + trained side net for the visual condition). LISA re-interprets the side branch as contributing an *implicit likelihood score* and adds an auxiliary loss that explicitly aligns side-branch features to an approximated likelihood-score target. Accelerates training, improves quality, costs ~0 at inference.

**Prereqs:** [_controllable-generation.md](_controllable-generation.md)
**Related:** [README.md](README.md)

---

## What it is

The dominant paradigm for visual-condition controllable generation freezes a pretrained diffusion / flow-matching model and trains a small *side network* that encodes the condition (depth map, edges, identity, pose, …) and injects features into the main network's intermediate layers. ControlNet, T2I-Adapter, IP-Adapter all follow this shape.

A score-based reading of why this works:

- The main (frozen) network supplies the **prior unconditional score** of the data distribution.
- The side (trained) network implicitly contributes the **likelihood score** of the condition given the noisy sample — the gradient that bends generation toward the condition.

LISA's claim is that this "implicit likelihood score" can be *explicitly* supervised, which makes the side network train faster and end up more disentangled.

## How it works

- Hook features from a designated intermediate layer of the side network.
- Project those features into the score latent space with a lightweight decoder head.
- Construct an **approximated likelihood-score target** for the current noisy sample and condition pair.
- Add an auxiliary regularization loss measuring the distance between the decoder's output and the target.
- Jointly optimize side network + decoder with the standard diffusion/flow loss **plus** the LISA regularizer.

The decoder head is discarded at inference, so generation cost is unchanged.

## Why it matters

- Gives a *theoretical reading* of dual-branch controllable generation (main = prior, side = likelihood) that previously was treated as a successful empirical trick.
- Works across image *and* video tasks, multiple architectures, and both diffusion and flow-matching backbones — i.e. it's a recipe-level improvement, not a single-model tweak.
- The disentanglement side-effect on side-branch features makes downstream conditional editing easier.

## Gotchas & tricks

- The "approximated likelihood score target" is itself an estimator — choice of estimator (e.g. denoising-score-style target vs flow-velocity-style target) matters and must match the backbone.
- The regularizer weight is a small hyperparameter; too large drowns out the main diffusion loss.
- Only one side-branch layer is supervised in the paper; multi-layer alignment is left for future work.

## Sources

- Paper: *LISA: Likelihood Score Alignment for Visual-condition Controllable Generation* — Wang, Chen, Liu, He, Liu, Wang, Chen, HKUST / Huawei, 2026 — arXiv:2606.27192.
