# Self-training for drift reduction
*Depth — training on corrupted histories and self-rollout residuals to close the train/inference gap in autoregressive world models.*

**TL;DR:** Autoregressive video world models drift at long horizons because they were trained conditioned on *ground-truth* histories that they'll never see at inference — at inference they condition on their *own* (imperfect) outputs. AlayaWorld's recipe: during training, feed the model **corrupted histories** and **prediction residuals collected from its own rollouts**, so the training-time distribution actually matches inference-time input.

**Prereqs:** *(none)*
**Related:** [../case-studies/alayaworld.md](../case-studies/alayaworld.md), [bounded-visual-context.md](./bounded-visual-context.md)

---

## What it is

Every autoregressive generative model has a training/inference distribution mismatch: it was trained on the true previous outputs (teacher forcing) but at inference sees its own imperfect outputs. In text this is often benign; in long-horizon video generation it's the dominant failure mode — small per-chunk errors compound into scene collapse.

Classical fixes (scheduled sampling, DAgger-style corrections) target this in RL and structured prediction. Self-training for drift reduction is the same idea, packaged specifically for autoregressive video-DiT world models.

## How it works

Two augmentation streams during base training:

1. **Corrupted histories.** Instead of always conditioning the next-chunk prediction on a *clean* ground-truth history, apply noise/perturbations to the history at training time. The model learns to be robust to imperfect conditioning — the state it will actually see at inference.

2. **Self-rollout prediction residuals.** Collect rollouts from the model itself. Compute the residual between what it produced and what a good next chunk should be. Add these (history, target) pairs to the training set. The model learns to *correct* its own mistakes when they compound.

Together, the model is exposed at training time to the *joint distribution* of imperfect histories and appropriate corrections — the joint distribution it actually operates in at inference.

## Why it matters

- **Directly targets the dominant failure mode.** Long-horizon drift is the number-one open problem in video world models. This is the first-order fix.
- **General across autoregressive video models.** Nothing about the recipe is AlayaWorld-specific.
- **Cheap compared to alternatives.** The alternative — training on much longer clips — hits memory ceilings; this scales with rollout count instead.
- **Analog of RL post-training's rejection sampling.** Sample from your model, keep the good, use the bad as correction signal.

## Gotchas & tricks

- **Corruption model matters.** Simple pixel noise doesn't approximate the actual failure modes; corruption should target the modes the model actually exhibits.
- **Residual collection is not free.** You have to roll out the model at training time. Batching / caching those rollouts is a real engineering item.
- **Distributional coverage.** The self-rollouts only cover trajectories the model already generates — this closes the loop but can leave rare failure modes uncovered.

## Sources

- Paper: *AlayaWorld: Interactive Long-Horizon World Modeling* — Zhang, Li, Zhan, Ge, Yin et al. (Alaya Lab), 2026 — [arXiv:2607.18367](https://arxiv.org/abs/2607.18367)
