# Flash-BoN
*Depth — cheap-draft Best-of-N cascade for inference-time scaling in diffusion models.*

**TL;DR:** For diffusion inference-time scaling, Flash-BoN sidesteps guided-search variants and re-establishes Best-of-N as the strongest baseline — provided the drafts are cheap enough. It stacks three acceleration tricks (**timestep truncation**, **layer skipping**, **activation proxies**) to produce a large pool of low-cost candidate images / videos, verifies with a multi-stage cascade, and fully denoises only the survivor. Under matched wall-clock budgets, this beats guided-search methods; it also accelerates RL post-training loops that use rollouts as rewards.

**Prereqs:** [../post-training/rejection-sampling.md](../post-training/rejection-sampling.md)
**Related:** [../post-training/grpo.md](../post-training/grpo.md) · [../post-training/cot-reward-model.md](../post-training/cot-reward-model.md)

---

## What it is

An inference-time scaling method for diffusion: instead of investing compute inside a single denoise trajectory (guided search, process-reward search), spend it across many *cheap* trajectories and pick the best. The cheapness comes from compounding three shortcuts that each degrade image quality individually but not so much that the ranking signal collapses.

## How it works

**Draft stage.** For each candidate:

1. **Timestep truncation** — run only $K \ll T$ denoising steps.
2. **Layer skipping** — skip a subset of transformer blocks per step.
3. **Activation proxies** — approximate late-stage feature maps from early-stage activations rather than computing them.

A single draft costs 10–20× less than a full denoise.

**Verify stage.** Multi-stage cascade:

1. Cheap scorer (e.g. CLIP-like) filters top-$M$ from $N$ drafts.
2. Mid-cost scorer (feature-space verifier) reduces to top-$K$.
3. Optional short-denoise refinement per survivor.

**Full denoise.** Run the standard full pipeline only on the single survivor.

## Why it matters

- **BoN wins under a wall-clock budget.** Guided search methods (best-first, PRM-guided) look strong under matched *step* count but lose under matched *wall-clock* — Flash-BoN gets the story straight.
- **RL-friendly.** Post-training loops that sample many rollouts per step (GRPO on diffusion) benefit directly: **10× faster convergence** vs prior inference-time-scaling schemes.
- **Composes.** Stacks with Reflection-Tuning and other post-training refinements.
- **Scales cleanly.** At larger models the gain grows (+8% AUC at the paper's largest scale), because cheap drafts amortize better.

## Gotchas & tricks

- Cheap drafts must preserve *ranking*, not quality — the scorer is what determines whether an aggressive draft is still useful.
- Timestep truncation and layer skipping compound multiplicatively — dropping to $K = 4$ steps *and* skipping every other layer is not always safe; ablate the two axes together.
- Activation proxies degrade smoothly but the cascade's cheap scorer must be robust to the proxy artifacts, else you filter out real winners.
- Best-of-N over cheap drafts is not universal — for tasks where verifier quality is the bottleneck (subtle preference reward), guided search may still win.

## Sources

- Paper: *Flash-BoN: Instant Drafts for Inference-Time Scaling in Diffusion Models* — Shirkavand, Paul, Wen, Huang, Chen, Goldstein, Somepalli — UMD / Hugging Face, 2026 — [arXiv:2607.04461](https://arxiv.org/abs/2607.04461).
- Project: [flash-bon.github.io](https://flash-bon.github.io/).
- Background: rejection-sampling family — see [../post-training/rejection-sampling.md](../post-training/rejection-sampling.md).
