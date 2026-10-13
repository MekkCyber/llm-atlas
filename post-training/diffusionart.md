# DiffusionART
*Depth — stable, efficient online RL fine-tuning for diffusion models without EMA tuning.*

**TL;DR:** Online RL for diffusion generators splits into *reverse-process* methods (e.g. Flow-GRPO) that credit-assign through full sampling trajectories but are slow, and *forward-process* methods (e.g. DiffusionNFT) that are efficient but unstable without carefully tuned EMA. DiffusionART identifies a residual **forward–reverse consistency force** inside NFT-style objectives and cancels it explicitly — with a shared per-prompt guidance direction plus a trust-region-masked restoration update. No EMA tuning. No differentiable reward model.

**Prereqs:** [grpo](grpo.md), [_rl](_rl.md)
**Related:** [rlvr](rlvr.md), [cot-reward-model](cot-reward-model.md)

---

## What it is

A forward-process RL algorithm for diffusion / flow-matching image and video generators. "Forward-process" means the loss operates on single noisy states (not whole denoising trajectories), which is why it is cheap. The paper's contribution is making that family *stable* without the EMA stabilization that previously made it brittle to hyperparameter choices.

## How it works

For each prompt the sampler draws several latents and the reward model scores the final decodings. DiffusionART then:

1. **Aggregates** each prompt's rewards into a single *shared guidance direction* used across all of that prompt's drafts (reduces variance; cancels the residual consistency force).
2. **Updates** the policy against that direction, like a normal NFT-style forward step.
3. **Restores** with a *second* update that is trust-region-masked: it pulls the policy back toward a stable reference (e.g. the pre-update weights) in regions where the first step overshot. This replaces the implicit stabilization that EMA was previously doing.

Reward model does not need to be differentiable (unlike DRaFT-style methods), making it compatible with black-box verifiers.

## Why it matters

Diffusion-RL is the current workhorse for image / video post-training (preference optimization, verifier-based improvement). The EMA sensitivity of NFT-style methods has been a practical barrier at scale. Removing it, and reporting large GPU-hour savings on HunyuanVideo-13B, makes forward-process diffusion RL the practical default when a differentiable reward isn't available.

## Gotchas & tricks

- **Trust-region mask radius.** Too tight and you lose the first update; too loose and restoration doesn't help. Paper gives sweep.
- **Shared guidance per-prompt only.** Across prompts the directions must stay independent — pooling them loses signal.
- **Does not help when the reward is poorly specified.** The method stabilizes training *given* a usable reward; it does not fix reward hacking on its own.

## Sources

- Paper: *DiffusionART: Efficient, Stable and Scalable Reinforcement of Diffusion Models* — 2026 — [arXiv:2610.12572](https://arxiv.org/abs/2610.12572)
- Prior: *DiffusionNFT* (forward-process RL baseline) and *Flow-GRPO* (reverse-process baseline), cited in the paper.
