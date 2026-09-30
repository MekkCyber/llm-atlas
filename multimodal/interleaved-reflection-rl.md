# Interleaved reflection RL for unified multimodal models
*Depth — training a single unified vision-language model to render → diagnose → revise → render its own image generations via end-to-end RL.*

**TL;DR:** Unified multimodal models can both perceive and generate images, so in principle they can critique and repair their own generations. UMM-Reflection makes that native: interleaved RL trains a single model to render, diagnose, and revise across a multi-turn trajectory, with the whole trajectory optimized against a final-image reward — not as a separate critic + generator handshake.

**Prereqs:** [../post-training/grpo.md](../post-training/grpo.md), [../post-training/ppo.md](../post-training/ppo.md)
**Related:** [../post-training/reasoning/long-cot-rl.md](../post-training/reasoning/long-cot-rl.md), [README](README.md)

---

## What it is

A post-training recipe for unified multimodal models (those with joint image-in and image-out heads) that teaches them to *reflect* on their own image generations: the model produces an image, describes what it got wrong, produces a revision, and repeats. The whole trajectory is trained with RL rather than by supervising each turn independently.

## How it works

- **Rollout.** Given a text prompt, the model alternates between render steps (produce an image) and diagnose steps (describe defects, decide edits). Each turn is a full model call; the sequence is a trajectory.
- **Reward.** A task-appropriate reward on the *final* image only — e.g. GenEval or a preference model. Intermediate turns get no direct reward.
- **Optimization.** Policy-gradient (PPO/GRPO-family) update over the whole trajectory. The gradient flows through both render and diagnose calls, so the model learns which self-critiques actually lead to better final images.
- **Interleaving.** The order and number of turns is unconstrained; the model can choose to stop refining. In practice a max-turns cap is used during training.

## Why it matters

Text reasoning models have long trained themselves to reflect on their own outputs; unified multimodal models let the same loop close in image space, *without* a separate scorer or refiner model. This eliminates the two failure modes of prior generate-then-critique pipelines: (a) the critic model has a different bias than the generator (mismatched trust), and (b) the critic-refiner handshake is trained on synthetic data whose distribution drifts from what the generator produces. Reported gains: +12.05 GenEval points over SFT on the same trajectories, transferring to WISE, OneIG-Bench, and T2I-CompBench++. Ablations show the interleaved-RL objective is doing the work — plain SFT on the same reflection traces recovers only a fraction of the gain.

## Gotchas & tricks

- **Reward hacking is easy** because the reward only sees the final image. Prompt-conditioned rewards (GenEval-style) are more robust than open-ended preference models; the latter can be gamed by making the critique text sound like the reward model's preferences.
- **Rollout cost.** Each rollout is `N * (text-decode + image-decode)`; use small `N` early, expand once the reflection head stabilizes.
- **Freezing decoders can help early training** — freeze the image generator for the first RL epochs so the model learns *when* to reflect before it learns *how* to revise pixels.
- Interleaved reflection works only if the model can both see and generate; asymmetric stacks (perception frozen, generation trainable) collapse to standard image RLHF.

## Sources

- Paper: Learning Native Reflection in Unified Models with Interleaved Reinforcement Learning — Fan, Huang, Cai, Li, Wen, Yin, Diao, Liu, 2026 — [arXiv:2609.35767](https://arxiv.org/abs/2609.35767)
