# AV-GRPO
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A GRPO variant for **joint audio-video diffusion**. Three tricks together: (1) *modality-anchored rollouts* disentangle heterogeneous multimodal reward signals so credit is clean; (2) *trajectory-locked frozen-tower optimization* keeps one modality's tower frozen while training the other, roughly halving cost; (3) *adaptive objectives* with modality-tailored perturbation strengths.

**Prereqs:** [../post-training/grpo.md](../post-training/grpo.md), [../post-training/_rewards.md](../post-training/_rewards.md)
**Related:** [README.md](README.md), [../post-training/rlvr.md](../post-training/rlvr.md)

---

## What it is

Joint audio-video generation stacks a video tower and an audio tower and asks a diffusion sampler to render both from the same prompt. Off-the-shelf RL post-training on this joint model has two problems:

1. **Reward entanglement** — audio quality, video quality, and cross-modal synchronization all show up in the same scalar reward. Credit assignment is a mess.
2. **Joint-tower cost** — training both towers simultaneously with divergent dynamics is expensive.

AV-GRPO restructures each rollout to give a clean per-modality signal while keeping the joint model.

## How it works

Given a joint audio-video generator and a group of `G` rollouts per prompt:

1. **Modality-anchored rollouts.** For each rollout, pin one modality's noise trajectory to a reference and vary only the other. The reward difference now attributes cleanly to the varied modality — no entanglement with the anchor.
2. **Trajectory-locked frozen-tower optimization.** During a training step for modality X, freeze modality Y's tower and pin its trajectory. Update X only. Alternate. Backprop cost is roughly a single-tower step, not a joint one.
3. **Adaptive objectives.** Each tower gets its own KL / perturbation strength; audio's smooth spectrum and video's high-frequency texture need different regularization.

Data-side: **5DAV**, a dataset decoupled along five axes for systematic control over training difficulty.

## Why it matters

- Beats LTX-2.3 in quality, semantic alignment, and cross-modal synchronization on **JavisBench** and **VABench**, both under LoRA and full fine-tuning.
- The design pattern — anchor one modality, vary the other, alternate — generalizes to any multi-tower generative model that needs RL post-training (text+image, image+3D, etc.).
- Tower-locking cuts RL cost enough to make joint audio-video RL routine instead of exceptional.

## Gotchas & tricks

- Anchoring bias: if the anchor's noise trajectory is bad for the pair, the varied modality is optimized to compensate for a mistake. Sample anchors from the current joint distribution, not a fixed reference.
- Alternation ordering matters early in training (one tower learning cross-modal cues before the other); the paper reports the schedule but treat it as tunable.
- Perturbation strength per modality is a live hyperparameter; audio typically wants weaker perturbations than video.

## Sources

- Paper: *AV-GRPO: Modality-Anchored Decoupling Diffusion Reinforcement Learning for Joint Audio-Video Generation* — Xu, Yan, Shi, Li, Liu, Lam, Cao — 2026 — [arXiv:2609.29816](https://arxiv.org/abs/2609.29816).
