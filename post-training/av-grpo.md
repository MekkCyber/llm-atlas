# AV-GRPO — Modality-Anchored Decoupled GRPO
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A GRPO variant for **joint audio-video diffusion** post-training. Naive GRPO on two modalities entangles per-modality rewards, blurs credit assignment, and doubles training cost. AV-GRPO alternates rollouts anchored on each modality (freezing the other tower), uses trajectory-locked frozen-tower optimization to cut compute, and adapts per-modality objectives and perturbation strengths — converting joint preference learning into unimodal subproblems.

**Prereqs:** [grpo](grpo.md), [_rl](_rl.md)
**Related:** [_rewards](_rewards.md), [rlvr](rlvr.md), [sc-grpo](sc-grpo.md), [rdpo](rdpo.md)

---

## What it is

Joint audio-video generation (e.g. Javis-style pipelines with paired video + soundtrack towers) has *heterogeneous* rewards: video fidelity, audio fidelity, cross-modal synchronization, semantic alignment. Bundling them into one scalar and running vanilla GRPO gives noisy per-modality advantages and unstable training. AV-GRPO factors the problem: at each RL step, anchor the update on *one* modality's reward while the *other* modality's tower is frozen (its outputs are re-used trajectories), and rotate.

## How it works

### Three modules

1. **Modality-anchored rollouts.** For each prompt, sample $G$ video-audio trajectories. On a video-anchored step, evaluate rewards on the video component (fidelity + alignment + sync), compute group-relative advantages *for the video tower only*, and update its weights. Alternate with audio-anchored steps.
2. **Trajectory-locked frozen-tower optimization.** During a video-anchored step, cache the audio-tower activations from the shared trajectories; skip audio-tower forward/backward passes. Roughly halves per-step compute versus joint updates.
3. **Adaptive per-modality objectives.** Video and audio diffusion towers have different signal-to-noise regimes; AV-GRPO tunes noise schedules, KL weights, and perturbation strengths per modality rather than sharing hyperparameters.

### 5DAV dataset

Companion dataset that decouples training samples across five dimensions (content, style, motion, timbre, sync) so the modality-anchored rollouts get clean training-signal separation.

## Why it matters

- **First clean GRPO recipe for multi-modal generation** with conflicting per-modality rewards.
- **Practical throughput.** Frozen-tower steps roughly halve compute vs. joint updates while preserving cross-modal sync.
- **Template.** The pattern (anchor on one modality, freeze the other, rotate) generalizes to video-image, image-text, and other joint-generation settings.

## Gotchas & tricks

- Alternation cadence matters — too rare and one modality drifts, too frequent and the frozen assumption breaks (activations become stale).
- Cross-modal sync reward must still be computed on the paired trajectory even during an anchored step, otherwise sync degrades.
- Full-model fine-tune and LoRA both work; LoRA is much cheaper and the paper's ablations show minimal quality gap.

## Sources

- Paper: *AV-GRPO: Modality-Anchored Decoupling Diffusion Reinforcement Learning for Joint Audio-Video Generation* — Xu et al., PolyU / ByteDance, 2026 — [arXiv:2609.29816](https://arxiv.org/abs/2609.29816).
- Base algorithm: [grpo](grpo.md) — DeepSeekMath, 2024.
