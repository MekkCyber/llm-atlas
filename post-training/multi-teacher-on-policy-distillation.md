# Multi-Teacher On-Policy Distillation (MOPD)
*Depth — on-policy distillation with a *router* that selects a task-specific teacher per sample, and a reverse-KL objective applied to the student's own rollouts.*

**TL;DR:** Distilling a multi-task model from a single generalist teacher caps the student at the teacher's weakest skill. **MOPD** instead maintains several specialist teachers (one per task family) and routes each training sample to the appropriate teacher. The student is distilled **on-policy** — against its own rollouts — using a **k3 reverse-KL** objective. Introduced in Qwen-Audio-3.1-ASR as the final post-training stage with task-routed teachers for ASR, SD, AU, and text/dialogue (Qwen3.5).

**Prereqs:** [_rl](_rl.md), [grpo](grpo.md), [rejection-sampling](rejection-sampling.md)
**Related:** [bridged-self-distillation](bridged-self-distillation.md), [cot-reward-model](cot-reward-model.md)

---

## What it is

Standard distillation trains a student to match a single teacher's output distribution — either offline (SFT on teacher samples) or on-policy (student rolls out, teacher scores). This has two problems in multi-task settings:

1. **No single teacher is strongest on every task.** A generalist teacher dominated by text underperforms on audio-event recognition; an ASR specialist is useless for dialogue.
2. **Static objective diverges from student distribution.** Offline distillation collects teacher outputs once and reuses them; the student's rollouts drift away.

MOPD addresses both:

- **Task-routed teachers.** The training set is annotated by task (ASR / SD / AU / text), and each sample is distilled against the teacher designated for its task.
- **On-policy.** The student generates rollouts; teachers score them in-place. No pre-collected teacher corpus.
- **Reverse-KL (k3).** The loss is a reverse-KL between student rollouts and teacher distribution, which penalizes the student placing probability mass where the teacher does not — the right direction for suppressing teacher-wrong tokens.

---

## How it works

For each training sample `(prompt, task_label)`:

1. **Student rolls out.** Produce a response token-by-token on-policy.
2. **Router selects teacher.** Based on `task_label`, pick the specialist teacher *T_task*.
3. **Teacher scores rollout.** Teacher computes its own token distribution over the rollout tokens.
4. **Reverse-KL loss.** Compute `D_KL(student || teacher)` with the k3 estimator (a biased-but-low-variance Monte Carlo estimator common in distillation literature). The gradient pushes student probability away from tokens the teacher assigns zero/low mass.
5. **Update student.**

Design choices from the Qwen-Audio report:

- **Four teachers for four task families** — ASR, SD, AU, and text/dialogue (Qwen3.5).
- **Rollouts rather than teacher outputs.** Keeps the training distribution matched to deployment.
- **k3 estimator specifically.** k3 reverse-KL is the Schulman-style estimator used in RLHF and PPO contexts — it has lower variance than naive MC KL and is a drop-in in frameworks like verl.

---

## Why it matters

Multi-task foundation models are the dominant target in 2026, and most of them share the same bottleneck: a single teacher caps multi-task performance. MOPD gives three things that solve it cleanly:

1. **Specialist knowledge preservation.** Each task is distilled against its best teacher; no averaging across teachers.
2. **On-policy = less drift.** Student distribution never leaves the training objective's support, which is the typical failure mode of offline distillation.
3. **Composable with RL.** MOPD runs after GRPO; the student has already shaped itself to the reward, and distillation sharpens it without undoing RL gains.

The approach is domain-agnostic — nothing about it is audio-specific. Any multi-task model with distinct specialist teachers can use this recipe.

---

## Gotchas & tricks

- **Router quality is the ceiling.** Mis-routing a sample distills against the wrong teacher. For most multi-task settings the task label is already in the data; use it directly instead of a learned router.
- **Reverse-KL asymmetry.** Reverse-KL is mode-seeking: the student will collapse to the strongest mode a teacher supports and *not* cover the full teacher distribution. That is usually what you want (don't imitate low-probability teacher tokens), but it costs diversity.
- **Teacher availability during training.** Each teacher must be inference-callable on every batch. Co-locating teachers on separate devices with vLLM-style rollout serving is the standard pattern.
- **Not a replacement for RL.** MOPD runs *after* GRPO; it sharpens, it does not reward-shape. Running MOPD without a prior RL stage misses the point of on-policy distillation.
- **Teacher conflicts near task boundaries.** If a sample looks like task A to the router but task B to the teacher, you can get oscillation. Add a soft-routing option (mix two teachers' logits) at known ambiguous boundaries.

---

## Sources

- Paper: *Qwen-Audio-3.1-ASR Technical Report* — Qwen Audio team, Alibaba, 2026 — [arXiv:2610.28711](https://arxiv.org/abs/2610.28711). Introduces MOPD as the stage-6 post-training step with ASR / SD / AU / text teachers.
- Background: k3 reverse-KL estimator (Schulman); verl framework for RL / distillation plumbing.
