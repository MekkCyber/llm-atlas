# Self-OPD — Teacher-free on-policy distillation for flow-matching models
*Depth — GRPO-style self-supervision for continuous-time generative models.*

**TL;DR:** On-policy distillation (OPD) for flow-matching / diffusion normally needs a task-specific teacher — expensive and prone to student–teacher drift. Self-OPD drops the teacher: at each timestep, branch the student's ODE step into K SDE candidates, score each, and update the velocity field with a pull-push loss driven by advantages against a deterministic self-reference baseline. Beats prior RL and OPD methods on single- and mixed-reward benchmarks.

**Prereqs:** [../multimodal/flow-matching.md](../multimodal/flow-matching.md), [grpo.md](grpo.md), [_rl.md](_rl.md)
**Related:** [rejection-sampling.md](rejection-sampling.md), [_rewards.md](_rewards.md)

---

## What it is

An RL-flavored alignment method for flow-matching generative models (image / video / audio DiTs) that removes the teacher requirement of standard OPD. The student model is its own baseline: it rolls out both a deterministic ODE trajectory and K noisy SDE branches, and learns which branches are better than its own default step.

## How it works

At each timestep `t` in the generation trajectory:

1. Compute the deterministic next-state prediction (ODE step) — this is the *self-reference baseline*.
2. Branch into `K` stochastic SDE candidates around the deterministic step.
3. Roll each branch to a completed sample via the ODE sampler; score with the reward model(s).
4. Compute normalized advantages `A_i = (r_i − r_ref) / σ_r` — GRPO-style, but with a deterministic baseline instead of a group mean.
5. Update the velocity field with an **all-branch pull–push loss**: high-advantage branches attract the student, low-advantage branches repel it, both scaled by direction-aware attenuation and SDE-variance normalization to prevent noisy updates from dominating.
6. For multi-objective alignment, fuse rewards at the *reward level* (normalize each reward, then combine into `r_i`) rather than the gradient level — avoids gradient-direction conflict between objectives.

## Why it matters

Flow-matching alignment has lagged LLM alignment because every task needed a bespoke teacher model. Self-OPD imports the GRPO recipe — group-based advantages, no critic — into continuous-time generative modeling, making per-task alignment 10×+ cheaper and reducing the compounding-error problem of student–teacher distribution mismatch.

## Gotchas & tricks

- The deterministic self-reference is what makes the baseline stable; a stochastic group mean (naive GRPO port) creates a moving target and destabilizes training.
- SDE-variance normalization matters — without it, larger-noise branches dominate the pull-push and the model drifts toward high-variance directions instead of high-reward ones.
- Reward-level (not gradient-level) fusion for multi-objective: gradient conflict was the main failure mode of prior teacher-based OPD.
- Compute per step scales with `K`; typical `K` is 4–8 in the paper's setup.

## Sources

- Paper: *Self-OPD: On-Policy Distillation for Flow Matching Models without Teacher* — Zhang et al., 2026 — [arXiv:2608.26872](https://arxiv.org/abs/2608.26872)
- GRPO precedent: [grpo.md](grpo.md)
