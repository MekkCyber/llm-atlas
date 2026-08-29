# Self-OPD — Teacher-Free On-Policy Distillation for Flow-Matching
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** On-Policy Distillation (OPD) for flow-matching image/video models normally needs a task-specific teacher, and the teacher-student distribution gap compounds along the generation trajectory. **Self-OPD** replaces the teacher with **stochastic branching from the student itself**: at each ODE timestep, branch the deterministic next-state prediction into K stochastic SDE candidates, roll them out to completion, compare their rewards to a deterministic self-reference baseline, and update the velocity field with an all-branch attract-repel objective. Beats prior RL and OPD baselines on single- and multi-reward image benchmarks with no separately-trained teacher.

**Prereqs:** [grpo.md](grpo.md), [rejection-sampling.md](rejection-sampling.md)
**Related:** [rlvr.md](rlvr.md), [_rl.md](_rl.md), [_rewards.md](_rewards.md)

---

## What it is

OPD for generative models: at each generation step, produce a teacher next-state (from a specialized teacher net trained for the reward) and pull the student's next-state toward it. Problems:
1. **Teachers cost money** — one per objective, trained separately.
2. **Compounding drift** — teacher and student diverge along the trajectory, so late-step supervision is off-distribution.

Self-OPD removes the teacher by turning the student's own stochastic exploration into its own dense supervision.

## How it works

Along the standard ODE sampling trajectory `x_T → x_{T-1} → … → x_0`, at each timestep t:

1. **Branch.** From `x_t`, take one deterministic ODE step to `x_{t-1}^{det}` (the *self-reference*), and K stochastic SDE steps to `x_{t-1}^{(1..K)}` (the *branches*).
2. **Roll out.** Continue each of the K+1 candidates to `x_0` with the deterministic ODE sampler.
3. **Score.** Compute reward `r` on each `x_0` (single or multi-objective).
4. **Advantage.** Normalize branch rewards against the deterministic self-reference reward: `A_k = (r_k − r_ref) / normalizer`. This is the flow-matching analogue of GRPO's group-relative advantage — the deterministic path is the baseline instead of the group mean.
5. **Attract-repel velocity update.** For each branch, update the velocity field `v_θ(x_t, t)` to move toward branches with `A_k > 0` and away from branches with `A_k < 0`, with two stabilizers:
   - **Direction-aware attenuation** — softer pulls when branch directions are close to the reference (avoid over-updating on marginal wins).
   - **SDE-variance normalization** — divide by branch variance so noisy branches don't dominate the gradient.
6. **Multi-objective fusion at reward level.** For multiple rewards, fuse the normalized *scores* (not gradients) before computing the advantage — avoids gradient conflict across objectives.

## Why it matters

- **Removes the per-objective teacher.** New rewards don't require training a new specialized teacher first.
- **Self-referenced baseline avoids OOD drift.** Because the reference and branches share `x_t` and only differ by one step, they're on the same trajectory manifold. Late-step supervision is not off-distribution.
- **Cross-modality echo of GRPO.** Same "group-relative advantage from student rollouts, no critic" recipe that made GRPO the default for LLM RL, ported to flow-matching's continuous action space.

## Gotchas & tricks

- **K controls the tradeoff.** Small K (2–4): cheap but noisy advantages. Large K (16+): stable but expensive rollouts per timestep.
- **Not every timestep needs branching.** Papers typically branch only on a sparse subset of timesteps to keep cost tractable.
- **Reward fusion is fragile with strongly conflicting rewards.** Score-level fusion prevents *gradient* conflict but doesn't resolve *preference* conflict — expect Pareto trade-offs, not a single optimum.
- **Requires deterministic ODE + stochastic SDE variants of the same sampler.** Not all trained flow-matching models expose both.

## Sources

- Paper: *Self-OPD: On-Policy Distillation for Flow Matching Models without Teacher* — Zhang et al. (Zhejiang / Alibaba), 2026 — arXiv:2608.26872.
