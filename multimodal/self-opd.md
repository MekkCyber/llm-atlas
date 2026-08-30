# Self-OPD: Teacher-Free On-Policy Distillation for Flow Matching
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** On-policy distillation (OPD) has worked well for LLMs and has recently been ported to flow-matching image/video models — but it needs a task-specific *teacher* per objective, which is expensive to train and introduces teacher/student distributional mismatch. Self-OPD kills the teacher: at each timestep it branches the deterministic next-state into $K$ stochastic SDE candidates, scores them against a deterministic self-reference baseline, and pushes the velocity field toward high-advantage branches (and away from low-advantage ones). It's **GRPO's baseline trick, ported to velocity fields**.

**Prereqs:** [../post-training/grpo.md](../post-training/grpo.md), [../post-training/_rl.md](../post-training/_rl.md)
**Related:** [README.md](README.md), [../post-training/rlvr.md](../post-training/rlvr.md)

---

## What it is

A reward-shaped update for pretrained flow-matching generators that replaces an OPD teacher with a **self-reference deterministic sample**. All rollouts, advantages, and updates come from the student itself — the reward model (or verifier) is the only external signal.

## How it works

Per training step, per timestep $t$:

1. **Deterministic self-reference.** From the current velocity field, roll out the ODE sampler forward to produce a baseline candidate $x_\text{ref}$.
2. **Stochastic branching.** Branch the current $t$-th state into $K$ SDE candidates $\{x_1,\dots,x_K\}$ (inject noise, follow the learned drift).
3. **Score.** Complete each branch with the ODE sampler; score each finished sample and $x_\text{ref}$ with the reward model.
4. **Normalized advantages.** $\tilde{A}_i = \frac{R(x_i) - R(x_\text{ref})}{\text{stddev over the } K \text{ scores}}$ — GRPO-style rebaseline against the deterministic self-reference.
5. **All-branch pull-push update.** Push the velocity field toward the direction of high-advantage branches, pull it away from low-advantage ones — a symmetric attractor/repeller update. **Direction-aware attenuation** shrinks the magnitude of updates that would collide with the manifold direction; **SDE-variance normalization** neutralizes branch scale.

**Multi-objective alignment.** With multiple rewards, Self-OPD fuses their normalized scores **at the reward level** (single scalar advantage per branch) — avoiding the gradient conflict that afflicts field-level fusion (per-reward velocity updates that fight each other on shared parameters).

## Why it matters

Removes the "train a teacher per task" bottleneck that has held back OPD adoption for diffusion/flow-matching alignment, and eliminates teacher-student mismatch as a compounding-error source along the generation trajectory. On single-reward and mixed-reward image benchmarks, beats prior RL and OPD methods without any task-specific teacher.

Also generalizes the GRPO recipe from discrete-token policies to continuous generative processes — the branching-and-rebaselining pattern is doing most of the work in both.

## Gotchas & tricks

- **Reward-level fusion is not optional for multi-objective.** Field-level fusion (per-objective velocity updates) empirically collapses due to gradient conflict — the paper explicitly rules it out.
- **$K$ scales like GRPO's group size.** Small $K$ (2–4) is too noisy; the paper uses larger $K$ typical of GRPO rollouts.
- **SDE-variance normalization matters.** Without it, branches with larger stochastic drift dominate the update, biasing away from the local manifold.
- **Direction-aware attenuation** is what keeps updates on-manifold in early training; it's the piece that most closely echoes the KL/trust-region role in language-model RL.

## Sources

- Paper: *Self-OPD: On-Policy Distillation for Flow Matching Models without Teacher* — Zhang et al., 2026 — [arXiv:2608.26872](https://arxiv.org/abs/2608.26872)
