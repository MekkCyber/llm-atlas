# ReOPD — Replayed-Prefix On-Policy Distillation
*Depth — reuse cached teacher trajectories as *replayed prefixes* so multi-turn on-policy distillation needs zero fresh environment interactions during student training.*

**TL;DR:** On-policy distillation (OPD) for agentic multi-turn tasks is expensive because each update requires fresh student rollouts through the environment and teacher queries at visited histories. **ReOPD** replaces live environment rollouts with pre-collected teacher trajectories that are replayed as prefixes: the student acts only at selected steps, the teacher supplies dense per-step supervision from cache, and no new tool calls happen during training. A step-decaying prefix sampler concentrates supervision at early, low-shift prefixes to avoid the *prefix trap*.

**Prereqs:** [_post-training](_post-training.md), [rejection-sampling](rejection-sampling.md)
**Related:** [../systems/partial-rollouts](../systems/partial-rollouts.md), [../agents/README](../agents/README.md)

---

## What it is

An off-environment training regime for distilling multi-turn agents. The student never actually calls a tool or environment during training; instead, it takes over at a chosen step inside a *cached teacher trajectory* and predicts what the teacher did next. Because the teacher's step-by-step targets were logged at collection time, dense per-step supervision is available without re-querying the teacher live.

## How it works

Given a fixed pool of teacher trajectories $\{\tau_1, \dots, \tau_N\}$ where each $\tau = (s_0, a_0, s_1, a_1, \dots)$ was collected by the teacher acting in the real environment:

1. **Sample a trajectory** $\tau$ and a **prefix length** $k$ from a schedule $p(k)$.
2. **Replay** $(s_0, a_0, \dots, s_{k-1}, a_{k-1})$ as the prefix — the student sees it as history.
3. The student produces its own action distribution at step $k$, and the loss is the KL / cross-entropy against the teacher's cached target at step $k$.

The **prefix trap**: pushing the prefix distribution toward the student's own occupancy makes the training data more student-relevant, but drifts it into histories where the teacher's cached targets are unreliable (regions the teacher never actually reached). This is a two-sided distribution shift.

ReOPD's fix is a *reliability-aware prefix design*: implement it with a **step-decaying sampler** that assigns more probability to smaller $k$ (early prefixes are closer to states the teacher actually visited, so its cached targets are trustworthy). Later steps get less mass, damping the drift.

## Why it matters

- **Zero tool calls during training.** Turns expensive environment interaction into a reusable offline resource; one teacher-rollout collection amortizes across many student runs.
- **≥4× faster per rollout** than fully-online OPD on math-with-Python and search environments, while preserving or improving OPD-level accuracy.
- **Cross-tool / cross-task scaling.** Teacher trajectories can be reused across student model families and scales.

## Gotchas & tricks

- **The prefix trap is real.** Uniform $p(k)$ over trajectory length blows up loss variance and, worse, teaches the student on histories the teacher never truly experienced. The step-decay schedule is not decoration.
- **Teacher determinism matters.** If the teacher was stochastic and only one sample was logged per step, its target is a noisy estimate. Prefer logging top-K logprobs per step at collection time.
- **Bookend with a small on-policy pass.** For extreme distribution shift, a short fully-online OPD phase after ReOPD closes any last mismatch cheaply.
- **Doesn't replace exploration.** ReOPD assumes the teacher already explored the interesting regions. If the student needs to visit new states, add on-policy RL — not another ReOPD round.

## Sources

- Paper: *Multi-Turn On-Policy Distillation with Prefix Replay* — Liao, Dong, Monz, Xu, Dong, Wei (MSR / U. Amsterdam), 2026 — [arXiv:2607.04763](https://arxiv.org/abs/2607.04763).
