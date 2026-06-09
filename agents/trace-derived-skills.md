# Trace-derived agent skills (Socratic-SWE)
*Depth — distill an agent's historical solving traces into structured skills, then synthesise weakness-targeted practice tasks.*

**TL;DR:** Static synthetic-task generators (random bug injection, mutation testing) don't know where the agent is weak. Socratic-SWE turns the agent's own solving traces into structured *skill descriptors* — recurring failure patterns and effective repair patterns — and uses them to synthesise targeted repair tasks in real repos. Candidate tasks are filtered by execution-based validation and a solver-gradient alignment reward. Pushes SWE-bench Verified to 50.40% after three iterations on the same compute budget as baseline self-evolution.

**Prereqs:** [rlvr](../post-training/rlvr.md), [grpo](../post-training/grpo.md)
**Related:** [self-improving-harness-and-weights](self-improving-harness-and-weights.md), [open-world-self-evolution](open-world-self-evolution.md), [livecodebench](../evaluation/livecodebench.md)

---

## What it is

A closed-loop self-evolution scheme for SWE agents where the *task curriculum* adapts to the solver, instead of being generated once and reused. The novel substrate is the agent's own historical solving traces — not just the success/failure label but the structured sequence of actions and outcomes.

## How it works

Each iteration:

1. **Trace collection.** Run the current solver on a base task pool; log full trajectories (tool calls, errors, retries, fixes).
2. **Skill distillation.** Cluster traces into structured skill descriptors: `{trigger, failure mode, effective repair pattern, evidence traces}`. These descriptors are concise and reusable across tasks.
3. **Task synthesis.** For each skill, generate candidate repair tasks in real repositories that should exercise that skill. Generators are skill-conditioned, not uniform.
4. **Filtering.** Execution-based validation (does the candidate task actually compile / fail meaningfully?) plus a solver-gradient alignment reward — keep tasks whose gradient on the current solver agrees with the desired improvement direction.
5. **Training.** RL update the solver on the retained tasks. New traces feed the next round.

## Why it matters

- Shifts SWE-agent training from "more synthetic bugs" to "weakness-targeted synthetic bugs", echoing the SFT→RLHF shift but at the *task-generation* layer.
- The solver-gradient alignment reward is independently interesting: a generic filter for "this synthetic task will actually move my model".
- Concrete gains: SWE-bench Verified, Lite, Pro, and Terminal-Bench 2.0 all improve over self-evolving baselines under matched compute.

## Gotchas & tricks

- **Trace fidelity drives skill quality.** Noisy or truncated traces produce noisy skills; tooling matters.
- **Skill clustering is a hyperparameter.** Too fine → no transfer; too coarse → no targeting.
- **The alignment reward needs a stable solver.** If the solver is changing rapidly, gradient estimates become noisy and the filter loses signal.
- **Real-repo synthesis can leak.** Validation must ensure new tasks aren't in the eval set's repo space — SWE-bench Verified contamination is a known risk.

## Sources

- Paper: *Socratic-SWE: Self-Evolving Coding Agents via Trace-Derived Agent Skills* — Xiao, Jiao, Wang, Wang, Zhao, Wei, Zhang, Qu — 2026 — [arXiv:2606.07412](https://arxiv.org/abs/2606.07412)
