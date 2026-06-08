# AdaPlanBench
*Depth — adaptive-planning agent benchmark where constraints are revealed only when the agent's plan would violate them.*

**TL;DR:** Most planning benchmarks reward one-shot synthesis: agent reads goal + constraints, outputs a plan, eval scores it. Real deployment is messier — constraints (physics, user preferences) are partially hidden and surface only when a proposed action would violate them. AdaPlanBench (Liu et al., 2026, UIUC) is a dynamic interactive harness over 307 household tasks where hidden constraints fire *as conflicts* when the agent commits to a violating action, forcing iterative re-planning. The best of ten leading LLM agents only reaches **67.75% accuracy**, with user constraints proving substantially harder than world constraints.

**Prereqs:** [README](README.md)
**Related:** [../agents/README.md](../agents/README.md), [ifeval](ifeval.md)

---

## What it is

A benchmark + harness pair targeting *adaptive re-planning* rather than upfront plan synthesis. Distinguishing features:

- **Progressive constraint disclosure.** A subset of constraints (world physics, user preferences) is hidden at start and revealed only when the agent's chosen action would violate them.
- **Two constraint sources.** *World* constraints (object affordances, physics, environment state) and *user* constraints (preferences, restrictions). Separately scored.
- **Scalable difficulty.** Number of deferred constraints and depth of plan revision required are tunable. 307 household tasks span the easy/hard ends.

## How it works

The harness is an interactive loop:

```
state = init_world(task)
plan  = agent.plan(task.goal, visible_constraints)
while not done:
    action = next(plan)
    if violates_hidden_constraint(action, hidden):
        agent.observe(violation_signal)
        plan = agent.replan(...)
        continue
    state = step(state, action)
done = goal_reached(state)
score = success_rate, with separate metrics for world-violations vs user-violations
```

Scoring breaks down accuracy by:
- Overall task completion.
- World-constraint compliance (did the agent eventually respect physics / object affordances?).
- User-constraint compliance (did the agent track the disclosed user preferences across replans?).

## Why it matters

- **Surfaces a real gap.** State-of-the-art LLM agents score 67.75% on AdaPlanBench, well below their one-shot planning scores on static benchmarks. Adaptive re-planning is genuinely under-served by current training.
- **User constraints are the hard axis.** The split between world and user constraints reveals that current models are better at physical-causal reasoning than at preference-tracking across revisions — a finding that doesn't show up on static benchmarks at all.
- **Calibrates agent eval to deployment.** Real apps disclose constraints in pieces; static benchmarks don't. AdaPlanBench narrows that gap.

## Gotchas & tricks

- **Harness costs.** Interactive eval is slower than static eval (many replans per task). Budget eval compute accordingly.
- **Re-planning != fresh planning.** A baseline that throws away the current plan and re-plans from scratch every violation is simple but wasteful. The benchmark rewards agents that *incorporate* the violation signal into a revision.
- **Constraint specification matters.** If hidden constraints are too "trivia" (random arbitrary rules), the eval becomes a memory game rather than a planning one. AdaPlanBench's hidden constraints are domain-grounded.
- **Avoid prompt leak.** A naive harness leaks hidden constraints through verbose error messages. Keep the violation signal minimal.

## Sources

- Paper: *AdaPlanBench: Evaluating Adaptive Planning in Large Language Model Agents under World and User Constraints* — Liu et al., UIUC, 2026 — [arXiv:2606.05622](https://arxiv.org/abs/2606.05622) — introduces the benchmark, harness, and the 67.75% SOTA result.
