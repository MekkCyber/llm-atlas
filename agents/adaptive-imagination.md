# Adaptive imagination for world-action models
*Depth — a learned stop-or-continue policy on world-model rollouts, from RISE.*

**TL;DR:** World Action Models (WAMs) improve planning by rolling out an imagined future before acting. Standard WAMs imagine for a fixed number of steps at *every* scene, wasting compute where imagination doesn't change the action and under-thinking on the ones where it would. RISE trains a stop-or-continue policy on top of the rollout: keep imagining while extra steps are likely to change the action, stop as soon as they aren't.

**Prereqs:** [../post-training/_rl](../post-training/_rl.md)
**Related:** [rlm-harness](rlm-harness.md)

---

## What it is

An adaptive-compute layer for imagination-based policies. Given a WAM that predicts future states step by step, adaptive imagination decides *when to stop* — dynamically per scene — instead of using a fixed imagination budget.

## How it works

- **Two components:** a WAM (predicts next state) and a *stop policy* (binary decision at each imagined step).
- **Stop policy input:** current imagined state, current planned action, and a running proxy for "planning value gained so far."
- **Decision rule:** stop when the expected marginal benefit of an extra imagined step is below a threshold. Otherwise continue rolling out.
- **Training signal:** counterfactual data — pairs of scenes where imagining longer would/would not have changed the action, together with expert risk annotations. RISE bundles this into the **CounterDrive** dataset for autonomous driving.
- At inference: imagination proceeds one step at a time; each step the stop policy is queried; when it stops, the planner commits its current action.

## Why it matters

The adaptive-compute story generalizes: any policy that can trade compute for prediction quality can borrow this pattern. In driving specifically, most scenes are trivial and don't need imagination; the ones that do need it need *more* than a fixed budget provides. Adaptive imagination captures both.

## Gotchas & tricks

- Stop-policy training is a supervised problem, not RL — you need labeled counterfactual data. CounterDrive provides one such dataset for driving; the same recipe requires task-specific data elsewhere.
- The stop policy is a tiny extra head on top of the WAM's shared features. It is cheap; the compute savings come from what it *doesn't* imagine.
- Reported gains include better planning accuracy on NAVSIM/nuScenes plus reduced imagination compute. Compute reduction alone (without accuracy gain) isn't the goal — the point is that imagination effort is redirected, not just cut.
- Composes with any planner–world-model pair, not just autonomous driving.

## Sources

- Paper: *RISE: Adaptive Imagination for World Action Models* — Lu et al., 2026 — [arXiv:2608.20430](https://arxiv.org/abs/2608.20430)
