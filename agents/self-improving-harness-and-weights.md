# SIA — self-improving via harness + weight updates
*Depth — a Feedback-Agent loop that updates both the scaffold and the weights of a task-specific agent on the same objective.*

**TL;DR:** Self-improving-AI research has two silos. Harness-update papers have a meta-agent rewrite prompts / tools / retry logic with the weights frozen; test-time-training papers update weights with the scaffold frozen. SIA argues these are complementary control surfaces and runs both in one loop. On three contrasting domains (legal classification, GPU kernel optimisation, single-cell RNA denoising) the combination beats scaffold-only baselines by a wide margin.

**Prereqs:** [rlvr](../post-training/rlvr.md), [grpo](../post-training/grpo.md)
**Related:** [open-world-self-evolution](open-world-self-evolution.md), [trace-derived-skills](trace-derived-skills.md)

---

## What it is

A loop in which a *Feedback-Agent* observes a task-specific agent's performance after each evaluation and decides whether to (a) rewrite the agent's harness — prompts, tool definitions, retry rules, search procedure — or (b) take a weight-update step on the agent's underlying model. The same Feedback-Agent makes both calls; the same reward feeds both updates.

## How it works

```
loop:
  trajectories = task_agent.run(eval_set)
  feedback     = feedback_agent.analyse(trajectories)
  if feedback.proposes_harness_edit:
      task_agent.harness = apply(feedback.harness_patch)
  if feedback.proposes_weight_step:
      task_agent.weights = rl_step(task_agent.weights,
                                   feedback.rl_signal)
```

The Feedback-Agent's choice between harness edits and weight steps is itself learned (or prompted) — it picks the cheaper, more effective lever each iteration. Harness edits buy *agency* (which tool to call, when to retry); weight updates buy *intuition* (domain heuristics no prompt can install).

## Why it matters

- Resolves a real silo: harness-update and test-time-training communities had been arguing past each other.
- Practical for deployment: weight updates are expensive, harness edits are cheap; SIA naturally schedules them by effectiveness.
- Reported gains are large: **+56.6% on LawBench**, **91.9% runtime reduction on GPU kernels**, **+502% on single-cell RNA denoising** over the initial baseline.

## Gotchas & tricks

- **Feedback-Agent stability.** A meta-agent that proposes wild harness rewrites can break the task agent between weight steps; constrain the edit space (allowed prompt templates, allowed tool deltas).
- **Catastrophic-forgetting risk on the weight track.** RL on a narrow eval set can degrade unrelated capabilities — keep a held-out general benchmark in the loop.
- **Compute accounting matters.** Harness edits look free vs. weight steps but a bad harness can multiply rollout cost; track wall-clock-per-improvement, not edits-per-iteration.

## Sources

- Paper: *SIA: Self Improving AI with Harness & Weight Updates* — Hebbar, Manawat, Verboomen, Ivanova, Palanimalai, Bhatia, Baskaran — 2026 — [arXiv:2605.27276](https://arxiv.org/abs/2605.27276)
