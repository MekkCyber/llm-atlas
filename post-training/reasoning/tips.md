# TIPS — Thinking-Induced Process Supervision
*Depth — training a process reward model with outcome-only RL.*

**TL;DR:** Process reward models (PRMs) usually need step-level annotations, which are expensive. TIPS trains a *generative* PRM with pure outcome-only RL: the model writes a chain of thought, emits step-level labels along the way, and emits a final outcome label. The reward depends only on whether the outcome label is correct — step-level labelling is implicitly reinforced because the outcome label depends on it. Tips-Qwen3-4B-Thinking-2507 reaches **85.2 F1 on ProcessBench** using only 3.2K outcome-labelled trajectories.

**Prereqs:** [prm](prm.md), [orm](orm.md), [rlvr](../rlvr.md)
**Related:** [grpo](../grpo.md), [cot-reward-model](../cot-reward-model.md), [long-cot-rl](long-cot-rl.md)

---

## What it is

A way to train a PRM without step-level supervised labels. The PRM is itself an LLM that generates a chain of thought while reasoning about a candidate solution, emits per-step verdicts along that CoT, and finally emits an outcome verdict (correct / incorrect). Training optimizes only the outcome verdict's accuracy, but the step verdicts "come along for the ride" because the outcome one is forced to depend on them.

## How it works

For each candidate solution to be labelled:

1. **PRM rolls out a chain of thought.** It thinks about the solution step-by-step, writing a short note per solution step and emitting that step's verdict (correct / incorrect).
2. **Outcome verdict.** After the step verdicts, the PRM emits an overall outcome verdict.
3. **RL update.** The reward is only +1 / -1 depending on whether the outcome verdict matches the ground-truth outcome label for this solution.
4. **Policy gradient flows back** through both step and outcome tokens. Because the outcome verdict is in the context *after* the step verdicts, and because the training distribution forces it to depend on them, the step verdicts are shaped by backprop even though they have no direct label.

GRPO-style group baselines are used for variance reduction. Only outcome-labelled trajectories are needed — no step annotations required.

## Why it matters

Collapses the data-cost of PRM training by one or two orders of magnitude. Outcome labels can come from automated verifiers (unit tests, numeric match) and are cheap. If step-level verdict quality holds up as reported (85.2 F1 ProcessBench), process-reward reasoning becomes affordable in domains where only outcome supervision was available — code, math contests, agent tasks.

## Gotchas & tricks

- **Spurious step verdicts.** The step verdicts only need to make the outcome verdict correct, so a sophisticated PRM can learn to use them as pure scratchpad with no human-interpretable meaning. Mitigation: evaluate on step-level benchmarks (ProcessBench) during training; add a mild consistency regularizer that penalises contradictory step/outcome patterns.
- **Needs CoT-capable base.** The method depends on the PRM being able to produce a usable reasoning trace; applied to a small non-reasoning base it may fail to credit-assign.
- **Does not replace step labels where they exist.** When step annotations are cheap and clean (e.g. for math competition problems with existing solutions), direct step supervision is still stronger per sample. TIPS wins on *total cost* because outcomes are far cheaper to collect.

## Sources

- Paper: *Inducing Process Supervision from Outcome-Only Reinforcement Learning* — Cong, Zhang, Chen, Lin, 2026 — [arXiv:2609.36641](https://arxiv.org/abs/2609.36641)
