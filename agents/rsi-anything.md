# RSI-Anything
*Depth — recursive self-improvement for agents using learned world models as the feedback signal.*

**TL;DR:** External feedback (an environment, a human, a verifier) is often the bottleneck when iteratively improving an agent. RSI-Anything replaces that signal with a *learned world model* that predicts action outcomes, and uses it to both pick actions during rollout and filter which trajectories to train on. The crucial addition is a **compatibility screen**: a new policy is only accepted if its paired world model still predicts its behavior accurately — preventing the usual RSI collapse where the learner outruns its own evaluator.

**Prereqs:** [grpo](../post-training/grpo.md), [_rl](../post-training/_rl.md)
**Related:** [rlvr](../post-training/rlvr.md), [rejection-sampling](../post-training/rejection-sampling.md)

---

## What it is

A general recipe for sustained recursive self-improvement (RSI) in agents when environment feedback is slow or expensive. The world model acts as a *rollout surrogate*: cheap to query, trained alongside the policy, and used to score candidate actions and candidate trajectories. The agent-plus-world-model is treated as a single object that is updated jointly.

## How it works

Three loops interleave:

1. **Rollout.** The agent generates candidate action sequences. The world model scores them for expected outcome; top-ranked actions are executed in the (expensive) environment to produce ground-truth returns.
2. **Policy update.** The policy is trained on high-return trajectories, with the world-model-predicted return used as a filter / baseline (reducing variance vs on-policy environment rollouts).
3. **World-model update + compatibility screen.** The world model is updated on the new environment data. The new (policy, world model) pair is *compared* to the previous one on a held-out set: if the world model no longer predicts the new policy's behavior well enough, the update is rejected and the previous checkpoint is kept.

The screen is what distinguishes RSI-Anything from naive dyna-style RL: it detects the drift that causes self-play collapse before it corrupts the next update step.

## Why it matters

RSI across heterogeneous domains — code, GUI, laboratory planning, chemistry, immunology — all without a shared external verifier. If world models can keep up with their paired policies, the agent-training budget becomes a function of world-model compute rather than environment-call latency. This changes the economics of long-horizon agent training.

## Gotchas & tricks

- **Compatibility threshold tuning.** Too loose → classic RSI collapse. Too strict → nothing ever accepts, no improvement.
- **Co-training balance.** The world model has to grow at least as fast as the policy on domain coverage; otherwise improvements become invisible to the screen.
- **Not a replacement for an environment.** You still need enough real rollouts to update the world model — the technique shifts *when* and *how often* you query the environment, not whether you need one.

## Sources

- Paper: *RSI-Anything: Unlocking Recursive Self-Improvement with World Models* — Cui, Zeng, Xu et al., 2026 — [arXiv:2610.13912](https://arxiv.org/abs/2610.13912)
