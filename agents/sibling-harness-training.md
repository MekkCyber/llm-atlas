# Sibling-harness agent training
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** An agent policy trained inside one scaffold ("harness") often fails on tasks a sibling scaffold would solve. Sibling-harness training collects successful rollouts from a *family* of harnesses built from a common base, then trains one policy on the aggregated trajectories so it absorbs the cross-harness skill set while still running under the native harness at inference. Introduced in Harness-S (2026).

**Prereqs:** [rejection-sampling](../post-training/rejection-sampling.md), [rl-prompt-curation](../post-training/rl-prompt-curation.md)
**Related:** [grpo](../post-training/grpo.md), [rlvr](../post-training/rlvr.md)

---

## What it is

An agent "harness" is the scaffolding around an LLM that implements tool schemas, memory, prompt format, and the control loop. Policies trained inside one harness rarely transfer: a model tuned for AppWorld's tool interface struggles under SWE-Gym's harness, even when the underlying capability (code reasoning, tool use) is the same. Sibling-harness training exploits the fact that *variants of the same base harness* (different tool sets, prompt orderings, memory layouts) succeed on complementary subsets of tasks.

## How it works

1. **Sibling construction.** Start from a base harness; produce N variants that change one axis at a time (tool subset, prompt template, memory structure, system prompt).
2. **Rollout collection.** On a shared training task set, run every sibling. Keep only rollouts that reach a successful terminal state (rejection sampling at the trajectory level).
3. **Aggregation + conditioning.** Union all successful trajectories; condition on a harness-identity token so the model learns which actions belong to which scaffold.
4. **Single-policy training.** Fine-tune one policy on the aggregate; at inference, run it under the native harness without the identity token leaking.
5. **Optional routing.** A tiny router at inference time picks which sibling's harness-identity token to inject per task, exploiting cross-harness complementarity.

The key observation is that sibling rollouts exhibit three properties: **locality** (siblings fail on a common subset), **complementarity** (some sibling solves most "base-failed" tasks), and **nonstationarity** (which sibling wins depends on the task).

## Why it matters

Agent training has been siloed by scaffold choice; models get re-trained from scratch every time a tool API changes. Sibling training treats scaffolds as a *data augmentation* axis — much the way instruction templates became an axis in SFT. Reported gains on AppWorld, SWE-Gym, and DBBench, with further gains from inference-time routing.

## Gotchas & tricks

- **Sibling diversity matters.** If sibling variants are too similar, complementarity is small. The paper recommends varying at least two axes simultaneously.
- **Rejection-sampling bias.** Only successful rollouts enter training, so the policy learns from an easier-than-real distribution. Mix in some negative-outcome rollouts with a lower weight to retain failure-recovery behaviour.
- **Routing cost.** Inference-time sibling routing adds a prompt-length overhead; worth it for DBBench, marginal for AppWorld in the paper's numbers.

## Sources

- Paper: Harness-S: Learning from Sibling Harnesses for Generalizable Agent Training — Jiao et al., 2026 — [arXiv:2611.04277](https://arxiv.org/abs/2611.04277)
