# Adaptive Tool Use
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Tool-augmented agents commonly show **net-zero gains** on average benchmarks — the wins on hard problems are cancelled by regressions on easy ones (the model calls a tool it didn't need and gets confused). **Beacon** (Peking U. / Kling, 2026) reframes agentic reasoning around two measurable properties — **Mode Adaptiveness** (invoke tools only when necessary) and **Tool Effect** (net capability change from tool use) — and trains for both with a **Necessity-Aware Adaptive Reward** and a **Hint-Guided Capability Expansion** RL scheme.

**Prereqs:** [../post-training/grpo](../post-training/grpo.md), [../post-training/_rewards](../post-training/_rewards.md)
**Related:** [../post-training/_rl](../post-training/_rl.md), [../multimodal/README](../multimodal/README.md)

---

## What it is

The unstated assumption in most tool-agent training: "more tool use is better." Empirically, it isn't — tool calls have overhead, tool outputs can mislead, and MLLMs often invoke tools defensively on problems they'd solve better in one shot. Adaptive tool use makes *when* to call a tool a first-class training target, not just *how*.

Two metrics operationalize this:

- **Mode Adaptiveness (MA):** does the model recognize tool necessity? Measured by comparing tool-call rate against oracle necessity.
- **Tool Effect (TE):** net capability change from tool use. Decomposed into gains on *hard* (tool-necessary) and losses on *easy* (tool-unnecessary) subsets.

## How it works

**Necessity-Aware Adaptive Reward.** For each rollout, a *necessity label* is estimated (from prior no-tool attempts or from the answer-only correctness of the same prompt). Reward is shaped:

- +reward for correct answers *when the tool call was necessary*.
- Penalty for tool calls on *unnecessary* prompts (even if answer is correct — the overhead is a cost).
- No penalty for skipping tools when unnecessary.

This directly targets MA rather than just correctness.

**Hint-Guided Capability Expansion.** The hard tail of tool-necessary problems is small in generic training data. During RL, the framework injects hints on the *hardest* problems, expanding the reachable set of tool-solvable tasks — the policy learns tool use on genuinely hard examples rather than just calibrating on easy ones.

Base algorithm is GRPO-style: group rollouts, group-normalized advantages, KL to reference.

## Why it matters

- **First clean measurement** of the actual failure mode — "tools don't help on average" was an open puzzle. MA + TE separates the two forces.
- **Reward shaping fixes it** — Beacon shows consistent MA and TE improvements across diverse benchmarks, not just aggregate accuracy.
- **Applies beyond visual tools** — the framework is agnostic to tool type; MA/TE are useful metrics for any tool-augmented agent (function calling, code interpreter, search, browser).

## Gotchas & tricks

- Necessity labels are noisy — a "no-tool" attempt may fail for reasons unrelated to tool need. Multiple no-tool attempts per prompt improves the label.
- Penalty magnitude is a tuning axis: too high and the policy stops using tools altogether; too low and it defaults to always-call.
- Hint injection can leak information at test time — carefully separate hinted training rollouts from held-out eval.
- MA and TE can be gamed independently — always report both together (a model with perfect MA but zero TE isn't useful).

## Sources

- Paper: *Beacon: Knowing When and How to Perform Agentic Visual Reasoning* — Wang et al., Peking U. / Kling Team, 2026 — [arXiv:2607.28595](https://arxiv.org/abs/2607.28595).
