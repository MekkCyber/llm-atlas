# Harness Scaling
*Depth — RL-train an agent across many harness configurations so the policy transfers to unseen harnesses.*

**TL;DR:** The *harness* is the execution scaffold around an LLM agent — context policy, tool-call schema, retry rules, output format. RL-training a policy inside a single fixed harness overfits to that harness; the same model in a different scaffold often collapses. **Harness Scaling** factorizes the harness into independent axes, builds a compact training set with balanced coverage of combinations, and samples a harness alongside each task during RL. Transfers to unseen white-box harnesses and to external black-box harnesses too.

**Prereqs:** [_rl](../post-training/_rl.md), [rlvr](../post-training/rlvr.md)
**Related:** [rl-prompt-curation](../post-training/rl-prompt-curation.md), [socratic-skill-learning](socratic-skill-learning.md)

---

## What it is

A real-world agent deployment has two moving pieces:

1. The **model** — the weights that generate actions.
2. The **harness** — everything around the model: how context is managed, what the tool-call API looks like, how outputs are parsed, when retries fire, how multi-turn history is formatted.

Standard RL post-training for agents fixes the harness and trains the model inside it. The failure mode: the model's behavior becomes *coupled* to the fixed harness. Swap the context policy or change the tool schema and the policy collapses. The paper documents this as "harness overfitting" — a sibling of distribution shift, but on the scaffold rather than the data.

Harness Scaling is domain randomization applied to the agent scaffold during RL.

---

## How it works

**Factorize the harness.** Identify independent axes:

- Context strategy (append / summarize / sliding window).
- Tool-call format (JSON / XML / natural language).
- Retry policy (fail-fast / up-to-N / backoff).
- History injection (full / abstracted / per-tool).
- Error presentation (raw / cleaned / coached).

**Pick a balanced training set.** A full Cartesian sweep over axes blows up; the paper selects a compact subset that is *balanced* over factorial coverage — every axis value appears about equally often, and pairs of axes are covered too.

**Sample harness per task.** During RL rollouts, each task is paired with harness configurations *structurally nearby* to each other — the agent sees the same task under several scaffolds in a batch, learning which behaviors are harness-robust vs. which depend on the scaffold.

**Train with any RL algorithm.** PPO, GRPO, DPO-style on-policy — the method is harness-level data generation, not a loss-level change.

---

## Why it matters

Agent frameworks (LangGraph, OpenAI Agents SDK, MCP, custom in-house) differ in scaffold details. A model that only works in one scaffold is a liability. Harness Scaling gives the first systematic recipe for harness robustness, and the empirical result (transfer to *unseen external black-box harnesses*) is strong: it's not just robustness within the training factory, but to external frameworks the model has never seen.

Secondary benefit: the factorization makes it obvious which scaffold choices matter — some axes are robust under randomization, some are not, and that information guides which parts of the harness to standardize across deployments.

---

## Gotchas & tricks

- **Coverage ≠ Cartesian.** The whole point is to not pay for the Cartesian product. Use balanced factorial / orthogonal arrays; the compact set is where the sample efficiency comes from.
- **Axis independence is approximate.** Some axes interact (e.g. tool-call format × error presentation). The paper handles this with pair-wise coverage; triple-wise is usually overkill.
- **Harness-paired batching.** Pair each task with structurally-nearby harnesses within a batch — the policy gradient benefits from seeing the same task under similar scaffolds back-to-back.
- **Does not fix tool overfitting.** Harness Scaling is about the scaffold, not about the toolset. Tool-set randomization is a separate (and often complementary) move.

---

## Sources

- Paper: *Harness Scaling for Generalizable Agentic Reinforcement Learning* — Huatong Song et al., IQuest Research / Beihang / RUC, 2026 — [arXiv:2610.29458](https://arxiv.org/abs/2610.29458).
- Background: domain randomization in robotics (Tobin et al., 2017); prompt-perturbation evaluations in LLM robustness work.
