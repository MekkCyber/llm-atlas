# Experience-Handoff RL (ACE-RL)
*Depth — learning *what to write* in an agent's shared playbook, not just how to act.*

**TL;DR:** Long-horizon agents often use a side-channel — a notebook, a playbook, a compacted memory — to pass experience from one attempt to the next. In most systems these artifacts are **hand-written** (rules for what to reflect, how to compact). ACE-RL freezes the Agentic Context Engineering (ACE) harness and trains **one model in three roles — actor, reflector, and playbook curator** — where each edit the reflector proposes to the shared playbook is scored by whether the *next attempt's task reward* improves. Turns passive reflection into a learned feedback loop and lifts held-out performance over hand-written baselines.

**Prereqs:** [../post-training/grpo](../post-training/grpo.md)
**Related:** [../post-training/_rewards](../post-training/_rewards.md), [../post-training/rlvr](../post-training/rlvr.md)

---

## What it is

Harness-level self-improvement. Policy-level RL (GRPO, PPO on actor rollouts) improves the agent's weights. ACE-RL improves what the *harness* hands to the agent across attempts — the playbook text that conditions future rollouts — and learns the editing policy with RL.

## How it works

- **Fixed harness.** The ACE scaffold (actor → attempt → reflector → playbook edit → next attempt) is kept as-is.
- **Three roles, one policy.** The same LLM plays:
  - **Actor** — performs attempts conditioned on the current playbook.
  - **Reflector** — reads an attempt's trajectory and environment feedback (including error logs, not just scalar rewards) and proposes playbook edits.
  - **Curator** — merges proposed edits into the shared playbook with some precedence rules.
- **Reward signal.** Each playbook edit e is scored by the delta in subsequent-attempt task reward: r(playbook ∪ e) − r(playbook). This is a dense, learned reward over the harness's own artifacts.
- **Training.** GRPO-family objective over three rollout types (acting, reflecting, curating), each with its own reward channel.

## Why it matters

- **Opens harness state to learning.** Agent frameworks have accumulated hand-written tricks (reflections, notebooks, scratchpads); ACE-RL is a path to letting the model learn what to put in them.
- **Preserves scalar-reward RL.** The environment reward still drives learning — ACE-RL just backpropagates it through the playbook-edit action instead of only the token-generation action.
- **Recursive self-improvement, bounded.** The harness is fixed, so capability growth is bounded to what the playbook text can represent — but inside that bound, improvement is self-driven.

## Gotchas & tricks

- The playbook becomes an unbounded growing artifact without a compaction policy; the curator role matters as much as the reflector.
- Credit assignment over harness edits is noisy: whether a given edit helped depends on the actor's next attempt, which itself has stochasticity. The GRPO-style group baseline mitigates this but doesn't eliminate it.
- Generalizes best when the environment surfaces structured feedback (error logs, trace snippets). On opaque, reward-only environments the reflector has less to work with.

## Sources

- Paper: *Experience-Handoff Reinforcement Learning for Recursive Self-Improvement* — Kim, Jeon, Zhang, Yang — KAIST AI, 2026 — arXiv 2611.16409 — https://arxiv.org/abs/2611.16409
- Agentic Context Engineering (ACE harness) — referenced baseline.
