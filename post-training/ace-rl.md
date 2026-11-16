# ACE-RL
*Depth — reinforcement learning over an agent's own playbook edits.*

**TL;DR:** Most RL on LLM agents optimizes the policy's actions. ACE-RL (Agentic Context Engineering with RL, from the Experience-Handoff paper) instead optimizes a *persistent playbook* that the agent writes for its own future rollouts. One model plays three roles — Actor, Reflector, Curator — and the Curator's edits to the playbook are the credit-assignment target: an edit is rewarded by the change it causes in held-out future rollouts' returns.

**Prereqs:** [grpo.md](grpo.md), [ppo.md](ppo.md)
**Related:** [_rl.md](_rl.md), [rl-prompt-curation.md](rl-prompt-curation.md)

---

## What it is

A training setup for LLM agents that separates in-episode acting from cross-episode experience accumulation. The persistent artifact is a *playbook* — a structured context the Actor conditions on at every rollout. ACE-RL's novelty is that this playbook is not hand-written; it is written and edited by the same model, and the edits are what RL optimizes.

## How it works

Three roles inside one Agentic Context Engineering harness, all served by the same LLM:

1. **Actor.** Takes an input task and the current playbook; runs the rollout.
2. **Reflector.** Reads the rollout's trajectory and outcome; proposes candidate playbook edits.
3. **Curator.** Decides which proposed edits enter the playbook. The Curator is the policy ACE-RL trains.

Credit assignment: each playbook edit (an insertion, deletion, rewrite) is scored by its *future-rollout effect* — the average return on held-out tasks after the edit is applied, minus the return before. That delta is the RL reward signal for the Curator's decision to accept or reject. The Actor and Reflector can be frozen or jointly trained; the paper focuses on the Curator.

## Why it matters

Pushes "recursive self-improvement" from a slogan to a trainable objective. Instead of optimizing the current rollout's actions (where credit is noisy), the training target is the *artifact* the model will keep relying on. Edits that generalize across many future rollouts are reinforced; one-off tricks fade. Fits long-running agent deployments where the playbook is the real system state.

## Gotchas & tricks

- The reward signal requires held-out rollouts per edit — expensive. The paper amortizes by batching edits; naive per-edit rollout is intractable.
- The Curator operates on structured edits (insert/delete/rewrite segments), not free text. Loosening this to "write arbitrary playbook text" degrades credit assignment.
- Playbook bloat is a real failure mode. Edits that are never rewarded net-negatively still accumulate unless the curator is penalized for size.
- Reflector quality bounds the Curator's search space — if the Reflector proposes only tiny edits, ACE-RL plateaus quickly regardless of how good the reward signal is.
- Held-out task distribution matters: an edit that helps on the training distribution can be penalized if the held-out set is narrow. Treat the held-out set as a first-class hyperparameter, not just an eval slice.

## Sources

- Paper: *Experience-Handoff Reinforcement Learning for Recursive Self-Improvement* — Kim, Jeon, Zhang, Yang (KAIST AI), 2026 — [arXiv:2611.16409](https://arxiv.org/abs/2611.16409)
