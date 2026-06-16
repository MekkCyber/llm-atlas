# DA-GRPO — Decomposition-Aware GRPO
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** GRPO variant for multi-agent orchestration: an orchestrator decomposes a task into sub-tasks and dispatches them to specialized sub-agents. DA-GRPO distributes the **group-relative advantage along the decomposition tree** so the orchestrator gets credit for its decomposition decisions and each sub-agent gets credit for executing its sub-task. The decomposition tree itself is part of the training signal, letting the orchestrator learn when *not* to decompose. Used to train Orchestra-o1-8B, reported +10.3% on OmniGAIA over single-agent baselines.

**Prereqs:** [grpo.md](grpo.md), [ppo.md](ppo.md), [_rl.md](_rl.md)
**Related:** [appo.md](appo.md), [../agents/agent-harness.md](../agents/agent-harness.md)

---

## What it is

Multi-agent orchestrators (a planner + specialized executors) introduce a credit-assignment problem GRPO doesn't address: when the final outcome is good or bad, *who* gets the credit? The orchestrator (for picking the right decomposition)? Each sub-agent (for executing well or poorly)? The default — assigning uniform advantage to every token in the joint rollout — gives the orchestrator no incentive to improve its decomposition policy and gives sub-agents noisy signal mixed with orchestrator decisions.

DA-GRPO makes the decomposition tree first-class: advantages are computed per-node of the tree, weighted by the node's contribution to the outcome, and applied to the orchestrator's decomposition tokens or the sub-agent's execution tokens accordingly.

---

## How it works

### Decomposition tree

Each rollout produces a tree:

- Root: the orchestrator's full output, including the sub-task list and final aggregation.
- Children: each sub-agent's full execution trace.
- Leaves: tool calls or final responses.

Tokens in the trace are tagged by which node owns them.

### Group-relative within decomposition class

For a group of $G$ rollouts on the same prompt, group them by **decomposition class** (e.g. "decomposed into 3 sub-tasks of type A/B/C" vs. "answered directly"). Compute group-relative advantages within each class. This avoids comparing apples to oranges — direct-answer rollouts get one baseline, decomposed rollouts get another.

The group-mean reward of each class also gives the orchestrator signal about *which class* to prefer for this prompt type.

### Advantage propagation along the tree

A successful sub-agent on a useful decomposition gets the full advantage signal. A successful sub-agent on a useless decomposition gets discounted credit (the outcome was hostage to orchestration choice). The orchestrator's decomposition tokens receive a weighted aggregate of the sub-agents' advantages plus a direct outcome contribution.

The remainder of the update is standard PPO-clipped with KL to the reference model.

---

## Why it matters

- **Trainable orchestration.** Most multi-agent systems are hand-orchestrated; DA-GRPO lets the orchestrator's policy be learned end-to-end.
- **Sub-agent specialization without per-agent supervision.** Each sub-agent's advantage is naturally specialized to its sub-task by the tree structure — no separate per-agent reward needed.
- **"Don't decompose" is learnable.** By comparing across decomposition classes, the orchestrator learns when the overhead of decomposition costs more than it saves.
- **Drop-in over GRPO.** Same outer loop, KL, reference model. The change is in advantage computation and where it's applied.

---

## Gotchas & tricks

- **Group-by-decomposition needs enough rollouts per class.** If one class has only 1–2 rollouts in the group, group-relative variance estimates blow up. Use larger $G$ or sample with class balancing.
- **Tree-depth heterogeneity.** Rollouts in the same group may have different tree depths; per-token credit needs careful weighting to avoid favoring deeper trees by sheer token count.
- **Sub-agent reward leakage.** A sub-agent receiving advantage from a parent's decomposition decision can game the system by emitting outputs that bias the parent's aggregation. Standard PPO clipping mitigates but doesn't eliminate this.
- **Multi-modal sub-agents.** Orchestra-o1's variant adds modality-typed sub-tasks (text / image / audio / video); DA-GRPO doesn't require this, but the tree structure assumes typed children.
- **Cold start needs a working orchestrator.** From scratch, the orchestrator produces incoherent decompositions and sub-agents get nonsense sub-tasks. Warm with SFT on hand-orchestrated traces before applying DA-GRPO.

---

## Sources

- Paper: *Orchestra-o1: Omnimodal Agent Orchestration* — Zhang, Qian, Li, *et al.*, CUHK · Lightspeed · PKU · THU, 2026 — [arXiv 2606.13707](https://arxiv.org/abs/2606.13707).
- Background: [grpo.md](grpo.md) for the group-relative baseline this extends; multi-agent RL literature for the broader credit-assignment problem.
