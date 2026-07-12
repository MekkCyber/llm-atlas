# Proactive Memory Agent
*Depth — memory as an active intervention policy for long-horizon agents.*

**TL;DR:** Long-horizon agents suffer *behavioral state decay*: task requirements, environment facts, prior diagnoses, and open subgoals get scattered across the trajectory or pushed out of context, and stop influencing decisions. Standard retrieval treats memory passively. The Proactive Memory Agent (PMA) runs a **separate memory agent** alongside an unmodified action agent — it maintains a structured memory bank from the recent trajectory and *decides on every step* whether to inject a memory-grounded reminder or stay silent. Plug-and-play with any frontier action agent; the memory policy is trained with SFT then GRPO.

**Prereqs:** [../post-training/grpo.md](../post-training/grpo.md), [../post-training/_rl.md](../post-training/_rl.md)
**Related:** [../post-training/reasoning/long-cot-rl.md](../post-training/reasoning/long-cot-rl.md) · [../post-training/rlvr.md](../post-training/rlvr.md)

---

## What it is

A two-agent architecture. The **action agent** is untouched — any frontier model (Claude, GPT, open-weights) with any harness. The **memory agent** observes the trajectory in a rolling window, extracts structured facts into a memory bank, and at each step outputs one of: (a) inject a reminder into the action agent's context, (b) do nothing. The memory-agent policy is trained on a dedicated task set (**SETA**) using supervised fine-tuning followed by GRPO with rewards on downstream task success.

## How it works

At each turn:

1. Memory agent reads the recent trajectory window and updates a structured bank keyed by fact type (open subgoals, environment observations, prior tool-error diagnoses, task constraints).
2. The memory agent's policy decides whether to fire a reminder. If yes, it retrieves the most relevant memory-bank entry and formats it as a short message injected into the action agent's context.
3. The action agent proceeds unmodified; it sees the reminder only if the memory agent chose to inject one.

Training: SFT on SETA teaches the memory agent to recognize decay-prone moments (subgoal drift, forgotten constraints); GRPO with task-success reward refines the *selective intervention* policy — inject only when it moves the needle. Ablations show selective intervention beats passive bank exposure, always-on injection, advisor-only guidance, and generic retrieval.

## Why it matters

- **Presence ≠ use.** Even with massive context windows, agents fail because relevant facts sit unused. A *trained* intervention policy fixes this at the architecture layer.
- **Composable.** Plugs in front of any harness (Terminal-Bench, τ²-Bench, agent frameworks like Aider / Cursor) without model retraining on the action side.
- **Measurable wins.** +8.3 pp pass@1 on Terminal-Bench 2.0 and +6.8 pp on τ²-Bench with both weaker and stronger action agents; open-weight Qwen3.5-27B memory policy transfers partially from SETA to Terminal-Bench.

## Gotchas & tricks

- Always-on injection *hurts* — it clutters the action agent's context. The learned selective policy is the whole point.
- The structured bank isn't a vector store — facts are typed. This is what lets the memory agent reason about *what kind* of fact would help.
- Memory-agent latency is small relative to action-agent turn cost; the two can run sequentially with negligible overhead.
- Transfer from SETA to production benches is *partial* — the memory-intervention policy generalizes but benefits from domain-specific fine-tuning.

## Sources

- Paper: *Remember When It Matters: Proactive Memory Agent for Long-Horizon Agents* — Zhang et al., Meta AI, 2026 — [arXiv:2607.08716](https://arxiv.org/abs/2607.08716).
- Background: GRPO training recipe — see [../post-training/grpo.md](../post-training/grpo.md).
