# Proactive Memory Agent
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A **two-agent architecture** for long-horizon agent runs that splits *acting* from *remembering*. An unmodified action agent runs normally; a **separate memory agent** watches its trajectory, maintains a structured memory bank, and — critically — decides on each step *whether to inject a memory-grounded reminder into the action agent's context or stay silent*. The intervention is **selective**, learned via SFT + GRPO on top of Qwen3.5-27B. Plug-and-play with any frontier action agent. +8.3 pp pass@1 on Terminal-Bench 2.0, +6.8 pp on τ²-Bench. Introduced by Zhang et al. (Meta AI), 2026 (arXiv 2607.08716).

**Prereqs:** [../post-training/grpo.md](./../post-training/grpo.md)
**Related:** [../post-training/rlvr.md](./../post-training/rlvr.md) · [../safety/cot-monitoring.md](./../safety/cot-monitoring.md)

---

## What it is

A memory system for LLM agents framed as an **intervention policy**, not passive retrieval. The paper names the failure mode it targets — **behavioral state decay**, where decision-relevant facts (requirements, environment quirks, prior diagnoses, open subgoals) accumulate faster than any fixed context can hold them and get pushed out of reach exactly when the agent needs them. Retrieval-based memory helps but misfires often; unconditional summarization or always-inject reminders eat context budget and derail the action agent. Proactive memory fixes this by learning *when* to speak.

## How it works

Two LLMs run alongside each other over a single agent trajectory.

**Action agent (unmodified).** A frontier model (e.g., Claude Code, Codex, any Terminal-Bench baseline) executing the task normally. No changes.

**Memory agent.** Runs on the same trajectory in parallel. On each step it does two things:

1. **Update the structured memory bank.** Extracts and re-organizes salient facts from the recent trajectory into a structured schema (task requirements, environment facts, prior attempts, diagnoses, open subgoals).
2. **Decide whether to intervene.** A learned policy over `{inject, silent}`. If it decides to inject, it writes a memory-grounded reminder into the action agent's next-turn context; otherwise it stays out of the way.

**Training the memory policy.** The paper trains Qwen3.5-27B as the memory agent, first with SFT on a dataset the paper calls SETA, then with GRPO. Validation reward improves and the trained policy shows partial transfer to Terminal-Bench.

**Plug-and-play.** Because the action agent is unmodified, the memory agent slots into any existing agent harness — its output is just an extra text block in the action agent's prompt.

## Why it matters

- **Names a real failure mode.** Behavioral state decay is a phenomenon every long-horizon harness has hit; giving it a name and a benchmark for it is half the value.
- **Selective intervention is the load-bearing choice.** Ablations show it beats: passive bank exposure (bank always visible), always-on injection (reminder every step), advisor-only guidance (memory agent only comments, doesn't act), and general retrieval. The learned "silence" decisions are what preserves the action agent's context budget.
- **Concrete numbers.** +8.3 pp pass@1 on Terminal-Bench 2.0 and +6.8 pp on τ²-Bench across both weak and strong action agents — the effect is not concentrated at one capability level.
- **Points at open-weight memory policies.** Training a memory policy is a smaller task than training a frontier action agent, and the paper shows Qwen3.5-27B is enough. This makes the recipe reproducible outside labs with GPU-heavy action-agent RL.

## Gotchas & tricks

- **The action agent still sees a bigger prompt.** The memory bank is structured but non-trivial; injection frequency and reminder length both compete for context budget. The learned policy is what makes this feasible — tune it, don't hard-code injection.
- **Memory agent latency matters.** Running a 27B model alongside the action agent adds latency and cost. Batching updates and asynchronous updates are practical necessities in a real deployment.
- **Reminder wording is not the same as bank contents.** A well-structured bank still has to be *rendered* as a reminder the action agent will attend to. The paper trains this end-to-end; hand-designed reminder formatters likely underperform.
- **Transfer is partial.** The policy trained on SETA transfers to Terminal-Bench but not perfectly — domain shift matters. Retraining on target-domain trajectories should help.

## Sources

- Paper: *Remember When It Matters: Proactive Memory Agent for Long-Horizon Agents* — Zhang, Zhou, Wang, Peng, Li, Fan, Zhao (Meta AI), 2026 — [arXiv 2607.08716](https://arxiv.org/abs/2607.08716).
- Related evals: Terminal-Bench 2.0 · τ²-Bench.
- Related: [../post-training/grpo.md](./../post-training/grpo.md) — GRPO used for the memory policy.
