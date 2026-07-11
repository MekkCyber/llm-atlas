# Proactive Memory Agent
*Depth — a separate agent that decides *when* to inject a memory-grounded reminder into an action agent.*

**TL;DR:** In long-horizon agent tasks, decision-relevant state gets buried deep in the expanding context (or pushed out of it entirely) — a failure mode the paper names **behavioral state decay**. The proactive-memory-agent recipe runs an unmodified action agent alongside a *memory agent* that maintains a structured memory bank and, at each step, actively decides whether to inject a reminder or stay silent. Trained end-to-end with SFT + GRPO. Introduced by Meta AI (2026).

**Prereqs:** [../post-training/grpo.md](../post-training/grpo.md), [../post-training/rlvr.md](../post-training/rlvr.md)
**Related:** [../post-training/reasoning/long-cot-rl.md](../post-training/reasoning/long-cot-rl.md), [README.md](./README.md)

---

## What it is

A two-agent architecture for long-horizon tasks (Terminal-Bench 2.0, τ²-Bench, agentic coding, computer-use). The action agent runs unchanged. A separate memory agent observes the same trajectory, updates a structured memory bank, and — at each step — chooses one of two actions itself: *inject a reminder* (a synthesized message with retrieved state) or *stay silent*. The "when to inject" decision is a learned policy, trained with reinforcement learning; the reward is the downstream task success.

Distinguished from passive-retrieval memory (RAG-style over a memory bank) by the **active injection policy**: it can decide the current step needs no reminder, or that a specific fact from 40 steps ago must be surfaced right now.

## How it works

- **Structured memory bank.** The memory agent parses the recent trajectory into structured slots — open subgoals, environment facts, prior attempts, diagnoses, task-requirement facts. This gives a queryable representation, not a raw log.
- **Injection policy.** At each step, the memory agent decides `inject | silent`. If `inject`, it also chooses *what* to inject — a synthesized reminder grounded in the memory bank.
- **Training pipeline (SETA).** SFT + GRPO. SFT bootstraps the injection policy from trajectories where a stronger model chose to inject; GRPO refines it against downstream task success. Trained on Qwen3.5-27B as an open-weight memory head.
- **Plug-and-play.** The memory agent sits alongside a frontier action agent (Claude / GPT-5 class) without modifying it. Injection happens by appending the reminder as a message in the shared context.

## Why it matters

- **Names the failure mode.** "Behavioral state decay" — decision-relevant state present in context but no longer *influencing decisions* because it's buried — is common in long agent runs; treating it as a specific target has been missing from the memory literature.
- **Active vs passive memory.** Prior work treats memory as retrieval (RAG over a memory store). This paper argues retrieval is insufficient — you also need a policy that decides *when* to inject. Ablations back this up: passive bank exposure, always-on injection, advisor-only guidance, and generic RAG all lose to selective injection.
- **Concrete gains.** +8.3 pp pass@1 on Terminal-Bench 2.0, +6.8 pp on τ²-Bench, with gains on both weak and strong action agents.

## Gotchas & tricks

- **Silent is a valid choice.** The point of the policy is to say nothing most of the time — always-on injection *hurts* by drowning the action agent in reminders.
- **Structured slots beat raw log.** Retrieving from a structured bank of (subgoals, facts, attempts) is more precise than semantic search over a trajectory log.
- **Reward shaping.** Because task-level reward is sparse and delayed, credit assignment to injection steps is noisy; the paper uses GRPO's group-relative advantages to reduce variance.
- **Transfer is partial.** The open-weight Qwen3.5-27B memory head trained on SETA partially transfers to Terminal-Bench; do not expect drop-in transfer across agent benchmarks.
- **The memory agent adds tokens.** Every step now runs the memory agent too; when the injection policy is silent, the cost is just one small forward pass per step.

## Sources

- Paper: *Remember When It Matters: Proactive Memory Agent for Long-Horizon Agents* — Zhang et al., Meta AI, 2026 — https://arxiv.org/abs/2607.08716
- Related: *Terminal-Bench 2.0* — long-horizon terminal task benchmark used for evaluation.
- Related: *τ²-Bench* — multi-turn agent benchmark used for evaluation.
