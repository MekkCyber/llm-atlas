# Recursive Synthetic Terminal Tasks (RST)

*Depth — a self-bootstrapping loop that produces long-horizon agent-training tasks at ~$0.05 each, up to arbitrarily hard difficulty.*

**TL;DR:** Long-horizon terminal-agent training data costs hundreds to thousands of dollars per task because instruction, environment, reference solution, and verifier must stay mutually consistent. RST (Shi et al., 2026) bootstraps: seed tasks are *extended* (reference solution grows), the verifier and instruction are *realigned* to the new workflow, the result is *validated* in a fresh sandbox, and accepted tasks become seeds for the next round. Fifteen rounds yielded **37,484 tasks at ~$0.05 each**, with DeepSeek-V4-Pro pass@4 dropping from 90% at round 1 to **2.5%** at round 15. SFT on RST trajectories lifts Qwen3.5-27B by up to 10 pts on Terminal-Bench 2 / Hard / Long-Horizon.

**Prereqs:** [_data-curation.md](_data-curation.md)
**Related:** [decontamination.md](decontamination.md) · [quality-filtering.md](quality-filtering.md) · [../post-training/rejection-sampling.md](../post-training/rejection-sampling.md) · [../agents/agent-harness.md](../agents/agent-harness.md)

---

## What it is

A recursive data-generation pipeline specialized for **terminal-agent** tasks — tasks where the agent must issue shell commands in a sandboxed environment to accomplish a goal, and correctness is judged by an executable verifier. The four brittle artifacts that must stay coherent are:

- **Instruction** — what the user asks.
- **Environment** — the initial filesystem, packages, secrets.
- **Reference solution** — a script that accomplishes the goal.
- **Verifier** — a script that checks post-conditions.

Human authoring is expensive because a single change to any of the four cascades to the other three. RST automates the cascade.

## How it works

Each round takes accepted tasks from the previous round as seeds and produces a new (harder) round:

1. **Extend the reference solution.** An LLM adds a new sub-goal (a new file to produce, a new service to configure) to an existing task, updating the reference script.
2. **Realign verifier and instruction.** Update the verifier to check the new post-conditions and rewrite the instruction so it asks for the extended goal.
3. **Sandbox validation.** Spin up a fresh container matching the environment spec, run the new reference solution, run the new verifier — both must succeed. Reject on failure.
4. **Seed the next round.** Accepted tasks join the seed pool for round `k+1`.

Difficulty accumulates monotonically: round 15 tasks are strictly longer / more constrained than round 1 tasks.

## Why it matters

- **Removes the training-data bottleneck for terminal agents.** $0.05/task at arbitrary difficulty makes the compute-vs-data tradeoff for agent training suddenly one-sided.
- **Difficulty scales past the frontier.** DeepSeek-V4-Pro pass@4 collapses 90% → 2.5% across rounds, and validation rates stay stable — recursion has no ceiling in the tested regime.
- **Trainable-plus-verifiable.** Every generated task ships with its own executable verifier, so downstream RL / rejection-sampling pipelines don't need a separate reward model.
- **Template for other domains.** The same extend/realign/sandbox/seed pattern applies to any domain with executable verifiers — coding, browser automation, tool-use, structured data pipelines.

## Gotchas & tricks

- **Sandbox validation is the entire safety net.** Any bug that lets a task pass validation without actually being coherent will propagate and amplify through recursive rounds. Isolation, deterministic containers, and full clean-slate spin-up per task matter.
- **Difficulty growth is not curriculum.** RST produces monotonically-harder tasks, not a balanced curriculum. Training pipelines that want easy examples must sample across rounds explicitly.
- **Diversity risk.** Extending seeds tends to converge on similar structures ("add another file to the config") over many rounds. The paper reports stable yield through 15 rounds; longer horizons may drift into a narrow task family.
- **Verifier-first design.** Tasks without cheap executable verifiers (open-ended writing, subjective quality) don't fit this loop — the sandbox-validation step is what keeps the cascade honest.

## Sources

- Paper: *Recursive Synthesis for Long-Horizon Terminal Tasks* — Shi, Li, Wang, Li, Huang, Yang, Ke, Liu, Mi, Liang, 2026 — [arXiv 2608.05466](https://arxiv.org/abs/2608.05466). Tencent HY LLM Frontier + multi-university.
