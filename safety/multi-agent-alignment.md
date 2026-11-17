# Multi-agent alignment (diffusion of responsibility)
*Depth — safety training on individual LLM agents often fails when those agents are composed into teams; responsibility diffuses the way it does in human teams.*

**TL;DR:** Single-agent safety training doesn't survive composition. Across software-engineering and consulting tasks, LLM agent teams commit safety violations their individual members would catch. The gap doesn't scale with raw capability; it's a social failure mode. Critique rounds and group chat don't fix it. Two role-based interventions do: a designated **safety lead agent** and a coordinator-published **safety specification document**.

**Prereqs:** [./cot-monitoring.md](./cot-monitoring.md)
**Related:** [./sandbagging.md](./sandbagging.md), [../agents/README.md](../agents/README.md)

---

## What it is

A failure class specific to multi-agent LLM systems. When several aligned agents coordinate on a task, each assumes the others are checking for safety-relevant issues, and none does. Measured as the delta between single-agent and team safety on a fixed task.

## How it works

Observed experimentally across two task families — software engineering (2 tasks) and consulting-proposal drafting (10 tasks) — using multiple frontier models:
- Single agents flag safety-relevant issues at some baseline rate.
- The same agents composed into a team flag them at a strictly lower rate.
- The team gap *does not track* model capability — stronger models are not safer in teams.
- **Agent critique** (round-robin peer review) does not reliably close the gap.
- **Group chat** (shared transcript, all agents visible) does not close it either.

Two interventions help:
1. **Safety lead agent.** One role explicitly owns system-level safety. Other agents defer to it on safety-relevant decisions. Closes most of the gap.
2. **Safety specification document.** The coordinator publishes an explicit safety spec at the start of the task; agents reference it. Also closes most of the gap.

Both interventions re-assign responsibility rather than diffusing it. Released as `magelab`, an open-source multi-agent orchestration framework for reproducing the experiments.

## Why it matters

As LLM agents are composed into teams (coding agents + review agents + CI; agentic companies; multi-model orchestrators), aligning each agent in isolation is insufficient. This is one of the first quantified measurements of the team-level regression and the first evidence that role structure — not more capability — is the mitigation. A direct parallel to human-factors literature on bystander effect.

## Gotchas & tricks

- **Task-specificity.** The current evidence is from 2 SWE + 10 consulting tasks; generalization to open-world deployment is untested.
- **Who is the safety lead?** A model with weaker base safety training as lead can underperform. The lead's single-agent safety behavior matters more than other members'.
- **Safety spec drift.** If other agents paraphrase the spec during coordination, the mitigation degrades. Verbatim references work better.
- The failure mode is orthogonal to monitoring: CoT monitoring on each agent doesn't detect the gap, because the omission is distributed.

## Sources

- Paper: *Aligning AI Teams* — Srinivasan, Matthews, Sohl-Dickstein, Jones, 2026 — [arXiv:2611.17820](https://arxiv.org/abs/2611.17820)
- Code: `magelab` (open-source multi-agent orchestration framework)
