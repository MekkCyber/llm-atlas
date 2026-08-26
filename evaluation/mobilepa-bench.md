# MobilePA-Bench
*Depth — stateful, sandbox-based benchmark for mobile planner agents.*

**TL;DR:** Existing mobile-agent benchmarks either test surface-level UI manipulation or rely on static offline API matching, both of which miss real runtime failure. MobilePA-Bench runs on an **executable sandbox** with live application databases and structured feedback, spanning 13 functional domains and 212 realistic mobile tools. It scores planning agents on sub-agent collaboration, memory use, and skill (macro) use in addition to basic tool calling.

**Prereqs:** [../agents/rlm-harness](../agents/rlm-harness.md)
**Related:** [ifeval](ifeval.md), [longrca-bench](longrca-bench.md)

---

## What it is

An interactive, stateful benchmark environment for planner agents in a simulated mobile OS. Unlike static function-call benchmarks, MobilePA-Bench's sandbox maintains live app state — contacts, calendars, files, messages — and the agent's actions actually mutate that state, returning structured feedback each step.

## How it works

- **Sandbox layer:** 212 realistic mobile tools across 13 domains (messaging, calendar, maps, photos, notes, health, e-commerce, …) backed by an executable simulation. Every tool returns typed, structured feedback; a permission model enforces realistic access boundaries.
- **Task instances:** long-horizon prompts spanning several tools and requiring memory. Evidence-based verification checks final DB state, not just tool-call transcripts.
- **Three "advanced" scoring axes** beyond basic tool use:
  - **Sub-agent collaboration** — the planner decomposes the task and delegates to specialized sub-agents.
  - **Memory usage** — recalling stored memories, user profile, past preferences to resolve implicit requests.
  - **Skill usage** — invoking pre-packaged composite skills instead of re-planning every step.
- **Failure modes** injected on purpose: strict tool ordering, permission limits, unexpected runtime errors. The benchmark reports both single-shot and repeated-attempt scores.

## Why it matters

Frontier LLMs remain unreliable under strict tool ordering, permission limits, and runtime errors — MobilePA-Bench surfaces this in a way GUI or static-API benchmarks cannot. It also doubles as a *training environment* for agent RL: the sandbox is fast enough for rollouts.

## Gotchas & tricks

- Evidence-based verification (final state check) is more robust than transcript matching, but requires the sandbox's DB to be reset per task instance. Slow if reset is naive.
- Skill-usage scoring depends on the agent knowing the skill library — provide it in-context to test recognition, not memorization.
- The stateful sandbox is what makes online RL practical here; treat evaluation runs as usable trajectory data by default.

## Sources

- Paper: *MobilePA-Bench: Benchmarking Mobile Planner Agents on Complex Real-World Tasks* — Zhu et al., 2026 (Alibaba Token Foundry / MAI Team) — [arXiv:2608.23035](https://arxiv.org/abs/2608.23035)
