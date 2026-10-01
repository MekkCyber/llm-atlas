# Agent Harness
*Depth — the engineered scaffolding around a base LLM that turns it into a working agent.*

**TL;DR:** The *harness* is everything around the model that isn't the model: the prompt scaffolding, tool schemas, orchestrator, scratchpad, memory, retry logic, verifier calls, and asset handling. Modern products (Claude Code, Devin, ChatGPT Agent) are 10% model and 90% harness. "Harness of harnesses" systems (Raven, 2026) treat the harness itself as a first-class, synthesized object: generate specialized sub-harnesses per domain and compose them under a Host Agent.

**Prereqs:** [README.md](./README.md)
**Related:** [context-language-models.md](./context-language-models.md), [omni-io-skills.md](./omni-io-skills.md), [asynchronous-agents.md](./asynchronous-agents.md), [hybrid-cua.md](./hybrid-cua.md)

---

## What it is

Everything that converts "a model that outputs tokens" into "an agent that gets work done": tool schemas, system prompts, planners, scratchpads, action-trace memory, verifiers, retry/fallback logic, context managers, and interface adapters (GUI/CLI/APIs). The harness is where most *actual* agentic capability lives — same base model + a better harness often beats a bigger model + a worse one.

A **single-domain harness** is hand-written for one workflow (coding, browsing, deep research). A **harness-of-harnesses** (Raven, EverMind AI, 2026) auto-constructs specialized harnesses per `(model, domain)` and composes them via a Host Agent + experience archive.

## How it works

Three recurring components:

1. **Procedure library** — serialized rollouts that worked once and can be invoked again (Raven's *Skill Forge*, Omni-IO's hierarchical Skills).
2. **State store** — a persistent record of past tasks, intermediate assets, and dependencies (Raven's *EverOS*, Omni-IO's *Asset Registry*).
3. **Orchestrator** — decomposes goals, matches sub-tasks to specialists, coordinates execution dependencies, integrates results (Raven's *Host Agent*, Omni-IO's *Declare Execution Graphs*).

An auto-synthesized harness runs a feedback loop: on each new task, propose a sub-harness (prompts + tools + procedure template), execute, measure, keep what worked.

## Why it matters

Harness engineering is currently the dominant source of agent capability variation between labs — and the dominant source of fragility (brittle prompts, broken tool schemas, invisible context overflow). Making the harness itself trained, synthesized, or at least first-class is how the field escapes "100 heroic prompt engineers per lab." Theory from Raven states sufficient conditions under which composing `(model, harness)` units expands reliable task coverage beyond any single harness at the same compute budget.

## Gotchas & tricks

- **The model/harness boundary matters.** Features baked into the harness are easy to ship but hard to transfer; features baked into the model (via training) transfer but are expensive to ship. The 2026 trend ([context-language-models.md](./context-language-models.md), [hybrid-cua.md](./hybrid-cua.md)) is to migrate specific harness responsibilities *into* the model via RL.
- **Experience archives decay without curation.** A Skill Forge full of partially-successful procedures teaches the agent to copy mistakes. Weight by downstream task success, not raw frequency.
- **Composition isn't free.** A Host Agent routing between specialists adds its own latency and failure modes. The theoretical gain assumes the router doesn't become the bottleneck.
- **The "right" harness depends on model size.** Small models need tighter scaffolding; large models can absorb more of the harness as learned behavior. Harness complexity should shrink as capability grows.

## Sources

- Paper: *Raven: The Harness of Harnesses for Composable Agentic Intelligence* — EverMind AI, 2026 — [arXiv:2609.33439](https://arxiv.org/abs/2609.33439).
- Paper: *Omni-IO Skills: Harnessing Your Agent Omni-Native* — Li et al., NUS, 2026 — [arXiv:2609.31847](https://arxiv.org/abs/2609.31847).
