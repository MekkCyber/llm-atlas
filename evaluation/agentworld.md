# AgentWorld
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A benchmark of 100 human-annotated tasks (plus 100 augmented variants) that evaluates *long-horizon multi-agent LLM collaboration* in an MMORPG-style sandbox. Tasks span 50+ interaction rounds and require 3–20 agents with **asymmetric roles and abilities** to coordinate through communication, joint planning, and resource sharing under a black-box setting (each agent sees only its own state). Introduces **Causal Collaboration Effectiveness (CCE)**, a graph-based metric that measures what fraction of team effort actually contributed to the outcome.

**Prereqs:** *(none)*
**Related:** [game-arena](./game-arena.md)

---

## What it is

Most multi-agent benchmarks either put agents in **competitive** settings, use **short** interactions (<20 steps), or report per-agent scores summed — none of which isolates *collaboration* as a capability. AgentWorld does three things differently:

1. **Long horizon.** Tasks require 50+ turns of interaction to complete.
2. **Asymmetric roles.** Agents receive different tools, information, and objectives; a task can only be solved by joint action across the team.
3. **Black-box coordination.** No agent sees another's internal state; communication is via natural-language messages the models emit.

Domains span exploration, resource gathering, combat, crafting, trading, and social negotiation in a text-rendered MMORPG environment.

## How it works

- **Tasks.** 100 human-annotated task templates; 100 augmented variants for robustness testing. Each task defines a success condition and a role assignment.
- **Teams.** 3–20 agents per task, with heterogeneous prompts and tool access.
- **Rounds.** Each round each agent observes, reasons, sends messages, and acts. Rounds continue until success, failure, or a step-count cap.
- **Metrics.**
  - **Task success rate** — the standard binary outcome.
  - **Causal Collaboration Effectiveness (CCE)** — build an action-graph across the team where edge $u \to v$ means agent $v$'s action was causally enabled by agent $u$'s prior message/action. CCE reports the fraction of team-effort nodes on paths that terminate in the success condition. Isolates *coordination* from *individual capability*: a team that solves the task by one agent doing all the work has low CCE even if success = 1.
- **Model set (initial report).** Gemini 3 Flash, Claude Haiku 4.5, GPT-5 Mini, DeepSeek R1-70B.

## Why it matters

- **First benchmark to cleanly isolate coordination as a capability.** Competitive game arenas measure strategic play; AgentWorld measures cooperative reasoning under asymmetric information.
- **CCE is portable.** The metric is defined on any multi-agent trajectory with action provenance; it generalizes beyond this sandbox.
- **Shows current models are far from ceiling.** Best model achieved only **52.0% task success**. Failure modes named in the paper: communication breakdowns, role confusion, inability to maintain shared plans across rounds.

## Gotchas & tricks

- **Prompt-harness sensitivity.** Small changes in the coordination prompt swing scores meaningfully; the paper standardizes a system prompt template per role.
- **Language sink.** Models tend to spend increasing tokens on natural-language coordination, exhausting per-round budgets before acting.
- **CCE requires action-provenance labels.** Not trivially extractable from a message stream; the paper's pipeline uses a rule-based tagger on the environment's action log.
- **Not diagnostic of individual capability.** A model that scores low here may be an excellent solo agent; use paired with individual benchmarks.

## Sources

- Paper: *AgentWorld: Benchmarking Long-Horizon Collaboration of Multi-agent LLMs* — Shu, Zhang, Cho, Yang et al. — Penn State + Penn, 2026 — https://arxiv.org/abs/2609.31590
- Code and data: open-sourced with the paper.
