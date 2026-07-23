# Typed-mutation agents
*Depth — LLM code-agents that emit structured platform edits rather than free-form scripts.*

**TL;DR:** A pattern for LLM agents that build workflows: instead of writing a script that the platform then re-parses, the agent applies typed, incremental mutations against a live platform state exposed via MCP. Each step produces a persistent, editable artifact. Named and demonstrated by DataFlow-Harness for data-processing DAGs, but the pattern generalizes to any agent whose deliverable is a platform-native structure.

**Prereqs:** *(none in the current graph)*
**Related:** [../agents/README.md](../agents/README.md)

---

## What it is

Coding agents typically produce scripts. The user gets a `.py` file that ran once. If they want to change one step, they either edit code or ask the agent again from scratch. The agent's output is code, not a first-class platform object. DataFlow-Harness calls this the **NL2Pipeline gap**: the gap between natural-language authoring and platform-native, editable artifacts.

Typed-mutation agents close that gap by making the agent's action space *the platform's own edit operations* — typed calls that mutate a live DAG, canvas, form, or model.

## How it works

Three ingredients (as instantiated in DataFlow-Harness):

1. **Typed action surface.** The platform exposes its state and legal edits as typed operations (add-node, connect, retype, …) rather than "run this code." The agent's calls are validated against the type system before any effect.
2. **MCP layer over live platform state.** Model Context Protocol carries the operator registry *and* the current DAG state into every agent turn, so the agent's next mutation is grounded in what already exists rather than in a re-imagined script.
3. **Synchronized UI.** A visual editor (here, DataFlow-WebUI) renders the same DAG the agent is mutating, so a human can take over at any step. Conversation and canvas stay in sync.

Optionally: a bank of procedural task patterns (DataFlow-Skills) the agent can call as macros for common multi-mutation sequences.

## Why it matters

- **Artifacts are persistent and editable.** Every step of the agent leaves behind a platform object, not a run-once script.
- **Human-in-the-loop is cheap.** The agent and the user share one representation.
- **Grounding beats hallucination.** Because MCP exposes the *live* platform state, the agent can't invent operators or reference nodes that don't exist.

The paper reports 93.3% end-to-end pass rate on a 12-task data-engineering benchmark — but the more durable claim is the interaction pattern.

## Gotchas & tricks

- Typed-mutation only works when the platform's edit surface is expressive enough that the agent doesn't fall back to shelling out to code.
- MCP registries need to be curated — a huge, unfiltered surface confuses the agent as much as a missing one.
- Skill libraries need versioning: patterns encoded as multi-step macros go stale when the platform's operators change.

## Sources

- Paper: *DataFlow-Harness: A Grounded Code-Agent Platform for Constructing Editable LLM Data Pipelines* — Wong et al., 2026 — [arXiv:2607.16617](https://arxiv.org/abs/2607.16617)
