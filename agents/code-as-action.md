# Code-as-Action
*Depth — using executable code as the agent's primary action interface instead of structured tool calls.*

**TL;DR:** Tool-augmented agents typically invoke tools through a structured schema (JSON function calls). For tasks that need composition over multiple primitives — spatial reasoning, data analysis, scientific computation — schemas are a bottleneck: each tool call is a discrete step, you can't easily chain intermediate values, and the schema fixes what's composable. **Code-as-action** replaces the schema with a persistent Python kernel: the agent writes code cells, observes intermediate results, and composes tools freely. SpatialClaw (2026) showed +11.2 points over schema-based agents on 20 spatial-reasoning benchmarks, training-free.

**Prereqs:** *(none)*
**Related:** [README.md](README.md) · [agent-environment-engineering.md](agent-environment-engineering.md)

---

## What it is

An agent action interface where every "action" is a snippet of code executed in a persistent stateful kernel. The kernel holds:
- Imported libraries (numerical primitives, perception tool wrappers).
- Variables from prior steps (intermediate results stay live across turns).
- An execution log (so the agent sees print/return values).

Each turn the agent reads the previous output, writes a new code cell, and the kernel executes it. The cycle continues until the agent emits a final answer.

---

## How it works

### The kernel

A long-lived Python process exposed to the agent. Perception tools (e.g. depth estimator, object detector) are exposed as Python functions; numerical primitives (numpy, math) are available natively. State carries across cells — the agent can compute `depth_map = run_depth(image)` in one cell and reference `depth_map` in the next.

### The agent loop

1. Agent receives task + kernel state summary.
2. Agent writes code (one or more statements).
3. Kernel executes; agent observes stdout, return values, errors.
4. Agent decides: write more code, or emit final answer.

### Why it beats schema-based tool calling

- **Composition is free.** Numerical operations on tool outputs don't need to go through the LLM; they happen in the kernel.
- **Intermediate values are addressable.** The agent can store, reference, and iterate over results rather than re-quoting them in JSON.
- **Errors are visible.** A Python stack trace is a richer error signal than a schema validation failure.
- **No schema lock-in.** Adding a tool means importing a function, not editing a system prompt and re-training.

---

## Why it matters

- **Spatial reasoning gained +11.2 points** training-free over schema-based spatial agents — a large gain in a domain where perception + arithmetic composition is the bottleneck.
- **Generalizes beyond spatial.** Any task that needs perception → arithmetic → control-flow loops benefits: physics, data analysis, scientific workflows.
- **Aligns with how humans use computers.** REPL-style problem-solving is what data scientists and engineers do; agents inherit the same affordances.

---

## Gotchas & tricks

- **Sandboxing is non-negotiable.** A persistent Python kernel is a security surface — pin the import list, sandbox filesystem/network access.
- **Long-running cells need timeouts.** Agents will write `while True:` loops; timeouts and step caps are mandatory.
- **The agent must see kernel state.** A summary of live variables (names + types + shapes) on each turn helps the LLM keep its mental model accurate.
- **Schema-based fallback for some tools.** Tools with side effects (sending emails, paying for things) are better gated behind structured calls so the kernel can't trigger them implicitly.
- **Trajectory length grows.** Code-as-action episodes are longer than schema-based ones; budget context accordingly.

---

## Sources

- Paper: *SpatialClaw: Rethinking Action Interface for Agentic Spatial Reasoning* — Hachiuma et al., NVIDIA, 2026 — [arXiv:2606.13673](https://arxiv.org/abs/2606.13673) — the most explicit head-to-head between code-as-action and schema-based tool calling on spatial reasoning.
