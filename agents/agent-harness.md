# Agent Harness
*Depth — the code around the model that constructs prompts, manages state, invokes tools, and coordinates execution.*

**TL;DR:** An agent's capability is a joint product of its foundation model and its *harness* — everything wrapping the model. The harness owns prompt construction, session/state management, tool dispatch, memory, retry/error policy, and the control loop that ties them together. Frontier agents share very similar base models but differ enormously in harness quality, which is why harnesses have become a moat and a bottleneck: they're large, tightly coupled, and behaviorally distributed across files, so localizing which code implements a target behavior is the central pain when they need to change.

**Prereqs:** none (folder root)
**Related:** [harness-handbook](harness-handbook.md), [on-device-agents](on-device-agents.md), [failure-attribution](failure-attribution.md), [../evaluation/agent-evaluation-infra.md](../evaluation/agent-evaluation-infra.md)

---

## What it is

The **harness** is the software that turns a stateless LLM into a stateful, tool-using agent. It contains at minimum:

- **Prompt construction** — assembling system prompt, tool schemas, memory, and the running conversation into each request.
- **Session / state management** — conversation history, scratchpad, per-task variables, run identifiers.
- **Tool invocation** — parsing tool calls from model output, dispatching to the underlying executor (shell, browser, MCP server, custom function), validating arguments, formatting results back into context.
- **Control loop** — deciding when to keep looping, when to stop, when to escalate, when to compact history.
- **Error / retry policy** — what happens on tool failure, model refusal, malformed output, budget exhaustion.
- **Observability hooks** — logs, traces, spans that make the run debuggable.

The model does not decide any of this on its own; the harness decides, and its choices shape what the agent can and can't do at least as much as the model does.

## How it works

A typical harness step:

1. Harness builds a prompt from `(system prompt, tool schemas, memory, history, latest observation)`.
2. Model returns a message that is either a natural-language reply, a tool call, or both.
3. Harness parses the message: if a tool call, validate the schema and dispatch to the executor; if a reply, decide whether to terminate.
4. Executor produces a structured result (stdout, JSON, error). Harness formats it back into the next observation.
5. Harness updates memory / scratchpad / task state and loops.

Harness design choices with outsized impact:

- **Prompt-template layout** — the ordering of tools, memory, and system rules; whether tool schemas are inlined or referenced.
- **History compaction policy** — sliding window, summarization, hierarchical (recent verbatim + summarized past), or full-context.
- **Tool boundary** — whether tools are opaque function calls, structured schemas, or fully-typed with pre- and post-conditions (see [on-device-agents](on-device-agents.md) for the "device as tool" version).
- **Control-loop stop conditions** — max steps, budget, model-signaled done, verifier-confirmed done.
- **Reflection / retry** — insert critique or replan turns after failures.

## Why it matters

- **The harness is the moat.** Two agents built on the same base model but different harnesses can differ by tens of points on the same benchmark. This is not folklore — it is what AgentCompass surfaces by making the harness axis independently swappable.
- **Harnesses evolve constantly.** Models, APIs, tools, and requirements change; the harness must too. Modification requests describe *what the system should do*; the repo is organized by files and modules. Bridging that gap — behavior localization — is expensive without dedicated tooling (see [harness-handbook](harness-handbook.md)).
- **Harness bugs look like model bugs.** Bad tool-schema formatting, wrong compaction, leaky control loop — all produce failure signatures easily blamed on "the model is dumb". Investing in harness diagnostics (see [failure-attribution](failure-attribution.md)) recovers the misattribution.

## Gotchas & tricks

- **Tool-schema drift.** When the tool executor changes but the schema in the prompt doesn't, the model calls the old shape and everything looks like model failure. Version the schema with the executor.
- **Silent history truncation.** Compaction that drops key observations without telling the model is a stealth capability loss. Prefer summarization that preserves references over raw truncation.
- **Non-deterministic loops.** A harness with no hard step budget and no verifier-based termination can loop forever on hard prompts. Always cap steps *and* wall-clock.
- **Cross-file behavior.** A single user-visible behavior often spans planner, executor, and formatter. Naive grep won't find it — a behavior-centric index (Harness Handbook style) will.
- **Debug logs are the harness's black box.** Structured trace logs of (prompt, tool call, tool result, next prompt) are the only way to reproduce agentic failures. Add them before you need them.

## Sources

- Paper: *Harness Handbook: Making Evolving Agent Harnesses Readable, Navigable, and Editable* — Wang et al., 2026 — [arXiv 2607.13285](https://arxiv.org/abs/2607.13285).
- Paper: *PalmClaw: A Native On-Device Agent Framework for Mobile Phones* — Cai et al., 2026 — [arXiv 2607.13027](https://arxiv.org/abs/2607.13027).
- Paper: *AgentCompass: A Unified Evaluation Infrastructure for Agent Capabilities* — Ding et al., 2026 — [arXiv 2607.13705](https://arxiv.org/abs/2607.13705). Formalizes the Benchmark / Harness / Environment split.
