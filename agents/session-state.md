# Session-Centered Runtime State
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Make agent runtime state a **first-class `Session` object** rather than letting it hide in implicit side channels (env vars, scratch files, framework-local thread context). The Session carries lineage, tool evidence, sandbox metadata, and usage records, so every tool call, sub-agent, and sandbox write has an auditable parent and can be replayed deterministically. Introduced in OpenRath (2026) as a remedy for the "hidden runtime state" problem in current agent frameworks.

**Prereqs:** [README.md](README.md)
**Related:** [../systems/README.md](../systems/README.md)

---

## What it is

Most agent frameworks (LangGraph, CrewAI, vanilla LLM scaffolds) pass *some* state explicitly — chat history, scratchpad — and let the rest live in:

- Process env vars and globals.
- Side files in a sandbox.
- Framework-internal caches that don't survive a restart.
- Tool clients with their own thread-local connection state.

When the agent run reproduces poorly, attribution gets murky, or replay is impossible, this scattered state is usually why. The Session abstraction collapses all of it into one explicit, serializable object that every tool call reads from and writes to.

## How it works

A `Session` is a structured record with at minimum:

- **Lineage** — the directed graph of which agent / which call produced which artifact.
- **Tool evidence** — every tool input and output, hashed and timestamped.
- **Sandbox metadata** — container ids, file system snapshots, network policy.
- **Usage records** — tokens, wall time, cost per sub-step.

Tool wrappers and sub-agent spawners take a `Session` argument and *must* update it. Replay reconstructs an execution by walking the lineage tree, re-invoking tools (or replaying cached results), and verifying that the resulting Session matches the original byte-for-byte where deterministic.

## Why it matters

- **Reproducibility:** a Session is the unit of replay. Two runs that produce different Sessions for the same prompt are a real divergence, not a debugging mystery.
- **Audit:** every artifact in the final answer points back to a specific tool call, sandbox, and timestamp.
- **Cost attribution:** per-sub-agent and per-tool cost falls out of the usage records for free.
- **Composition:** multi-agent frameworks can plug into a shared session contract instead of inventing their own thread-local context.

## Gotchas & tricks

- Session bloat is the obvious failure mode — long-running agents with deep tool chains can generate sessions of millions of entries. The reference design hashes evidence and lazily reifies it on replay.
- Wrapping every tool call in Session updates adds latency; the design pushes the update off the critical path where possible (write-after-return semantics).
- The Session must be *append-only* during the agent's run for lineage to be meaningful. Mutating earlier entries on the fly defeats audit.

## Sources

- Paper: *OpenRath: Session-Centered Runtime State for Agent Systems* — Wen, Wang, Xu, Tsinghua, 2026 — [arXiv:2606.19409](https://arxiv.org/abs/2606.19409).
