# Self-State Attacks on Self-Hosted AI Agents

*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Self-hosted AI agents read and write their own memory and configuration files during normal operation. An attacker who can trigger legitimate OS syscalls — e.g., via a compromised prompt or a piggy-backed tool call — can corrupt those files to persistently subvert the agent. The paper introduces this class as "self-state attacks," decomposes it along four axes (Target, Mechanism, Granularity, Temporal), and evaluates OS-layer defenses against a 23-cell attack matrix.

**Prereqs:** [_attacks.md](./_attacks.md)
**Related:** [sleeper-agents.md](./sleeper-agents.md) · [safety-case.md](./safety-case.md) · [cot-monitoring.md](./cot-monitoring.md)

---

## What it is

An attack class distinct from prompt injection and jailbreaks: the compromise vector is a *legitimate* syscall (open/write/rename/truncate) against the agent's own state files (system prompt, memory store, config, plugin/tool registry). The threat model requires that the attacker has already achieved some code-execution or prompt-injection foothold; self-state attacks describe how that foothold is *persisted* and *amplified*.

## How it works

The paper decomposes the space along four axes:

| Axis | Values |
| --- | --- |
| **Target** | Instruction file, configuration, memory store |
| **Mechanism** | Legitimate syscall (write / rename / truncate / append) |
| **Granularity** | Whole-file rewrite → surgical byte-level edit |
| **Temporal** | One-shot injection vs. persistent recurring modification |

This yields a 23-cell matrix of concrete attack primitives, instantiated as 43 real file-mutation operations, injected into live traces from a representative self-hosted agent.

Defenses evaluated in layers:

1. **Access-control prevention** on the instruction and configuration layers (deny writes from the agent's own uid to files that shouldn't be self-modified at runtime).
2. **Workload-conditioned detection** on the memory layer (a syscall pattern that would be anomalous *given the current workload profile* triggers a flag).
3. **Periodic backup** for recovery (rollback to a known-good state at fixed intervals).

## Why it matters

The industry is shipping "AI agents that run on your machine" — Devin, Manus, Claude Desktop's MCP servers, Codex-CLI-style tools. Nobody had published an OS-layer threat model at this granularity. The paper's punchline is that a layered defense stack neutralizes most attack cells, *but a residual set is structurally indistinguishable from normal behaviour at the OS layer* — catching those requires agent-internal monitoring (the agent knows it didn't mean to write that byte). This is likely to become a reference decomposition for red-teaming agent installers.

## Gotchas & tricks

- Access control breaks agents that legitimately self-modify (e.g., memory updates). Distinguishing legitimate self-writes requires either capability-token schemes or agent-side co-signing.
- Workload-conditioned detection needs a stable notion of the agent's current task — a scheming agent could deliberately obfuscate its workload to hide.
- Periodic backup limits blast radius but doesn't prevent *the current session* from doing harm; useful for state integrity, not for action safety.
- The residual "structurally indistinguishable" attacks are exactly where agent-internal monitors (chain-of-thought monitoring, action auditing) become necessary — this paper motivates the composition.

## Sources

- Paper: *Self-State Attacks on Self-Hosted AI Agents: How Far Can OS Defenses Go?* — Yimeng Chen, Nathanaël Denis, Roberto Di Pietro, Jürgen Schmidhuber (KAUST, IDSIA-USI/SUPSI), 2026 — [arXiv:2607.17986](https://arxiv.org/abs/2607.17986) · [HF](https://huggingface.co/papers/2607.17986)
