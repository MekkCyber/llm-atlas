# Agent Memory (Evolution-Aware)
*Depth — a memory paradigm where the agent records and queries environment changes as structured update history.*

**TL;DR:** Long-running agents need to track state that changes over time — files edited, preferences updated, schemas migrated. Flat chat-buffer memory loses the temporal structure: the agent sees "the state at turn N" but can't easily ask "what changed between turn N and N+5". Evolution-aware memory stores each environment change as a **patch** with provenance, and exposes a query interface so the agent can reason about *what changed*, not just *what is*. Introduced in EvoMem (2026) and benchmarked on EvoArena.

**Prereqs:** *(none)*
**Related:** [README.md](README.md) · [../evaluation/evoarena.md](../evaluation/evoarena.md)

---

## What it is

A memory data structure that an LLM agent reads from and writes to over a multi-turn interaction with an environment that itself evolves. Two design choices distinguish it from chat-buffer memory:

1. **Patches, not snapshots.** Each update is recorded as `(before, after, cause, turn)` rather than overwriting state in place.
2. **Queryable history.** The agent can ask the memory for diffs between arbitrary time points, not just retrieve the current value.

Used in: terminal agents (filesystem evolves), software agents (codebase migrates), personal assistants (user preferences shift).

---

## How it works

### Write path

When the environment reports a state change — file edited, config updated, user preference revealed — the agent emits a structured patch:

```
{ "key": "...", "before": ..., "after": ..., "evidence": "...", "turn": N }
```

Patches are append-only. Latest-value lookups walk the history backward.

### Read path

Three query types:
- **Current state**: latest patch for a key.
- **Historical state**: walk back to the turn-bounded version.
- **Diff between turns**: enumerate patches in `(t1, t2]`.

The agent's prompt template surfaces all three so the LLM can choose which to use.

### Integration with reasoning

The agent's planner sees the memory schema, decides which queries to issue (typically 1–3 per turn), and folds the results into its chain-of-thought. EvoMem reports that *evidence capture* — whether the patch contains enough detail to reconstruct the change — is the bottleneck, not retrieval.

---

## Why it matters

- **Dynamic environments are the default in production**, not the exception. Static-environment benchmarks overstate agent capability.
- **Reasoning over change is qualitatively different from reasoning over state.** "What broke after the migration" is a diff question; a flat memory answers it only by accident.
- **Compositional tasks need it.** EvoArena's chain-level accuracy (consecutive subtasks that depend on each other's evolution) improves +3.7% with EvoMem — the kind of gain that compounds across long horizons.

---

## Gotchas & tricks

- **Patch fidelity is the limiting factor.** If `before`/`after` are summaries instead of literal values, downstream diff queries lose precision. Store the literal where the budget allows.
- **History compaction is unsolved.** Append-only patches grow linearly. Production deployments need a compaction policy (e.g. squash patches older than N turns into checkpoints) without losing diff queries that cross the boundary.
- **Doesn't replace tool memory.** Tool outputs still need their own short-term cache; evolution memory is for environment *state*, not for "what did `ls` return".
- **Evaluation is hard without an evolving benchmark.** Standard agent benchmarks freeze the world; EvoArena is the first to require evolution-tracking explicitly.

---

## Sources

- Paper: *Tracking Memory Evolution for Robust LLM Agents in Dynamic Environments* — Xu et al., 2026 — [arXiv:2606.13681](https://arxiv.org/abs/2606.13681) — introduces EvoMem and EvoArena.
