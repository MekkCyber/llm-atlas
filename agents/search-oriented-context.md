# Search-Oriented Context Management (SOCM)
*Depth — externalizing the implicit state of an information-seeking agent into shared, persistent data structures.*

**TL;DR:** Multi-agent RAG systems fail by re-searching the same failed queries or losing track of what's been covered. **SOCM** replaces implicit "conversation history" state with four explicit, persistent, shared objects — a Frontier Task list, an Evidence Graph, a Coverage Map, and a Failure Memory — plus a middleware harness that intercepts tool calls to record grounded evidence and react to stalls. Introduced with SearchOS-V1.

**Prereqs:** [../post-training/grpo.md](../post-training/grpo.md)
**Related:** [../systems/partial-rollouts.md](../systems/partial-rollouts.md)

---

## What it is

Traditional agentic RAG stores the search state inside the LLM's chat history — a linear, opaque log. As the log grows, agents lose track of what they've tried, repeat failed searches, and get trapped in loops.

SOCM is a **state externalization pattern**: everything the agent needs to plan the next step lives in typed data structures outside the model, updated by tool-side effects rather than reconstructed from context. The pattern was introduced by SearchOS-V1 for open-domain information seeking, but the shape is general to multi-step tool-using agents.

## How it works

### The four state objects

1. **Frontier Task.** The next hypothesis or unresolved question to investigate. Popped by whichever sub-agent is free; pushed to when new subquestions surface.
2. **Evidence Graph.** A graph of `(entity, attribute, value, source)` tuples with citations. New evidence lands as a node; contradictions become explicit edges. This is the grounded output that final answers draw from.
3. **Coverage Map.** A checklist of what's been searched (which query variants, which subtopics, which entity slots). Prevents duplicate work and highlights unresolved gaps for the scheduler.
4. **Failure Memory.** A log of *search patterns that failed*, indexed by pattern rather than by exact query. Future agents avoid replaying them.

### The reformulation of the task

SearchOS reframes open-domain information seeking as **relational schema completion**: agents discover entities, populate attributes across linked tables, and anchor each value to source evidence. The Evidence Graph is a natural fit for schema completion; the Coverage Map tracks per-slot progress.

### The middleware harness

A **Search Tool Middleware Harness** wraps every model↔tool interaction:

- Records the returned evidence into the Evidence Graph with citations.
- Detects stalls (repeated queries, exhausted budgets, low-yield paths) and injects control signals.
- Provides a reusable **hierarchical skill system**: strategy skills (planning-level moves) and access skills (concrete tool sequences) that agents compose instead of writing raw queries.

### Pipeline-parallel scheduling

The scheduler runs sub-agents in a pipeline: as soon as one completes or stalls, its slot is refilled with a task targeting an unresolved coverage gap. This turns idle GPU time into new search progress and keeps the Frontier Task queue drained.

## Why it matters

- **Kills the loop-forever failure mode.** Failure Memory + Coverage Map make it structurally impossible to repeat known-bad search patterns.
- **Grounding is a side effect, not a target.** Because every evidence tuple is stored with its source at write time, the final answer can cite without extra work.
- **Scales to many sub-agents.** Externalized state is the natural coordination substrate; agents don't need to see each other's histories.
- **Beats single- and multi-agent baselines** across every metric on WideSearch and GISA.

## Gotchas & tricks

- **State schema is domain-specific.** The four objects are the pattern; the exact shape of the Evidence Graph and Coverage Map must be designed per task class.
- **Failure Memory needs pattern matching, not exact-query matching.** Slight query rewrites will bypass exact-string dedup; SearchOS indexes by search *pattern*.
- **Middleware is where correctness lives.** If the harness doesn't record evidence with citations, or doesn't detect stalls, the whole pattern degenerates back to implicit history.
- **Pipeline-parallel scheduling amplifies bugs.** A subtly-wrong Coverage Map update propagates faster when many sub-agents are pulling from the frontier in parallel.

## Sources

- Paper: *SearchOS-V1: Towards Robust Open-Domain Information-Seeking Agent Collaboration* — Gao, Wu, Fan, Zhang, et al. — Renmin University of China / Ant Group, 2026.
