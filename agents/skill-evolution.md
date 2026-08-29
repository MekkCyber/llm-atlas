# WikiSkill — Agent Skill Evolution with a Persistent Wiki
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Agent-skill libraries let insights die inside optimization histories: the reasoning that produced skill v1 is not available when a later iteration decides how to write v2. **WikiSkill** co-evolves the skill library with a **persistent wiki** — a human-readable knowledge base that accumulates lessons across iterations — and explicitly separates three layers: raw execution experience, accumulated wiki knowledge, and executable skills. Subsequent skill-update passes read from the wiki, not from raw traces. Consistently beats state-of-the-art skill-evolution baselines; ablations show the persistent wiki is the critical ingredient. Skills transfer across models and families, and skills evolved by *another* model can outperform self-evolved skills.

**Prereqs:** [README.md](README.md)
**Related:** [pilot-supervisor-worker.md](pilot-supervisor-worker.md), [skill-graph-retrieval.md](skill-graph-retrieval.md)

---

## What it is

Automatic skill discovery from agent traces is now standard: run the agent, inspect the trace, distill a reusable skill, add it to a library, run again. What's usually missing is a shared *intermediate representation* between "raw trace" (too specific) and "executable skill" (too crystallized). The reasoning behind skill decisions — *why* a step matters, *when* it fails, *what* alternatives were considered — evaporates once the skill is written.

WikiSkill introduces a persistent wiki as that intermediate representation. Every iteration writes into the wiki; every skill update reads from it. The wiki survives changes to the skill library the same way a codebase's design docs survive refactors.

## How it works

Three explicit layers:

1. **Raw execution experience** — full trajectories from agent runs, tagged with outcome.
2. **Wiki** — a persistent, structured knowledge base written in natural language. Each wiki entry records a lesson: a task pattern, a failure mode with symptoms and correction, a precondition to a known-good approach, an environmental quirk. Entries are updated (not overwritten) as new evidence arrives.
3. **Executable skills** — the current library of callable skills. Each skill has a spec and code; skills are versioned.

**Iteration loop:**

- Run the agent on a task batch, producing new raw traces.
- **Consolidation pass** (LLM-driven): read new traces + relevant existing wiki entries → propose wiki edits (new entries, additions, corrections). Prior wiki state is context, not overwritten.
- **Skill update pass** (LLM-driven): read the *updated wiki* → propose skill additions, edits, or deprecations. Skills are compiled from the wiki, not from raw traces.
- Deploy new skills, repeat.

Because the skill update reads from the wiki (which carries cross-iteration context) rather than from just the most recent traces, later iterations can reuse insights from much earlier ones.

## Why it matters

- **Cross-model skill transfer.** Skills evolved by model A improve model B, and can outperform B's self-evolved skills. Strong evidence that WikiSkill captures task structure rather than model idiosyncrasies.
- **Skill evolution complements scaling.** Larger models benefit more from evolved skills; smaller models with skills can outperform substantially larger models without them.
- **Ablations validate the wiki.** Removing the persistent wiki (skill updates read only from recent traces) collapses most of the gain — the intermediate representation is the load-bearing element, not the two-pass structure by itself.

## Gotchas & tricks

- **Wiki growth is unbounded.** Without pruning, the wiki bloats and consolidation-pass cost grows. The paper uses relevance-scored retrieval to bound context per pass.
- **Human-readable is a design choice with a cost.** Structured natural language is more expensive to update than an embedding index but makes the wiki debuggable and transferable across models.
- **Update conflicts.** When new evidence contradicts an existing entry, the paper appends rather than overwrites — history is preserved for later re-adjudication.
- **The consolidation-pass LLM matters.** A weak consolidator produces a noisy wiki that misleads later skill updates. Match the consolidator's capability to the domain complexity.

## Sources

- Paper: *WikiSkill: Compiling Agent Experience into Persistent Knowledge for Skill Evolution* — Tang et al. (Google Research / Virginia Tech), 2026 — arXiv:2608.27454.
