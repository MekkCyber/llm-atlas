# AgenticSTS

*Depth — a bounded-memory testbed and typed-retrieval framing for evaluating long-horizon LLM agents.*

**TL;DR:** AgenticSTS reframes long-horizon agent evaluation around **memory as a contract**: instead of appending everything the agent has ever seen, each future decision assembles its prompt from a *typed retrieval* over past items (observations, tool calls, reflections, plans). The typed decomposition makes it possible to ablate one memory type at a time and see its downstream effect on decisions — impossible under append-all baselines.

**Prereqs:** [../agents/_agent-memory.md](../agents/_agent-memory.md)
**Related:** [../agents/automem.md](../agents/automem.md)

---

## What it is

A benchmark and framing for long-horizon LLM agents. Each memory item at each step carries an explicit **type** — observation, tool call, reflection, plan, environment feedback, etc. The prompt for step $t$ is not the concatenated history; it is the union of results from typed retrieval queries fired against the memory store. Removing one type surgically isolates its causal effect on the agent's next action.

## How it works

### Typed retrieval

For each step $t$, the agent (or the harness) fires one query per memory type. Each returns a bounded number of items of that type. The prompt is assembled from those results plus the current observation. Every prompt is **fresh** — nothing is silently carried from the previous step's context, so anything that affects the current decision must have been retrieved by name.

### Ablations become tractable

Because the read policy is typed, an ablation like "no reflections in this run" is a one-line change (drop that type's query). The differential effect on task success is directly attributable to that type. Contrast with append-all setups where dropping "reflections" also drops formatting cues and positional-order signals confounded with them.

### Task suite

Long-horizon environments where the naive append-all baseline breaks — either by hitting the context window or by drowning meaningful items in noise — and where correctly-scoped typed retrieval makes the difference.

## Why it matters

- **Isolates memory contributions.** For years agent papers have claimed "we made memory better" and confounded retrieval, storage, and prompt-assembly at once. Typed retrieval separates them at the eval layer.
- **Bounded-memory is realistic.** Real deployed agents cannot fit an unbounded history into every prompt. A testbed that enforces the bound while making the read policy inspectable matches the production constraint.
- **Grounds the next wave of memory training.** Techniques like [AutoMem](../agents/automem.md) need an eval harness that can *tell* whether their memory changes help. AgenticSTS is a step toward that.

## Gotchas & tricks

- **Typing discipline is load-bearing.** If reflections silently include tool-call summaries, the ablation of "tool-call type" leaks reflection content and results are muddy. Enforce strict type schemas at write time.
- **Retrieval quality is a confound.** A weak per-type retriever will make all types look uninformative. Sanity-check retrieval hit rate before drawing conclusions.
- **Bounded memory is not a *hard* memory limit.** The bound is per-type item count in the prompt, not total store size. Long-horizon runs can still explode the store; garbage-collect old items.

## Sources

- Paper: *AgenticSTS: A Bounded-Memory Testbed for Long-Horizon LLM Agents* — Cheng et al., 2026 — [arXiv:2607.02255](https://arxiv.org/abs/2607.02255).
