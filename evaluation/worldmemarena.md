# WorldMemArena

*Depth — a stage-decomposed benchmark for multimodal agent memory, comparing long-context, RAG, external-memory, and harness-based architectures.*

**TL;DR:** Existing agent memory benchmarks roll memory failures into one end-of-task accuracy and reduce visual observations to captions, so you can't tell *which* stage of memory broke. WorldMemArena formalizes agent memory as an Action–World Interaction Loop with a four-stage lifecycle (write / maintain / retrieve / use), instantiates it in 400 multi-session multimodal tasks, and annotates each instance with gold memory points, updates, distractors, and evidence chains — enabling stage-level diagnosis.

**Prereqs:** [README.md](README.md)
**Related:** [../agents/README.md](../agents/README.md)

---

## What it is

A benchmark that treats agent memory as a *lifecycle* rather than a black box. Designed to support head-to-head comparison of (a) long-context-only agents, (b) RAG agents, (c) external-memory systems, and (d) harness-based agents that author their own memory.

## How it works

**Two task families.**

- *Lifelong Evolution* — evolving personal and task states; the agent must track what changes over sessions.
- *Agentic Execution* — memory from real observations, actions, and feedback during task execution.

**400 multi-session multimodal tasks.** Each task spans multiple sessions to force write / maintain / retrieve cycles rather than collapsing into one-shot recall.

**Annotation per instance** — every task instance includes:

- Gold memory points (what should be remembered)
- Update events (when stored memory should be revised)
- Distractors (plausible but irrelevant items that bad retrievers would surface)
- Evidence chains (which stored items support the final answer)

**Stage-level scoring.** Localize each failure to write (was it stored?), maintain (was it kept correctly across updates?), retrieve (was the right item surfaced?), or use (did the agent ground the answer in retrieved evidence?).

## Why it matters

- First benchmark that lets you compare architecturally different memory systems on the same instances with the same stage-level diagnostics.
- Findings refute several common assumptions: (1) better memory writing does *not* guarantee better task performance; (2) multimodal memory underuses visual evidence even when present; (3) harness-based self-managing memory is more flexible but costlier and less reliable than well-designed RAG.
- Provides actionable engineering signal — if your agent fails most at "use," investing in better retrieval is wasted; fix the grounding step instead.

## Gotchas & tricks

- Stage labels assume a clean separation that some architectures violate. Long-context-only "agents" don't have a distinct write stage; score them on a re-mapped lifecycle (write = the input concatenation step).
- Multimodal evidence is only useful if the model can attend to it. Several systems score well on text-evidence tasks and collapse on image-evidence tasks — separate the two when interpreting.
- "Harness-based memory" covers many designs. Disaggregate harness scores by sub-strategy (memory-update rules, eviction policy) when comparing across papers.
- 400 tasks is small relative to LLM training scales — useful for evaluation, not for training.

## Sources

- Paper: *WorldMemArena: Evaluating Multimodal Agent Memory Through Action–World Interaction* — 2026 — [arXiv 2605.29341](https://arxiv.org/abs/2605.29341).
