# Zero-Token Memory (Zero-Mem)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** An agent-memory scheme that eliminates every LLM call *except* the final answer reader. Raw interaction traces are indexed twice — as an entity–context graph (cross-interaction relations) and as a temporal hierarchy (locality + session state) — and retrieval routes through both structures deterministically. Cuts memory-operation time by 57.6% vs the fastest LLM-mediated baseline at competitive answer quality.

**Prereqs:** [_agent-memory](_agent-memory.md)
**Related:** none yet

---

## What it is

Most agent-memory systems (MemGPT, summary-based memory, graph-extractor stacks) burn LLM calls on write (summarise, decide-what-to-store) or read (rewrite query, choose retriever, judge relevance). Zero-Mem's claim: those calls aren't necessary — a well-structured index over the *raw* traces plus a deterministic scoring rule matches quality without any of them.

## How it works

**Two indices, both computed from raw traces without LLM invocation:**

1. **Entity–Context Graph.** Nodes are entities mentioned across interactions; edges are co-occurrence relations weighted by proximity and frequency. Extracted via encoder-based NER and simple co-occurrence — no generative LLM. Enables cross-session recall ("what did the user say about X three sessions ago").
2. **Temporal Hierarchy.** Sessions → turns → utterances, with session-level state (goals, unresolved threads) surfaced by structural rules. Preserves conversational locality — cheap for "what did they just say" recall.

**Retrieval path (per query):**

1. Score query against both indices in parallel.
2. Weight the two view scores by a query-dependent coefficient — entity-heavy queries pull from the graph, "recent context" queries pull from the temporal hierarchy.
3. Retrieve raw evidence spans from both, following the index structure (graph neighbours or temporal neighbours) to expand context.
4. **Deterministic calibration**: drop retrieved spans that conflict with each other, keep only mutually consistent evidence.
5. **Final QA reader** — the only LLM call in the pipeline — answers grounded in the retrieved spans.

## Why it matters

- **Cost floor.** Zero token spend on memory ops is a hard lower bound; systems can't beat it. On typical agent workloads this is >50% of memory-related tokens.
- **57.6% wall-clock reduction** over the fastest compared LLM-mediated baseline, same reader and context budget.
- **Reproducible retrieval.** Deterministic indexing means the same query always returns the same evidence — helpful for debugging and evaluation.
- **Falsifies a widely-held assumption.** "You need an LLM to write memory representations" turns out to be wrong for QA workloads.

## Gotchas & tricks

- **Encoder is not free.** Zero-Mem "accounts for encoder computation separately" — the entity extractor and embedder run on every write. It's smaller than a generative LLM call but non-trivial at high write rates.
- **No LLM-driven abstraction.** For traces where the useful memory is *implicit* (a user's evolving preference across many turns, never stated directly), a purely retrieval-over-raw system will miss it. Add an occasional LLM-mediated summary pass if this matters.
- **Deterministic calibration is aggressive.** Dropping "conflicting" evidence can throw out genuinely correct answers that clash with earlier contradicted claims. Tune the conflict threshold.
- **Query-dependent view weighting is the fragile part.** The paper trains this weighting; a naïve fixed 50/50 loses noticeable quality.
- **Read the raw code before adopting.** "Zero LLM calls" is a headline claim; the released implementation may include small LLM calls for edge cases.

## Sources

- Paper: *Zero-Mem: Zero-Token Memory Operations for LLM Agents* — Xiao, Zhu, Zhang et al., arXiv:2607.29377, 2026.
- Code (post-review): [github.com/TheMoon0815/Zero-mem](https://github.com/TheMoon0815/Zero-mem).
