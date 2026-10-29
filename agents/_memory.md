# Agent Memory

*Taxonomy — storage and retrieval systems that let an LLM agent carry context across turns, sessions, and long horizons.*

**TL;DR:** An agent "memory" is anything beyond the model's own context window that persists information between turns. The field is split along two axes: **what is stored** (raw turns, summaries, structured facts, learned embeddings) and **how it's retrieved** (RAG, hierarchical summary, graph walk, learned recall). Modern evaluations show that memory systems built on top of RAG-over-history often *hurt* net performance because distillation/compression steps strip evidence — the right baseline is RAG-over-raw-history, not no-memory. See [memory-tax.md](memory-tax.md) and [../evaluation/habit-bench.md](../evaluation/habit-bench.md).

**Related taxonomies:** *(none yet)*
**Depth files covered here:** [memory-tax.md](memory-tax.md) · [../evaluation/habit-bench.md](../evaluation/habit-bench.md)

---

## The problem

LLM agents operating over many turns or sessions need to recall facts, preferences, and past decisions that fall out of the context window. The naive fixes — stuff everything into context, or forget everything — both fail: the first is expensive and dilutes attention; the second makes the agent useless for long-horizon work.

What goes wrong:

- **Context pressure.** Linear growth of raw history quickly exceeds budget; even with long context, attention degrades on the oldest material.
- **Signal dilution.** When memory stores paraphrases or summaries, specific details needed for a later task are irreversibly lost.
- **Habit induction.** Simple retrieval doesn't infer repeated preferences from scattered weak evidence (see [../evaluation/habit-bench.md](../evaluation/habit-bench.md)).

## The shared pattern

```
      ┌─────────── store ──────────┐              ┌─────── recall ───────┐
turn →│  raw / summary / graph /   │   query  →   │  RAG / graph walk /  │ → context
      │  learned-embedding record  │              │  learned fetch       │
      └────────────────────────────┘              └──────────────────────┘
```

Every memory system is a (write, read) pair over some representation. The representation decides what can be recalled.

## Variants

| Technique | What it stores | How it retrieves | Main tradeoff | When it wins |
| --- | --- | --- | --- | --- |
| **RAG over raw history** (baseline) | Full turn log, embedded | Dense retrieval, top-k | Linear store, retrieval scales with history | **The right baseline.** Hard to beat on fact recall |
| **Hierarchical summarization** | Nested rolling summaries | Walk + top-level dense | Compression removes specific evidence (see [memory-tax.md](memory-tax.md)) | Long sessions where fact recall isn't critical |
| **Entity / graph memory** | Structured triples or scene graphs | Graph walk + dense | Structuring step is brittle; may miss unstructurable detail | Role-play, long-running personae |
| **Episodic memory** | Chunked episodes + metadata | Metadata filter + dense | Episode boundaries are heuristic | Multi-day workflows, personal assistants |
| **Learned recall** | Latents from a side model | Trained recall head | Expensive and opaque; evaluation is hard | Research; occasionally wins on narrow benchmarks |
| **Habit induction** (no depth file yet) | Repeated-pattern distillations | Pattern match + dense | Current systems top out at ~34% on [HABIT-Bench](../evaluation/habit-bench.md) | Not yet production-ready |

## How to choose

**Default:** RAG over raw history with a strong embedder and generous context. [Memory Tax](memory-tax.md) shows this is a hard baseline to beat.

**Add structure only when you can measure net value.** Any layer that reorganizes, summarizes, or distills the history has a *memory tax* — it loses evidence as the price of a smaller store. Only ship such a layer if it clears RAG-over-history on your target evals.

**For personae and long characters,** entity/graph memory earns its tax by making role-consistency queries cheap. For factual recall, it doesn't.

**For habit induction** (preferences, routines), nothing current works well. Treat this as an open problem, not a product feature.

## Adjacent but distinct

- **Long context.** Just enlarging the window is not memory — it's a budget change. Memory systems become necessary again once the budget is spent.
- **KV cache compression.** An inference-time optimization over the same context; doesn't change what the agent *remembers* across sessions.
- **Fine-tuning on user data.** Permanent, not retrievable. Also raises privacy issues RAG doesn't.
- **Tool / function calling for state.** Writing to a user-visible store (notes, calendar) via tools is memory *in the world*, not in the agent. Both can coexist.

## Sources

- Paper: *Memory Tax: Dissecting the Net Value of Agent Memory* — He et al., 2026 — [arXiv:2610.34073](https://arxiv.org/abs/2610.34073)
- Paper: *HABIT-Bench: Evaluating Habit Induction from Longitudinal Weak Evidence in Agent Memory* — Zhang et al., 2026 — [arXiv:2610.24520](https://arxiv.org/abs/2610.24520)
