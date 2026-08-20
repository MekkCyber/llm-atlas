# Agent Memory Substrates

*Taxonomy — how long-horizon LLM agents persist and recall state across turns.*

**TL;DR:** "Memory" for LLM agents is a *substrate* decision, not a single mechanism: dense retrieval, sparse retrieval, text records, structural stores, hierarchical stores, refinement-based memories, parametric updates, and activation/context caches all trade differently across regimes. No substrate dominates; the right one depends on task type (QA vs decision-making) and horizon length. Treat memory as a **routing** problem.

**Related taxonomies:** [../post-training/_rl.md](../post-training/_rl.md) (post-training that can update parametric memory)
**Depth files covered here:** none yet — this taxonomy anchors future depth files.

---

## The problem

LLM agents that run for many turns exceed the context window. Recompressing prior history into the prompt every turn either (a) drops crucial state or (b) inflates every subsequent call. Every "memory" system is a way to move some state out of the working prompt and get it back cheaply enough that it earns its keep. The failure mode is universal: memory that helps user-facing QA (recall dense factual context) often *hurts* sequential decision-making by dragging attention away from action-critical tokens.

## The shared pattern

Every memory substrate is a **write path × read path × store**:

- **Write path** — when do we commit? (every turn, on success, on failure, on user cue.)
- **Store** — where does state live? (index, text file, DB row, graph, adapter weights, KV cache.)
- **Read path** — how does the agent surface state next turn? (retrieve by query, follow a link, load an adapter, reuse a KV block.)

The class of the substrate is set by *what state actually lives in the store*: text, structured records, vectors, weights, or activations.

## Variants

| Substrate | Key idea | Main tradeoff | When it wins |
| --- | --- | --- | --- |
| Dense index (RAG-style) | Embed text, retrieve by cosine over a vector store | Recall-only; brittle on structured queries | Long-context factual QA |
| Sparse index (BM25-style) | Token-overlap retrieval over documents | Fails on paraphrases | Named entities, code identifiers |
| Text records | Append raw turn text keyed by session / user | Grows unboundedly; needs compaction | User-personalisation memory |
| Structural store | Typed rows / graph nodes; queried by SQL/Cypher | Requires schema; write cost | Multi-entity workflows, CRM-style |
| Hierarchical store | Summaries at multiple granularities | Summary drift; compaction bias | Very long horizons where full retrieval is too costly |
| Refinement memory | Rewrite prior memory on new evidence | Overwrites correct history; expensive | Belief-tracking tasks |
| Parametric update | Fine-tune / LoRA / adapter into weights | Slow to write; catastrophic forgetting | Skills the agent uses many times |
| Activation / KV cache | Reuse KV blocks or prefix embeddings across turns | Tied to model version and tokeniser | Interactive latency-sensitive loops |

## How to choose

Start from the regime, not from the technique:

- **User-centric QA over long history** → dense index + text records, top-k retrieval, aggressive summarisation once history exceeds ~50k tokens.
- **Sequential decision-making** → *minimise* retrieval; prefer hierarchical summaries and structural stores. Broad retrieval *hurts* — Huang et al. (2026) show it shifts attention off action-critical context.
- **Repeatable expertise** → parametric update (LoRA / adapter) once a behaviour stabilises; keep the substrate small and offline-trainable.
- **Very long horizons (>>100k tokens of cumulative history)** → substrates that win at moderate horizons (e.g. flat dense indices) become brittle or costly; move to hierarchical or refinement stores.
- **Multi-substrate agents should treat substrate choice as a routed decision** — a first-class action of the agent policy, not a static architectural pick.

## Adjacent but distinct

- **Skill libraries** (see `agent-skills.md`) — a specific substrate variant tuned for *procedural* memory rather than general state.
- **KV cache management** in serving stacks — an inference optimisation, not a memory system per se, though hierarchical/paged KV caches shade toward this taxonomy.
- **Long-context modelling** — architectural work (RoPE-scaling, YARN, DCA) that expands the *implicit* memory. Complementary; not a substrate.

## Sources

- Paper: *Harness the Memory: A Holistic Evaluation of Memory Substrates in Memory Agents* — Huang, Zhang, Wu, Chen, Jiang, Yang, Yang, Zou, Zhang, Wu, Wu, Chang, Yu, Liu, Caliskan — UIUC / UCLA / U-Washington / McGill, 2026 — https://arxiv.org/abs/2608.15008
- MemGPT, LongMem, Reflexion, Voyager — earlier memory-substrate proposals along individual points of the taxonomy.

---

## Conventions

- **Filename:** `_memory-substrates.md` (taxonomy in `agents/`).
- **Scope:** substrates specifically for LLM-agent-durable state; excludes context-window architecture work (that belongs in `architectures/`) and serving-side KV cache work (that belongs in `inference/`).
