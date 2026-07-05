# Agent Memory

*Taxonomy — how a long-horizon LLM agent decides what past context each future decision is allowed to see.*

**TL;DR:** For a multi-turn agent, memory is not "the context window." It is a *contract* deciding which observations, tool calls, plans, and reflections each next-step prompt gets to include. The dumb contract (append everything) makes decisions unattributable to individual memory items and blows the context budget. Real systems pick one of three shapes — **typed retrieval**, **explicit file-system memory the agent operates on itself**, or **hand-designed sliding-window schemas** — and trade capacity, latency, and controllability differently.

**Related taxonomies:** [../post-training/_post-training.md](../post-training/_post-training.md)
**Depth files covered here:** [automem](automem.md)

---

## The problem

A long-horizon agent (thousands of steps in Crafter, NetHack, or a real workflow) generates far more context than any window can hold. Every step, something has to decide what the model sees next. Failure modes are asymmetric: **too much** and the model drowns in noise / hits context limits; **too little** and the agent forgets its own prior tool calls and reflections. Compounding the pain: a single bad memory decision at step 100 may not surface until step 3000, making the loop hard to debug or train.

## The shared pattern

Every agent memory system decomposes into:

1. **A store** — where past items live (context, external DB, file system).
2. **A write policy** — what gets added, at what granularity (per token, per observation, per reflection).
3. **A read policy** — what gets pulled into the *next* prompt, and how (append-all, similarity retrieval, typed retrieval, explicit file reads).

The interesting design axis is the read policy: it decides whether the effect of any single memory item on downstream behavior can be isolated at all.

## Variants

| Technique | Key idea | Main tradeoff | When it wins |
| --- | --- | --- | --- |
| Append-all context | Concatenate every past observation and tool call into every prompt | Simple, unattributable, hits context limits fast | Short trajectories; baselines |
| Typed retrieval (AgenticSTS-style) | Each memory item carries a type (observation, tool-call, reflection); retrieval is per-type | Enables per-type ablation; requires typing discipline | Bounded-memory benchmarks; research on which memory matters |
| Similarity retrieval / RAG-over-history | Embed past items; retrieve top-k by cosine to the current step | Cheap; ignores structural types | Chatbots with long history but simple task shape |
| [AutoMem](automem.md)-style file system | File-system operations are first-class *actions* the agent chooses alongside task actions | Memory becomes a trainable skill; slow to train | Very long horizons where memory decisions must adapt over episodes |
| Structured hand-designed schemas | Fixed slots per step ("last plan", "last error", "goal") | Easy to reason about; brittle to new tasks | Production agents with a stable task shape |

## How to choose

Default to typed retrieval for research and benchmarking — it's the only design that isolates each memory type's contribution and makes ablation studies valid. Move to file-system-memory-as-actions ([AutoMem](automem.md)) when the horizon is long enough that the memory *policy itself* needs to improve with training, not just the task actions. Fall back to hand-designed schemas only when the task shape is fixed and you need predictable latency.

Whatever you pick, keep the memory policy **separable** from the task policy: they train and fail on different timescales, and conflating them makes both harder to improve.

## Adjacent but distinct

- **KV cache** — a hardware-level context store, not a memory *policy*. See [../architectures/mla.md](../architectures/mla.md) for KV compression.
- **Long-context training** — extending the base model's window (RoPE scaling, YaRN). Complements memory policies but doesn't replace them; even at 1M tokens, a 10k-step agent still needs one.
- **Retrieval-augmented generation** — memory of an external corpus, not of the agent's own history. Same read-policy machinery, different store.

## Sources

- Paper: *AgenticSTS: A Bounded-Memory Testbed for Long-Horizon LLM Agents* — Cheng et al., 2026 — typed-retrieval framing.
- Paper: *AutoMem: Automated Learning of Memory as a Cognitive Skill* — 2026 — memory-as-trainable-skill; see [automem](automem.md).
