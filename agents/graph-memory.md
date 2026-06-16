# Graph Memory (Cue–Tag–Content + Active Reconstruction)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Replace static "retrieve-then-reason" agent memory with an **associative graph** (Cue → Tag → Content nodes) walked iteratively by the LLM during reasoning. Memory is *reconstructed* per query — the agent expands promising tags and prunes the rest based on accumulated evidence — rather than retrieved as a fixed top-k context dump. Reported up to **+23%** on memory-augmented agent benchmarks with reduced compute (no fixed retrieval cost).

**Prereqs:** [README.md](README.md)
**Related:** [../post-training/_post-training.md](../post-training/_post-training.md)

---

## What it is

Standard agent memory: embed past interactions, retrieve top-k by similarity to current query, prepend to context, reason. Failure modes: retrieved context is fixed before reasoning starts; irrelevant matches consume tokens; the agent can't drill into a thread it discovers mid-thought.

Graph memory replaces the retrieval index with a graph whose nodes have three types:

- **Cue** — what triggered storage (the query / event that produced this memory)
- **Tag** — a semantic bridge label (often LLM-generated)
- **Content** — the actual recalled detail

Edges connect a Cue to its Tags and Tags to their Contents. The graph is sparse and overlap-rich: many Cues can share a Tag, many Contents can be tagged multiply. The Tag layer is the lossy compression that lets reasoning find content it didn't know to search for.

---

## How it works

### Construction (write path)

When the agent stores an interaction:

1. Encode the interaction as a Cue node.
2. LLM-generate a small set of Tags (3–10) summarizing semantic facets.
3. Extract Contents (factual atoms, conclusions, decisions) and link each to relevant Tags.
4. Add edges; merge Tags that semantically match existing ones (small embedding ANN over tag strings).

The write path is cheap: a few LLM calls per stored interaction, amortized.

### Active reconstruction (read path)

When the agent reasons over memory:

1. Start with the current query state.
2. LLM proposes candidate Tags relevant to the query — *not* a similarity lookup, an LLM-internal generation step.
3. Look up matching Tag nodes in the graph; expand to their Contents and connected Cues.
4. Feed the expanded subgraph fragment back to the LLM's reasoning step.
5. The LLM updates state, prunes paths that didn't help, and proposes the *next* round of Tags.
6. Iterate until reasoning concludes.

The graph walk is driven by the *current reasoning state*, not the original query. New evidence opens new paths; the agent reconstructs memory rather than receiving a precomputed snapshot.

---

## Why it matters

- **Discovers paths the query alone wouldn't surface.** A retrieve-then-reason pipeline only ever sees what looked relevant up-front. Active reconstruction follows leads as they appear in reasoning.
- **Token-efficient.** No fixed top-k dump; only the expanded fragments enter context, and only when the LLM asks.
- **Aligns with cognitive-science framing.** Memory in humans is constructive (cue-driven, partially generative), not stored-and-fetched. The Cue–Tag–Content split makes that machinery operational for agent stacks.
- **Composable with vector stores.** Embedding indices can sit *under* the Tag layer to scale Tag lookup; the graph is the new control surface, not a replacement for ANN.

---

## Gotchas & tricks

- **Tag quality is the ceiling.** Garbage Tags from a weak LLM at write time hide useful Contents permanently. Use a stronger model for tagging than for runtime reasoning.
- **Graph cleanup is real work.** Merging near-duplicate Tags ("payment date" vs. "billing date") needs an explicit reconciliation pass — embeddings + LLM judge — or the graph fragments by synonym.
- **Termination criteria.** Open-ended reconstruction can spiral; cap iterations (paper reports a small step budget) or require explicit "stop" decisions from the LLM.
- **Cold start is rough.** With few Cues, the Tag layer is sparse and reconstruction degenerates to direct content lookup. Bootstrap by mass-tagging an initial document corpus before the agent runs live.
- **Provenance.** Every reconstructed fragment should carry its Cue / Tag / Content provenance for auditability — particularly important in regulated agent deployments.

---

## Sources

- Paper: *Memory is Reconstructed, Not Retrieved: Graph Memory for LLM Agents (MRAgent)* — 2026 — [arXiv 2606.06036](https://arxiv.org/abs/2606.06036).
- Background: associative-memory literature (Hebbian, Hopfield) for the cognitive framing; modern parallels in retrieval-augmented agent stacks.
