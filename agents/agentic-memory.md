# Agentic Memory (Evolvable Embeddings)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Static embedding models encode each segment in isolation, so they can't reflect "what the agent knows now." **Evolvable embeddings** maintain a continuously-updated latent memory that is read alongside the raw content when emitting an embedding — so the same query retrieves different things as context evolves. Introduced as EvoEmbedding (2026) for long-context retrieval and agentic memory.

**Prereqs:** [README.md](README.md)
**Related:** [../inference/README.md](../inference/README.md), [session-state.md](session-state.md)

---

## What it is

The standard retrieval stack treats embeddings as a function of a *fixed string*: `emb(text)`. For an agent that accumulates state — what tools it has already used, what facts it has confirmed, what user preference it has inferred — this is wrong. The agent's *knowledge state* is not the literal text it stored.

Evolvable embeddings parameterise embeddings as a function of *content + running memory*: `emb(text, memory)`. The memory is updated sequentially as the agent processes inputs, so the embedding emitted at time `t` carries the agent's state at `t`.

## How it works

A recurrent / stateful embedding model:

1. Maintain a latent memory `m_t` (a fixed-dim vector or short token sequence).
2. At each new input `x_t`, compute `e_t = f(x_t, m_t)` — the embedding is conditioned on both.
3. Update memory: `m_{t+1} = g(m_t, x_t)`.
4. Index `e_t` in the retrieval store.

At query time, the query is embedded with the current `m_t`, so it lands in the slice of embedding space that reflects the agent's present state. The same literal query string maps to different points over time.

## Why it matters

- **No re-indexing on state change.** Static embeddings would have to be recomputed every time the agent's understanding shifts. Evolvable embeddings absorb the shift into the memory rollout.
- **Better long-context retrieval.** Late items in a context are embedded with awareness of what came earlier, which is exactly the information a retriever needs to disambiguate.
- **Plumbing for long-running agents.** A foundation for the broader "agentic memory" stack — episodic memory, working memory, semantic memory layers can all sit on top of an evolvable embedding base.

## Gotchas & tricks

- **Memory drift.** Long sequences let the memory drift far from the original embedding space, hurting cross-agent retrievability. Periodic memory resets or projection back to a base distribution help.
- **Indexing semantics.** A standard vector index expects embeddings of `text`; here embeddings are of `(text, memory)`. Storing `m_t` alongside `e_t` and exposing both at retrieval time keeps the index honest.
- **Retraining cost.** Going from static to evolvable embeddings is a new model — not a thin shim over an existing encoder.

## Sources

- Paper: *EvoEmbedding: Evolvable Representations for Long-Context Retrieval and Agentic Memory* — anonymous, 2026 — [arXiv:2606.21649](https://arxiv.org/abs/2606.21649).
