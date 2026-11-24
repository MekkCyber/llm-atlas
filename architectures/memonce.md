# MemOnce
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A long-context memory architecture that caches a small subset of **final-layer hidden states** as compact memory tokens, then **re-expands them into per-layer K/V on demand** via tied projections. Sparse attention uses retrieval indices computed **once across all layers**, amortizing retrieval cost. At 1M-token contexts, reports lower memory and latency than full-KV Transformers and linear-time baselines, with zero-shot transfer to retrieval tasks.

**Prereqs:** [attention](../fundamentals/attention.md), [mla](mla.md)
**Related:** [dca](../fundamentals/dca.md)

---

## What it is

Standard transformers cache per-layer K and V for every token — memory scales as `2 · L · d_kv · seqlen`. Linear-attention / SSM alternatives give O(1) state at the cost of fine-grained recall.

MemOnce aims for middle ground: **store memory compactly as final-layer hidden states**, and reconstruct per-layer K/V only for the small retrieved subset at each query step. The memory tokens are a subset of the sequence selected by the model — not every token is memorized.

---

## How it works

**Memory writing.** As the model processes a long prefix, a lightweight selector picks which tokens' final-layer hidden states `h_t^L` to add to the memory pool. Memory size is a tunable budget (e.g. a few thousand tokens for a 1M-token context).

**Memory reading via tied projections.** When a query at decode time needs memory, a retrieval step picks the top-`k` memory tokens by similarity. For each retrieved token, a set of **per-layer tied up-projections** reconstructs `(K_l, V_l)` at every layer `l` from the single stored `h^L`:

```
K_l = U_K^l · h^L
V_l = U_V^l · h^L
```

The `U_K^l`, `U_V^l` are learned once, shared across all retrieved tokens. Reconstruction is a cheap matmul per retrieved token per layer.

**One retrieval across layers.** The retrieval indices are decided **once per query** (at the top layer) and reused by every layer. This is where the "once" in MemOnce comes from — classical per-layer retrieval would redo the sparse attention sort `L` times. Here it's one sort, amortized.

**Attention path.** Non-memory context (recent tokens) uses normal dense attention. Memory tokens contribute via the reconstructed K/V, mixed into the normal attention computation.

---

## Why it matters

Opens a design axis distinct from KV compression (MLA, GQA), KV eviction (StreamingLLM, H2O), and linear-time models (Mamba, GLA). MemOnce keeps full-fidelity attention over a **learned subset**, with the memory itself as a one-vector-per-remembered-token structure.

Zero-shot retrieval-task transfer is a nice side effect: the memory tokens behave like a learned index, so the model can answer "where in the context was X?" without a separate retriever.

---

## Gotchas & tricks

- **Memory selector quality matters.** If the selector misses relevant tokens, no amount of reconstruction will recover them — this is the main failure mode at 1M scale.
- **Tied projections mean per-layer specialization is limited.** Compensated by keeping recent-token attention dense.
- **One-retrieval-across-layers assumes top-layer similarity ≈ right answer.** For heavy multi-hop tasks the assumption breaks; the paper reports mitigation via query reformulation.

---

## Sources

- Paper: *MemOnce: A Scalable Memory Architecture for Large Language Models* — Fu, Liu, Hu, Shan, 2026 — [arXiv:2611.27973](https://arxiv.org/abs/2611.27973)
- Project page: https://clare-nie.github.io/MemOnce
