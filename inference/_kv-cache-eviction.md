# KV Cache Eviction

*Taxonomy — how to bound the KV-cache memory footprint while preserving long-context quality, by deciding which past tokens to drop or compress.*

**TL;DR:** Long-context LLM inference makes the KV cache the dominant GPU-memory consumer. Eviction policies decide which key-value entries to keep per step under a fixed (or dynamic) budget. The space ranges from oblivious sliding windows (fast, lossy) to attention-mass scoring (H2O), to confidence-aware dynamic budgets (Conf-KV). Mixed-precision storage is the orthogonal axis — INT8 or lower for the bulk, FP16 for a protected recent window.

**Related taxonomies:** none yet in this folder.
**Depth files covered here:** [conf-kv.md](conf-kv.md)

---

## The problem

For a context of $L$ tokens, $N$ layers, and head dimension $d$, the KV cache holds $2 \cdot L \cdot N \cdot d$ values per request. At long $L$ this dominates GPU memory and serializes per-token attention reads. You need to *bound* what you keep without destroying long-range recall.

## The shared pattern

Every eviction policy answers three questions:

1. **What is the budget?** Static (fixed K tokens kept) or dynamic per-step.
2. **How are tokens ranked?** Recency, accumulated attention, learned importance, or a composite.
3. **What is the storage format for kept tokens?** Uniform FP16, mixed FP16/INT8, mixed precisions by layer, or compressed.

A protected "recent window" of $W$ tokens always kept in full precision is near-universal — local coherence depends on it.

## Variants

| Technique | Key idea | Main tradeoff | When it wins |
| --- | --- | --- | --- |
| Sliding window | Keep last $K$ tokens, drop the rest | Loses all distant context | Streaming text, very tight budget |
| H2O (heavy hitters) | Rank by accumulated attention mass; keep top-K | Single static rank; can miss tokens that become important later | Mid-context, well-behaved attention patterns |
| StreamingLLM (attention sink) | Keep first few tokens + recent window | Stabilizes generation past trained length; doesn't recover distant facts | Open-ended generation past context limit |
| [conf-kv](conf-kv.md) | Per-step budget driven by model confidence; composite recency × attention rank; mixed FP16/INT8 | Adds per-step overhead; depends on accurate confidence | Long-horizon agentic tasks where uncertainty varies |
| SnapKV / pyramidal | Layer-specific budgets (smaller toward upper layers) | Hyperparameter-heavy | Mixed when paired with another base policy |

## How to choose

- **Default for chat / open-ended generation:** StreamingLLM-style sink + recent window. Cheap, robust.
- **Long-context recall (needle-in-haystack, long-document QA):** confidence-aware (Conf-KV) or H2O-style composite. The model's own uncertainty is a strong signal for *when* to keep more context.
- **Tight memory budgets:** add INT8 storage for the non-recent portion. The recent window stays FP16. The PPL hit is small (Conf-KV reports within 1.5–2.1 PPL points of full KV at ~512-token sliding-window footprint).
- **Don't stack three policies blindly.** Sliding window + H2O + confidence-aware is rarely a Pareto improvement over a single well-chosen policy; the composite scoring becomes opaque.

## Adjacent but distinct

- **Paged attention** (vLLM) is *allocation*, not eviction — it manages KV memory in pages to reduce fragmentation, but does not drop tokens.
- **Speculative decoding** is decode-time compute reduction; orthogonal to KV memory.
- **KV quantization** (e.g., KIVI, AWQ-KV) reduces per-token bytes; orthogonal to which tokens are kept. Mixed-precision storage in Conf-KV bridges both axes.

## Sources

- Paper: *H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models* — Zhang et al., 2023.
- Paper: *Efficient Streaming Language Models with Attention Sinks (StreamingLLM)* — Xiao et al., 2023.
- Paper: *Conf-KV: Confidence-Aware KV Cache Eviction with Mixed-Precision Storage for Long-Horizon LLM Inference* — 2026 — [arXiv 2605.24786](https://arxiv.org/abs/2605.24786).
