# KV-cache compression

*Taxonomy — methods that shrink the transformer KV cache at inference to serve longer contexts at fixed memory.*

**TL;DR:** The KV cache dominates GPU memory during autoregressive inference — one entry per layer per head per token. Compression methods either **evict** tokens (drop past K/V), **quantise** their storage, or **share** across layers/heads. Modern default for long-context serving is a mix: paged attention for allocation, GQA/MLA for architectural sharing, low-bit K/V quantisation for storage, and eviction (attention-sink + recency, or importance-scored) when the budget still isn't enough.

**Related taxonomies:** [_number-formats](../quantization/_number-formats.md)
**Depth files covered here:** [beacon-kv](beacon-kv.md)

---

## The problem

For a model with `L` layers, `H` heads, and head dim `d`, decoding a sequence of length `T` keeps `2·L·H·d·T` KV values in memory. At 70B / 32K tokens / batch 32 this exceeds most single-GPU budgets before you even count activations. Recomputing K/V per step trades quadratic FLOPs for the cache and is a non-starter beyond a few K tokens.

## The shared pattern

Every method answers two questions: **what to keep** (all tokens vs. some) and **how to store what's kept** (full precision vs. compressed). Layered together they compose: e.g. paged allocation + FP8 K/V + importance-based eviction.

## Variants

| Technique | Key idea | Main tradeoff | When it wins |
| --- | --- | --- | --- |
| **Attention-sink + recency** (StreamingLLM) | Keep the first few tokens + a sliding window | Loses long-range recall | Streaming decoding, chat |
| **Heavy-hitter eviction** (H2O, Scissorhands) | Score KV entries by past attention mass; evict low scorers | Uses recent queries as a proxy — misses future revisits | Non-reasoning long contexts |
| [beacon-kv](beacon-kv.md) | Cluster query history; score entries against per-cluster beacon queries | Adds a small beacon set to manage | Long CoT / reasoning traces with revisits |
| **KV quantisation** (KIVI, FP8 KV) | Store K/V at 4-bit or FP8 | Small quality drop; needs per-token or per-channel scales | Any long-context serve; compose with eviction |
| **Cross-layer sharing** (YOCO, CLA) | Reuse a single KV cache across many decoder layers | Requires trained-in support | Purpose-built architectures |
| **Architectural sharing** (GQA, MQA, MLA) | Fewer K/V heads than Q heads; MLA compresses via low-rank latent | Requires training | Any new model — this is now default |
| **Paged attention** (vLLM) | Non-contiguous block allocation | Doesn't shrink the cache, but enables batching | Every modern serving stack |

## How to choose

- **New model design:** GQA is table stakes; MLA if you can afford the retrain cost — see [../architectures/mla.md](../architectures/mla.md).
- **Serving an existing model, standard context:** paged attention + FP8/INT4 KV.
- **Serving an existing model, long context, non-reasoning:** add attention-sink + recency (StreamingLLM) or heavy-hitter eviction (H2O).
- **Serving a reasoning model with long CoT:** BeaconKV — recency-based eviction is the wrong policy for revisit-heavy traces.
- Compose freely: quantisation is orthogonal to eviction; both are orthogonal to paging.

## Adjacent but distinct

- **Prompt compression** (LLMLingua, RECOMP) — shrinks the *input* text, not the cache. Complementary.
- **Speculative decoding** — reduces the number of forward passes, not the per-pass memory. Composes cleanly with cache compression.
- **Prefix caching** — reuses cache across requests; is a scheduler/allocator concern, not a compression method.

## Sources

- Xiao et al., *StreamingLLM* — 2023 — https://arxiv.org/abs/2309.17453
- Zhang et al., *H2O* — 2023 — https://arxiv.org/abs/2306.14048
- Liu et al., *KIVI: 2-bit Quantization of Key-Value Cache* — 2024 — https://arxiv.org/abs/2402.02750
- Kwon et al., *vLLM / Paged Attention* — 2023 — https://arxiv.org/abs/2309.06180
- Kim et al., *BeaconKV* — 2026 — https://arxiv.org/abs/2609.04971
