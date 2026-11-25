# KV-Cache Eviction

*Taxonomy — methods for throwing away cached KV entries when the budget is below the sequence length.*

**TL;DR:** Long contexts blow the KV-cache budget; eviction policies decide which tokens to drop. The field has settled around two anchors — H2O (score by accumulated attention) and StreamingLLM (keep a recency window plus a few attention sinks) — and NoHairCache (2026) unifies them as endpoints of a single power-law family controlled by one age exponent. The modern take: calibrate once per model, then evict with the best $\alpha$.

**Related taxonomies:** —
**Depth files covered here:** —

---

## The problem

Autoregressive decode caches each position's key and value vectors so attention at step $t$ reads $O(t)$ entries. For context length $N$ and heads $H$, the KV cache grows as $O(NH)$ per layer — at 128K context and 70B params, this is tens of GB. Serving frameworks cap the budget and must evict.

Evicting hurts quality in two ways: the model loses access to dropped tokens, and attention patterns can become degenerate if critical tokens (often early "attention sinks") are removed. Every variant below is a different scoring rule for *which* tokens to drop first.

## The shared pattern

All variants score cached tokens with some rule $s_i$ and keep the top-$B$ by score, where $B$ is the cache budget. The scoring rule has two inputs:

- **How much attention token $i$ has attracted** (cumulative attention mass, $A_i$).
- **How old token $i$ is** (positions since arrival, $t - i$).

Different policies weigh these two differently.

## Variants

| Technique | Key idea | Main tradeoff | When it wins |
| --- | --- | --- | --- |
| H2O (no depth file yet) | Score by accumulated attention: $s_i = A_i$ | Early attended tokens survive forever; recent irrelevant tokens evicted first | Workloads with a few persistent important tokens (code, instructions) |
| StreamingLLM (no depth file yet) | Keep a recency window + fixed attention sinks | Simple, no scoring needed; drops anything old except sinks | Streaming / infinite-context workloads where recent tokens dominate |
| NoHairCache (no depth file yet) | Power-law score $s_i = A_i \cdot (t - i)^{-\alpha}$, with one $\alpha$ calibrated per model | Needs one-time calibration, generalizes across datasets | Everything in between; uniformly matches or beats H2O and StreamingLLM |

## How to choose

**Modern default:** NoHairCache with $\alpha$ calibrated on a short WikiText run. The published measurements (Qwen2.5-1.5B-Instruct: $\alpha \in [0.556, 0.822]$) are consistent across five models (410M–1.5B) and transfer across target datasets without re-tuning.

Use **plain StreamingLLM** when you want zero calibration overhead and a strictly streaming workload — recency-only loses ~7 points on NarrativeQA (91.1% vs 98.2% full-cache retention at 25% budget) but is one parameter simpler.

Use **plain H2O** when you have workloads with very long-lived attended tokens and short recency windows hurt (e.g., long-form reasoning over a specification). In the NoHairCache analysis this corresponds to $\alpha = 0$.

**Combinations:** All three are orthogonal to paged attention, quantization, and speculative decoding — the eviction policy is a budget filter, not a serving mechanism.

## Adjacent but distinct

- **KV compression** (e.g., per-token quantization, multi-head KV tying) reduces per-entry size rather than count. Composes with eviction.
- **MLA / GQA** change the KV footprint architecturally — fewer heads to cache in the first place.
- **Sliding-window attention** at training time hard-codes a recency window into the model, rather than evicting at inference. StreamingLLM approximates this post-hoc for models that weren't trained with it.

## Sources

- Paper: H2O: Heavy-Hitter Oracle for Efficient Generative Inference of LLMs — Zhang et al., 2023 — https://arxiv.org/abs/2306.14048
- Paper: Efficient Streaming Language Models with Attention Sinks (StreamingLLM) — Xiao et al., 2023 — https://arxiv.org/abs/2309.17453
- Paper: NoHairCache: Calibrate-Once KV Cache Eviction with One Power-Law Exponent — 2026 — [arXiv:2611.29006](https://arxiv.org/abs/2611.29006)
