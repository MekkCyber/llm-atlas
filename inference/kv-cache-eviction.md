# KV Cache Eviction
*Depth — dropping cached key/value tokens to shrink the memory footprint of long-context decoding.*

**TL;DR:** During generation the KV cache grows linearly with context length and dominates VRAM. Eviction methods delete a chosen subset of cached tokens per attention head so decoding can continue in a bounded budget. Most published methods **score** each token's future importance (attention mass, recency, sink status, learned salience) and evict the lowest-scored — but Salesforce's *Random Attention* (2026) shows uniform random eviction matches the strongest scored method once the prompt is protected, thanks to redundancy in the reasoning trace and across heads.

**Prereqs:** [attention](../fundamentals/attention.md), [multi-head-attention](../architectures/multi-head-attention.md)
**Related:** [long-cot-rl](../post-training/reasoning/long-cot-rl.md)

---

## What it is

A serving-time compression on the per-head KV tensor. For head $h$ with cache of length $L$ and budget $B < L$, an evictor picks $B$ tokens to keep and drops the rest. Kept tokens continue to be attended to on future decode steps; evicted tokens are gone forever (unlike offloading, which trades memory for bandwidth). Some schemes reserve a **prompt** region that is never evicted, others reserve **attention sinks** (the first few tokens, whose keys act as an absorbing state for the softmax).

Distinct from prefix caching (reuse across requests), paged attention (physical layout), and quantized KV caching (fewer bits per token, no tokens dropped).

## How it works

The generic per-head, per-step recipe:

```
score(token_i) := f(attention_history_i, position_i, is_prompt_i)
keep := top_B({score(i) : i in cache})
if len(cache) > B: cache <- cache[keep]
```

Common scoring functions:

- **Recency / sliding window** — keep the last $B$ tokens.
- **Attention-mass (H2O, Scissorhands)** — keep tokens with high accumulated attention weight in recent decode steps.
- **Streaming-LLM** — always keep the first $k$ sink tokens plus a sliding window.
- **Learned salience** — a small predictor scores tokens.
- **Random Attention** — no scoring; sample $B - |\text{prompt}|$ tokens uniformly per head, always keep the prompt.

The Random Attention analysis identifies the failure mode of scored methods as *prompt loss*, not *bad selection*: when a scored evictor removes prompt tokens, quality falls; when it keeps the prompt, its choice among trace tokens barely matters, because both the text and the multi-head cache carry redundant copies of what the model still needs.

## Why it matters

The KV cache is the dominant memory line for long-context reasoning (long CoT, agent trajectories, retrieval-augmented decoding). A well-chosen budget $B \ll L$ can enable 32K–128K contexts on cards that would otherwise OOM, and improve throughput because per-step attention cost scales with $B$ not $L$. Vs. quantized KV, eviction loses information but frees VRAM for larger batches or longer contexts; the two compose.

The Random Attention result is a null result for a whole family of methods: expensive per-token scores may be paying for what redundancy already provides for free. If you are building a KV compressor, protect the prompt first, then measure whether your scorer beats random by more than seed noise — often it will not.

## Gotchas & tricks

- **Protect the prompt.** Every published evictor that beats a naïve top-$B$ over the whole cache in practice does so mainly by keeping the prompt region.
- **Per-head, not per-layer.** Redundancy across heads is what makes random eviction survive; if you evict the same positions in every head you lose that redundancy.
- **Reasoning helps eviction.** Long CoT restates intermediate state in text, so evictors survive it much better than tasks with dense one-shot answers.
- **Interaction with speculative decoding.** Evicting tokens that the draft model still attends to causes acceptance drops; keep draft/target evictors aligned.
- **Cache warmup.** Very early eviction (first few tokens after prefill) can lose attention sinks — reserve $k=4$ sink tokens explicitly.

## Sources

- Paper: *Random Attention: Rethinking KV Cache Eviction for Efficient Reasoning* — Wang et al., Salesforce AI Research, 2026 — the scoreless baseline result — https://arxiv.org/abs/2609.03430
- Paper: *H2O: Heavy-Hitter Oracle for Efficient Generative Inference of LLMs* — Zhang et al., 2023 — attention-mass scoring.
- Paper: *Efficient Streaming Language Models with Attention Sinks* — Xiao et al., 2023 — the sink phenomenon and Streaming-LLM.
- Paper: *Scissorhands: Exploiting the Persistence of Importance Hypothesis for LLM KV Cache Compression* — Liu et al., 2023.
