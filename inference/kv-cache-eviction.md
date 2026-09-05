# KV cache eviction
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Long-CoT inference blows the KV cache past the memory budget, so serving stacks drop tokens partway through generation. Every "smart" evictor scores cached tokens by expected future importance and keeps the top-k. **Random Attention** (Wang et al., 2026) shows that once the prompt is protected, uniform random eviction inside each attention head *matches* the strongest score-based evictor across four models × six reasoning tasks — with 32–43% higher throughput because it computes no score. The default assumption in the literature (importance scoring is worth its cost) is thus falsified.

**Prereqs:** [attention](../fundamentals/attention.md), [multi-head-attention](../architectures/multi-head-attention.md)
**Related:** [long-cot-rl](../post-training/reasoning/long-cot-rl.md), [mla](../architectures/mla.md)

---

## What it is

During autoregressive decoding, each layer caches one K and V per past token per head. For long reasoning traces this grows into tens of GiB per sequence. KV eviction chooses, at each step, which cached tokens to *drop* so the cache stays under a budget. The design axis: what signal do you evict by?

Prior evictors used variants of "attention-weight × recency" scores (H2O, SnapKV, Scissorhands, PyramidKV) — token *i* is evicted when its accumulated attention received from later tokens is low. The intuition: unattended tokens won't be needed.

Random Attention's contribution is a null baseline that isn't a null: **keep the prompt in full, then evict uniformly at random within each head, independently per layer**. No scoring, no ranking, no state.

## How it works

At each decode step, for every head and every layer:

```
budget_per_head = B
if cache_size[head] > B:
    # protect the prompt; only decode-generated tokens are eligible
    candidates = decode_tokens[head]
    evict_n = cache_size[head] - B
    drop uniformly at random from candidates
```

Two design choices matter:
- **Prompt protection.** Never evict prompt tokens; only decode-time tokens are eligible. Removing this collapses quality — the prompt carries the task and few-shot examples.
- **Per-head independence.** Each head runs its own uniform draw. Heads whose induction/retrieval patterns want specific tokens keep them in *some* head by chance; different heads keep different tokens, so the ensemble still covers most of the useful signal.

Because there is no scoring pass, decoding stays kernel-bound on the matmul; no auxiliary scan or top-k on the K-tensor is needed. That is where the 32–43% throughput gain over the best score-based evictor comes from.

## Why it matters

Prior evictors bake an assumption — "importance is estimable and worth estimating" — into every kernel. Random Attention shows the assumption is empirically wrong for reasoning workloads. Once the prompt is safe, the cache has enough redundancy across tokens and heads that specific-token identity barely matters. This reframes the KV-eviction problem: the interesting axes are (i) *what to protect* (prompt vs. summary vs. anchor tokens), and (ii) *the head-independence trick*, not the scoring function.

For deployed inference stacks, this is nearly a free win — you can rip out the scoring pass in vLLM's cache manager and gain throughput at the same quality.

## Gotchas & tricks

- **Do protect the prompt.** Uniform eviction over *all* tokens tanks quality. The prompt slot is load-bearing.
- **Per-head, not global.** A single global random-drop matches poorly; head-independent draws are what makes the ensemble robust.
- **This is not a general "attention doesn't matter" result.** The finding is specifically about cache eviction under the "top-k relevance" family of scorers. Different eviction primitives (block-wise, offloading, prefix-caching) aren't tested here.
- **Untested with sinks / streaming.** Sink-token schemes (StreamingLLM) protect the first few tokens too; Random Attention's protect-the-prompt rule already subsumes this for typical templates but hasn't been benchmarked against streaming-with-sinks.

## Sources

- Paper: *Random Attention: Rethinking KV Cache Eviction for Efficient Reasoning* — Wang, Qiu, Zhao, Qian, Yang, Chen, Han, Ji, Savarese, Heinecke, Wang, 2026 — [arXiv:2609.03430](https://arxiv.org/abs/2609.03430).
- Related prior work: H2O, SnapKV, Scissorhands, PyramidKV — score-based evictors this paper competes against.
