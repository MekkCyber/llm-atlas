# Prompt compression
*Depth — reducing long-context input size by scoring and dropping less-important units.*

**TL;DR:** Long-context inference cost grows with input length. Prompt compression scores tokens, sentences, or chunks by importance under a budget, keeping the highest-scoring units and dropping the rest. "Hard" compression drops units entirely (vs "soft" compression, which distills the whole context into a shorter representation). Standard families rank chunks with a small embedding model (Beaver-style Qwen3-0.6B-embedding scoring) or a specialized compressor LLM (LLMLingua-family).

**Prereqs:** [README.md](README.md), [../fundamentals/attention.md](../fundamentals/attention.md)
**Related:** [referential-dangling.md](referential-dangling.md), [../architectures/mla.md](../architectures/mla.md)

---

## What it is

At long context, most tokens contribute little to the answer for a given query. Prompt compression exploits this by ranking units of the input (tokens / sentences / chunks) by an importance score, then keeping only the top-scoring units under a length budget. Serving cost drops roughly linearly with the compression ratio.

Two paradigms:
- **Hard compression:** drop unranked units entirely; downstream LLM sees the compressed prompt as if it were the original.
- **Soft compression:** compile the context into a shorter latent / summary representation (survey embedding).

This depth file covers hard compression, which is what production long-context serving pipelines mostly use.

## How it works

Given input $X$ (tokens $t_1, \dots, t_n$), a query $q$, and a compression ratio $\rho \in (0, 1]$:

1. **Segment.** Break $X$ into units — tokens, sentences, or coherent chunks. Chunk-level is the standard trade-off (chunks preserve local coherence while keeping the scoring problem tractable).
2. **Score.** Assign each unit $u_i$ an importance score $s_i$. Common scorers:
   - **Perplexity-based:** score by how much the unit reduces query-conditional perplexity (LLMLingua family).
   - **Embedding-similarity:** dot-product similarity of unit embedding to query embedding under a small embedding model (Beaver / Qwen3-0.6B).
   - **Attention-based:** for tokens, use attention weights from a probe model.
3. **Select.** Keep the top-scoring units up to the budget $\rho n$. Preserve original order for the downstream LLM.
4. **Concatenate.** Emit the compressed prompt.

Scoring is done independently per unit — units don't know about each other's scores. This is exactly what makes compression fast and what causes [referential-dangling](referential-dangling.md).

## Why it matters

- **Linear cost reduction.** At $\rho = 0.3$, decoder attention cost drops by 3× (or more if KV-cache-limited).
- **Deployable on any served model.** Compression sits in front of any downstream LLM; no training required.
- **Composable with KV-cache compression** and streaming-friendly serving.

## Gotchas & tricks

- **Independent scoring breaks reference chains.** When one unit defines an entity and another uses it, keeping only one severs the reference — see [referential-dangling](referential-dangling.md). Rate on multi-hop QA can hit 34–54% at $\rho = 0.3$ for Beaver.
- **Chunk boundary sensitivity.** Compression quality depends on whether your chunker keeps sentences / bridging references together.
- **Query-dependent vs query-independent.** Query-dependent scoring is better per-query but blocks caching; query-independent scoring is worse but pre-computable.
- **Compression ratio has a knee.** $\rho > 0.5$ usually preserves quality; $\rho < 0.2$ typically breaks. The middle band ($0.3$–$0.5$) is where per-task tuning matters.
- **Soft compression is the alternative when quality matters more than latency** — but requires training a compressor and gives up interpretability.

## Sources

- Paper: *Referential Dangling as a Paradigm-Level Failure Mode in Hard Prompt Compression* — Hu et al., 2026 — arXiv:2608.04569 — the failure-mode analysis of the paradigm.
- Prior: *LLMLingua* — Jiang et al., 2023 — perplexity-based compressor family.
- Prior: *LongLLMLingua*, *Selective Context* — subsequent variants in the perplexity-scoring line.
