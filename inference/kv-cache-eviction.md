# Query-Agnostic KV Cache Eviction
*Depth — serving-time KV cache compression that reuses the same compressed cache for arbitrary future queries.*

**TL;DR:** During long-context serving, the KV cache blows up linearly with context length and dominates GPU memory. **KV cache eviction** trims the cache by dropping less-important (K, V) pairs after prefill. The *query-agnostic* variant compresses **once**, then reuses the compressed cache for *arbitrary* future queries — critical for chat, RAG, and agent scenarios where a shared long context is queried many times. The trade-off: tighter budgets cause abrupt quality collapse, since you can no longer know at compression time which context was going to be relevant to the query. Recent work (RestoreKV, 2026) shows the *selection-based* framing has been leaving quality on the table — a small learned reconstruction head can recover most of it.

**Prereqs:** [../fundamentals/attention.md](../fundamentals/attention.md)
**Related:** [../architectures/mla.md](../architectures/mla.md) · [../architectures/multi-head-attention.md](../architectures/multi-head-attention.md)

---

## What it is

The KV cache stores $(K_l, V_l)$ tensors for every layer $l$ and every past token position, so that decoding step $t+1$ can attend to positions $1..t$ without re-running attention on the whole prefix. Memory is $O(L \cdot T \cdot d \cdot 2)$ per sequence — for a 32K context on a mid-size model it's tens of GB.

Two regimes of KV eviction:

- **Query-aware.** Given the *current query*, decide which cached positions to keep. Great quality at a given budget; only works for the current query — the compressed cache is not reusable for the next question about the same context.
- **Query-agnostic.** Decide once at prefill time which positions to keep, then reuse the compressed cache for *any* future query. Only regime that composes with cache reuse across turns / users. Where the pressure lives.

Base signals for what to keep: attention-magnitude scores (H2O, TOVA), attention entropy, position priors (StreamingLLM's attention sinks + recency window), or hybrid scoring (KVzip, SnapKV).

## How it works

The **selection-based** template:

1. **Prefill** the long context through the model, computing full attention. Collect per-token importance scores from a chosen signal (accumulated attention weights, magnitude, layer-specific priors).
2. **Score.** Aggregate scores into a per-position importance vector.
3. **Evict.** Keep the top-$k$ positions per layer / per head (subject to structural constraints like keeping the first few "sink" tokens and a recent window). Drop everything else.
4. **Reuse.** All future queries over this context attend only to the retained KV pairs.

The compression ratio and layer-wise budget schedule are the two main design knobs. Some methods keep the same budget per layer (H2O); others allocate hierarchically (KVzip, SnapKV).

The **selection-plus-restoration** template (RestoreKV, 2026) adds one more step:

3.5. **Restore.** After prefill, a small number of *restore tokens* attend to the full pre-eviction KV cache in a single LoRA-adapted pass, producing a compact, context-conditioned *restore cache* held alongside the retained KV pairs. Trained by parameter-efficient self-distillation from the frozen full-cache model — 0.4% of parameters, no task-specific tuning. The base scorer and eviction rule are unchanged; the restore cache is a shared complement.

## Why it matters

- **Serving-time reusability.** Query-aware compression forces you to re-compress on every query. Query-agnostic compression is what makes long-context multi-turn chat and RAG affordable at all.
- **Hard budget regime.** At 5–10% retention budgets, selection-only methods degrade sharply on retrieval-style long-context evaluations (RULER, LongBench). This is where restoration helps most.
- **RestoreKV's numbers.** Across 4 backbones × 4 long-context benchmarks: improved 59 of 60 paired budget-matched settings vs five base eviction methods. On Qwen3-4B at a 5% budget, raised KVzip from **38.2 → 73.2** on RULER-4K. Applied to KVzip+, reached **86.4** RULER accuracy at 16× compression with <0.5% one-time overhead in 32K-context evaluation.
- **Compositional with design-time KV compression.** MLA (design-time), quantized cache (post-training), and eviction (serving-time) all attack KV bloat at different layers and stack.

## Gotchas & tricks

- **Attention sinks matter.** Dropping the first few positions collapses many models (Xiao et al., StreamingLLM). Every eviction method leaves them in — treat this as an invariant, not an optimization.
- **Layer-wise heterogeneity.** Not all layers need the same budget. Later layers tend to be more retrieval-critical; naive uniform eviction over-compresses them.
- **Budget vs benchmark type.** Query-agnostic eviction is far worse on retrieval / passkey benchmarks than on summarization / language modeling — retrieval punishes any lost position that happens to hold the answer. Report both.
- **Reference-free eviction is hard to evaluate.** Query-agnostic methods can look great on benchmarks where the query happens to align with the retained positions. Paired budget-matched evaluation across many query distributions is the right protocol.
- **RestoreKV's <0.5% overhead is one-time.** It's paid at prefill (during the restore-token pass); decoding uses the frozen cache. Don't confuse with a per-step overhead.

## Sources

- Paper: *H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models* — Zhang et al., 2023 — [arXiv 2306.14048](https://arxiv.org/abs/2306.14048). Foundational attention-based eviction.
- Paper: *Efficient Streaming Language Models with Attention Sinks (StreamingLLM)* — Xiao et al., 2023 — [arXiv 2309.17453](https://arxiv.org/abs/2309.17453). The attention-sink invariant.
- Paper: *SnapKV: LLM Knows What You are Looking for Before Generation* — Li et al., 2024 — [arXiv 2404.14469](https://arxiv.org/abs/2404.14469). Prefill-time scoring for eviction.
- Paper: *KVzip: Query-Agnostic KV Cache Compression* — 2025. Strong query-agnostic baseline.
- Paper: *RestoreKV: Recovering Full-Cache Behavior Under Aggressive Query-Agnostic KV Cache Eviction* — 2026 — [arXiv 2608.01247](https://arxiv.org/abs/2608.01247). Selection + learned restoration.
