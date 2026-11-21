# Sparse Attention

*Taxonomy — attention mechanisms that compute a *subset* of query-key interactions to escape the $O(T^2)$ cost of full attention.*

**TL;DR:** Full attention is quadratic in sequence length $T$, which becomes the dominant cost past ~32k tokens. Sparse attention mechanisms restrict each query to a chosen subset of keys — a fixed local window, a learned top-$k$ of blocks, hash-bucketed neighbours — trading recall of long-range interactions for sub-quadratic compute and smaller KV traffic. Modern variants cluster into **block-sparse routed** attention (MoBA, Native Sparse Attention), **structured sliding-window + landmark** designs (Mistral, Jamba), and **learned-dense retrieval at inference** (Quest, MInference).

**Related taxonomies:** [_moe.md](_moe.md) — the MoE/sparse-attention design mindset is analogous: a cheap router picks a subset, then full-strength computation on the subset.
**Depth files covered here:** [moba](moba.md) · [flashmorph2](flashmorph2.md)

---

## The problem

- **FLOPs: $O(T^2 d)$.** At 128k tokens, attention dominates both training and inference FLOPs.
- **KV memory bandwidth.** Decode reads the full KV cache per token; even when FLOPs are fine, memory bandwidth isn't.
- **Long-context quality.** Models trained on 2k–8k windows degrade past the window; retrofitting long context without a sparsity structure is expensive and fragile.

Sparse attention attacks the first two directly. A good sparse mechanism also preserves the third by giving the router enough recall.

## The shared pattern

Every variant has three parts:

1. **Partitioning.** How keys/values are grouped — fixed-size blocks, sliding windows, hash buckets, retrieval-indexed chunks.
2. **Routing.** How each query picks which partitions to attend to — top-$k$ by block score, threshold, hash collision, learned classifier.
3. **Scheduling.** How the kernel iterates — query-centric (one query, many blocks) vs KV-centric (one block, many queries).

Variants differ on each axis and in whether routing is **trained** (MoBA, NSA) or **inferred post-hoc** (Quest, MInference).

## Variants

| Technique | Key idea | Main tradeoff | When it wins |
| --- | --- | --- | --- |
| **Sliding-window attention** (Mistral 2023, Longformer) — *no depth file yet* | Each query attends to the last $w$ keys | No long-range recall without landmark heads | Default for mid-range (16k–64k) when quality loss is tolerable |
| **Block-sparse routed (MoBA)** — [moba](moba.md) | Learned top-$k$ KV-block router, trained end-to-end | Needs training with the sparsity; block-granular granularity | Long context (≥128k) with quality parity to dense |
| **Native Sparse Attention (NSA)** (DeepSeek 2025) — *no depth file yet* | Hierarchical cluster + top-$k$ routing, end-to-end trained | Similar; different partitioning | Long context; close MoBA cousin |
| **MInference / Quest** (MSR 2024, Infinigen 2024) — *no depth file yet* | Learned sparsity patterns at inference time only — no retraining | Pattern quality capped by base model's attention entropy | Retrofitting existing dense models for long context |
| **Hybrid full + linear attention** — [flashmorph2](flashmorph2.md) | Keep a few layers full-attention, swap the rest for a linear-attention variant | Needs recall-sensitive layer selection | Serving-time compromise between dense quality and linear's throughput |
| **Big Bird / Longformer** (2020) — *no depth file yet* | Global + random + window sparsity patterns | Pre-training with the pattern | Historical; superseded by learned-router variants |

## How to choose

- **Default for 128k+ context, training from scratch:** MoBA or NSA. Both route at the block level with a trained scorer; both are kernel-friendly on Hopper.
- **Retrofitting a dense model:** Quest / MInference-style post-hoc sparsity — no retraining, but caps out below trained sparse models on quality.
- **Serving-time cost-sensitive, quality-insensitive:** sliding-window + a handful of full-attention layers (Jamba-style). Cheap and simple.
- **Mid-depth hybridisation:** [flashmorph2](flashmorph2.md) picks which Transformer layers to keep full-attention and swaps the rest for linear-attention variants, trained on-policy. Good when you have a full pretrained model and want to cut cost post-hoc.

## Adjacent but distinct

- **Linear attention / state-space models** (Mamba, RWKV, GLA). These don't "sparsify" full attention — they replace the attention operator with a recurrence. Distinct category; hybrid designs ([flashmorph2](flashmorph2.md)) mix the two.
- **Mixture-of-Experts** ([_moe.md](_moe.md)). Same "route to a subset" mindset, but applied to FFN weights, not attention keys.
- **KV-cache compression / quantisation.** Reduces per-key cost without changing which keys are attended to. Composes with sparse attention.
- **Speculative decoding** ([../inference/_speculative-decoding.md](../inference/_speculative-decoding.md)). Reduces per-step forward count, not per-step cost — orthogonal lever.

## Sources

- Paper: *MoBA: Mixture of Block Attention for Long-Context LLMs* — Kimi, 2025.
- Paper: *Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention* — DeepSeek, 2025.
- Paper: *Quest: Query-aware Sparsity for Efficient Long-Context LLM Inference* — Tang et al., 2024.
- Paper: *MInference 1.0: Accelerating Pre-filling for Long-Context LLMs via Dynamic Sparse Attention* — Microsoft Research, 2024.
- Paper: *Mistral 7B* — Jiang et al., 2023 — sliding window at scale.
- Paper: *On-Policy Morphing into Hybrid Attention Models* — Wang et al., ByteDance Seed / NTU, 2026 — the hybrid-selection recipe.
- Paper: *Accelerating Mixture of Block Attention via Dynamic Routing, KV-Centric Execution and Runtime Pipelining* — Tencent, 2026 — threshold routing + scheduling for MoBA.
