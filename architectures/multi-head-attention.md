# Multi-Head Attention
*Depth — splitting attention into parallel specialized heads.*

**TL;DR:** Instead of one attention operation with full dimension, run `h` parallel attention operations each with a smaller dimension, then concatenate and project. Different heads learn to specialize — one for syntax, one for coreference, one for position — which is the inductive bias that makes Transformers actually work.

**Prereqs:** [attention](../fundamentals/attention.md)
**Related:** [transformer-block](transformer-block.md)

---

## What it is

Given input `X ∈ ℝ^(n × d_model)`:

1. Project into `h` separate `(Q, K, V)` triples, each of dimension `d_k = d_model / h`.
2. Run scaled dot-product attention independently on each head.
3. Concatenate the `h` outputs back to `d_model`.
4. Apply a final output projection `W_O`.

```
head_i = Attention(X W_Q^i, X W_K^i, X W_V^i)
MultiHead(X) = Concat(head_1, ..., head_h) W_O
```

Total compute is roughly the same as one full-dimension attention — the splitting is free in FLOPs.

## How it works

In practice, the `h` heads are implemented as a single big matmul with a reshape:

- Project `X → QKV` of shape `(n, 3 · d_model)`.
- Reshape to `(h, n, d_k)` per Q, K, V.
- Run batched attention across the head dimension.
- Reshape back to `(n, d_model)`.
- Multiply by `W_O`.

So "multi-head" is really "one attention computation with an extra batch dimension."

## Why it matters

- **Specialization.** Different heads learn different functions. Interpretability research (Olsson et al., induction heads; Elhage et al., mathematical framework) depends entirely on this fact. Remove it and you can't do circuit analysis the same way.
- **Subspace attention.** A single high-dim attention can only form one similarity structure at a time. `h` heads let the model attend to `h` different kinds of relationships in parallel.
- **Enables later efficiency tricks.** MQA and GQA (below) only make sense because attention is already split into heads — they share K/V across groups of heads.

## Variants you need to know

- **Multi-Query Attention (MQA)** — one K and one V head shared across all Q heads. Cuts KV cache size by `h×`, which dominates memory at inference. Quality cost is noticeable at scale.
- **Grouped-Query Attention (GQA)** — the compromise now standard in modern LLMs (LLaMA 2+, Mistral, Qwen). `h` Q heads share K/V across `g` groups (e.g., 32 Q heads, 8 KV heads). ~Same quality as full multi-head, ~4× smaller KV cache.
- **Multi-head Latent Attention (MLA)** — DeepSeek's compression of KV into a low-rank latent. Dramatically smaller KV cache, used in DeepSeek-V2/V3.

## Gotchas & tricks

- **Head count vs. head dim tradeoff.** More heads with smaller `d_k` = more specialization but each head has less capacity. The typical choice is `d_k = 64` or `128`; head count follows from `d_model / d_k`.
- **`W_O` is not optional.** Without the output projection, the concat is a fixed splicing of subspaces. `W_O` lets heads recombine.
- **KV cache memory scales with number of KV heads**, not Q heads. This is why MQA/GQA exist and why choosing the right group count matters for serving cost.
- Interpretability claims ("this head does X") are easy to overfit. Verify with ablations and multiple prompts.

## Sources

- Paper: *Attention Is All You Need* — Vaswani et al., 2017.
- Paper: *Fast Transformer Decoding: One Write-Head is All You Need (MQA)* — Shazeer, 2019.
- Paper: *GQA: Training Generalized Multi-Query Transformer Models* — Ainslie et al., 2023 — https://arxiv.org/abs/2305.13245
- Paper: *DeepSeek-V2 (MLA)* — DeepSeek, 2024.
- Paper: *In-context Learning and Induction Heads* — Olsson et al., 2022 — why head specialization matters.
