# Sparse Attention

*Taxonomy — attention layers that compute exact attention over a sparse subset of key-value positions, rather than all of them.*

**TL;DR:** Dense self-attention is $O(L^2)$ in sequence length $L$, which dominates compute at long context (100k+ tokens). Sparse-attention layers select **a small subset of key-value positions per query** and compute exact attention only over that subset — the rest are skipped, not approximated. Modern frontier-scale variants ([MiniMax Sparse Attention](minimax-sparse-attention.md), Native Sparse Attention, etc.) score *blocks* of KV cheaply with an index branch and then run exact attention on the top-scoring blocks. Drop-in for GQA / MLA at long context with no measured quality regression.

**Related taxonomies:** [_moe.md](_moe.md) (the other "sparse-activation" family — but over FFN experts, not KV positions).
**Depth files covered here:** [minimax-sparse-attention.md](minimax-sparse-attention.md) · [mla.md](mla.md) (compression-based long-context, adjacent)

---

## The problem

At sequence length $L$ and head dim $d$, dense attention costs $O(L^2 d)$ per layer and the KV cache costs $O(L)$ per layer per head. Both blow up past ~100k context — both for training and for serving. Approximating attention (linear attention, kernels) trades quality for cost; sparse attention instead **keeps attention exact** but **drops most of the entries** that would have contributed nothing anyway.

What goes wrong if you do it naively:

- **Static patterns under-attend.** Hand-coded sparse patterns (sliding window, dilated) miss the long-range hits attention is supposed to capture.
- **Learned selection is unstable.** Discrete top-$k$ selection isn't differentiable; surrogate scoring (softmax over candidates, Gumbel) introduces its own noise.
- **GPU efficiency.** Sparse attention beats dense only if the kernel is efficient on the irregular access pattern — typically requires *block*-sparse rather than per-token-sparse.

## The shared pattern

```
queries Q                  KV blocks B_1, ..., B_M
   │                              │
   ▼                              ▼
Index-Branch: score s_m = ScoreFn(Q, B_m)  (cheap; per-block)
                                  │
                       Select top-K' blocks per query (per group)
                                  │
                                  ▼
Main-Branch: standard softmax attention over selected blocks only
```

Every modern sparse-attention variant has two passes: an **index** pass that scores blocks cheaply, and a **main** pass that runs exact softmax attention over only the selected blocks. The variants differ in (i) what the index function is, (ii) how blocks are defined, (iii) whether selection is per-head or per-group ([GQA](multi-head-attention.md)-style amortization).

## Variants

| Technique | Key idea | Main tradeoff | When it wins |
| --- | --- | --- | --- |
| [**MiniMax Sparse Attention**](minimax-sparse-attention.md) (MiniMax, 2026) | Block-sparse over GQA: learned Index Branch scores KV blocks per group; Main Branch runs exact block-sparse attention | Index Branch is extra parameters and adds overhead; needs care to keep block selection differentiable | Frontier-scale (~100B) long-context (≥256k) training and serving |
| **Native Sparse Attention** (DeepSeek, 2025) — *no depth file yet* | Coarse + fine selection: token-cluster summaries pick coarse blocks, then per-token fine selection | More complex than single-level selection | Pretraining new models with sparse attention from scratch |
| **MoBA** (Moonshot, 2025) — *no depth file yet* | Hierarchical block routing — mixture-of-block-attention | Two-level routing introduces extra hyperparameters | Long-context fine-tuning of existing dense models |
| **Sliding window + sinks** (Mistral, StreamingLLM) — *no depth file yet* | Hand-coded local window + a few "attention sink" positions | Static pattern misses long-range hits | Cheap, robust default for streaming serving |

## How to choose

**Default for long-context pretraining (2026):** learned block-sparse with an index branch on top of GQA — [MiniMax Sparse Attention](minimax-sparse-attention.md) is the canonical example. Compatible with standard kernels at block granularity.

**Default for retrofitting existing dense models:** sliding-window + sink-tokens is the cheapest add-on; MoBA-style hierarchical routing if quality matters.

**Sparse vs. KV compression.** Sparse attention drops *positions*; KV compression ([MLA](mla.md)) keeps all positions but in a smaller representation. They compose: a frontier long-context recipe combines GQA + MLA-style compression + block-sparse selection.

## Adjacent but distinct

- **Linear attention / kernel attention** — *approximate* attention via factorization. Sparse attention is exact; linear is not.
- **MoE** ([_moe.md](_moe.md)) — also "sparse activation," but over FFN experts, not KV positions. Different problem, different mechanism.
- **Paged attention** — a serving-side memory layout, not an attention math change.

## Sources

- Paper: *MiniMax Sparse Attention* — MiniMax, 2026 — [arXiv:2606.13392](https://arxiv.org/abs/2606.13392).
- Paper: *Native Sparse Attention* — DeepSeek, 2025.
- Paper: *MoBA: Mixture of Block Attention* — Moonshot, 2025.
- Background: *Generating Long Sequences with Sparse Transformers* — Child et al., 2019.
