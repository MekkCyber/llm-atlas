# RAT+ — recurrence-augmented attention
*Depth — a transformer backbone with an exponentially-decaying recurrent memory alongside standard attention, enabling flexible dilated attention and friendlier KV sparsity.*

**TL;DR:** RAT+ bolts an exponentially-decaying recurrent memory state onto the standard attention block. The memory subsidises information that vanilla attention would otherwise need a dense KV cache to access, enabling flexible dilated attention at inference time and improving accuracy under query-aware KV sparsity (Quest, MoBA, SnapKV) at every budget. Validated on released RAT+ checkpoints and on OLMo2-7B continually pretrained for 10B tokens with the memory module added.

**Prereqs:** [attention](../fundamentals/attention.md), [multi-head-attention](multi-head-attention.md)
**Related:** [mla](mla.md), [query-aware-kv-sparsity](../inference/query-aware-kv-sparsity.md)

---

## What it is

A backbone variant that augments each attention block with a parallel recurrent path carrying an exponentially-decaying memory state. The block's output sums (or otherwise combines) a standard attention term and a memory term, where the memory is updated by a cheap recurrence over the token stream.

## How it works

For each layer, alongside the standard self-attention `Attn(Q, K, V)`:

```
m_t = α · m_{t-1} + (1 − α) · f(x_t)     # exponentially-decaying recurrent state
y_t = Attn(Q_t, K_{≤t}, V_{≤t}) + g(m_t) # combine standard + memory term
```

The decay `α` is per-head (and/or per-channel) and learned. The memory term costs O(d) per step regardless of sequence length — much cheaper than attention's O(seq × d).

Key implication: tokens that are far in the past are still represented (in compressed form) inside `m_t`, even if their KV positions are dropped by sparse inference.

## Why it matters

- **Backbone-side amplifier for sparse inference.** Query-aware KV sparsity methods drop information; the recurrent memory partially recovers it. Reported gains: consistent accuracy improvements across Quest / MoBA / SnapKV at all sparse budgets, on 8 needle-in-a-haystack tasks.
- **Cheap to bolt on.** OLMo2-7B continually pretrained with the memory module for 10B tokens already shows the gains — no from-scratch retraining needed.
- **Enables flexible dilated attention at inference**, since the memory carries the global context that dilation skips.

## Gotchas & tricks

- **Decay schedule matters.** Too aggressive a decay forgets long-range structure; too gentle and the memory becomes a constant.
- **Stability during continual pretraining.** Adding the memory mid-training risks destabilising activations; warm-up schedules on the memory term help.
- **Composes with KV-cache compression.** RAT+ doesn't conflict with [MLA](mla.md)-style KV compression — the two attack the cost from different angles.

## Sources

- Paper: *RAT+* — Wei & Gulcehre — 2026 — origin of the architecture.
- Paper: *Augmenting Attention with Exponentially Decaying Memory Improves Query-Aware KV Sparsity* — Wei & Gulcehre — 2026 — [arXiv:2605.28640](https://arxiv.org/abs/2605.28640) — sparse-inference analysis + OLMo2-7B validation.
