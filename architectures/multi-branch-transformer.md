# Multi-Branch Transformer
*Depth — a transformer scaling axis orthogonal to width and depth.*

**TL;DR:** Instead of making a transformer block wider or stacking it deeper, run `B` parallel transformer blocks on the same residual stream and sum their outputs back. Branches learn diverse features without MoE routing. A **Joint-KV** pooled cache keeps decode memory from scaling with `B`.

**Prereqs:** [transformer-block](transformer-block.md), [multi-head-attention](multi-head-attention.md)
**Related:** [_moe](_moe.md), [hyper-connections](hyper-connections.md)

---

## What it is

A **block-level composition primitive**. A standard transformer stage is one block per residual stream position. MBT replaces it with `B` independent blocks (each with its own attention and FFN parameters) that read the same residual, compute in parallel, and whose outputs are summed before the residual add. It is **not** a mixture of experts: every branch runs on every token.

MBT gives three scaling axes instead of two:
- **Depth** — number of stages.
- **Width** — hidden size / FFN inner size.
- **Branches** — parallel blocks per stage.

## How it works

At stage `l`:

```
h ← residual stream
for b in 1..B:  (parallel)
    y_b = Attn_b(h) + FFN_b(h)      # each branch has its own weights
h ← h + sum_b y_b
```

Each branch is a full transformer block. Parameters scale linearly in `B`; wall-clock does not, since branches are embarrassingly parallel.

**Joint-KV.** A naive cache would hold one KV per branch — a factor-`B` memory blow-up. Joint-KV projects every branch's K/V through a shared pooled space; the attention logits mix branch-specific queries with the shared cache. This keeps decode memory the same as a single-branch transformer at the cost of some branch specialization.

## Why it matters

Width-vs-depth scaling has diminishing returns on reasoning-heavy evals. Branches are a cheap third knob: at matched training compute, MBT improves language-modeling quality vs. a baseline transformer, and Joint-KV plus parallel branch execution raises decoding throughput on a single H200. The axis is *compute-additive* (branches run together) rather than *compute-multiplicative* (depth adds latency), which is attractive for latency-sensitive serving.

## Gotchas & tricks

- **Branches drift.** Without regularization, branches tend to co-adapt and lose their diversity advantage. The paper uses standard dropout + initialization scaling to keep them differentiated.
- **Joint-KV vs. independent KV is a quality/memory knob.** Joint-KV is cheaper but gives up some branch-specific attention. For training, independent KV wins; for serving, Joint-KV is almost always the right choice.
- **Not MoE.** Every branch runs every step — this is dense compute, unlike MoE's routed sparse compute. The two can compose (MoE inside each branch).
- **Interacts with hyper-connections.** Both are residual-stream variations; combining branching with multi-stream hyper-connections is unexplored.

## Sources

- Paper: *Multi-Branch Transformer* — Li, Chen, Deng, Xu, Chen, De Sa, He, Wang, Tong (UIUC / Rice / Michigan / Cornell), 2026 — [arXiv:2611.19800](https://arxiv.org/abs/2611.19800)
