# HLA — Hybrid Linear Attention

*Depth — hybrid attention where the softmax/linear mix is chosen **per chunk, dynamically** by a small router.*

**TL;DR:** Fixed-ratio hybrid attention (every Nth layer softmax, the rest linear) overpays softmax on local content and underpays it on retrieval-heavy content. HLA partitions the sequence into fixed-size chunks and lets a lightweight router produce a per-chunk mixing weight between the linear and softmax branches, so softmax budget flows to the chunks that need it.

**Prereqs:** [linear-attention.md](./linear-attention.md), [multi-head-attention.md](./multi-head-attention.md)
**Related:** [mla.md](./mla.md), [_moe.md](./_moe.md), [attention.md](../fundamentals/attention.md)

---

## What it is

A hybrid attention block that computes **both** a linear-attention output and a softmax-attention output and takes a weighted combination per chunk:

```
chunk  c = tokens[t·C : (t+1)·C]           (chunk size C fixed)
α_c     = σ(router(h_c))                   (scalar mixing weight in [0, 1])
y_c     = α_c · softmax_attn(c) + (1 − α_c) · linear_attn(c)
```

The router is a cheap pooling + MLP on the chunk's hidden states. It is trained end-to-end — no auxiliary balance loss required, because the softmax branch is still bounded in cost: at full `α_c = 1` the chunk pays `C²` softmax, not `N²`.

## How it works

### Chunked mix

Chunk size `C` is a fixed hyperparameter (typical `C = 128` to `512`). Within a chunk the two branches operate on the same `C` tokens and the mix is applied at the chunk's output. Across chunks the softmax branch sees only local-chunk context; the linear branch's running state carries global context.

### The router

Per chunk, the router computes a single `α_c ∈ [0, 1]`:

```
summary_c = pool(hidden_c)                 (mean + max + first + last, say)
α_c       = σ(W_router · summary_c)
```

This is small enough to be essentially free. The paper reports that the router learns a non-trivial, interpretable allocation: retrieval-heavy and long-range-dependency chunks get `α` near 1; local-syntax chunks get `α` near 0.

### Cost accounting

Per-layer compute is roughly

```
Cost(HLA) ≈ (N/C) · C² · 𝔼[α]      (softmax, chunk-local)
          +  N · d²                  (linear, over whole sequence)
          =  N · C · 𝔼[α] + N · d²
```

With `C ≪ N` and `𝔼[α] < 1`, HLA stays sub-quadratic in `N`. Pure softmax would be `N²`; pure linear would be `N · d²` only.

### Training

HLA is trained end-to-end, from scratch or from a pretrained hybrid baseline. The router gets gradient through the `α · softmax + (1−α) · linear` combiner. No auxiliary load-balance loss is reported in the paper.

## Why it matters

- **Allocates attention precision where it's needed.** On RULER-style retrieval benchmarks HLA closes most of the gap to full softmax at a fraction of the cost; on local-syntax tasks it matches linear attention's throughput.
- **The pattern generalizes.** Per-chunk routed mixing of two attention operators is MoE-shaped — the "experts" are attention types, the "router" selects per-chunk. Same structure could blend any pair of attention operators (full/windowed, dense/sparse, etc.).

## Gotchas & tricks

- **Chunk size `C` is the main knob.** Small `C` lowers the softmax `C²` cost but amplifies router decisions; large `C` makes softmax expensive in high-`α` chunks. The paper recommends `C` in the 128–256 range for 100k-context inference.
- **Router collapse is a risk** at high `α` or low `α` ends. The paper regularizes with a mild diversity term implicitly through the loss landscape; keep an eye on average `α` during training.
- **Caching:** the softmax branch's KV cache is only within-chunk, so cache size is bounded. The linear branch's state is the usual `d²` running tensor. Together, HLA has a **bounded-per-chunk** KV footprint — a win over full softmax for long context.
- **Hybrid ratio compared to fixed-schedule designs.** Jamba / Zamba / Griffin use fixed layer alternation; HLA's per-chunk dynamic mix is a strictly finer-grained lever at the same compute.

## Sources

- Paper: *HLA: Expressive Hybrid Linear Attention via Chunk-Wise Dynamic Mixing* — Chen et al., Monash + Zhejiang, 2026 — [arXiv:2610.05842](https://arxiv.org/abs/2610.05842).
- Related: *Jamba, Griffin, Zamba* — fixed-ratio hybrid baselines.
- Related: *Gated Linear Attention Transformers with Hardware-Efficient Training* — Yang et al., 2024 — linear branch used by HLA.
