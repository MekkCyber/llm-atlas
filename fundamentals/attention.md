# Scaled Dot-Product Attention
*Depth — the attention operation from the original Transformer.*

**TL;DR:** A mechanism that lets every token in a sequence look at every other token directly and mix their information, weighted by learned similarity. It replaces recurrence with a fully parallel operation where any two positions are one hop apart.

**Prereqs:** none (this is a foundational primitive)
**Related:** [multi-head-attention](../architectures/multi-head-attention.md), [transformer-block](../architectures/transformer-block.md), [positional-encoding](positional-encoding.md)

---

## What it is

Given a sequence of token representations, attention computes a new representation for each token as a weighted combination of all tokens' values, where the weights come from how similar each token's *query* is to every other token's *key*.

The formula:

```
Attention(Q, K, V) = softmax(QKᵀ / √d_k) V
```

- `Q`, `K`, `V` ∈ ℝ^(n × d) are linear projections of the input sequence — queries, keys, values.
- `QKᵀ` is an `n × n` matrix of similarity scores between every (query, key) pair.
- Softmax over each row turns scores into a probability distribution.
- Multiplying by `V` produces a weighted sum of values for each query.

**Causal (masked) attention**: for autoregressive decoding, set the upper triangle of `QKᵀ` to `-∞` before softmax so position `i` can only attend to positions `≤ i`.

## How it works
"
1. Project input `X ∈ ℝ^(n × d_model)` into three matrices: `Q = XW_Q`, `K = XW_K`, `V = XW_V`.
2. Compute raw scores: `S = QKᵀ` (shape `n × n`).
3. Scale: `S / √d_k`. Without scaling, dot products grow with dimension, softmax saturates, gradients vanish. This step is load-bearing.
4. Optional mask (causal, padding): add `-∞` to disallowed positions.
5. `A = softmax(S)` row-wise — now each row sums to 1.
6. Output: `O = AV` (shape `n × d`).

Complexity: **O(n² · d)** in time and memory — this is the quadratic cost that every efficiency paper since (FlashAttention, paged attention, linear/sparse attention, Mamba) fights.

## Why it matters

- **Parallelism.** Unlike RNNs, every position is computed simultaneously — GPUs get saturated.
- **Path length.** Any two tokens are 1 attention hop apart vs. O(n) through an RNN. Long-range dependencies stop being a bottleneck.
- **Scales cleanly.** No hidden assumption that breaks at large size — this is what made GPT-style scaling possible.

Every modern LLM is, at its core, a stack of attention operations. Everything else is plumbing around them.

## Gotchas & tricks

- **Forgetting `√d_k`** — without the scale, training becomes unstable or stalls at high dim.
- **Numerical stability**: the softmax needs to subtract the row max before exp (standard trick). FlashAttention's clever part is doing this *tiled* without materializing the full `n × n` matrix.
- **Attention is permutation-equivariant** — it has no idea of token order. You must inject position information separately (see [positional-encoding](positional-encoding.md)).
- **Attention scores ≠ feature importance.** A tempting interpretation but often misleading — the value vectors and subsequent layers matter too.

## Sources

- Paper: *Attention Is All You Need* — Vaswani et al., 2017 — https://arxiv.org/abs/1706.03762
- Annotated: *The Annotated Transformer* — Harvard NLP — http://nlp.seas.harvard.edu/annotated-transformer/
- Illustrated: *The Illustrated Transformer* — Jay Alammar — https://jalammar.github.io/illustrated-transformer/
