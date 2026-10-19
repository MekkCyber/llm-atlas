# Copy-Paste Language Models (CPLM)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A lightweight head on a standard LM that predicts the next token by **pointing to an earlier position in the context** rather than scoring a vocabulary distribution. Interpolated with the standard LM head via a learned gate. The pointer path trains fast, improves perplexity and long-context retrieval, and matches full-attention performance under sliding-window attention — because it adds a second gradient path that bypasses the large-vocabulary softmax bottleneck.

**Prereqs:** [../fundamentals/attention.md](../fundamentals/attention.md), [multi-head-attention.md](./multi-head-attention.md)
**Related:** [transformer-block.md](./transformer-block.md), [../fundamentals/_tokenization.md](../fundamentals/_tokenization.md)

---

## What it is

Transformers are already good at *using* information from the context, but their output layer is a softmax over the full vocabulary (often 100k+ entries). For tokens the model should just copy from the context (names, numbers, rare words, retrieved spans) that vocabulary head is doing a lot of implicit work: it has to re-identify the token in a huge output space.

CPLM adds a parallel **copy head** that, given the final hidden state, scores every position in the context and copies the token at the chosen position. The next-token distribution becomes a learned mixture:

$$
p(x_t \mid x_{<t}) = (1 - \lambda_t) \cdot p_{\text{vocab}}(x_t \mid x_{<t}) + \lambda_t \cdot p_{\text{copy}}(x_t \mid x_{<t})
$$

where $\lambda_t \in [0,1]$ is a per-token gate learned from the hidden state.

---

## How it works

For a decoder-only transformer with hidden states $h_1, \ldots, h_t$:

1. **Vocab head.** Standard $p_{\text{vocab}} = \mathrm{softmax}(W_{\text{vocab}} \cdot h_t)$.
2. **Copy head.** Score each context position:

$$
s_i = h_t^\top W_{\text{copy}} h_i, \quad p_{\text{copy}}(x_t = v) = \sum_{i: x_i = v} \mathrm{softmax}(s)_i
$$

i.e., attention-like scoring over positions, then aggregating by token identity. Shapes make this linear in context length and introduce only a small projection matrix $W_{\text{copy}}$.

3. **Gate.** Linear + sigmoid on $h_t$ outputs $\lambda_t$.
4. **Train** on the mixed likelihood with the standard next-token NLL.

The authors argue the copy head provides a *second gradient path* that eases the vocab-softmax bottleneck: tokens that should be copied no longer have to be re-encoded through the full vocabulary distribution.

---

## Why it matters

- **Long-context retrieval wins.** CPLMs substantially improve retrieval metrics vs matched baselines, where exact copying from the context dominates.
- **Pairs with sliding-window attention.** CPLM + sliding-window matches full-attention baseline perplexity and downstream performance at much lower memory — the pointer path handles the long-range copying that would otherwise force full attention.
- **Minor added parameters, no serving changes.** The extra head is small; inference is a cheap additional matmul plus a dot-product over context positions.
- **Pointer networks mainstreamed.** Pointer networks have been around since 2015 (Vinyals et al.); CPLM makes them a drop-in LM head rather than a seq2seq module — "the long-context retrieval problem is partly a loss-function problem, not just an attention problem."

---

## Gotchas & tricks

- **Rare / out-of-vocab wins the most.** The copy path dominates for tokens that are rare in the vocab softmax but easy to point to — names, hex strings, long numerical IDs.
- **Not a free retrieval mechanism.** CPLM only copies tokens *already in context*. It is not a replacement for retrieval augmentation; it just makes the attention-driven copying inside context more efficient.
- **Gate collapse.** If $\lambda_t$ pins to 0 or 1 the copy head is wasted. Standard mitigations: temperature on the gate, auxiliary loss on gate entropy.
- **Vocab head still essential.** Novel generation (not copy) depends on the vocab path. The two paths are complementary, not competitive.

---

## Sources

- Paper: *Copy-Paste Language Models* — anonymous HF submission, 2026 — arXiv [2610.19247](https://arxiv.org/abs/2610.19247).
- Precursor: *Pointer Networks* — Vinyals, Fortunato, Jaitly, 2015.
- Precursor: *Pointer Sentinel Mixture Models* — Merity, Xiong, Bradbury, Socher, 2016 — mixture of vocab + pointer for language modeling.
