# Sparse attention

*Taxonomy — attention variants that skip most of the KV context to scale beyond dense attention's quadratic cost.*

**TL;DR:** Dense attention costs $O(n^2)$ compute and $O(n)$ KV per layer, which becomes the bottleneck at long context. Sparse-attention methods have each query attend to only a subset of past tokens or chunks. Variants differ in *how the subset is chosen* — fixed patterns (sliding window, local + global), learned static heads, or content-based retrieval — and in whether they train from scratch or convert a dense checkpoint. The 2026 default for long-context production is **learned chunk-wise retrieval trained end-to-end** (HiLS).

**Related taxonomies:** [_positional-encoding](../fundamentals/_positional-encoding.md)
**Depth files covered here:** [hils-attention](hils-attention.md) · [mla](mla.md) *(KV-side sparsity via compression, sibling axis)*

---

## The problem

Dense attention over $n$ tokens costs $O(n^2)$ in compute and $O(n \cdot d)$ per-layer in KV cache. Both scale poorly past ~32k tokens on production hardware. Positional encoding tricks (RoPE scaling, YaRN, DCA) buy some length extrapolation but don't reduce cost. Something has to give: either the query stops attending to most of the past (sparse attention) or the KV representation shrinks (KV compression / MLA).

## The shared pattern

All variants restrict each query's attention to a subset $S \subset [1, n]$ of past positions, then compute standard scaled dot-product attention only over $S$. They differ in:

1. **How $S$ is chosen** — fixed window, static routing, or dynamic retrieval.
2. **Whether $S$'s choice is *learned*** — via auxiliary loss, LM loss end-to-end, or not at all.
3. **Whether the KV for out-of-$S$ tokens is materialized** — training-time only, or discarded entirely.

## Variants

| Technique | Key idea | Main tradeoff | When it wins |
| --- | --- | --- | --- |
| Sliding window (Longformer, Mistral) | fixed local window + optional global tokens | can't span long-range deps | short-to-medium context, streaming |
| Streaming LLM | keep first-k "attention sinks" + rolling window | no true recall of the middle | infinite streaming, chat |
| InfLLM (training-free) | retrieve chunks by content similarity at inference | retrieval not trained | drop-in retrofit of a dense checkpoint |
| Native Sparse Attention (NSA) | learned chunk router trained w/ auxiliary loss | complex training pipeline | train from scratch, high recall |
| [hils-attention](hils-attention.md) | learned chunk retrieval, end-to-end under LM loss, hierarchical fusion | needs sparse-attention kernel | best in-domain quality **and** extrapolation, convertible from dense checkpoint |
| MLA (KV compression, sibling) | compress KV to a low-rank latent — dense attn over compact KV | not truly sparse; KV shrinks not selects | production serving where KV size dominates cost |

## How to choose

- **Long-context production LLM you're training from scratch** → HiLS-style learned chunk attention. End-to-end training under LM loss is the current best-quality path.
- **You already have a dense-pretrained checkpoint and can spend a few billion tokens on adaptation** → HiLS (with continued pretraining). Preserves in-domain quality; adds >64× extrapolation.
- **Truly training-free retrofit** → InfLLM or a Streaming LLM-style sink+window. Quality gap to dense grows with context.
- **Chat / streaming latency is the constraint** → Streaming LLM. Constant memory, no retrieval.
- **KV size is the bottleneck, not compute** → MLA (dense attention, small KV). Composable with sparse attention.

Sparse compute and KV compression are **orthogonal**; production systems often stack both (MLA + sliding-window locality, e.g.).

## Adjacent but distinct

- **[_moe](./_moe.md)** — sparse *in experts*, not in attention. Orthogonal.
- **KV cache eviction** — drops low-utility past KV entries at inference (H2O, StreamingLLM). More of a serving strategy than an architectural change.
- **Linear / kernel attention** — replaces softmax with a decomposable kernel to hit $O(n)$; a different route to the same goal.
- **State-space models (Mamba, S4)** — solve long-context without attention at all.

## Sources

- *Longformer / BigBird* — Beltagy et al. 2020 — fixed-pattern sparse attention.
- *Streaming LLM* — Xiao et al. 2023 — attention sinks + rolling window.
- *InfLLM* — Xiao et al. 2024 — training-free content-based chunk retrieval.
- *Native Sparse Attention* — DeepSeek 2024 — learned chunk router with auxiliary loss.
- *Hierarchical Sparse Attention Done Right (HiLS)* — Hu et al. 2026 — [arXiv:2607.02980](https://arxiv.org/abs/2607.02980).
