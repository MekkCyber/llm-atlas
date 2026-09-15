# Sparse attention

*Taxonomy — post-training and training-time mechanisms that let a query attend to a **subset** of context units instead of all of them, to escape the quadratic attention cost at long context.*

**TL;DR:** Full-context softmax attention is $O(N^2)$; sparse attention picks $k \ll N$ context units per query. The design space splits along three axes: **who selects** (fixed pattern vs data-dependent selector), **when training happens** (post-training vs from-scratch), and **what loss trains the selector** (dense-attention distillation vs the LM loss itself). Modern trainable post-training sparsifiers converge on end-to-end LM-loss training with a memory-efficient FlashAttention-style kernel.

**Related taxonomies:** [_moe](_moe.md)
**Depth files covered here:** [sparse-attention-sas](sparse-attention-sas.md)

---

## The problem

Attention cost scales as $O(N^2)$ in both compute and KV traffic. For long-context models (100k+ tokens) and agentic workloads (accumulating screenshots or tool outputs), this bounds serving throughput and effective context. KV-cache compression (MLA, GQA) attacks the *storage* side; sparse attention attacks the *compute-and-traffic* side by keeping the number of attended units small.

## The shared pattern

Every variant answers three questions:

1. **What is a "context unit"?** Individual tokens, fixed-size blocks, or learned segments.
2. **How is the subset chosen per query?** By a fixed pattern (sliding window, strided, block-sparse), by a fast selector network scoring units, or by hashing / clustering.
3. **How does the subset get gradient signal at training time?** Not at all (post-hoc pruning), distilled from dense-attention distributions, or **end-to-end from the LM loss**.

## Variants

| Technique | Key idea | Main tradeoff | When it wins |
| --- | --- | --- | --- |
| Sliding-window | Fixed local window per query | No context selection intelligence | Local-only tasks; RWKV-style long-context prefixes |
| Block-sparse (Longformer, BigBird) | Fixed pattern of blocks + a few global tokens | Rigid; can't adapt per prompt | Documents with known structure |
| Top-K (dense-distill) | Selector picks top-K units; distilled to match dense-attention distributions | Ranking objective is a *proxy* — not aligned with LM-loss impact | Post-training sparsification with modest budgets |
| [sparse-attention-sas](sparse-attention-sas.md) | Continuous selector scores injected into softmax; trained **end-to-end with the LM loss** | Requires a custom kernel to stay FlashAttention-fast | Tight-budget post-training sparsification |
| Native sparse attention (no depth file yet) | Sparsity trained from scratch, not post-hoc | Requires re-pretraining a base model | Building a long-context model from scratch |
| Retrieval-augmented attention (no depth file yet) | Attend to a retrieved *set* of external units | Retrieval quality is the bottleneck | Very long / external knowledge |

## How to choose

- **Working with a pretrained model, need long context now.** Post-training sparsification — the [SAS](sparse-attention-sas.md) family (end-to-end LM-loss selector) if a training budget is available; simple Top-K + dense distillation otherwise.
- **Tight compute budget (very sparse).** End-to-end LM-loss training pulls decisively ahead — the proxy-signal gap in dense-distill methods widens as $k$ shrinks.
- **Building a long-context model from scratch.** Native sparse attention or hybrid attention (mix of dense and sparse layers) beats retrofitting.
- **Local-only tasks.** Sliding-window suffices; the selector overhead isn't worth it.

## Adjacent but distinct

- [MLA](mla.md) / [multi-head-attention (GQA)](multi-head-attention.md) — attack the *storage* side of long context (compress KV cache). Composes with sparse attention: MLA-style latents plus SAS-style selector run happily on the same layer.
- [_moe](_moe.md) — sparse across *experts* (FFN routing), not context units. Same "sparsity" word, different axis.

## Sources

- Paper: *SAS: Simple Attention Sparsification via End-to-End Optimization of Context Ranking* — Li et al., 2026 — [arXiv:2609.13141](https://arxiv.org/abs/2609.13141).
- Paper: *Longformer: The Long-Document Transformer* — Beltagy et al., 2020 — fixed block-sparse pattern baseline.
- Paper: *Big Bird: Transformers for Longer Sequences* — Zaheer et al., 2020 — random + window + global.
