# Hierarchical Landmark Sparse (HiLS) Attention
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A chunk-wise sparse attention where **chunk selection is learned end-to-end under the LM loss**. Each query independently attends inside every retrieved chunk to extract per-chunk outputs, which are then fused as a **retrieval-score-weighted sum**. The retrieval scores appear in the forward pass, so gradients flow through them — the model learns *which* chunks matter without an auxiliary retrieval loss. Matches full attention in-domain and extrapolates >64× the training context length at 90% retrieval accuracy.

**Prereqs:** [attention.md](../fundamentals/attention.md), [rope.md](../fundamentals/rope.md)
**Related:** [_sparse-attention.md](./_sparse-attention.md), [mla.md](./mla.md), [dca.md](../fundamentals/dca.md), [mtp.md](../pre-training/mtp.md)

---

## What it is

Chunk-wise sparse attention splits the KV context into fixed-size chunks and lets each query attend to only a small set of "important" chunks. The historical problem: how do you *choose* which chunks are important? Prior methods rely on heuristics (local + a few landmark tokens), gated networks trained with auxiliary losses, or content-based retrieval that can't be optimized under the LM loss. All have inaccuracies that widen the gap to full attention as context grows.

HiLS closes this gap by making chunk selection differentiable under the standard next-token loss: retrieval scores enter directly into the attention output.

## How it works

**Hierarchical factorization.** For each query $q$:

1. Compute a **chunk retrieval score** $s_c$ for each candidate chunk $c$ (typically via a landmark token summarizing the chunk).
2. Select the top-$k$ chunks by score.
3. For each selected chunk $c$, run a **local attention** between $q$ and the tokens of $c$ → per-chunk output $o_c$.
4. Fuse: $o = \sum_c \tilde s_c \cdot o_c$, where $\tilde s_c$ is a normalized retrieval score (softmax over the selected chunks).

Because $\tilde s_c$ multiplies $o_c$ in the forward pass, $\partial L / \partial s_c$ is non-zero — the LM loss backprops into the chunk-selection module. No auxiliary "which chunk should we pick" supervision is needed.

**Native sparse training.** Only the top-$k$ chunks are ever materialized in KV — no dense softmax across the full context is computed at any point. This preserves the efficiency claims during training, not just inference.

**Continued-pretraining conversion.** A pretrained dense-attention checkpoint can be converted by (a) attaching landmark heads, (b) short continued pretraining under the HiLS objective. In-domain quality is preserved and long-context extrapolation is acquired.

## Why it matters

- **Efficiency-quality Pareto break.** Prior chunked sparse attention *always* lost to full attention in-domain; HiLS matches or beats it at training context lengths.
- **Extrapolation without RoPE tricks.** >64× training length at 90% retrieval accuracy — a regime out of reach for full attention, and beyond what YaRN/DCA-style scaling reliably deliver.
- **Sparse in both compute *and* KV access.** Fits the disaggregated prefill/decode serving pattern well; KV pressure at long contexts drops with $k$.

## Gotchas & tricks

- **Top-$k$ hard selection is not differentiable in $k$**, but the softmax over selected $s_c$ scores is — HiLS backprops through the softmax weights, not the identity of selected chunks. Selection acts like a straight-through estimator in practice.
- **Landmark drift.** The landmark tokens summarizing each chunk are also trained; if the chunk-size hyperparameter is misspecified for the target sequence length distribution, retrieval accuracy degrades. The paper uses a fixed chunk size but tuning is likely needed per domain.
- **Not a drop-in for attention kernels.** Requires a custom sparse-attention kernel with per-query chunk gathers; FlashAttention-style dense kernels don't cover this pattern.
- **Continued-pretraining budget.** "Lightweight" in the paper is not free — expect several billion tokens of adaptation to convert a dense checkpoint. Don't confuse with plug-in inference-only methods (Streaming LLM, InfLLM).

## Sources

- Paper: *Hierarchical Sparse Attention Done Right: Toward Infinite Context Modeling* — Hu et al., Tencent HY / ShanghaiTech / HKUST / UCSD, 2026 — [arXiv:2607.02980](https://arxiv.org/abs/2607.02980).
- Related: *DCA — Dual Chunk Attention* — the earlier training-free chunk-scaling approach documented in [dca.md](../fundamentals/dca.md).
