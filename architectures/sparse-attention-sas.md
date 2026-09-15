# SAS — Simple Attention Sparsification
*Depth — a post-training sparse-attention selector trained **end-to-end with the language-modeling loss** via a softmax-gated continuous-score design.*

**TL;DR:** Post-training sparse-attention methods usually train the selector to distill layer-wise **dense** attention distributions and then apply hard Top-K at inference. That ranking objective is a proxy: it doesn't line up with which context units actually change predictions under a fixed budget. SAS injects the selector's continuous scores into attention logits during training — inside the softmax, in log form — so standard backprop from the LM loss updates the ranking directly. Ships with a FlashAttention-style Triton kernel for long sequences.

**Prereqs:** [attention](../fundamentals/attention.md), [multi-head-attention](multi-head-attention.md)
**Related:** [_sparse-attention](_sparse-attention.md), [mla](mla.md)

---

## What it is

Trainable sparse-attention pipelines have converged on this pattern:

```
selector(query, context)  -> scalar score per context unit
score -> Top-K            -> keep k units, drop the rest
loss   = distillation vs the pretrained model's dense attention
```

Hard Top-K blocks gradients from downstream layers back into the selector, so the selector never sees the LM loss. Distillation fills the gap, but "match the dense-attention distribution" ranks context units by dense attention weight, which is only weakly correlated with which units matter for the prediction *under a tight budget*.

SAS closes that loop: the selector's score enters the attention softmax as a **continuous gate**, so gradients from the LM loss flow through both the attention values and the selector's rank.

## How it works

### Softmax-gated attention

Let $s_j$ be the selector's log-score for context unit $j$. The attention logit for query $i$, unit $j$ becomes:

$$
\ell_{ij} = \frac{q_i \cdot k_j}{\sqrt{d_k}} + s_j
$$

The softmax is taken over all units. Because $s_j$ is inside the softmax, it monotonically re-weights units without hard truncation: units with strongly negative $s_j$ drop out of the softmax's mass, low-scoring units contribute little, and high-scoring units contribute more.

At inference, hard Top-K is recovered by sorting on the *combined* $\ell_{ij}$ (or the selector score alone if it dominates), then truncating.

### The three load-bearing design choices

1. **Gate inside the softmax, in log form.** Placing the score as an additive log-gate keeps the operation numerically stable, kernel-friendly, and monotone.
2. **Normalize against an always-retained current block.** The current block (recent tokens the model always attends to) serves as an anchor: gate scores are normalized so historical context is calibrated against it, preventing the selector from silently dropping every historical unit.
3. **Preserve continuous scores end-to-end.** Not just for training — even the inference-time ranking uses continuous scores, so the model learns *relative priorities* rather than a hard classification into keep/drop.

### Memory-efficient kernel

A Triton kernel integrates the softmax gate into FlashAttention-style computation so long-sequence training doesn't materialize the $N \times N$ score matrix. The gate becomes an extra reduction inside the same tile loop.

## Why it matters

- **Alignment with the actual objective.** LM-loss gradient flow through the selector fixes the proxy-signal problem that hobbles dense-distillation sparsifiers.
- **Consistent gains under tight budgets.** SAS outperforms trainable sparse-attention baselines across attention budgets in reasoning, long-context understanding, and agentic tasks, **with the largest margins where the budget is smallest** — exactly where sparse attention matters.
- **Post-training, not from-scratch.** Applies to already-trained Transformers; no re-pretraining.

## Gotchas & tricks

- **Score scale interacts with softmax.** If selector scores drift too large, the gate saturates the softmax and blocks gradient to $q, k$. The normalization against the current block anchors the scale.
- **Kernel is required for long sequences.** Naive PyTorch is fine at $N \le 4k$; for 32k+, the FlashAttention-style Triton kernel isn't optional.
- **Not a replacement for KV-cache compression.** SAS trims *what is attended*, not *what is stored*. Compose with [MLA](mla.md) or GQA for the storage side.
- **Continuous inference matters.** At inference, the selector's continuous score can be re-thresholded per query — the model learned relative priorities, not a hard partition. Hard Top-K at inference works, but soft top-p or budget-aware thresholding usually recovers additional quality.

## Sources

- Paper: *SAS: Simple Attention Sparsification via End-to-End Optimization of Context Ranking* — Li, Zhu, Gu, Hu, Wang, Mi, Han, Liang, Guo, 2026 — [arXiv:2609.13141](https://arxiv.org/abs/2609.13141).
- Background: *FlashAttention* — Dao et al., 2022 — the kernel style SAS's Triton implementation inherits.
