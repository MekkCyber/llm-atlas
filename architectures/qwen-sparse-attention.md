# Qwen Sparse Attention (QSA)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Sparse attention that replaces full attention at **continued-pretraining** time (not from scratch). Context is broken into **micro-blocks**; a **compressed lightweight indexer** scores each block for relevance to the current query and only the top-scored blocks are attended to. Introduced in Qwen3.8-Next as the sparse-attention layer in a hybrid Gated-DeltaNet + attention backbone.

**Prereqs:** [../fundamentals/attention.md](../fundamentals/attention.md), [multi-head-attention.md](multi-head-attention.md)
**Related:** [mla.md](mla.md) · [../case-studies/qwen3-8-next.md](../case-studies/qwen3-8-next.md) · [gated-residual.md](gated-residual.md)

---

## What it is

Full self-attention over $L$ tokens costs $O(L^2)$ compute and $O(L)$ KV memory per head. Prior sparse-attention work tackled this from scratch — training with sparse patterns from step 0 — and struggled to match dense-attention quality at frontier scale. QSA takes a different tack: **train dense, then swap to sparse at continued pretraining**, using a lightweight learned indexer to pick which context blocks each query attends to.

The choice of *when* to introduce sparsity is the key design point. Dense pretraining lets the model learn long-range attention dependencies unconstrained; the subsequent QSA phase asks the indexer to preserve those dependencies while attending to only a small fraction of the context.

## How it works

**Micro-block segmentation.** The context is split into small fixed-size blocks (micro-blocks). Each token is assigned to one block. Block size is small enough that intra-block information is dense, coarse enough that per-block scoring is cheap.

**Compressed lightweight indexer.** For each query token, an **indexer** — a very small learned module (much smaller than the attention it gates) — produces a compressed representation of the query and scores each micro-block for relevance:

$$
s_{qb} = \text{indexer}(q, \text{block}_b)
$$

Only the top-$k$ scored blocks contribute to attention. Everything else is masked out. Because the indexer is small, its cost is a rounding error next to the attention it saves.

**Continued-pretraining swap.** During dense pretraining, standard full attention runs at every attention layer. When the continued-pretraining stage begins, the full-attention layers are replaced with QSA; the indexer is trained to reproduce the attention distribution the dense model learned, then fine-tuned end-to-end. The dense pretrained weights of the attention layer itself (Q, K, V projections) are retained.

**Where in the backbone.** In Qwen3.8-Next, attention is 1-in-4 layers (the rest are Gated DeltaNet). QSA replaces only those attention layers; GDN layers are untouched.

## Why it matters

- **Sparsity without from-scratch pain.** Learning long-range attention patterns from scratch under a sparse mask is hard; QSA sidesteps that by letting dense pretraining teach the patterns and the indexer preserve them.
- **Indexer is cheap.** A lightweight compressed indexer costs a small fraction of the attention it gates; net compute savings are large.
- **Composes with hybrid attention backbones.** In a GDN-heavy backbone, only the (few) attention layers need to be sparse for total compute to drop meaningfully.
- **Preserves long-range dependency modelling.** Reported at 125B-total / 6B-active scale, downstream benchmarks match or beat a larger dense-attention baseline.

## Gotchas & tricks

- **Continued-pretraining budget is a hyperparameter.** Too short and the indexer hasn't learned the dense-attention patterns; too long and you've burnt the compute you were trying to save. The paper's numbers correspond to a specific budget not fully disclosed in the abstract.
- **Block size trades granularity for indexer cost.** Small blocks give the indexer finer control but more blocks to score; large blocks are cheaper to index but coarser.
- **top-$k$ selection is not always the right rule.** Threshold-based selection (all blocks above a score) matches variable relevance better; top-$k$ is what the paper uses for predictable cost.
- **Not from-scratch usable as-is.** The paper's swap-at-continued-pretraining recipe is what makes QSA work; attempts to train QSA from step 0 are not what the paper reports.
- **KV cache implications.** Sparse attention still needs the full KV cache in principle (unless you're willing to drop unindexed tokens permanently). Memory savings are compute-side, not memory-side.

## Sources

- Paper: *On the Design of Qwen3.8-Next Architecture: Evaluation, Efficiency, and Training Stability* — Qiu, Wang, Li, et al. — Qwen team / Alibaba, 2026 — arxiv.org/abs/2608.30320.
