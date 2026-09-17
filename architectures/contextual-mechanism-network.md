# Contextual Mechanism Network (CMN)
*Depth — tabular-foundation-model backbone with typed row/column attention that models a conditional data-generating mechanism.*

**TL;DR:** A CMN is a transformer variant tailored for **tabular in-context learning**. It layers two attention axes — a **row-attention** that lets a query row attend to a set of context rows, and a **column-attention** that models feature ↔ target relationships within a row — and enforces asymmetric information flow (query row sees context rows, not the other way around). This shape is what enables joint-distribution pretraining ([ccmm](../pre-training/ccmm.md)) to produce mechanism-oriented representations and, in LimiX-2, recover causal skeletons from feature-attention patterns.

**Prereqs:** [multi-head-attention](multi-head-attention.md), [transformer-block](transformer-block.md)
**Related:** [../pre-training/ccmm.md](../pre-training/ccmm.md), [../case-studies/limix-2.md](../case-studies/limix-2.md)

---

## What it is

A backbone for tabular foundation models. Inputs are cells arranged in rows × columns; each cell is embedded with a column-id and column-type embedding. Instead of flattening the table into a sequence and using standard causal attention, CMN interleaves two structured attention operations that reflect the table's geometry:

- **Row-attention**: how a query row's cells attend to the context rows' cells (in-context learning axis).
- **Column-attention**: how cells within a row attend to each other (feature-target interaction axis).

The routing rules enforce a **context ⇒ query** direction: query-row cells can read from context rows, but context rows are self-contained. This asymmetry keeps the in-context aggregation clean and lets pretraining use context rows as a truly "conditioning" set.

## How it works

A CMN block, in schematic pseudocode:

```
# X: [B, N, F, d]  — batch × rows × features × hidden
# where N = 1 (query) + K (context) rows

# Row-attention: query row attends to context rows, per feature column
X_query  = row_attn(X_query, keys=X_ctx, values=X_ctx)   # per-column

# Column-attention: within each row, features attend to each other
X = col_attn(X)                                          # per-row

# Standard MLP + residual + norm
X = mlp(X) + X
```

Key implementation notes:

- **Typed embeddings.** Column-type embeddings differentiate numeric, categorical, and text-lite columns. Column-id embeddings preserve identity across permutations.
- **Row-attention is per-column.** The context rows contribute per-column keys/values, so the query row's `feature f` cell aggregates over the same feature across context rows plus any feature via subsequent column-attention.
- **Column-attention is per-row.** All features in a single row (query or context) attend to each other, letting the model represent feature ↔ target dependence structure.
- **Asymmetric routing.** Context rows do not receive updates from the query row within a block. This keeps context rows as a stable conditioning set.
- **Position invariance.** CMN is designed to be permutation-invariant across rows and (with column-id embeddings) permutation-equivariant across columns.

## Why it matters

The typed row/column attention gives CMNs several properties that flat transformers over serialized tables don't have:

- **Scales to variable row and column counts.** No re-training when the table shape changes at inference time.
- **Enables joint pretraining.** Because attention respects table geometry, masking arbitrary cells (CCMM) trains a coherent conditional model, not a bag-of-cells classifier.
- **Interpretable feature-attention.** Feature-attention weights over columns approximate feature dependencies. In LimiX-2, these weights recover the **causal skeleton** of the underlying data-generating mechanism with high accuracy — a side effect of joint mechanism-oriented pretraining.
- **Beats target-only PFNs.** On TabArena / TALENT / BCCO, CMN backbones trained with CCMM outperform target-only tabular PFNs.

## Gotchas & tricks

- **Row-attention cost.** For datasets with many context rows, per-column row-attention is O(K · F · d) per block. Bucket by row count during pretraining.
- **Column identity.** Column-id embeddings drift if the model sees very high column counts at inference that were rare in pretraining. Curriculum with wide graphs helps.
- **Asymmetric routing breaks with naive KV cache.** Sharing KV across query rows is fine only if the queries do not update the context; the design already handles this, but reimplementations often get it wrong.
- **Type heterogeneity.** Mixed numeric / categorical columns need per-type heads; sharing a single head degrades numeric calibration.
- **Not for text.** CMN is not a language model backbone. Feeding it long-text columns without a text encoder wastes capacity.

## Sources

- Paper: *LimiX-2: A Contextual Mechanism Network Towards General Structured-Data Intelligence* — Zhang, Ren, Yuan, Zou, … Cui — 2026 — [arXiv:2609.17488](https://arxiv.org/abs/2609.17488).
- Prior tabular PFNs: TabPFN v2 (Hollmann et al., 2025), TabICL, TabDPT.
