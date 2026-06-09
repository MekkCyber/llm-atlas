# EmbedFilter — Unembedding-Matrix Subspace Removal for Text Embeddings
*Depth — a one-shot linear projection that nulls the high-frequency-token subspace from LLM-derived embeddings.*

**TL;DR:** Off-the-shelf LLM hidden states make poor text embeddings because they over-weight a low-rank subspace dominated by high-frequency tokens — visible by projecting through the model's own unembedding matrix (an "unembedding-as-feature-lens" reading). EmbedFilter learns or constructs a rank-r projection orthogonal to that subspace and applies it as a post-hoc transform, simultaneously improving zero-shot embedding quality and reducing dimensionality.

**Prereqs:** [attention](../fundamentals/attention.md), [logit-lens](logit-lens.md)
**Related:** [_data-curation](../data/_data-curation.md)

---

## What it is

LLMs are increasingly used as drop-in text encoders by pooling final-layer hidden states. They under-perform contrastively-trained encoders on MTEB. EmbedFilter identifies one specific reason — the embedding vector aligns with directions in the residual stream that the model uses to *write* high-frequency tokens (function words, punctuation) into the output distribution. These directions live in a low-rank subspace of the unembedding matrix `W_U`. Filtering them out leaves room for semantic content.

## How it works

1. Pool an embedding `e` from an LLM (e.g. mean of last-layer hidden states).
2. Project `e` through `W_U` to inspect which tokens it expresses. Frequent tokens dominate.
3. Identify a rank-`r` subspace of `W_U` spanning the rows for the top-`k` most frequent tokens (or, more generally, the top singular directions of `W_U` weighted by frequency).
4. Build `P = I − UUᵀ` where `U ∈ ℝ^{d×r}` is an orthonormal basis of that subspace.
5. Output `e' = P · e`. Optionally drop to `d − r` dimensions by dropping the `U`-aligned coordinates.

Step 3 is one-shot: no gradient training is required, only an SVD of a slice of `W_U`. The transform composes with downstream similarity search unchanged.

## Why it matters

- Closes part of the gap between "LLM-as-embedder" and contrastively-trained encoders without any fine-tuning.
- The dimensionality drop is essentially free — index storage and ANN latency go down while quality goes up.
- Gives a mechanistic story: "embedding quality is suppressed by frequent-token writing directions", which is testable beyond the proposed fix.

## Gotchas & tricks

- Rank `r` is a hyperparameter; too aggressive nulls semantic directions that happen to overlap with frequent-token writing.
- The "frequent token" set depends on the pretraining corpus — using a domain-specific frequency table can help on out-of-domain retrieval.
- The technique is logit-lens-flavoured; failure modes include LLMs whose unembedding is tied to embedding (where `W_U` is shared with `W_E`) — there the subspace is hardly orthogonal to semantic structure and the gains shrink.

## Sources

- Paper: *Your UnEmbedding Matrix is Secretly a Feature Lens for Text Embeddings* — 2026 — [arXiv:2606.07502](https://arxiv.org/abs/2606.07502)
- Code: https://github.com/CentreChen/EmbFilter
