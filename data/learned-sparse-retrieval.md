# Learned Sparse Retrieval (LSR)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Retrieval where each document (or query) is represented as a **sparse vector over vocabulary** — most entries zero, non-zero entries are learned term weights. Combines the interpretability and inverted-index efficiency of BM25 with the semantic matching power of neural models. SPLADE is the canonical variant; the pattern now also lives inside decoder-only multimodal embedders (UEmbed).

**Prereqs:** none (helpful: BM25 / TF-IDF familiarity)
**Related:** [../multimodal/README.md](../multimodal/README.md), [_data-curation](_data-curation.md)

---

## What it is

Dense retrieval represents queries and documents as continuous vectors (BERT-style), and matches by cosine similarity. Fast at query time but throws away lexical structure and hurts interpretability. Sparse retrieval (BM25) uses vocabulary-length vectors with only a few non-zero entries — interpretable, blazingly fast on inverted indices, but stuck with exact-lexical matching.

**Learned Sparse Retrieval** unifies the two: use a neural model to produce vocabulary-sized *sparse* vectors, where non-zero entries can be terms *not* in the source text (semantic expansion) and their weights are trained rather than TF-IDF-derived.

## How it works

The canonical SPLADE recipe:

1. Encode a query/document with a bidirectional transformer.
2. For each output position, produce logits over the full vocabulary (mask-language-modelling head).
3. Aggregate across positions with a max or log-sum-exp pooler to produce a single sparse vector over vocab.
4. **Sparsity regulariser**: FLOPS-loss (a differentiable proxy for average non-zero count) forces most entries to zero.
5. Train with the usual contrastive retrieval objective (in-batch negatives, hard negatives).

Result: each doc becomes a vocabulary-sized vector with ~100–500 non-zero entries, indexable in a standard inverted index (Lucene, PISA) at the same performance as BM25 but with expanded, learned term weights.

**Multimodal / decoder-only variants (UEmbed style):** replace the bidirectional encoder with a causal decoder, use two projection heads (sparse + dense), and train jointly on multimodal contrastive + LSR losses. Enables one model to serve dense, sparse, and hybrid retrieval regimes.

## Why it matters

- **Inverted-index compatible.** Unlike dense retrieval, LSR indices live in existing search infra. Deployment is a plug-in, not a rewrite.
- **Interpretable.** Non-zero entries are vocabulary words with weights — you can literally read what the model thinks the document is "about."
- **Robust to out-of-distribution queries.** LSR generalises better than dense retrieval on domains far from training data (BEIR benchmark evidence).
- **Hybrid wins are cheap.** Combining LSR + dense scores at query time consistently beats either alone.
- **Now inside multimodal decoders.** Extending LSR to a causal-decoder multimodal setup (UEmbed) removes the auxiliary cross-modal encoder that image-LSR previously needed.

## Gotchas & tricks

- **Sparsity is a soft target.** The FLOPS regulariser produces expected sparsity, not enforced. Occasional documents come out dense; cap non-zero entries or truncate top-K.
- **Vocabulary choice matters.** Using WordPiece leaks subword artefacts into the retrieval index. Prefer a whole-word vocab for interpretability if that's a goal.
- **Multi-vector variants** (ColBERT-style multi-token) sit between LSR and dense. Don't confuse the two — different index shapes.
- **Training data quality.** Contrastive retrieval training is dominated by the negative-mining strategy; hard negatives (BM25 top-k that don't answer the query) are essential.
- **Latency vs storage.** Sparse vectors with more non-zeros retrieve better but bloat the index. 100–500 non-zeros is the standard sweet spot for text; higher for multimodal.
- **Freshness.** LSR indices update the same way BM25 indices do — no re-embedding of the whole corpus needed on model changes if you can accept a partial re-encode.

## Sources

- Paper: *SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking* — Formal et al., 2021.
- Paper: *SPLADE v2 / v3* — 2022 / 2024 — the current strong baselines.
- Paper: *UEmbed: Unified Sparse and Dense Multimodal Embeddings* — arXiv:2608.02583, 2026 — decoder-only multimodal LSR + dense.
- Benchmark: BEIR — 2021 — heterogeneous retrieval benchmark where LSR shines.
