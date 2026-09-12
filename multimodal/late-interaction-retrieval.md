# Late-Interaction Retrieval (and Generative Compression)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Late-interaction retrieval — the ColBERT / ColPali family — indexes documents as *sets* of per-token vectors and scores queries against the whole set (max-sim, sum-max). It dominates single-vector retrieval on quality, especially for visual documents, but storage grows linearly in tokens per page — the operational blocker at fleet scale. **Generative Late-Interaction Embeddings (GLIE)** compresses each page down to *k* representative "seed" vectors and *regenerates* the full embedding set on-demand for the top-k search candidates. At **k = 4 vectors/page**, GLIE retains ~80% of uncompressed retrieval quality.

**Prereqs:** [../fundamentals/attention](../fundamentals/attention.md)
**Related:** [README](README.md), [../inference/README](../inference/README.md)

---

## What it is

**Late-interaction retrieval.** A query is $q = (q_1, \ldots, q_m)$ (token vectors); a document is $d = (d_1, \ldots, d_n)$. Similarity is $\mathrm{sim}(q, d) = \sum_i \max_j \langle q_i, d_j \rangle$. Because every token pair is scored, semantic nuance is preserved that single-vector encoders average away. For visual documents (PDF pages, screenshots) the version is ColPali/ColVQA: replace text tokens with vision-transformer patch embeddings. Quality gains have made late-interaction the SOTA for enterprise visual doc search.

**The storage tax.** A single page becomes hundreds of high-dimensional vectors. At web scale this multiplies the on-disk index by 100× vs single-vector retrieval — the pain point every deployment hits.

## How it works

**GLIE compression.**

1. **Analyze embedding geometry.** Across trained visual encoders the full per-page vector set concentrates near a low-dimensional manifold — the vectors are highly correlated in structured ways.
2. **Seed storage.** Store only $k$ representative vectors per page (e.g. via PCA or a learned seed selector). Typical $k = 2$–$8$.
3. **Two-stage query.**
   - *Stage 1 (seed match).* Score the query against the compressed $k$-vector seeds for every page; take the top-$K$ candidates.
   - *Stage 2 (regenerate).* For each top-$K$ candidate, a small **generative decoder** conditioned on the $k$ seed vectors expands them back into the full per-token embedding set. Then run standard late-interaction max-sim scoring on the regenerated set.
4. **Rank on regenerated scores.** Only the top few candidates pay the regeneration cost.

The decoder is trained to reconstruct the original full embedding set from the seeds under a per-vector L2 loss plus a retrieval-quality loss on downstream ranking.

## Why it matters

- **Storage collapse.** From hundreds of vectors/page to $k$; at $k = 4$, ~80% of uncompressed retrieval performance survives.
- **Regenerate only what matters.** The full embedding set is expensive; regenerating only for top-$K$ candidates amortizes the decoder cost.
- **Cross-modal applicability.** The manifold observation generalizes: text ColBERT indexes exhibit similar structure, so GLIE-style compression likely ports to text late-interaction and cross-modal retrieval.
- **Complements early-exit ranking.** Stage 1 seed scoring is cheap and can be run on more candidates than a full late-interaction index.

## Gotchas & tricks

- **Seed selection matters.** Random PCA loses on rare-content pages; a learned seed selector conditioned on the page content generalizes better.
- **Decoder capacity.** Too small and reconstruction is lossy; too large and you're back to storage cost via decoder params. Amortize decoder over the corpus.
- **Top-K vs $k$.** These are different — $k$ is per-page seed count, $K$ is the number of pages regenerated per query. Tune independently.
- **Manifold breaks under domain shift.** Fine-tune the seed encoder + decoder on your corpus's domain, not on a generic training set.
- **Not a full replacement for the index.** GLIE assumes late-interaction is the right retrieval model — if a task is well-served by single-vector (dense), skip the whole family.

## Sources

- Paper: *Generative Late-Interaction Embeddings For Visual Document Retrieval* — Eltahir, Aloushan, Khairoalsendi, Shata, Alhassan, Alrehaili, Hussain, Khan (KAUST), 2026 — [arXiv:2609.11808](https://arxiv.org/abs/2609.11808)
- Related: ColBERT (Khattab & Zaharia, 2020), ColPali (Faysse et al., 2024) — the late-interaction and visual-late-interaction ancestors.
