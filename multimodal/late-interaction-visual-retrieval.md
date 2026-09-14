# Late-Interaction Visual Retrieval (with generative compression)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** ColPali-style late-interaction retrievers score visual documents with many vectors per page, giving high accuracy at brutal storage cost. **Generative Late-Interaction Embeddings (GLIE)** learn a small basis set and regenerate the full per-page embedding on demand — compress at storage time, generate at query time — recovering ~80% of uncompressed retrieval performance with only 4 vectors per page.

**Prereqs:** [README](README.md), [../fundamentals/attention](../fundamentals/attention.md)
**Related:** [../inference/README](../inference/README.md)

---

## What it is

**Late interaction** for retrieval keeps many vectors per document (one per patch / token) and scores queries against them individually, then aggregates — instead of collapsing the document to one vector (dense retrieval). It wins on accuracy but multiplies index size by 100–1000×. For visual document retrieval (ColPali and successors) the storage cost is a hard deployment ceiling.

GLIE is a **generative compressor** for the late-interaction index. It replaces per-page multi-vector storage with a small learned basis and coefficients; the full multi-vector embedding is regenerated on demand when a query comes in.

## How it works

- **Structural observation.** Page-embedding vectors lie on the unit sphere and concentrate near a low-dimensional manifold.
- **Basis learning.** Fit a small learned basis $B$ (a few hundred vectors) that spans the manifold well; each page's multi-vector representation is written as a low-dimensional coefficient tuple over $B$.
- **Storage.** Store only `(page_id, coefficients)` — orders of magnitude smaller than the full multi-vector representation. The basis itself is shared across the whole corpus.
- **Regeneration at query time.** Reconstruct the multi-vector representation from `(B, coefficients)` on the fly. Late-interaction scoring proceeds as usual against the regenerated vectors.
- **No encoder retraining.** The visual encoder is frozen; GLIE plugs on top of whatever ColPali-family model produced the original vectors.

## Why it matters

Late-interaction visual retrieval is the current SOTA for document QA (invoices, financial filings, scientific papers), but storage cost has blocked it from web-scale deployment. Trading storage for a cheap generation step is exactly the move needed to make ColPali-class systems shippable, and the basis-regeneration idea generalizes to any late-interaction index — text as well as vision.

Reported: ~80% of uncompressed performance at 4 vectors per page, beating existing compression approaches across multiple encoders and benchmarks.

## Gotchas & tricks

- **Basis size vs. quality is the main knob.** Too small → coefficients cannot represent the manifold; too large → storage win shrinks.
- **Encoder specificity.** The manifold's low-dimensional structure is encoder-specific — one basis per encoder family (ColPali, ColQwen, ColSmol) rather than one universal basis.
- **Regeneration cost lands at query time.** Retrieval latency grows slightly; batch queries, cache popular pages' regenerations.
- **Late interaction ≠ ColBERT.** Vision-side late interaction operates over image patches with different statistics than text tokens; compression schemes borrowed from text ColBERT don't transfer directly.
- **Combine with product quantization** for a further storage cut — orthogonal axis.

## Sources

- Paper: *Generative Late-Interaction Embeddings For Visual Document Retrieval* — Eltahir, Aloushan, Khairoalsendi, et al. (KAUST), 2026 — https://arxiv.org/abs/2609.11808
