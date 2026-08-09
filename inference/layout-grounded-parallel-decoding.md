# Layout-Grounded Parallel Decoding

*Depth — decode multiple regions of a structured document concurrently by first predicting the layout tree, then decoding each region under a shared encoder.*

**TL;DR:** End-to-end document parsers serialize the whole page into one autoregressive stream and pay the full sequential decoding cost. PaDoc (2026) breaks that by decoding **layout-first, content-in-parallel**: a layout head predicts the region tree, then a shared encoder feeds a bank of decoders that each generate their region's content concurrently. The scheme trades a small header pass for large end-to-end throughput on structured documents while maintaining or improving parsing quality.

**Prereqs:** *(basic decoder-only transformer decoding)*
**Related:** [README.md](README.md), [../multimodal/README.md](../multimodal/README.md)

---

## What it is

A decoder architecture for document parsing where the output has *known branching structure* — a page decomposes into regions (blocks, tables, figures, formulas), and regions are largely independent given the layout. Instead of concatenating all region contents into one AR stream, decoding proceeds in two phases:

1. **Layout phase.** Predict the region tree — a compact tokenization of "here's the page geometry and region types."
2. **Content phase.** Each region's content is decoded by a decoder head conditioned on the shared encoder state + the region's layout descriptor. Region decodes run in parallel (batched together) because they don't depend on each other's content.

---

## How it works

**Encoder.** A vision encoder (or vision + text hybrid) produces token representations of the page image, plus positional / structural embeddings.

**Layout head.** A small AR decoder emits a serialized region tree: `[region_1_bbox, region_1_type], [region_2_bbox, region_2_type], ...`. Typically short — a page has O(10²) regions, not O(10⁵) tokens.

**Content decoders.** All regions are batched into a single forward pass; each region's decoder attends to the shared encoder state through region-conditioned attention (bbox / type embeddings as conditioning). Because there is no cross-region content dependency, decoding is embarrassingly parallel across regions.

**Loss.** Sum of layout AR loss + per-region content AR loss.

## Why it matters

- **Trades ordering assumptions for throughput.** The AR ordering across regions was never load-bearing for document parsing; PaDoc names and exploits that.
- **Concrete quality gains alongside speed.** Reports 91.1 layout F1 on OmniDocBench Full and top-tier 94.24 Overall among end-to-end parsers with best Text Edit (0.038) and Formula CDM (95.59). Not a speed-vs-quality tradeoff.
- **Extends the "parallel decoding" family.** Sits alongside speculative decoding, medusa-style multi-head, and diffusion-style token-parallel generation; distinct because the branching structure comes from the *output modality*, not from a draft model or a diffusion process.

## Gotchas & tricks

- **Requires layout supervision.** Training needs region-tree labels; free with document-parsing datasets that expose bounding boxes, hard to bootstrap without them.
- **Cross-region dependencies limit applicability.** Works when regions are near-independent (blocks in an invoice, cells in a table row). Breaks for outputs where region B semantically depends on region A (a long narrative flow across columns).
- **Region decoder size.** Sharing weights across region decoders is the natural default; heterogeneous region types (table vs. formula) sometimes benefit from small type-specific heads.
- **Layout errors cascade.** A missed region in the layout phase silently drops its content. Practical systems add layout confidence thresholds and a fall-back to full AR decode when confidence is low.
- **Serialization for evaluation.** Even though decoding is parallel, benchmark comparison usually needs a canonical serial output — pick a deterministic region ordering (top-to-bottom, then left-to-right).

## Sources

- Paper: *PaDoc: Layout-Grounded Parallel Decoding for Document Parsing* — Yu et al., Tsinghua + industry co-authors, 2026 — [arXiv:2608.06146](https://arxiv.org/abs/2608.06146). Evaluated on OmniDocBench Full.
