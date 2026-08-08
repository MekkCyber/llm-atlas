# Layout-Grounded Parallel Decoding
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A decoding pattern for structured outputs (document parsing, forms, structured extraction): predict the output's **structural layout** first, then decode each region's content in parallel branches conditioned on a shared prefix. Decoding depth becomes the longest layout→content path instead of the total content length. Introduced in PaDoc for full-page document parsing; generalizes to any output with a known DAG.

**Prereqs:** [../fundamentals/attention.md](../fundamentals/attention.md).
**Related:** [../multimodal/README.md](../multimodal/README.md) · [sparse-context-routing.md](./sparse-context-routing.md) · [README.md](./README.md)

---

## What it is

**End-to-end parsers** serialize a page (or a JSON, or any structured object) into one autoregressive sequence — depth = total tokens. **Crop-based two-stage parsers** decode regions in parallel but redo the visual prefill per region and lose full-page context. Layout-grounded parallel decoding gets both: one visual prefill, one layout prediction, then parallel region decoders sharing the same page representation.

## How it works

**Step 1 — visual prefill.** Encode the full page once (image tokens + positional info) into KV cache. Shared across all subsequent decoding.

**Step 2 — layout stream.** An autoregressive head predicts the **branching structure**: a list of regions (bounding boxes) and the DAG that says which regions depend on which. This is a short sequence — depth of the layout tree, not depth of the content.

**Step 3 — region branches.** Each region's content is decoded in parallel by a separate branch. Each branch conditions on:
- The shared page KV cache.
- Its own region's layout descriptor (bounding box, region type).
- Its parent regions' *layout* only (not their content, under the region-sufficiency assumption).

**Region-sufficiency assumption.** For document-like structures, a region's content is well-approximated as conditionally independent of siblings' content *given* the shared page representation and the region's layout. This is what makes parallel decoding sound. Doesn't hold in all structured settings — validate per domain.

**Decoding depth.** `depth = max(layout_depth + region_depth)`. For typical document pages: layout is ~O(#regions), region content is O(tokens-in-longest-region). Sum is much less than total-tokens-in-page.

## Why it matters

- **Removes cross-region autoregressive dependency** without giving up full-page context — the classical tradeoff of document parsers.
- **Structure-aware alternative to speculative decoding.** Speculative decoding is structure-agnostic (guess and verify tokens); layout-grounded decoding exploits *known* output structure to parallelize.
- **Generalizes to structured outputs beyond documents:** JSON with known schema, tool-call argument lists, tabular extraction, code with known scaffold.
- Substantial latency reduction vs serial parsers on realistic document parsing workloads.

## Gotchas & tricks

- **Region-sufficiency assumption is the risk.** In documents with cross-references (footnotes, running headers), region content depends on siblings. Approximation may drop quality on those cases; papers report per-region-type numbers.
- **Layout head is the new critical path.** If it's slow or wrong, downstream branches decode wrong content. Small dedicated layout head with strong supervision.
- **KV cache size.** Sharing a large page KV across many region decoders can pressure memory — worth quantizing the shared cache.
- **Branch scheduling on the accelerator.** GPU throughput depends on how many branches you can pack in parallel; tune batch shape to hardware.
- **Complements speculative decoding.** The two are orthogonal: use spec decoding inside each region branch for further speedup.

## Sources

- Paper: *PaDoc: Layout-Grounded Parallel Decoding for Document Parsing* — Yu et al., Tsinghua + industry, 2026 — [arXiv:2608.06146](https://arxiv.org/abs/2608.06146).
