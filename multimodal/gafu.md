# GAFU
*Depth — using generation-side features as retrieved evidence for fine-grained visual understanding in unified multimodal models.*

**TL;DR:** Unified multimodal models carry two feature streams for the same image — an understanding-side stream and a generation-side stream — and the generation side turns out to carry fine-grained detail the understanding side misses. GAFU is a post-training framework that compresses generation-side features into a small set of G tokens and lets the understanding head retrieve task-relevant evidence from them. Beats same-data SFT with up to 20.00% relative gain on VStar-Bench, 17.94% on OmniSpatial, 8.62% on VSR-Bench.

**Prereqs:** [attention.md](../fundamentals/attention.md)
**Related:** [cot-monitoring.md](../safety/cot-monitoring.md)

---

## What it is

A post-training recipe for unified multimodal models. Rather than tuning the understanding head alone, GAFU treats the model's own generation-side features as an additional, higher-resolution visual feature source. These features are compressed into a bounded set of `G` tokens; the understanding head attends to these G tokens as retrieved evidence during both training and inference.

## How it works

1. **Dual-stream extraction.** For an input image, pull hidden states from both the understanding-side encoder and the generation-side decoder of the unified model.
2. **Compress to G tokens.** The generation-side features are mapped to a fixed-size set of `G` tokens via a lightweight compressor. Spatial organization of the generation features is preserved — flattening without positional structure removes most of the gain.
3. **Retrieve.** For a given question, the understanding head attends to the `G` tokens as external evidence, in addition to its own hidden states.
4. **Post-train end-to-end.** The compressor and the retrieval head are trained on SFT-style multimodal data. The base unified model stays mostly frozen.

## Why it matters

- **Repurposes an unused signal.** Unified MMs were argued-for on efficiency grounds; GAFU shows the generation pathway is *useful* for understanding, not just a convenience.
- **Fine-grained perception gains:** +20.00% on VStar-Bench, +17.94% on OmniSpatial, +8.62% on VSR-Bench vs same-data SFT.
- **No new data.** The gain comes from a different *path* through the same model, not from more labels.
- **Generalizes.** Any unified model with a generation pathway can expose this signal; GAFU's compressor is a template.

## Gotchas & tricks

- Spatial structure matters: compressing with a positional-aware attention beats a plain pooling. If `G` tokens lose spatial order, gains disappear.
- `G` is a budget; too small and the detail is lost, too large and inference cost blows up. Paper uses modest sizes (~tens of tokens).
- Only works where a generation side exists. Non-unified VLMs (CLIP-encoder + LLM) have no generation features to tap.
- Potential leakage: generation-side features may encode content that doesn't appear in the image; evaluate for hallucination explicitly.

## Sources

- Paper: *What Generation Knows: Unlocking Fine-Grained Understanding in Unified Multimodal Models* — 2026 — [arXiv:2610.23987](https://arxiv.org/abs/2610.23987)
