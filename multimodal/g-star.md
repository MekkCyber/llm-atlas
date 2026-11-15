# G-STAR — Group-Scaled Token Capacity for Deeply Compressed AR Image Generation
*Depth — product-quantized visual tokens with a group-scaling rule for high-compression autoregressive image generation.*

**TL;DR:** An autoregressive image generator built on **product-quantized** visual tokens. The paper's core claim is a scaling rule: as per-axis spatial compression grows by factor s, the per-position **group size** of the product quantizer should scale **quadratically** in s, so the AR decoder sees enough bits per spatial position. A **semi-autoregressive decoder** handles the intra-group dependencies this creates — groups are decoded in parallel inside, and sequentially across spatial positions. Delivers effective 64× compression and ~20× speedup at 1024×1024 (2.5s per image on one H200), competitive with prior discrete AR image models on five text-to-image benchmarks.

**Prereqs:** [../fundamentals/_tokenization](../fundamentals/_tokenization.md)
**Related:** [../fundamentals/attention](../fundamentals/attention.md)

---

## What it is

Discrete autoregressive image models trade sequence length for quality. Long sequences (fine spatial grids) preserve detail but slow generation; aggressive spatial compression (fewer positions) reduces capacity because each token has to carry more information.

G-STAR resolves the tradeoff by increasing the token's *bit budget* instead of the sequence length. Each spatial position is encoded as a **group** of K sub-codes from a product-quantized codebook. As compression s grows, K grows as s² so bits per position keep up with pixels per position.

## How it works

- **Tokenizer** — product quantization splits a position's latent into K sub-vectors, each quantized against its own codebook.
- **Scaling rule** — if the tokenizer compresses the image by s× on each spatial axis, use K ∝ s² sub-codes per position. Keeps bits-per-pixel constant across compression settings.
- **Decoder** — within a group the K sub-codes are not independent. A **semi-autoregressive** decoder emits all K sub-codes of a position in parallel (intra-group), but iterates autoregressively across positions. The intra-group heads share context and attend to prior positions.
- **Training** — standard cross-entropy on the discrete codes, text-conditioned via cross-attention.

## Why it matters

- **Reopens discrete AR for high-res image generation.** 1024×1024 image in 2.5s on one H200 with a 20× speedup over previous discrete AR models.
- **Clean tokenizer for unified multimodal LLMs.** A product-quantized visual tokenizer is a natural fit for a text-image unified vocabulary; the compression knob makes it tunable per task.
- **The rule itself transfers.** The s² group-size relation is a tokenizer-design principle, not specific to one model family — likely to show up in the next round of unified MMLM tokenizers.

## Gotchas & tricks

- The intra-group heads add parameters even though they run in parallel; the total training compute is modestly higher per step than a single-code AR model at matched sequence length.
- "Effective 64× compression" (quoted) refers to the combined spatial × codebook-depth savings; raw spatial compression in the paper is 32×.
- Not a diffusion replacement at arbitrary quality — competitive on T2I benchmarks, not always the frontier.

## Sources

- Paper: *G-STAR: Group-Scaled Token Capacity for Deeply Compressed Autoregressive Generation* — Huang et al., Peking U. / Meituan Longcat, 2026 — arXiv 2611.13004 — https://arxiv.org/abs/2611.13004
