# Hyper-Spherical Codes for Visual Tokenizers
*Depth — a visual tokenization scheme (dRAE) that eliminates codebook collapse by constraining codewords to a unit hypersphere.*

**TL;DR:** Discretize visual features by placing codewords on a unit hypersphere (unit-norm) and looking them up by angular similarity. This geometric constraint eliminates the "unused low-norm region" that drives VQ-VAE codebook collapse, and reports **100% codebook utilization** at vocabulary sizes up to **131,072 tokens** — with monotone gains in downstream understanding and generation as vocab grows.

**Prereqs:** [../multimodal/README.md](../multimodal/README.md), [../fundamentals/_tokenization.md](../fundamentals/_tokenization.md)
**Related:** [../quantization/_number-formats.md](../quantization/_number-formats.md)

---

## What it is

Vector-quantized visual tokenizers (VQ-VAE, VQGAN, FSQ, LFQ) map a continuous feature to the nearest codebook vector. Under standard training, codebooks *collapse* — most codes are never used because the encoder distribution concentrates in a small region and low-norm codewords sit in dead zones. Practitioners paper over this with dead-code resets, perplexity boosters, EMA tricks, and small vocabularies. Hyper-spherical codes remove the root cause by constraining every codeword to unit norm.

## How it works

Two changes over standard VQ:

1. **Unit-norm codebook.** Every codeword is projected to the unit sphere after each update. The lookup becomes an *angular* (cosine) nearest-neighbor query rather than Euclidean.
2. **Unit-norm encoder outputs.** Feature vectors are normalized before lookup so encoder and codebook share the same geometric surface.

Because both the query and every code live on the sphere, there are no low-norm dead zones, no norm-scale race between encoder and codebook, and the effective "distance" is fully captured by angle. Empirically the training pipeline simplifies — no perplexity boosters, no dead-code resets.

## Why it matters

Codebook collapse has been the reason unified multimodal LLMs use tiny visual vocabularies (a few thousand codes) despite text tokenizers running at 100k+. Hyper-spherical codes report **100% utilization at 131,072 tokens** with monotone quality gains as the vocab scales. That's enough headroom to start treating visual codes as first-class "words" and to build unified multimodal decoders with a single, large tokenizer.

## Gotchas & tricks

- Angular lookup is cheap on GPU with normalized vectors — implement as a plain dot product.
- The unit-norm constraint interacts with the reconstruction decoder — it needs to be trained to consume unit-norm latents, which is trivial but a common bug source when swapping in a hyper-spherical codebook post hoc.
- Scaling the vocabulary past 131K keeps giving returns in this regime — a change from the "vocab plateaus" mental model of VQ-VAE.
- Downstream LLMs consuming these tokens need to be aware of the expanded vocabulary; the language-side tokenizer still owns text codes.

## Sources

- Paper: *dRAE: Representation Autoencoder with Hyper-Spherical Codes* — Ma et al., 2026 — [arXiv:2607.22148](https://arxiv.org/abs/2607.22148)
