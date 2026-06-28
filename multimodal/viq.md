# ViQ
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A discrete visual tokenizer that produces *text-aligned* codes at *native* image resolution. Targets the long-standing semantics-vs-detail tradeoff in VQ-style tokenizers so the same tokens can drive both vision-language understanding and pixel-faithful reconstruction in an any-to-any multimodal LLM.

**Prereqs:** [_visual-tokenizers.md](_visual-tokenizers.md), [_tokenization.md](../fundamentals/_tokenization.md)
**Related:** [README.md](README.md)

---

## What it is

Discrete visual tokenizers convert images into integer codes a language model can ingest. The literature splits into two camps:

- **Reconstruction-oriented** (VQ-VAE, VQ-GAN, FSQ): codes preserve pixel detail; weak semantic alignment with text.
- **Semantics-oriented** (e.g. SigLIP-style features quantized): codes align with text/CLIP space; severe loss of local detail.

ViQ ("Visual Quantized") targets both at once and adds *native-resolution* support so high-detail inputs aren't forced into 224/336-square crops.

## How it works

- A vision encoder produces continuous features; a quantizer maps them to a discrete codebook.
- During training the codebook is regularized to be **text-aligned**: codes are pushed to live in a representation space close to a text encoder's embedding space, so each visual code carries semantic content directly comparable to language tokens.
- A reconstruction branch ensures the same codes are sufficient to decode back to the image — this is what prevents the usual detail collapse when you optimize for semantics.
- The pipeline accepts inputs at native resolution rather than forcing a fixed square; spatial layout is preserved in the token sequence, so dense / small-detail visual content survives into the LLM context.

## Why it matters

- A *single* discrete tokenizer suitable for both **understanding** (multimodal LLM inputs) and **generation** (image outputs from token sequences) is the missing piece for clean any-to-any models. Most prior work uses two separate tokenizers.
- Native-resolution support sidesteps the cropping/resizing tax that hurts OCR, fine-grained recognition, and high-resolution reasoning.
- Matches or beats prior visual tokenizers on both reconstruction and downstream multimodal benchmarks — i.e. it actually closes the tradeoff rather than just naming it.

## Gotchas & tricks

- Codebook size and the strength of the text-alignment regularizer are the two main knobs; too strong an alignment collapses to a SigLIP-like representation with poor reconstruction.
- Native-resolution sequences vary in length — downstream LLMs need positional/length handling that scales (RoPE-style or learned 2D positions).
- Like all VQ schemes, codebook collapse is a failure mode; the paper relies on commitment-loss and replacement tricks borrowed from VQ-VAE literature.

## Sources

- Paper: *ViQ: Text-Aligned Visual Quantized Representations at Any Resolution* — Yu, Liu, Yang, Dong, Qian, Lu, Hu, Rao, Tencent HY Vision / Tsinghua / CAS / NTU, 2026 — arXiv:2606.27313.
