# Visual Tokenizers

*Taxonomy — discrete representations of images for use inside multimodal LLMs and any-to-any models.*

**TL;DR:** A visual tokenizer maps an image to a sequence of discrete codes that an LLM can ingest (and, for generative models, decode back from). The space splits on two axes: *codebook objective* (reconstruction vs semantic alignment) and *spatial handling* (fixed-square crops vs native resolution). Modern any-to-any models want a *single* tokenizer that's good at both — historically the hardest combination.

**Related taxonomies:** [_tokenization.md](../fundamentals/_tokenization.md)
**Depth files covered here:** [viq.md](viq.md)

---

## The problem

LLMs consume sequences of integer token IDs. To put an image inside the same LLM you either (a) keep continuous visual features and use a projection layer (no discretization) or (b) discretize the image into integer codes so it lives in the same vocabulary as text. Option (b) — discrete visual tokens — is the cleanest for *unified* generation+understanding models, but every discretizer that gains semantic alignment seems to lose pixel detail, and vice versa. Plus most schemes force fixed input resolution, which destroys fine-grained content.

## The shared pattern

Every visual tokenizer has three parts:

1. **Encoder.** Image → continuous features (CNN or ViT).
2. **Quantizer.** Continuous features → discrete codes (VQ, FSQ, residual VQ, lookup-free).
3. **Decoder (optional).** Codes → reconstructed image, used when the tokenizer must also support generation.

Training pressure for each step decides whether the codes lean *reconstruction* (decoder loss dominates) or *semantic* (alignment loss against a text encoder dominates).

## Variants

| Technique | Quantization | Optimized for | Tradeoff |
| --- | --- | --- | --- |
| VQ-VAE / VQ-GAN | Vector quantization with codebook | Reconstruction | Weak semantic alignment; great pixel fidelity |
| FSQ (Finite Scalar Quantization) | Per-dim scalar quantization | Reconstruction, simple codebook | No codebook collapse, but still semantic-poor |
| MAGVIT / MAGVIT-v2 | Lookup-free quantization for video | Video reconstruction | Excellent generation; not text-aligned |
| ViT-VQGAN | ViT encoder + VQ codebook | Better scalability of VQ-GAN | Same tradeoff axis |
| SigLIP-features-VQ | Quantize semantic features | Vision-language understanding | Severe detail loss, can't decode images well |
| [viq](viq.md) | Text-aligned VQ with reconstruction branch, native resolution | Both reconstruction *and* semantic alignment | Newer; requires careful regularization to avoid collapse |

## How to choose

- **For understanding-only multimodal LLMs:** continuous features + a projection layer often still wins on quality; reach for a discrete tokenizer only if you need a unified vocabulary with text.
- **For any-to-any (generate + understand) models:** you need one tokenizer that does both; reconstruction-only (VQ-GAN, FSQ) and semantic-only (SigLIP-quantized) each fail half the job. A *text-aligned reconstructive* tokenizer like [ViQ](viq.md) is the modern target.
- **Native resolution > fixed-square crops** for tasks with dense detail (OCR, charts, small objects). Fixed-resolution tokenizers can be retrofitted with tiling, but pay a context-length tax.
- **Codebook size** is a hyperparameter, not a free win — larger codebooks improve reconstruction but make the LLM's vocabulary blowout worse. 8k–32k visual codes is typical.

## Adjacent but distinct

- **Continuous projection layers** (LLaVA-style): no discretization. Different paradigm; less elegant for *generation*, simpler for understanding.
- **Per-patch features without a codebook**: not discrete; treated as soft tokens by the LLM, not in the vocabulary.
- **Diffusion latent encoders** (the VAE in latent diffusion): continuous-latent tokenizers, not discrete. Adjacent design space but for diffusion not LLMs.

## Sources

- Paper: *Neural Discrete Representation Learning* — van den Oord et al., 2017 — VQ-VAE.
- Paper: *Taming Transformers for High-Resolution Image Synthesis* — Esser et al., 2021 — VQ-GAN.
- Paper: *Finite Scalar Quantization: VQ-VAE Made Simple* — Mentzer et al., 2023 — FSQ.
- Paper: *MAGVIT-v2* — Yu et al., 2023 — lookup-free quantization for video.
- Paper: *ViQ: Text-Aligned Visual Quantized Representations at Any Resolution* — Yu, Liu, Yang et al., 2026 — text-aligned native-resolution tokenizer.
