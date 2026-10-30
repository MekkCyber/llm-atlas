# Semantic Image Tokenizer
*Depth — image tokenizers trained with language supervision instead of pixel reconstruction.*

**TL;DR:** The classical VQ-VAE / VQGAN recipe trains an image tokenizer with a **pixel reconstruction loss**; the token codebook ends up encoding texture and local structure. A **semantic image tokenizer** drops the reconstruction target entirely and trains with **language supervision alone** (text-image contrastive or caption prediction) so that each code encodes a semantic unit rather than a pixel pattern. "Know, then Draw" (Ge et al., Adobe / UCSD, 2026) argues this is the *right* tokenizer for autoregressive generative pretraining: semantic token quality — not reconstruction PSNR — predicts final generation performance when the pipeline is AR-pretrain → diffusion-post-train.

**Prereqs:** *(none)*
**Related:** [../fundamentals/bpe.md](../fundamentals/bpe.md), [vla.md](vla.md)

---

## What it is

An image tokenizer is a learned encoder `E: image → codes` that maps continuous pixels to a discrete code sequence (VQ codebook) or a short continuous latent sequence. Three families:

| Family | Training objective | Codes encode |
| --- | --- | --- |
| **Reconstruction** (VQ-VAE, VQGAN, FSQ) | pixel reconstruction (+ adversarial, perceptual) | low-level texture + structure |
| **Hybrid** | reconstruction + language or classifier head | mixed |
| **Semantic** | language supervision only (CLIP-style contrastive or captioning) | high-level semantics |

Semantic tokenizers were historically used for *understanding* (CLIP, SigLIP) but discarded for *generation* on the intuition that you can't reconstruct pixels from semantic codes. "Know, then Draw" argues the premise is wrong: in a staged AR → diffusion pipeline, the AR stage *does not need* reconstruction-grade codes — the diffusion stage will fill in high-frequency detail.

---

## How it works

### Training objective (Know, then Draw)

Given an image–text dataset `{(I, c)}`:

```
tokens = E(I)                       # discrete codes
loss = CE(decoder(tokens), c)       # autoregressive caption prediction
                                    # OR CLIP-style contrastive on pooled tokens
```

No pixel loss. No GAN discriminator. The codebook is updated through text gradients only.

### Pipeline position

```
Image → Semantic Tokenizer E (frozen)
      → AR Transformer trained on next-token-prediction over E(I)       [knowledge stage]
      → Diffusion Head conditioned on AR hidden states                   [rendering stage]
      → Pixels
```

The AR stage learns *what to draw*; the diffusion stage learns *how to draw it*. The tokenizer's job is purely to provide a stable semantic basis for the AR objective.

### Scaling behavior

The paper fits compute-optimal scaling laws for the two-stage pipeline. Headline results on a 3B model trained for ~6,600 H100 GPU-hours: **GenEval 0.86**, **DPG-Bench 87.78**, competitive with much larger diffusion-only models.

---

## Why it matters

- **Removes the VQ-VAE reconstruction bottleneck for AR generation.** AR tokenizers with reconstruction loss are notorious for low codebook utilization and blurry low-frequency content. Semantic codes side-step both issues.
- **Semantic quality predicts generation quality.** The paper's main empirical claim is that downstream GenEval / DPG-Bench performance tracks tokenizer *semantic* metrics (text-image retrieval accuracy on held-out captions), not reconstruction PSNR / FID.
- **Separates concerns cleanly.** AR learns the discrete "concept distribution"; diffusion learns the continuous "detail distribution". Each paradigm plays to its strength — a cleaner factorization than monolithic diffusion.
- **Opens AR-pretraining for images with LLM-grade recipes.** With a semantic tokenizer, image AR looks structurally like text AR: next-token prediction over discrete semantic units. Scaling-law machinery transfers directly.

---

## Gotchas & tricks

- **Needs a diffusion post-training stage.** Semantic codes alone cannot be decoded back to high-fidelity pixels. The paper's pipeline is explicitly two-stage; using a semantic tokenizer without a rendering head gives low-quality outputs.
- **Codebook size tradeoff.** Smaller codebooks (8K–16K) force semantic abstraction; larger ones (64K+) can memorize texture-like features and partially reconstruction-ify. Semantic-only training usually benefits from the smaller end.
- **Language-supervision bias.** If the captioner is biased (short captions, missing attributes), the codes inherit that bias. This is the image-tokenizer analogue of pretraining data bias in LLMs.
- **Not useful for pure image-reconstruction tasks.** If you want a tokenizer for compression or in-painting without a diffusion head, stick to reconstruction tokenizers.

---

## Sources

- Paper: *Know, then Draw: A New Learning Paradigm for Visual Generation* — Ge, Tu, Tan, Jiang, Adobe Research / UCSD, 2026, [arXiv 2610.37626](https://arxiv.org/abs/2610.37626) — introduces the semantic-only tokenizer argument and reports the GenEval / DPG-Bench numbers.
- Related: VQ-VAE (van den Oord 2017), VQGAN (Esser 2021), FSQ (Mentzer 2023) — reconstruction baselines.
- Related: CLIP (Radford 2021), SigLIP (Zhai 2023) — language-supervised image encoders, the ancestry.
