# Visual Tokenization

*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** The image analogue of text BPE: encode an image into a sequence of discrete tokens drawn from a learned codebook, so a multimodal LLM can treat vision symmetrically with text. The long-standing tradeoff has been **detail vs semantics** — reconstruction-trained codebooks lose meaning; semantic-trained codebooks lose detail. **ViQ** (2026) proposes a text-aligned VQ encoder that supports **native resolution** and balances both.

**Prereqs:** [../fundamentals/_tokenization.md](../fundamentals/_tokenization.md)
**Related:** [danceopd](danceopd.md), [in-context-world-modeling](in-context-world-modeling.md)

---

## What it is

A discrete image encoder $E: \text{image} \to (z_1, \dots, z_n) \in \{1, \dots, V\}^n$ that turns a 2D image into a sequence of integer codes from a vocabulary of size $V$. Paired with a decoder $D$ that reconstructs an image from the code sequence. The output sequence is what a unified text+vision autoregressive model consumes alongside text tokens.

Two design objectives compete:

- **Reconstruction.** The decoder must produce a faithful image. Codes have to carry low-level texture/edge/color information.
- **Semantic alignment.** The codes have to be useful to a downstream LLM. They should land near text directions in semantic space.

VQ-VAE / VQ-GAN-style encoders win on reconstruction but their codes are semantically unstructured; CLIP-quantized encoders win on semantics but blur reconstruction. ViQ argues you don't have to pick.

## How it works

The ViQ design has three pieces:

1. **Text-aligned codebook.** The codebook is trained so that code-vector centroids align with directions in a text embedding space (CLIP-style supervision). This forces the discrete tokens to carry semantic structure usable by an LLM directly.
2. **High-detail decoder branch.** A separate decoder path is trained against pixel-level reconstruction losses (L1 + perceptual). The encoder's residual capacity is split — semantic features go to the LLM-facing codebook, fine-detail features feed the decoder.
3. **Any-resolution input.** Earlier VQ encoders fix a square crop and a fixed grid. ViQ supports native input resolutions — important for downstream multimodal LLMs that need to ingest screenshots, documents, charts, etc., where center-crop loses critical structure.

The encoder is positioned as a unified discrete representation for arbitrary visual inputs (T2I conditioning, multimodal LLM input, retrieval, control conditioning).

## Why it matters

- **Unblocks unified autoregressive text+image models.** The "tokenize images like text" agenda has been stuck on the detail-vs-semantics frontier for years. ViQ pushes that frontier.
- **Native resolution removes a UX-blocking constraint.** Downstream VLMs ingesting documents, code snippets, GUIs, or charts can't tolerate center-crop; ViQ lets the tokenizer match the input's aspect/scale.
- **Bridges two architectures.** Continuous-projection VLMs (BLIP-2, LLaVA) treat vision as soft features. ViQ-style discrete tokens make the same VLM tractable as a pure autoregressive model — relevant to "any-to-any" frontier designs.

## Gotchas & tricks

- **Codebook collapse** is the classical failure mode of VQ training — most codes go unused. Mitigations include code reset, codebook regularization, and EMA codebook updates.
- **Tokens-per-image budget.** More tokens → better detail → much higher LLM context cost downstream. Picking the right budget is workload-dependent; native-resolution support also means *variable* token counts per sample.
- **Reconstruction-quality alone is a misleading metric.** ViQ explicitly argues that the right downstream metric is multimodal-LLM accuracy on the resulting tokens, not PSNR.
- **Composes with separate continuous projection.** Some recent VLMs use a hybrid (discrete codes + continuous residual) to recover the last few PSNR points — orthogonal to the ViQ contribution.
- **Vocabulary size matters.** Too small → loses semantic discrimination; too large → poor codebook usage. ViQ tunes this carefully; transfer the choice to your own setup with care.

## Sources

- Paper: *ViQ: Text-Aligned Visual Quantized Representations at Any Resolution* — Yu, Liu, Yang, Dong, Qian, Lu, Hu, Rao, 2026 — [arXiv:2606.27313](https://arxiv.org/abs/2606.27313).
- Background: *Neural Discrete Representation Learning (VQ-VAE)* — van den Oord et al., 2017 — foundation of the family.
- Background: *Taming Transformers for High-Resolution Image Synthesis (VQ-GAN)* — Esser et al., 2021.
- Background: *MAGVIT-v2: Language Model Beats Diffusion* — Yu et al., 2024 — earlier high-fidelity LLM-friendly tokenizer.
