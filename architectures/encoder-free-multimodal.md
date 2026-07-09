# Encoder-free multimodal architecture
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Drop separate vision/audio encoders. Instead, tokenize raw image patches and raw audio directly into the LM's input stream and let the same transformer weights process every modality. Pursued by Fuyu and Chameleon at smaller scale; Gemma 4's 12B variant is the largest open-weight instance to date. Simplifies the modality-fusion pipeline at the cost of forcing the LM to learn low-level modality features itself.

**Prereqs:** [transformer-block.md](./transformer-block.md), [../multimodal/README.md](../multimodal/README.md)
**Related:** [../multimodal/unified-multimodal-generation.md](../multimodal/unified-multimodal-generation.md), [../architectures/_moe.md](./_moe.md)

---

## What it is

The dominant multimodal LM design uses a **separate vision encoder** (a ViT, SigLIP, EVA-CLIP, etc.) that produces feature vectors → a **projection layer** that maps them into the LM's token space → the LM. Audio and other modalities follow the same pattern with their own encoders.

Encoder-free multimodal removes the encoder step entirely. Raw image patches (pixel blocks) or raw audio waveforms are tokenized (via a light linear layer or per-modality tokenizer) directly into the LM's input embedding space. The LM's own transformer layers do all subsequent modality processing.

## How it works

For images:

1. Split the image into a grid of patches (e.g., 16×16 pixels per patch).
2. Flatten each patch and apply a linear projection to LM hidden size — no ViT.
3. Add 2D positional encoding (row + column of patch).
4. Concatenate with text tokens in the input stream.

For audio: analogous — small-time-window frames or a light spectrogram tokenizer, no separate audio encoder.

The LM's attention layers then handle intra-modal and cross-modal fusion in the same computation. No modality-specific parameters exist beyond the tokenizer.

## Why it matters

- **Removes an entire artifact from the pipeline.** The vision encoder is separate code, separate training, separate serving. Its removal simplifies the whole system.
- **Modality parameters unified.** For MoE routing, positional encoding scaling, quantization, and inference optimization, you only need to handle one artifact.
- **Better cross-modal grounding potentially.** With a separate encoder, cross-modal features are compressed before they meet text; encoder-free preserves raw signal further into the network.
- **Scale is the open question.** Fuyu (8B) and Chameleon (7B+34B) showed encoder-free works. Gemma-4-12B pushes the class up; if the 12B matches or beats its encoder-equipped Gemma 4 siblings, encoder-free wins the design debate.

## Gotchas & tricks

- **The LM absorbs low-level features.** With a separate ViT, image low-level features (edges, textures) are handled in the encoder. Encoder-free means the LM's early layers do this — potentially costing capacity that would otherwise go to language modeling.
- **Positional encoding for images needs 2D structure.** RoPE, ALiBi, and other 1D positional encodings need adaptation. Common: 2D RoPE, or interleaving row/col into the standard 1D scheme.
- **Longer input sequences.** A 224×224 image at 16×16 patches = 196 tokens per image, before considering audio. Long-context scaling matters more than in text-only.
- **Requires training from scratch (or near).** Retrofitting an existing LM to be encoder-free is expensive — the LM's early layers weren't trained to handle raw patches. Fuyu and Chameleon trained from scratch; Gemma 4's 12B likely did too.
- **Not a universal win.** For pure image-in-language-out tasks (VQA, captioning), a well-trained ViT encoder still gives strong features cheaply. Encoder-free's payoff is in image-out and mixed-modal generation, where the encoder-decoder split becomes a bottleneck.
- **Distinct from "unified multimodal generation."** The output side of that concept ([../multimodal/unified-multimodal-generation.md](../multimodal/unified-multimodal-generation.md)) is orthogonal — you can be encoder-free on the input and still have task heads on the output. Gemma 4 does both.

## Sources

- Paper: *Gemma 4 Technical Report* — Gemma Team, Google DeepMind, 2026 — [arXiv:2607.02770](https://arxiv.org/abs/2607.02770). 12B variant.
- Paper: *Fuyu-8B: A Multimodal Architecture for AI Agents* — Adept, 2023 — smaller-scale precursor.
- Paper: *Chameleon: Mixed-Modal Early-Fusion Foundation Models* — Meta AI, 2024 — 7B and 34B encoder-free.
