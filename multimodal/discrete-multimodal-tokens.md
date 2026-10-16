# Discrete Multimodal Tokens
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Add image and audio as **extra entries in the LM's vocabulary** rather than through a separate encoder → projection → cross-attention pipeline. Modality content is quantized into **discrete codes** (image codes, audio codes) that occupy new embedding-matrix rows; the LM head stays text-only, so the model understands multimodal input but only generates text. Used by Apertus 1.5 (8B / 70B).

**Prereqs:** [../fundamentals/_tokenization.md](../fundamentals/_tokenization.md), [../fundamentals/bpe.md](../fundamentals/bpe.md)
**Related:** [README.md](./README.md) · [../case-studies/apertus-1-5.md](../case-studies/apertus-1-5.md)

---

## What it is

Standard VLM / omni-modal architectures use a vision encoder (ViT, SigLIP, …) whose output is projected into the LM's residual stream through a learned adapter / Q-Former / cross-attention. Modalities enter the model through activations, not tokens.

Discrete-multimodal-tokens inverts this: the modality is **quantized** into discrete codes (e.g. VQ or RVQ for images and audio), and those codes are added as **new rows in the embedding matrix**. From the Transformer's perspective, multimodal input is just more tokens with their own ids.

## How it works

### Vocabulary layout

Apertus 1.5's concrete layout:

- 131,072 original text entries (inherited unchanged from Apertus 1.0).
- 200 reserved multimodal slots (structure markers: `<image>`, `<audio>`, start/end markers, bucket ids).
- **131,072 image codes** — the output alphabet of the image tokenizer.
- **4,096 audio codes** — the output alphabet of the audio tokenizer.
- Total: 266,440 real entries; the released embedding matrix is 266,752 rows (312 padding for alignment).

### Encoder side

Modality-specific tokenizers (image VQ codebook; audio frame-level codebook) convert raw inputs to code sequences. Each code is just an integer id in the extended vocabulary. Structure markers surround the content codes to tell the model where each modality block starts/ends.

### Decoder side

**The LM head is restricted to the original text vocabulary.** The output softmax is over text ids only. Multimodal codes have embeddings for *input*, no output rows. Consequence: the model is **multimodal-input, text-output**.

### Training

Continued pretraining mixes text with multimodal-code-interleaved sequences. The LM loss is computed only on text positions (loss over multimodal code positions is masked). Attention is full across modalities so the model can condition text generation on image / audio context.

## Why it matters

- **No cross-modal adapters, no projection stack.** The architecture is literally the text Transformer; only the vocabulary and the input tokenizers change. Less to tune, less to break.
- **Reuses the full pretrained text model.** Continued pretraining upgrades a text LM in place without redesigning the model body.
- **Natural fit for continued pretraining.** All losses stay the same (text next-token prediction) because outputs are text; the model learns to attend over multimodal tokens as context.
- **Open-source friendly.** No encoder weights to source, no cross-modal data format to invent — just extend the vocabulary.

## Gotchas & tricks

- **Needs a good discrete tokenizer per modality.** The modality tokenizer bounds representation quality; a weak codebook is a hard ceiling. Apertus 1.5 uses 131k image codes and 4k audio codes.
- **Can't generate modality content.** Because the LM head is text-only, the model does not emit images or audio. For generation across modalities, a separate decoder or detokenizer is needed.
- **Code-sequence length can balloon.** An image of 131k codes is a long sequence; multimodal continued pretraining pairs well with long-context extension (Apertus 1.5 extended to 262k partly to accommodate this).
- **Structure markers matter.** The 200 reserved slots do real work — the model uses them to segment modality blocks and to know when to shift attention regime.
- **Embedding matrix grows significantly.** 266k entries vs 131k roughly doubles embedding parameters. For small models this is a nontrivial fraction of total parameters; worth budgeting.
- **Loss masking is non-optional.** Training the LM to predict image/audio codes degrades text generation and wastes compute; mask the multimodal positions.

## Sources

- Paper: *Apertus v1.5 Technical Report: From Open Text Models to Open Multimodal Reasoners* — Xu, Turki, Sabolčec, Pyatkin, Bosselut, Jaggi, Schlag et al., 2026 — [arXiv:2610.15542](https://arxiv.org/abs/2610.15542). Vocabulary layout: 131,072 text + 200 markers + 131,072 image codes + 4,096 audio codes = 266,440 entries. Output head restricted to text.
