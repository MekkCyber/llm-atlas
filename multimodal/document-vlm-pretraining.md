# Document-VLM Pretraining (Dual Generation + Reconstruction)
*Depth — pretraining a document-native vision encoder with joint image→text generation and pixel-level reconstruction.*

**TL;DR:** Standard vision encoders (CLIP, DINO, SAM) are pretrained on natural images and drop the character strokes and layout that document AI depends on. The dual-objective recipe from MonkeyOCRv2 fixes this by jointly training an encoder with (a) autoregressive image→text generation to align features with semantics, and (b) pixel-level document reconstruction to preserve strokes and layout. The trained encoder can then be **frozen** and paired with a small LM, and it outperforms general-purpose encoders on document parsing and understanding at a fraction of the size.

**Prereqs:** [transformer-block](../architectures/transformer-block.md)
**Related:** *(no other multimodal depth files yet)*

---

## What it is

A pretraining recipe for a vision encoder specialized for documents: dense text, formulas, tables, handwriting, and multi-language scripts. The encoder is domain-native — pretraining data is *document images*, not natural photographs — and the objectives are chosen so the learned features are both semantically aligned (they know what the text says) and pixel-faithful (they preserve strokes so the LM head can recover characters accurately).

Sits inside the multimodal stack in the same slot as CLIP or SigLIP in a general VLM: encoder → projection → language model.

## How it works

Two heads share a single vision backbone and are optimized jointly:

1. **Image → text generation.** An LM head autoregressively decodes the document's transcribed text from the encoder features. This pulls the visual representation toward *what the document says* — high-level semantic alignment.
2. **Pixel-level reconstruction.** A decoder reconstructs the document image (or a masked/downsampled version) from the same encoder features. This penalizes throwing away stroke-level detail, which the generation head alone doesn't punish.

The two losses are summed. MonkeyOCRv2 pretrains on **MonkeyDoc v2**, ~113M document images across 17 languages — mix of digital-born (clean rendered PDFs) and photographed (real-world). After pretraining the encoder is used two ways:

- **Frozen + tiny LM head** for a specialist document parser.
- **Frozen** as a drop-in replacement for CLIP/DINO/SAM in a general document-understanding VLM.

## Why it matters

Document AI has been treated as an adapter-on-top-of-CLIP problem: take a natural-image encoder, hope the projection layer learns character-level detail. It doesn't fully — dense text and fine strokes are outside CLIP's pretraining distribution. Retraining the encoder on documents with the right objectives closes a large gap:

- A 0.7B MonkeyOCRv2 parser beats the 3B dots.mocr on MDPBench by **+2.8** absolute with an encoder ~11× smaller.
- The frozen encoder beats CLIP, DINO, and SAM across 8 benchmarks under identical training.

The generalizable lesson: when the target domain is far from natural-image statistics, domain-native visual pretraining beats domain adaptation of a general backbone.

## Gotchas & tricks

- **Reconstruction loss is what saves the strokes.** Pure image→text pretraining lets the encoder throw away pixels it doesn't need to produce text; that hurts formula recognition and tampering detection where geometry matters.
- **Multi-language document data is essential.** English-only pretraining tanks on non-Latin scripts even with the same objectives.
- **Freezing the encoder is the point.** The paper's efficiency win (11× smaller encoder) depends on the encoder being frozen and paired with a lightweight LM — training the encoder end-to-end per task would erase the gains.
- **Digital-born vs. photographed is a real distribution shift.** Both must be in pretraining; otherwise the encoder collapses on the missing side.

## Sources

- Paper: *MonkeyOCRv2: A Visual-Text Foundation Model for Document AI* — Liu et al., HUST, 2026 — arXiv 2607.11562.
- Data: MonkeyDoc v2 — 113M document images, 17 languages.
- Benchmark: MDPBench — multi-language document parsing.
