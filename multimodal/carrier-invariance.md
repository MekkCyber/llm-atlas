# Carrier Invariance (LoMo)

*Depth — a data-curation paradigm that trains VLMs to treat semantically identical text and rendered-image content symmetrically.*

**TL;DR:** Replace a textual prompt with an image rendering of the same text and most VLMs collapse — they handle "What is the capital of France?" but fail on the same question rendered as an image. LoMo names this *carrier sensitivity* and traces it to a structural bias in training data (text is always the query, images are always references). It fixes the bias by dynamically substituting text spans with rendered-image versions in training prompts, forcing cross-modal representational invariance without architecture changes.

**Prereqs:** [README.md](README.md)
**Related:** [../data/_data-curation.md](../data/_data-curation.md)

---

## What it is

A data-curation recipe for VLM training that breaks the role-asymmetry between text and image carriers in standard multimodal corpora. Architecture-agnostic — drops into any existing VLM training run as a data transformation.

## How it works

Standard image-text training data (captioning, VQA, OCR, interleaved web) systematically casts text as the linguistic query and images as the visual reference. VLMs trained on this corpus learn *modality-asymmetric routing*: they read text deeply and treat images as auxiliary references, even when the content is identical.

LoMo's transformation: given a textual training prompt, dynamically pick spans within the prompt and re-render those spans as images. Splice the rendered images back into the prompt at the original positions, yielding an interleaved multimodal sequence with the same semantics as the original text. The model trains on the mixture of:

- The original text-only prompt.
- The fully-image-rendered version.
- The interleaved hybrid versions (random subsets of spans rendered).

All three carry the same target answer. The training objective is unchanged — supervised loss against the answer. The model must learn that the substituted spans are *semantically* identical to their text originals, which forces cross-modal representational invariance.

Implementation cost is small: a text renderer (PIL/font rendering) and a span sampler.

## Why it matters

- Today's "multimodal fusion" appears shallower than benchmark numbers imply — VLMs largely route around the visual stream when text is available. Carrier sensitivity is the diagnostic.
- A pure data fix that any open VLM can adopt without architecture or training-loop changes.
- Generalizes to interleaved-document tasks where text and figures co-occur (slides, papers, web pages) — the same invariance is what the deployment needs.

## Gotchas & tricks

- The rendering style matters. A single font/size produces a trivial invariance ("text rendered in Arial 12" ≠ "general image"); vary the font, size, and background to force genuine modality invariance rather than a surface-style shortcut.
- Span selection should favor semantically meaningful units (noun phrases, named entities, code identifiers). Random character-level substitution produces nonsense and degrades reading.
- The text-only and image-only extremes anchor the invariance; dropping either degrades transfer. Train on all three.
- Some carriers genuinely deserve different processing (handwritten notes ≠ Arial text). Carrier invariance is about *semantic* equivalence; don't over-apply.
- Pairs cleanly with vision-encoder choice (CLIP, SigLIP, native-pixel) — the encoder's text-image alignment determines how cheaply LoMo can close the gap.

## Sources

- Paper: *LoMo: Local Modality Substitution for Deeper Vision-Language Fusion* — Han, Zhang, Liang, Wang, Wang — Fudan / Shanghai Innovation Institute / SJTU / USTC / JD.COM, 2026 — [arXiv 2605.30265](https://arxiv.org/abs/2605.30265). Project: https://maplebb.github.io/LoMo
