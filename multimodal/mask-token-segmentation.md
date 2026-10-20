# Mask-Token Segmentation (Shared-Representation Mask Supervision)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A way to fold pixel-level segmentation into a vision-language model's **autoregressive output stream**, with no separate segmenter. The model emits a special `<|mask|>` token; its hidden state is read by a lightweight **mask decoder** that fuses the state with vision-encoder patch features and produces a mask. Mask loss flows back through the LLM, so pixel supervision updates the representations used for language generation — including mid-reasoning states inside `<think>` blocks.

**Prereqs:** _(none yet — multimodal folder is being seeded)_
**Related:** [xiaomi-lens-0 case study](../case-studies/xiaomi-lens-0.md)

---

## What it is

Standard VLM-plus-segmentation stacks bolt a specialist decoder onto a frozen vision encoder: the LLM produces a reference ("the red cup on the left") and the external segmenter does the pixel work. Two drawbacks: the LLM never sees pixel loss, so its representations are not refined by segmentation, and the segmenter can't be conditioned on the LLM's chain of thought mid-reasoning.

Mask-token segmentation moves the mask decoder *inside* the LLM's generation loop. A special `<|mask|>` token appears in the response. Its hidden state is passed to a mask decoder that fuses it with patch features from the shared vision encoder, producing a mask. Loss on that mask backpropagates through the backbone.

## How it works

**Token-level mask trigger.** The model is trained to emit `<mask><|mask|></mask>` whenever a mask is called for. The `<|mask|>` token can appear anywhere in the response — including inside a `<think>` block — so a chain of thought can produce several masks, each conditioned on everything thought so far.

**Mask decoder.** A lightweight head reads two inputs:
1. The **hidden state at the `<|mask|>` position** — the "visual latent" produced by the LLM.
2. The **patch-level features** from the vision encoder (reused, not re-run).

A small cross-attention or FiLM-style fusion is enough; the heavy lifting is already in the backbone and the vision encoder.

**Loss.** Standard mask losses (BCE + Dice on the predicted mask vs ground-truth) are added to the language modeling loss. The mask loss has a direct path back through the mask decoder to the `<|mask|>` hidden state and from there into the shared backbone.

**Boxes alongside.** The same stream carries boxes as text: `<bbox>[x1, y1, x2, y2]</bbox>`. Box supervision is pure cross-entropy on the box-token coordinates, no new decoder. Boxes and masks can both appear in a response.

## Why it matters

- **Shared gradients, not shared only features.** The LLM's representations receive pixel-level signal. Reported ablations show mask supervision also lifts general perception benchmarks (MME-RealWorld, CV-Bench), not just segmentation metrics.
- **Mid-reasoning masks.** A `<|mask|>` inside `<think>` produces a mask conditioned on the ongoing chain of thought. External segmenters cannot do this cleanly without a round-trip through text.
- **No specialist encoder.** The segmentation decoder is lightweight; no SAM-sized model is needed at inference. The vision encoder is already paying its cost for VQA.
- **One stream, many outputs.** Text, boxes, and masks share a single autoregressive output, which simplifies serving (no multi-head routing, no modality switch).

## Gotchas & tricks

- **Mask-token placement in training data.** The target position of `<|mask|>` matters — supervising a mask when the preceding tokens haven't yet identified a target gives noisy gradients. Data pipelines synthesize instructions so the mask token always follows a well-formed reference.
- **Patch-feature caching.** The vision encoder runs once per image; the mask decoder re-reads its cached features at each `<|mask|>` position.
- **Three-way loss weighting matters.** Mask / box / text terms must be balanced; the paper's details are partial but the resulting checkpoints are an equal-weight average of three Stage-2 runs.
- **Not for pure-pixel tasks.** Dense segmentation benchmarks (ADE20K, COCO panoptic) still favor specialist models. The design pays off when segmentation is in the loop with reasoning or VQA, not as a standalone task.

## Sources

- Paper: *Xiaomi-Lens-0 Technical Report* — Xiaomi, 2026 — [arXiv:2610.23279](https://arxiv.org/abs/2610.23279)
- Prior art: LISA (large-language segmentation assistant, 2023), GLaMM (2024), SAM-3 (used as a mask generator in the data pipeline).
