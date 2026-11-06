# Native (Encoder-Free) Vision-Language-Action Models
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Standard VLA models bolt a frozen vision encoder (CLIP, SigLIP, DinoV2) onto an LLM, which compresses pixels into a fixed representation *before* action reasoning starts. A **native** VLA removes the encoder: a thin patch-conv tokenizer feeds raw image patches into the shared Transformer, and the action head reads visual features from *multiple* depths of the backbone — not just the final layer. Layer-wise probing shows that control-relevant signal often peaks before the final visual layer, so encoder-based VLAs are throwing it away.

**Prereqs:** [../architectures/multi-head-attention.md](../architectures/multi-head-attention.md), [../fundamentals/attention.md](../fundamentals/attention.md)
**Related:** [../post-training/dpo.md](../post-training/dpo.md) · [../architectures/transformer-block.md](../architectures/transformer-block.md)

---

## What it is

A VLA (Vision-Language-Action) is a policy conditioned on images + instructions that outputs robot actions. The dominant recipe since RT-2 / OpenVLA is:

```
frozen vision encoder  →  projector  →  LLM  →  action head
```

The frozen encoder is a design choice inherited from LLaVA-style VLMs, not a necessity. A **native** VLA replaces it with lightweight patch convolutions whose weights are learned jointly with the rest of the model, and gives the action expert multi-depth access to the backbone.

## How it works

**Patch-conv tokenization.** A small stack of conv layers (one or two) slices each frame into fixed-size patches and projects them into the Transformer's hidden size. No semantic pretraining; the tokens are "early" representations.

**Multi-depth action expert.** A flow-matching or diffusion-policy head attends to hidden states at *several* depths of the shared backbone — e.g. layers $\{L/4, L/2, 3L/4, L\}$ — rather than only the top layer. In NEO-VLA this is a flow-matching head using cross-attention into the backbone.

**Joint training.** Visual tokens, language tokens, and action tokens all go through one Transformer. Gradients flow end-to-end into the patch-conv tokenizer. No stage of frozen weights.

```
patches  ─┐
tokens    ├─►  shared Transformer (L layers)
          │        │   │   │
          │        ▼   ▼   ▼    multi-depth taps
          │      action expert (flow matching)
          │                │
          └────────────────┴──► action chunk
```

## Why it matters

The frozen-encoder pattern has been the universal baseline for VLAs, but it bakes in two unexamined assumptions: that **high-level visual features suffice** for control, and that **vision and language representations should live in different parameter sets**. Native VLAs challenge both. Empirically, action-relevant information peaks *before* the final visual layer — a bottleneck the frozen encoder hides and the native architecture can exploit. The practical payoff is robustness under distribution shift (new lighting, textures, camera placement) and smaller models at matched capability.

## Gotchas & tricks

- Patch-conv tokenizers need more data than frozen encoders early on — if the training mix is small, warm-start the Transformer from an LLM and only train the patch-conv + late layers.
- Multi-depth cross-attention adds compute to the action expert proportional to the number of taps; two or three is usually enough.
- Native VLAs break interoperability with pretrained visual tools (grounding, OCR adapters) that expect CLIP-space features — those have to be relearned inside the shared backbone or wired in as separate heads.

## Sources

- Paper: *From Pixels to Actions — Towards Native Vision-Language-Action Policies* — Fan et al., 2026 — https://arxiv.org/abs/2611.06918
