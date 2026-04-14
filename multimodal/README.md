# Multimodal

*How models learn to see and read — vision encoders, projection layers, interleaved pretraining, and the data, training, and evaluation challenges unique to vision-language models.*

---

## What This Is

Multimodal models add a perceptual front-end to a language model. The architecture choices (encoder vs. native, early vs. late fusion, token budget per image) and the training recipes (interleaved image-text pretraining, multimodal SFT, visual RLHF) determine what the model can actually see and reason about.

---

## What Belongs Here

- **Vision encoders** — CLIP, SigLIP, native-pixel encoders.
- **Projection & fusion** — linear projections, Q-former, cross-attention, early vs. late fusion.
- **Multimodal pretraining** — interleaved image-text corpora, captioning, grounding.
- **Multimodal post-training** — SFT, preference learning with image inputs.
- **Video** — temporal encoders, frame sampling, long-video understanding.
- **Multimodal evaluation** — VQA, grounding, chart/document understanding.

## Reading Order

1. Vision encoders (CLIP and successors)
2. Projection & fusion strategies
3. Multimodal pretraining data & objectives
4. Multimodal post-training
5. Video-language models
6. Multimodal evaluation

---

## Related

- [architectures/](../architectures/) — how vision components integrate with Transformer blocks.
- [pre-training/](../pre-training/) — multimodal pretraining mirrors text pretraining's scaling questions.
- [data/](../data/) — image-text corpora curation.
