# Native Multimodal Pre-Training
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Train a multimodal model on interleaved multimodal inputs *from step 0*, rather than stitching a vision encoder onto a pre-trained LM (late fusion). Native pre-training removes optimization asymmetries between modalities and reaches deeper cross-modal integration. The 2026 study formalizes its scaling laws — fitting compute-vs-loss exponents across model size, data budget, and modality mix — and shows native catches up with and overtakes late fusion at scale on cross-modal tasks.

**Prereqs:** [../pre-training/README.md](../pre-training/README.md)
**Related:** [../architectures/transformer-block.md](../architectures/transformer-block.md)

---

## What it is

A multimodal foundation-model training recipe with **no unimodal pre-training warmup**. From the first optimizer step, the model sees interleaved image/video/text tokens. Contrasts with the dominant *late-fusion* paradigm (pretrain a strong text LM; attach a vision encoder + projection; fine-tune) and *early-fusion* variants (train the vision encoder and LM together but still initialize the LM from a text-only checkpoint).

## How it works

- **Interleaved corpora.** Training data is interleaved from the start — image-text pairs, VQA, document-with-figures, video-with-captions.
- **Single tokenizer / projection stack.** No separate warmup for a text-only pass; the model learns cross-modal token statistics jointly.
- **Scaling-law fitting.** The paper fits compute-optimal exponents (à la Chinchilla) on the native regime and reports where the frontier crosses late-fusion baselines.
- **Optimization symmetry.** Late fusion has a known asymmetry — the LM's gradient dominates because it was pretrained; native pretraining has no such asymmetry, so per-modality gradients are on comparable footing throughout training.

## Why it matters

Most frontier VLMs (Qwen-VL, InternVL, Gemini-Nano-family) take the late-fusion path because a strong text LM is cheap to reuse. A rigorous scaling-law study of the native path is what decides whether the field should invest in from-scratch multimodal foundation models. The study's finding — native scales more favorably on cross-modal tasks — is not by itself a mandate, but it sets a baseline others must beat.

## Gotchas & tricks

- **Data mixture is a first-class hyperparameter.** Late-fusion pipelines get to tune this at fine-tune time; native pipelines commit early and pay for it later.
- **Early-checkpoint text-only capability is worse.** Native models trail text-only baselines on pure-text tasks for a longer wall-clock window; this can panic teams into aborting the run.
- **Encoder architecture is still not settled.** Fully native tokenization (pixel patches directly through the LM) trades quality for simplicity vs. a shared encoder with cross-modal supervision.

## Sources

- Paper: *Scaling Native Multimodal Pre-Training From Scratch* — Wu, Wu, Wang, Wu, Ou, Yu (CUHK / Tencent LLM Department), 2026 — [arXiv:2607.22043](https://arxiv.org/abs/2607.22043).
