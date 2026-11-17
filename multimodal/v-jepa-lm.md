# V-JEPA-LM (counterfactual supervision for dynamics-aware VLMs)
*Depth — teach a video-language model to read the dynamics signal already present in a predictive video encoder, using counterfactual (video, perturbed-video) pairs.*

**TL;DR:** Video encoders like V-JEPA learn scene dynamics, but video-language models trained on captions/QA mostly ignore them. The paper shows the dynamics information sits in a *subspace shared across videos* for predictive encoders (V-JEPA) but not for frame-wise encoders (SigLIP). Counterfactual supervision — training pairs where a pixel transformation flips the correct answer — forces the LM to read that subspace. V-JEPA-LM at 1B–8B leads its size classes across 11 video benchmarks.

**Prereqs:** [./README.md](./README.md)
**Related:** [./README.md](./README.md)

---

## What it is

A recipe for turning an already-strong video encoder (V-JEPA family) into a video-language model that *uses* the encoder's dynamics signal, rather than shortcutting through captions.

## How it works

**Diagnosis.** Probe where each video encoder stores motion information.
- **V-JEPA** (predictive, trained to predict masked future features) places dynamics in a low-dimensional subspace shared across videos — a few directions carry "what changes" cleanly.
- **SigLIP** (frame-wise, contrastively aligned with captions) does not have such a shared subspace; dynamics information is diffuse and tangled with per-frame content.

**Counterfactual pair construction.** Start with a public video. Apply a pixel transformation that changes only dynamics — reverse time, mask trailing frames, swap frame order — producing a counterfactual video. Generate (video, caption-or-QA) pairs where the correct answer differs between original and counterfactual (e.g. "the ball moves toward the goal" vs "away from the goal"). The LM can only get both right by reading the dynamics subspace.

**Training.** Standard VLM projector + LM fine-tuning, with the counterfactual pairs added to the supervision mix. The projector learns to route the dynamics subspace into token embeddings the LM actually attends to.

## Why it matters

Pins down a concrete reason mainstream VLMs stall on temporal reasoning: the encoder *has* the signal, the objective doesn't exercise it. The counterfactual-supervision pattern is general — the same recipe (construct pairs that differ only along a known-informative axis, train to flip the answer) can force any multimodal model to attend to any identified subspace. Transferable beyond video: audio dynamics, 3D geometry, retrieval-provenance.

## Gotchas & tricks

- **Only works if the encoder has the signal.** Running this recipe on a SigLIP backbone wouldn't help — the subspace isn't there. Encoder choice matters.
- **Transform selection.** Transformations that change only dynamics (reverse, reorder, mask-trailing) must leave static features intact; naïve augmentations (color-jitter, crop) also change appearance and confound the supervision.
- **Benchmark saturation risk.** Dynamics-heavy benchmarks reward this recipe heavily; static-VQA benchmarks see less lift. Expect gains to concentrate where temporal reasoning is tested.
- **Scales cleanly** from 1B to 8B; the dynamics subspace is a small compute tax on top of ordinary VLM training.

## Sources

- Paper: *V-JEPA-LM: Video-Language Models That Read Dynamics Beyond Static, Text-Aligned Representations* — Kim, Kim, Choi, 2026 — [arXiv:2611.17831](https://arxiv.org/abs/2611.17831)
- Related: V-JEPA (predictive video encoder), SigLIP (frame-wise contrastive encoder).
