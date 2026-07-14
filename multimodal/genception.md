# GenCeption
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Reuse a pretrained **text-to-video diffusion backbone** as a general-purpose vision encoder: attach a small feed-forward head, condition on task text, and read off depth, surface normals, camera pose, expression-referring segmentation, and 3D keypoints from the *same* frozen weights. Argues that large-scale text-to-video generation is the vision analogue of next-token prediction — the scalable pretraining objective that yields a truly *generalist* visual foundation model.

**Prereqs:** none.
**Related:** [../pre-training/README.md](../pre-training/README.md)

---

## What it is

Modern computer vision still relies on task-specialist backbones (DepthAnything for depth, SAM for segmentation, VGGT for pose, etc.). LLMs took the opposite path — one pretraining objective (next-token prediction) that yields one generalist backbone, adapted downstream with minimal task specialization.

GenCeption's thesis: the vision equivalent is **text-to-video generative pretraining**. A diffusion backbone trained to generate video conditioned on text already encodes the spatiotemporal, geometric, and vision-language priors that structured perception tasks need. All you have to do is *read them off*.

---

## How it works

### The generalist recipe

1. **Backbone**: a pretrained text-to-video diffusion model (DiT-style), frozen or lightly adapted.
2. **Head**: a small **feed-forward perception head** that projects backbone features to task-native outputs (depth map, normal map, 3D keypoints, segmentation mask).
3. **Task selection**: **text conditioning** — the same weights answer "output depth" vs "output surface normals" vs "output 3D keypoints" based on the text prompt. No task-specific fine-tuning.

Concretely: the diffusion backbone runs a single forward pass; features from a chosen layer are read out; the head maps them to the task-native tensor. The head is task-family-shared, not per-task.

### Why generative pretraining transfers

A text-to-video generator has to internally model:
- **Spatial geometry** — objects have consistent shape across frames.
- **Temporal dynamics** — motion is coherent.
- **Vision-language alignment** — text prompts specify content, style, camera, motion.
- **Scale** — trained on internet-scale video corpora.

These are exactly the priors dense perception needs. GenCeption operationalizes the observation.

---

## Why it matters

- **Matches or beats specialists with 7×–500× less data.** GenCeption reports SoTA on multiple benchmarks (matting, depth, referring segmentation) while using drastically less task-specific training data than baselines like DepthAnything3, SAM3, VGGT-Ω, Sapiens, Lotus-2.
- **Beats other pretraining paradigms.** Under matched settings, the video-generative backbone outperforms V-JEPA and Video MAE as a pretraining objective for perception.
- **Emergent OOD generalization.** A model trained *only* on synthetic human videos generalizes to real-world footage and out-of-distribution categories (animals, robots).
- **Collapses the specialist zoo.** If replicated, the vision-model stack becomes one backbone + task-conditioned heads, mirroring how LLMs eat the NLP task landscape.

---

## Gotchas & tricks

- **Depends on the video-generative backbone being strong.** GenCeption's transfer quality tracks generation quality of the underlying model — expect the recipe's ceiling to rise as text-to-video models scale.
- **Text-conditioned task selection has to be interpretable to the model.** The paper uses natural-language task descriptions; ambiguous prompts degrade quality.
- **Not a real-time system yet.** Diffusion backbones are expensive to forward; the perception pass inherits that cost. Head-only distillation is a natural next step.
- **Scaling behavior is preliminary.** The paper reports scaling *trends* (data efficiency, backbone size) but doesn't run a full scaling-law sweep — the strong claim is directional, not law-level.
- **Generative vs discriminative pretraining is the deeper question.** GenCeption's headline claim is that generative pretraining beats self-supervised discriminative (V-JEPA, Video MAE); community consensus on this is still forming.

---

## Sources

- Paper: *Video Generation Models are General-Purpose Vision Learners* — Zhang, Kabra, Uijlings, Waslander, Zisserman, Carreira, He, Andriluka, Bazavan, Zanfir, Sminchisescu — Google DeepMind — [arXiv:2607.09024](https://arxiv.org/abs/2607.09024).
- Baselines cited: DepthAnything3, SAM3, D4RT, VGGT-Ω, Sapiens, Lotus-2, V-JEPA, Video MAE.
