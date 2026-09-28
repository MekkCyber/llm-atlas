# WROP — World Reasoning with Object Permanence
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A benchmark for evaluating **object permanence and solidity** in video-generation models, framed as a **cognitive-science-derived taxonomy**. 150 hand-designed tasks across six cognitive categories, each rendered by Blender generators that randomize speed, lighting, camera, and other nuisance parameters while preserving the task's core cognitive structure. Includes a 1.5M-sample training corpus and a 300-question **blind pairwise Elo exam** covering reference-to-video, edit, and continuation model classes.

**Prereqs:** [README.md](README.md)
**Related:** [../multimodal/README.md](../multimodal/README.md)

---

## What it is

Video generators are increasingly cast as "world models," but there is no clean way to check whether they encode developmentally-basic priors — objects continue to exist when occluded, solids can't pass through each other. WROP borrows the classical inventory of object-permanence probes from cognitive-science research and renders each as a video-model task.

## How it works

- **Six cognitive categories** derived from the object-permanence literature (containment, occlusion, solid-passing, and three others).
- **150 hand-designed tasks**, each defined by a *cognitive structure* rather than a specific rendering.
- **Blender procedural generators** per task that randomize surface-level nuisance parameters (lighting, camera angle, speed, textures) while preserving cognitive structure. Yields 10,000+ samples per task, ~1.5M total for training.
- **300-question exam** for evaluation.
- **Blind pairwise Elo protocol** to score models: pit two models' outputs on the same task, blind human judgment, aggregate.

Three model classes are compared: reference-to-video, edit, and continuation. The paper's own **PWM-WROP** (a 16B continuation model trained on the 1.5M corpus) ranks 1st among continuation models and 3rd overall, statistically tied with two reference-to-video models.

## Why it matters

- First cognitively-motivated exam for "does your video model actually understand object permanence?" — a prerequisite for using video generators inside agent training loops as world models.
- The **procedural-generator + core-cognitive-signal** design is a template for building benchmarks that resist surface-feature shortcuts.
- Elo-based blind evaluation avoids the metric-hacking problems of automated scorers on generative video.

## Gotchas & tricks

- Elo scores are population-relative; adding new models to the pool can move existing rankings. Report the model pool alongside the score.
- Blender-rendered scenes may share stylistic tells that a model can shortcut; watch for cross-task style-based scoring rather than genuine cognitive-property scoring.
- Continuation models benefit from access to a starting frame; the pairwise comparison across classes is not a clean apples-to-apples comparison and should be read alongside per-class rankings.

## Sources

- Paper: *Training Object Permanence in World Models* — Zhang, Yu, Luo, Sun, Zhao, Gao, Li, An, Qin, Zhang, Jiang, Feng, Zhang, Guo, Wang, Yang, Mei, Wang, Ji, Kumar, Shi, Sripada, Muller, Torr, Yuille, Kriegeskorte, Juefei-Xu, Zhang, Chen, Du, Deng — 2026 — [arXiv:2609.28654](https://arxiv.org/abs/2609.28654).
