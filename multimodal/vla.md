# VLA — Vision-Language-Action models
*Depth — foundation models that map (image, instruction) → action tokens for embodied control.*

**TL;DR:** A VLA is a single transformer that takes a visual observation plus a natural-language instruction and outputs an action (typically as tokens over a discretized action space). The class emerged from VLMs by adding an action head / tokenizer, and now spans robotic manipulation, GUI agents, and video-game control. Modern variants are trained on internet video *plus* teleop or synthetic trajectories, often with in-context task specification.

**Prereqs:** [README.md](README.md)
**Related:** [../agents/README.md](../agents/README.md), [../agents/_world-models.md](../agents/_world-models.md)

---

## What it is

A **vision-language-action** model unifies perception, language grounding, and control in one autoregressive decoder. The inputs are image observations (single frame, video, or interleaved with proprioception) and a task instruction; the outputs are actions represented as discrete tokens — either a native action vocabulary (RT-2 style) or open-ended text that a downstream parser converts to control signals.

## How it works

**Backbone:** a VLM (usually a frozen or lightly tuned vision encoder + LLM). Actions are added by:

1. **Action tokenizer** — quantize each continuous action dimension into `N` bins, giving a `D × N` action vocabulary. Actions are then tokens the LLM emits alongside language.
2. **Joint training** — mix internet-scale (image, text) pretraining data with action-labeled trajectories (teleop, demos, or simulator rollouts). Losses are the same next-token cross-entropy, with weight sharing across modalities.
3. **Chunking** — emit `H`-step action chunks per inference call to amortize latency and improve consistency across timesteps.

Recent variants add **in-context conditioning** (Zero-WAM): pass a human demonstration video as context tokens; the VLA imitates the demo on a new task without weight updates. This is ICL, but for policies.

## Why it matters

VLAs are the substrate for a unified embodied stack. Same architecture serves robot arms, browsers, and game controllers — the differences live in the action tokenizer and the data mix. If in-context task specification generalizes, per-task fine-tuning largely goes away: a single deployed model can be steered by demo videos or text instructions at inference time.

## Gotchas & tricks

- Discretization granularity is a real hyperparameter — too coarse and control is jittery, too fine and the vocabulary blows up. 256 bins per dimension is a common default.
- Latency scales with chunk length and image resolution; deployed systems chunk 4–16 steps and cache attention across chunks.
- Domain gap between internet video and control data is large; loss weighting and staged curricula matter.
- In-context task specification (à la Zero-WAM) requires the pretraining mix to have shown many (demo, execution) pairs; a VLA trained only on instructions won't gain ICL for free.

## Sources

- RT-2: *Vision-Language-Action Models Transfer Web Knowledge to Robotic Control* — Google DeepMind, 2023 — [arXiv:2307.15818](https://arxiv.org/abs/2307.15818)
- OpenVLA: Kim et al., 2024 — [arXiv:2406.09246](https://arxiv.org/abs/2406.09246)
- Recent in-context variant: *Zero-WAM* — Zhou et al., 2026 — [arXiv:2608.26103](https://arxiv.org/abs/2608.26103)
