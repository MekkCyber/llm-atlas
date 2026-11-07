# Vision-Language-Action (VLA) Models
*Depth — the model class of foundation policies that take image + text and emit robot / UI actions.*

**TL;DR:** A **Vision-Language-Action (VLA) model** is a VLM-style model that outputs an **action** instead of (or in addition to) text — joint angles for a robot arm, waypoints for a vehicle, mouse clicks for a computer-use agent. The modern recipe inherits the VLM pretraining (image encoder + LLM backbone) and adds an action head trained on expert-demonstration / RL data. Representative open examples: RT-2, OpenVLA, π0 / π0.5, Octo. Two 2026 directions converge on this page: continual-pretraining alignment with 3D perception (**SLA**) and structure-aware fast action decoding (**VFlash**).

**Prereqs:** [attention](../fundamentals/attention.md)
**Related:** [multi-turn-editing](./multi-turn-editing.md), [step-distillation](./step-distillation.md), [vla-continual-pretraining](./vla-continual-pretraining.md), [vla-flash-decoding](./vla-flash-decoding.md)

---

## What it is

A VLA factors:

$$
\pi(a \mid o_{\text{img}}, o_{\text{text}}) = \text{Action-Head}(\text{VLM-Backbone}(o_{\text{img}}, o_{\text{text}}))
$$

- **Image encoder:** ViT or convolutional backbone, often shared with a pretrained VLM (SigLIP, CLIP).
- **LLM backbone:** a decoder-only transformer that consumes vision tokens + text tokens.
- **Action head:** emits a sequence of **action tokens** that discretise or parameterize the action space (joint angles, waypoints, click coords, trajectory points).

Action tokenization is the main design choice. Three patterns:

| Pattern | Idea | Example |
| --- | --- | --- |
| **Discretized action vocab** | Quantize each dimension into bins; emit as LLM tokens. | RT-2 |
| **Continuous head** | Replace the LM head with a regression / diffusion head over the continuous action. | π0, Octo |
| **Primitive tokens** | Emit entire structured primitives (bbox, trajectory) atomically. | **VFlash** |

## How it works

### Standard decoding

An autoregressive VLA emits action tokens one at a time, like a text LM. For a trajectory primitive (e.g. a 32-waypoint path, each waypoint 3D), this is 32×3 = 96 sequential LM steps per primitive. Latency-bounded.

### Continual-pretraining alignment

SLA ([vla-continual-pretraining](./vla-continual-pretraining.md)) adds a *continual pretraining* stage that aligns the VLM's intermediate visual tokens with 3D spatial embeddings from a frozen perception backbone, jointly with action prediction. This installs explicit spatial reasoning into a VLA without touching the base architecture.

### Flash primitive decoding

VFlash ([vla-flash-decoding](./vla-flash-decoding.md)) replaces sequential autoregressive emission of structured primitives with a single parallel forward pass that predicts all coordinates of a primitive jointly, validated by a structural verifier (ordering, bounds, type). The speedup on 3D trajectory decoding is ≈25×.

## Why it matters

- **Dominant foundation-policy format.** As of 2026, the leading open robot-policy stacks (π0.5, Octo) and computer-use agents (Molmo, Gemini Robotics) are all VLAs.
- **Shared infrastructure with text LLMs.** Pretraining, scaling, quantization, inference-serving techniques built for text LLMs carry over almost unchanged.
- **Action decoding is the latency bottleneck.** The VLM backbone is fast per-step; sequential coordinate emission is where real-time control pipelines lose seconds. Primitive-level and structured decoding are active research directions.

## Gotchas & tricks

- **Action tokenization is not reversible.** Discretizing too coarsely bakes in control resolution; too finely blows up the sequence length. Project-specific tuning.
- **Vision-token density matters more than for VLMs.** Robot manipulation tasks need pixel-accurate affordances; the default 256–576 vision tokens may not be enough.
- **Simulation-to-real gap.** Most VLA eval is in sim or structured tabletop benchmarks (LIBERO, RoboCasa, RoboTwin). Real-robot generalisation is the hard case.
- **No universal action space.** A VLA trained on Franka joint angles doesn't directly transfer to a UR5 or a mobile base. Standard fix: train with normalized action spaces + explicit embodiment tokens.
- **Alignment is still structurally open.** The analogue of RLHF-for-text is not yet standard for VLAs; verifier-based RL (RLVR-style) is the current direction.

## Sources

- Paper: *RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control* — Google DeepMind, 2023 — the canonical VLA paper.
- Paper: *π0 / π0.5* — Physical Intelligence, 2024–2025 — the open-weights continuous-head VLA family.
- Paper: *SLA: Scaling Spatial-Language-Action Model via Continual Pre-training* — Song et al., 2026 — [arXiv:2611.08373](https://arxiv.org/abs/2611.08373).
- Paper: *VFlash: Fast, Accurate Visual Planning via Flash Primitive Decoding* — Wang et al., 2026 — [arXiv:2611.08449](https://arxiv.org/abs/2611.08449).
