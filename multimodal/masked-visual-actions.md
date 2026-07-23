# Masked Visual Actions
*Depth — a pixel-space action interface that unifies forward and inverse dynamics in a video world model.*

**TL;DR:** Express action as a **partially revealed pixel-space trajectory** of an arbitrary entity in a video. Reveal the robot's motion → the model behaves as a forward dynamics predictor of the scene. Reveal a desired object motion → the same model behaves as an inverse controller that synthesizes robot behavior consistent with that outcome. Fine-tuned from a general video model with only 15 hours of masked examples.

**Prereqs:** *(none in current graph)*
**Related:** [../multimodal/README.md](../multimodal/README.md)

---

## What it is

Video models pick up rich priors over how the visual world moves, but bolting *action* onto them is the hard part of using them as robot world models. Existing recipes either encode actions as discrete tokens (breaks visual grounding) or as language (imprecise, high-latency).

Masked visual actions is a pixel-space representation of action: a subset of pixels along the trajectory of a chosen entity is *revealed* (kept visible) while the rest is *masked* out. The mask itself is the action.

## How it works

- **One shared conditioning signal.** During training, the model is fine-tuned to fill in the video given a partially revealed trajectory of one entity. Which entity is revealed is chosen per-example.
- **Forward mode.** Reveal the robot's motion trajectory. The model completes the scene — its response predicts how the world reacts to that robot motion. Serves as a **forward dynamics model** for planning.
- **Inverse mode.** Reveal a desired object motion trajectory. The model completes the video — its response synthesizes robot behavior consistent with the target object trajectory. Serves as an **inverse model** for control.
- **One checkpoint, both modes.** The training data mixes revealed-robot and revealed-object examples; a single model handles both at inference.
- **Data efficiency.** Only 15 hours of masked examples (real videos + simulation) are needed to fine-tune a general video model into a controllable world model.

## Why it matters

- **Unifies forward/inverse dynamics.** Traditionally two different models; here one checkpoint plays both roles just by changing conditioning.
- **Pixel-space grounding.** Action lives in the same representation as the video prior, so it inherits the prior's structure for free.
- **Downstream use.** Imagined rollouts correlate with real-world execution well enough to rank candidate futures in model-based planning; inverse mode enables policy synthesis from desired object motion.
- **Embodiment-agnostic.** A single checkpoint handles multiple embodiments — controllability isn't tied to a specific robot morphology.

## Gotchas & tricks

- Choice of "which entity to reveal" is data engineering. Too much robot-reveal → weak inverse model; too much object-reveal → weak dynamics predictor.
- The mask needs to be temporally continuous — sparse point supervision breaks the video prior.
- Pixel-space actions inherit video-model limitations: fine motor control below the model's spatial resolution is invisible.

## Sources

- Paper: *Masked Visual Actions for Unified World Modeling* — Alzayer et al., 2026 — [arXiv:2607.19343](https://arxiv.org/abs/2607.19343)
