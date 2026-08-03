# World Models
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A **world model** predicts how a scene evolves over time given an initial state and (optionally) an action. Modern approaches split into three families: **pixel-space video diffusion** (implicit physics, opaque, expensive), **latent-space rollout models** (dynamics inside a compressed latent), and — new — **reason-then-render** models that predict a *discrete symbolic sequence* describing the transition and only then render it into video. PhiZero (CAS, 2026) is the reference for the third family; ShadowDancer (2026) is a leading approach to action-conditioned interactive world models via appearance-dynamics disentanglement.

**Prereqs:** [README](README.md), [../fundamentals/_tokenization](../fundamentals/_tokenization.md)
**Related:** [../architectures/multi-head-attention](../architectures/multi-head-attention.md)

---

## What it is

A world model $p(x_{t+1:T} \mid x_{\le t}, a_{t:T})$ predicts future observations given past ones and (optionally) an action stream. Uses include:

- **Video generation as prediction** — long, physically plausible video from a text or image prompt.
- **Model-based RL / planning** — roll out imagined trajectories to score actions.
- **Interactive simulators** — game engines, robotics environments, self-driving worlds.

The design axis that separates modern approaches is **the intermediate representation** — pixels, latents, or explicit symbols.

## How it works

**1. Pixel-space diffusion.** A DiT or UNet predicts the next frames directly from noise, conditioned on prior frames. Dynamics are implicit in the pixel-predictor's weights. Expensive; opaque.

**2. Latent-space rollout.** Encode frames to a latent stream; a Transformer or state-space model rolls out latents autoregressively; decoder renders. Cheaper than pixel-space; still opaque — dynamics live inside the latent predictor.

**3. Reason-then-render (PhiZero).** Two stages:
- *Reasoning stage:* generate a **physical-language** sequence — a compact discrete tokenization of world-state transitions learned self-supervised from in-the-wild video — that explicitly encodes how objects move, deform, and interact.
- *Rendering stage:* a video model renders the transitions specified by the physical-language sequence.

The intermediate is inspectable, interpretable, and reasoning-cheap compared to pixels.

**Action conditioning (ShadowDancer).** For interactive world models, the challenge is transferring actions across scenes. Shadow pairs — video pairs replaying the same dynamics with independently resampled appearance — let the model learn dynamics *by construction*: predicting one shadow from the other discards appearance, keeps the action. Any demonstration clip becomes a reusable, replayable action asset.

## Why it matters

- Physically consistent video generation is the practical bar for T2V, and pixel-space models struggle at it.
- A learned discrete "physical language" is a real third path — closer to program synthesis for physics than to pixel prediction.
- Shadow-pair training unlocks *any-action* control without hand-labelled action data — a general recipe for interactive world models.
- Long-horizon simulation for RL, robotics, and self-driving all hinge on world-model quality.

## Gotchas & tricks

- Physical-language vocabulary size matters — too small and it can't express complex dynamics; too large and the reasoning stage overfits. Learn it from data, don't hand-design.
- Reason-then-render adds latency (two stages) — for real-time interactive use, cache the reasoning output and only re-render on action change.
- Shadow-pair construction requires a **Shadow Library** that can resample appearance while preserving dynamics — a nontrivial data-engineering artifact.
- Evaluation is hard — PhyGenBench and VBench-2.0 are current physics-aware benchmarks; single-frame FID is misleading.

## Sources

- Paper: *PhiZero: A World Model Built Around Physical Language* — Shang et al., CASIA NLPR, 2026 — [arXiv:2607.28624](https://arxiv.org/abs/2607.28624).
- Paper: *ShadowDancer: Teaching Video World Models Any Action by Learning Unified Dynamics Representations from a Video and Its Shadow* — 2026 — [arXiv:2607.28362](https://arxiv.org/abs/2607.28362).
- Paper: *VideoCoCo: Code-as-CoT for Physically-Consistent Video Generation* — Ren et al., 2026 — [arXiv:2607.27380](https://arxiv.org/abs/2607.27380) — an orthogonal path using executable Blender code as the intermediate.
