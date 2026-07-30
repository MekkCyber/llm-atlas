# Video World Models
*Depth — camera-controllable interactive video generators that let a user "walk around" a scene.*

**TL;DR:** A **video world model** takes an image or a short conditioning clip and produces a *playable* video: the user steers the camera in real time, revisiting regions and discovering new ones. To be usable it needs (1) continuous camera conditioning, (2) a memory of already-explored regions so revisits are consistent, and (3) fast enough sampling for interactivity. Wonder is the current state of the art at 16 FPS on minute-long clips.

**Prereqs:** [README](README.md)
**Related:** [parallel-decoding-distillation](parallel-decoding-distillation.md)

---

## What it is

Not just a text-to-video generator, and not a game engine. A video world model:

- **Ingests** an image or a short video plus a stream of camera poses (position, orientation, focal length).
- **Emits** a video frame stream conditioned on the current pose given previously produced frames.
- **Persists** enough state that when the camera revisits a region, the same content reappears with consistent geometry.

Compared to a text-to-video model, the input is a pose stream instead of a prompt; compared to a NeRF-style reconstruction, the scene is not fixed — the model *invents* unseen regions on demand.

## How it works

Three composable ingredients (Wonder is the current instantiation):

1. **Dense coordinate-field camera conditioning.** Encode camera pose as a per-pixel coordinate field that the diffusion / flow model consumes as a conditioning channel — continuous rather than a discrete pose token.
2. **Sparse attention over a memory of past views.** Store a bounded set of previously generated frames (or their latents); at each new frame, attend sparsely so revisits pull consistent content rather than hallucinating.
3. **Refined distillation for real-time throughput.** A multi-step diffusion / flow-matching teacher distilled to a few-step student (see [parallel-decoding-distillation](parallel-decoding-distillation.md)) so the loop hits interactive framerates.

## Why it matters

- Moves video generation from "watch a fixed clip" to an *interface* — user input steers the model.
- Playable worlds are the natural next step above text-to-video: same generative backbone, richer control signal.
- Feeds into VLA / embodied-agent work — a differentiable interactive world simulator is a cheap RL environment.

## Gotchas & tricks

- **Consistency at revisits is the hard part.** Memory has to store enough state (latents, features, or reconstructed geometry) that a returned camera pose regenerates the same content. Naive per-frame conditioning drifts.
- **Latency budget dominates architecture choices.** 16 FPS at 512×512 = ~30ms/frame. This rules out large diffusion backbones without heavy distillation.
- **Long-horizon coherence vs. novelty.** Too much reliance on memory → the model can't invent new regions when the camera moves out; too little → catastrophic forgetting of visited areas.
- **Pose vs. semantic control.** A pose stream controls *viewpoint* but not *content*. Adding a text prompt for content on top adds another control dimension; combining them is nontrivial.
- **Evaluation is unsettled.** Standard video FVD doesn't measure revisit consistency or camera-following fidelity. Custom protocols per paper.
- **Memory is bounded.** The sparse-attention memory is O(#stored frames). Very long sessions require compression or dropping old frames — with the risk of forgetting a region the user might revisit.

## Sources

- Paper: *Wonder: Video World Model Done Better* — Jiang et al., 2026 — [arXiv:2607.26037](https://arxiv.org/abs/2607.26037).
- Related: Genie (DeepMind), GameGen-X, and the broader interactive-video-generation literature.
