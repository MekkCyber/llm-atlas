# Latent Action Models (DreamX)
*Depth — repurpose a pretrained video diffusion model's latents as a compact, cross-embodiment action language.*

**TL;DR:** A pretrained video diffusion model already encodes motion structure across many visual appearances and embodiments. A small projector compresses each video-frame latent **300× (1200 → 4 tokens)** into a "latent action" that preserves motion but drops appearance. These latent actions **transfer across embodiments** (human ↔ robot), support zero-shot action transfer, and can be used as the action space for policy learning. DreamX (KAIST AI, 2026) sets SOTA among latent action models on LARY-Bench.

**Prereqs:** [README.md](README.md)
**Related:** none yet

---

## What it is

Vision-language-action (VLA) models typically learn an action head jointly with the vision encoder on robot trajectory data. The action space is model-specific, embodiment-specific, and small compared to the amount of motion information in web video.

Latent action models invert this: use a *generative* video model (trained on web-scale cross-embodiment video) as the action backbone. The model's internal latents already represent motion patterns across appearance, embodiment, and task. Extract and compress those latents into a compact per-frame action token, and you have a cross-embodiment action language that scales with video pretraining.

---

## How it works

Three pieces:

1. **Base video diffusion.** A pretrained video diffusion model (image-to-video or text-to-video) with a learned temporal latent space. DreamX uses a public backbone; the method is designed to be backbone-agnostic.

2. **Latent-action projector.** A small MLP or transformer that takes a per-frame video latent (e.g. 1200 tokens) and projects it to a compact action vector (4 tokens). Trained to preserve **motion-salient** information: the paper uses a mix of (a) reconstruction loss on predicted next-frame pixels from action + previous frame, (b) contrastive loss against appearance-perturbed negatives (same motion, different appearance → same action).

3. **Downstream heads.** Policy learning on the latent action space. For robotics: sample a latent action trajectory conditioned on goal/instruction, decode via a small action-to-joint-commands head trained with a few demonstrations.

### What's preserved and what's dropped

- **Preserved**: temporal dynamics, motion direction, speed, trajectory shape, coarse end-effector pose change.
- **Dropped**: visual appearance, object identity (unless motion-informative), lighting.

This is the key for cross-embodiment transfer: the action for "move left arm forward" is nearly the same latent whether the arm is human, humanoid robot, or 7-DoF arm — appearance is factored out.

---

## Why it matters

- **Scales with video pretraining.** Robot demonstration data is a bottleneck; web video is not. If the latent-action approach holds up, the frontier of robotics follows the frontier of video generation.
- **Cross-embodiment by construction.** No per-embodiment fine-tuning of the action encoder — the projector is trained once; the embodiment-specific decoder is small and cheap.
- **Any-to-any video transfer falls out.** Because latent actions are appearance-free, transferring a motion sequence from one video to a new appearance (different human, different robot, different scene) is a zero-shot operation.
- **Compact policy input.** 4 tokens per frame is cheap for downstream RL/IL compared to full latents or image features.

---

## Gotchas & tricks

- **Compression ratio is tuned.** 300× compression is aggressive; too-small codes lose motion fidelity, too-large codes retain appearance. The paper's ratio works for LARY-Bench; new backbones need re-tuning.
- **Projector training is the only moving part.** The video diffusion base is frozen; the projector is light. All quality gains and failures live in projector design + loss weighting.
- **Contrastive negatives are load-bearing.** Appearance-only negatives (same motion, different appearance) are what force the projector to drop appearance. Trajectory-only negatives (different motion, same appearance) force it to keep motion. Both are needed.
- **Downstream decoder is embodiment-specific.** The action-to-joint-commands head is still trained per robot. "Universal" refers to the latent action space, not the full stack.
- **Doesn't help if the base video diffusion is weak in a domain.** If the pretraining corpus underrepresents dexterous manipulation (common), the latent action space there is sparse. Scale the base.
- **Latent actions are not interpretable.** 4-token vectors are opaque; debugging policy failures requires rendering back through the video decoder.

---

## Sources

- Paper: *DreamX: Video Diffusion Model is a Universal Latent Action Learner* — Kim, Kim, Kim, Park, Kim, Hong, Son, Yang, Kim (KAIST AI), 2026, arXiv 2611.05377 — primary source.
