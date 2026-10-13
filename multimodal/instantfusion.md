# InstantFusion
*Depth — a shared latent interface across heterogeneous frozen image generators.*

**TL;DR:** Independently trained diffusion and flow image generators share more latent structure than their differing architectures suggest. InstantFusion aligns their mid-sampling latents into a shared space via per-model autoencoders trained with two objectives: match states at matched noise levels, and match the *local velocity* of denoising. Once aligned, a sampler can hand the next denoising step from one frozen generator to another without a dedicated per-pair mapping — enabling acceleration, multi-preference composition, cross-architecture on-policy distillation, and cross-model composition.

**Prereqs:** [_rl](../post-training/_rl.md)
**Related:** (diffusionart — once that depth file lands)

---

## What it is

A protocol, more than a model. Each candidate generator keeps its weights frozen; a *small autoencoder* per generator maps its latents into (and out of) a shared space. The autoencoders are trained jointly so that one generator can produce a latent, hand it to another's encoder, and have the second generator continue denoising from there.

## How it works

Two alignment objectives:

1. **Noise-level matching.** For a shared noise level σ, both generators denoise the same source noise independently. Their intermediate states are projected through the respective autoencoders; the shared-space representations are trained to match.
2. **Velocity matching.** The local denoising *direction* at matched σ — the per-step update vector — is also projected and matched. This is what makes continuations compatible: if generator A's velocity at a point matches generator B's, B can take over the next step without a sharp discontinuity.

Once aligned, the four applications fall out:

- **Acceleration** — hand steps to a faster generator.
- **Multi-preference composition** — alternate steps across models tuned to different objectives.
- **Cross-architecture on-policy distillation** — teacher and student can be in different families.
- **Cross-model composition** — mid-trajectory handoffs produce new visual mixtures.

## Why it matters

If generators have compatible latent geometry, the ecosystem can treat them like microservices rather than monolithic models. Mid-trajectory handoffs are a far cheaper form of composition than ensembles over final outputs. Cross-architecture distillation is especially useful: today's distillation setups are usually constrained to teacher/student in the same family.

## Gotchas & tricks

- **Autoencoder expressivity.** If the per-model AEs are too lossy, handoffs introduce artifacts; too expressive and they absorb all the alignment work.
- **Alignment set must cover the relevant σ range.** Mismatched denoising schedulers across generators have to be bucketed to the nearest shared σ.
- **No one generator "wins."** The protocol doesn't improve any single model's quality in isolation — it only unlocks cross-generator applications.

## Sources

- Paper: *InstantFusion: A Shared Latent Interface for Heterogeneous Image Generators* — Bu, Zhou, Liu et al., 2026 — [arXiv:2610.13944](https://arxiv.org/abs/2610.13944)
