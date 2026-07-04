# MrFlow — Multi-Resolution Flow Matching for Training-Free Diffusion Acceleration
*Depth — a staged low→high resolution pipeline that speeds up pretrained flow-matching text-to-image models 10–25× without any training.*

**TL;DR:** Text-to-image diffusion is slow. Prior training-free acceleration via multi-resolution generation upsamples in the *latent* space and blurs. **MrFlow** stages the process in *pixel* space: draft the main structure at low resolution, super-resolve with a lightweight pretrained GAN, re-inject low-strength noise so high-frequency detail resamples rather than upsamples, then refine a few steps at full resolution. 10× end-to-end speedup on FLUX.1-dev and Qwen-Image with <1 % OneIG-benchmark drop; up to 25× stacked with step distillation.

**Prereqs:** (no diffusion depth pages yet in the graph)
**Related:** (no diffusion depth pages yet in the graph)

---

## What it is

Diffusion / flow-matching text-to-image models sample by iterating a learned velocity field over many denoising steps. Two orthogonal acceleration levers:

1. **Reduce steps** (timestep distillation, consistency models).
2. **Reduce per-step cost** (feature caching, low-resolution generation).

Multi-resolution generation has been an especially attractive #2: sampling at low resolution is quadratically cheaper in tokens *and* often needs fewer steps. But prior training-free work upsampled *in the latent space* and applied selective region modification — leaving noticeable blur and artifacts. MrFlow fixes this by moving the upsampling operation into pixel space and injecting a resampling step.

## How it works

Four sequential stages:

1. **Low-resolution structure.** Sample the flow-matching model at (e.g.) 512² — few steps, cheap. Get the main composition and rough content.
2. **Pixel-space super-resolution.** Pass the low-res image through a small pretrained GAN-based SR model (not a diffusion SR). Fast, sharp, but locally hallucinates texture.
3. **Low-strength noise injection.** Add mild noise to the SR output, corresponding to a small time offset $\tau$ in the flow-matching schedule. This turns the SR output into a starting point *for the flow model at high resolution*.
4. **High-resolution refinement.** Run the flow model for a small number of steps from $\tau$, refining details at full resolution. Because the high-resolution schedule only runs from $\tau$ onward, per-step count is minimal.

Result: most of the *steps* happen at low resolution (cheap tokens), most of the *quality* comes from the final high-res refinement.

## Why it matters

- **10× end-to-end speedup, no training.** Any released flow-matching model gets the multiplier for free. Zero fine-tuning, zero calibration, zero runtime dynamic-region detection.
- **Stacks with orthogonal tricks.** Combined with pretrained step distillation, up to **25× total** — the two levers compose because they attack different parts of the cost curve.
- **Quality retained.** Within 1 % on OneIG relative to unaccelerated baseline; qualitatively free of the blurring / artifacts characteristic of prior latent-space multi-resolution methods.

## Gotchas & tricks

- **Pixel-space SR is the crux.** Latent-space upsampling loses too much high-frequency detail for the refinement stage to recover. A GAN SR is fast and sharp enough; a diffusion SR reintroduces most of the cost.
- **Noise strength $\tau$ is a hyperparameter.** Too small, artifacts from SR carry through; too large, the low-res structure is destroyed and quality drops.
- **Model-specific tuning.** FLUX.1-dev and Qwen-Image both work with the same recipe, but optimal step counts (low-res vs high-res) shift with the flow-matching schedule of the base model.
- **Not applicable to autoregressive image models.** The pixel-space resampling step relies on the flow field being continuous in noise time; discrete AR pipelines don't have this handle.

## Sources

- Paper: *Multi-Resolution Flow Matching: Training-Free Diffusion Acceleration via Staged Sampling* — Zheng et al., 2026 — [arXiv:2607.01642](https://arxiv.org/abs/2607.01642).
- Related: *FLUX.1* — the flow-matching text-to-image base evaluated.
- Related: *Consistency Models / Rectified Flow* — step-reduction methods that compose with MrFlow.
