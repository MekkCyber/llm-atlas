# Spectral Alignment for Diffusion (SPA)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Exposure bias in diffusion models is *frequency-structured*, not a uniform noise mismatch. Spectral Alignment fits a per-timestep **spectral prior** offline from training-data intermediate predictions, then at inference uses an FFT-based gradient step to pull the network's outputs toward that prior. 3–4% inference overhead, complements existing exposure-bias fixes, works across DDPM, ADM, Stable Diffusion variants, and flow-matching.

**Prereqs:** none — assumes diffusion basics.
**Related:** [tbsm.md](./tbsm.md)

---

## What it is

A guidance-style correction for **exposure bias** — the training-vs-inference mismatch where a diffusion sampler compounds small errors because it never saw its own imperfect predictions during training. SPA reframes the mismatch in the frequency domain: certain frequency bands drift systematically at certain timesteps, and drift is different per model.

## How it works

- **Offline.** For each model and timestep, fit a **spectral prior** — the empirical power-spectrum distribution of intermediate predictions on training data. Cheap: forward-pass the training set once, FFT the intermediates.
- **Online.** At inference, at each denoising step, compute the FFT of the current prediction, compare against the timestep's spectral prior, and take a small guidance gradient step in the direction that reduces the spectral distance.
- **Composability.** SPA is additive to other exposure-bias mitigations (input-perturbation training, corrector steps); it doesn't replace them.
- **Applicability.** Works across DDPM, ADM, Stable Diffusion variants, and flow-matching without architectural changes.

## Why it matters

Exposure bias is a stubborn residual problem for diffusion samplers, and most fixes require training-time changes. SPA is a shelf-ready tool: fit the prior once per model, apply at inference with ~3–4% overhead. Also methodologically interesting — spectral analyses of diffusion outputs are underutilized, and a per-timestep spectral prior turns them into an operational tool.

## Gotchas & tricks

- **Prior fitting is per-model, per-timestep.** No cross-model transfer; refit whenever the model version changes.
- **Guidance strength is a knob.** Too weak and it barely moves; too strong and it flattens output spectra artificially, hurting perceptual quality.
- **FFT overhead is real for large latents.** The 3–4% figure is for standard image resolutions; scale carefully for large video latents.
- **Composes but doesn't replace.** Best used with input-perturbation training, not instead of it.

## Sources

- Paper: *Spectral Prior for Reducing Exposure Bias in Diffusion Models* — Sony AI, 2026 — [arXiv:2607.22091](https://arxiv.org/abs/2607.22091).
