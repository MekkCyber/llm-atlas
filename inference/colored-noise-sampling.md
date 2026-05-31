# Colored Noise Diffusion Sampling (CNS)

*Depth — a training-free SDE sampler for diffusion models that injects frequency-shaped noise instead of white noise, allocating energy to still-unresolved spectral bands.*

**TL;DR:** Conventional SDE samplers inject uniform white noise at every timestep. But diffusion models resolve low frequencies first and high frequencies later — the noise budget should match this schedule. CNS replaces white noise with a *colored* noise schedule (timestep- and frequency-dependent) that directs energy toward the bands the model hasn't yet resolved. Training-free, plug-in sampler. Drops in for ODE/SDE solvers in SiT, JiT, FLUX with substantial unguided-FID improvements.

**Prereqs:** none (drops into existing samplers)
**Related:** none yet in this folder

---

## What it is

A stochastic diffusion sampler — a replacement for the noise term in DDPM-class SDE solvers. Operates at inference only; no retraining, no extra parameters. Targets the gap between the model's *inherent* spectral bias (low-freq early, high-freq late) and the *isotropic* noise that standard samplers inject.

## How it works

The denoising SDE has the standard shape:

$$
dx = f(x, t)\,dt + g(t)\,d W_t
$$

In a conventional solver, $dW_t$ is white Gaussian noise — flat power across all frequencies. CNS replaces $dW_t$ with a noise process whose power spectral density depends on both the timestep $t$ and the frequency $\omega$:

$$
dW_t \to \int \sqrt{\Sigma(t, \omega)}\, d\widetilde{W}_\omega
$$

The schedule $\Sigma(t, \omega)$ is computed (not learned) from the diffusion model's denoising statistics — early timesteps inject more low-frequency energy (where the model is still finalizing global structure), late timesteps shift energy to high frequencies (where fine details emerge). Implementation: apply an FFT, scale by $\sqrt{\Sigma(t, \omega)}$ per frequency, inverse FFT, inject.

This actively exploits, rather than fights, the model's spectral bias. White noise wastes part of its budget injecting energy where the model has already converged.

## Why it matters

- Free FID improvements at inference time. ImageNet-256 unguided FID: SiT-XL/2 8.26 → 6.27; JiT-B/16 32.39 → 26.69; JiT-H/16 11.88 → 8.31. Gains persist under classifier-free guidance.
- Reframes sampler design as a *frequency allocation* problem. The standard isotropic-noise assumption was leaving FID on the table for years across most diffusion samplers.
- Architecture-agnostic. Works on SiT (DiT-style), JiT (vanilla DiT), and FLUX (MMDiT) without modification — the schedule is computed from the model's own statistics.

## Gotchas & tricks

- The FFT step adds per-step overhead, but tiny relative to the model forward pass (microseconds vs. hundreds of milliseconds). Negligible cost in practice.
- The schedule depends on the model's noise schedule — for non-standard noise schedules (rectified flow, EDM-style), the energy-budget derivation needs to be re-done. The paper covers the common cases.
- High-resolution generation amplifies the gains because the spectrum is wider; small-image generation sees less benefit.
- Pairs with classifier-free guidance straightforwardly — the colored-noise term sits in the stochastic part of the SDE, CFG modifies the drift term. Independent levers.

## Sources

- Paper: *Colored Noise Diffusion Sampling* — Davidson, Issachar, Benaim — Hebrew University of Jerusalem, 2026 — [arXiv 2605.30332](https://arxiv.org/abs/2605.30332). Project page: https://hadardavidson.github.io/CNS/
