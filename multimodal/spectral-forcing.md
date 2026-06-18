# Spectral Forcing for Pixel-Space Diffusion
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Pixel-space diffusion models waste capacity learning to denoise high-frequency bands that, at most time steps, carry no signal. Spectral Forcing is a **parameter-free, time-conditional 2D-DCT low-pass operator** applied to the noisy input *before* the patch embedder so the denoiser sees only the frequency bands with non-trivial data-to-noise ratio. Improves FID and Inception Score on ImageNet-256 at zero extra parameters.

**Prereqs:** (no diffusion concept pages yet in the graph)
**Related:** (none yet)

---

## What it is

Under **rectified-flow** (or any diffusion) noising of natural images, the noise is white but the signal has a **power-law spectrum** $|\hat{x}(k)|^2 \propto k^{-\alpha}$. As time $t \to 1$ (full noise), the data-to-noise contour separating signal from noise climbs to lower and lower frequencies:

$$
k^*(t) = (1-t)^{-2/\alpha}
$$

Below $k^*(t)$ the band still carries signal; above $k^*(t)$ it's noise-dominated for that time step. The standard practice — feeding the full noisy image to the patch embedder at every $t$ — asks the denoiser to spend capacity denoising bands where there is nothing to learn.

Spectral Forcing makes the boundary explicit: zero out frequencies above $k^*(t)$ before the patch embedder, leaving the model with the actually-signal-bearing input.

---

## How it works

### The operator

A 2D discrete cosine transform (DCT) of the noisy latent / image $z_t$, multiplied by a hard low-pass mask thresholded at $k^*(t)$, then inverse-DCT back:

$$
\tilde z_t = \mathrm{IDCT}\!\left( M_{k^*(t)} \odot \mathrm{DCT}(z_t) \right)
$$

$M_{k^*(t)}$ is 1 below the threshold frequency and 0 above. The threshold is set analytically from the rectified-flow noise schedule and the empirical power-law exponent $\alpha$ (≈ 1 for natural images).

The operator has:
- **No learned parameters.** $\alpha$ is data-derived once.
- **No extra FLOPs after the embedder.** Only the embedder sees a clipped input.
- **A clear analytic justification** rather than a learned mask.

### Insertion point

The DCT low-pass sits between the noisy input and the patch embedder. Everything downstream — the diffusion transformer (e.g., JiT-700M/32 in the paper), the loss — is unchanged.

### Per-step adaptive

Because $k^*(t)$ depends on $t$, early (low-noise) timesteps see almost all frequencies, and late (high-noise) timesteps see only the lowest few bands. The model's capacity allocation across $t$ is therefore *automatic*: at each $t$ the input is the maximum-signal projection.

---

## Why it matters

- **Capacity-efficient pixel-space diffusion.** Latent diffusion (e.g., Stable Diffusion) gets around the noise-dominated-band problem by running the denoiser in autoencoder latent space; this paper shows you can recover much of that efficiency *in pixel space* by zeroing the right bands.
- **Parameter-free.** Drop-in modification; consistent FID and IS gains across ImageNet-256 training epochs on JiT-700M/32.
- **Analytically grounded.** A rare diffusion modification with a closed-form justification (the data-to-noise crossover frequency under the rectified-flow schedule) rather than a hyperparameter to tune.

---

## Gotchas & tricks

- **Tied to the noise schedule.** The crossover $k^*(t)$ is derived from rectified-flow + natural-image $\alpha$. Different schedules (e.g., variance-preserving DDPM) or different data spectra (line art, scientific imagery) require re-derivation.
- **Hard mask, not soft.** The paper uses a binary mask. A soft (e.g., Butterworth) mask is mentioned as a future direction; for natural images the hard cut is enough.
- **Doesn't help latent diffusion.** Latent-space diffusion already operates in a learned compressed representation where the data-to-noise structure differs. Spectral Forcing is specifically a pixel-space rescue.
- **Patch-embedder interaction.** Because the DCT is global but the embedder is patch-local, there is a small spatial-locality interaction. The paper reports no degradation but suggests the patch size be ≤ Nyquist of the lowest pass band.

---

## Sources

- Paper: *Show the Signal, Hide the Noise: Spectral Forcing for Pixel-Space Diffusion* — Anonymous (HF page lists no authors), 2026 — [arXiv:2606.15236](https://arxiv.org/abs/2606.15236).
- Background: rectified-flow diffusion — Liu et al., 2022 — the schedule whose noise structure makes the analytic crossover clean.
