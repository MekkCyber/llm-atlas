# x₀-Prediction for High-Dimensional Latents
*Depth — clean-data-prediction flow matching for diffusion in high-dimensional representation-autoencoder latents.*

**TL;DR:** Diffusion models trained in the latent space of pretrained visual encoders (Representation Autoencoders) suffer when the encoder is finetuned to reconstruct faithfully — reconstruction-finetuning reduces effective dimensionality, and standard **velocity-prediction flow matching** then has to fit orthogonal noise directions outside the low-dimensional signal manifold. Switching the parameterization to **x₀-prediction** (predict the clean sample directly) focuses learning on the manifold and consistently improves text-to-image generation across strong-reconstruction encoders.

**Prereqs:** *(none listed yet in the KG)*
**Related:** *(add sibling parameterization files when landed)*

---

## What it is

A parameterization choice for training diffusion / flow-matching models in a high-dimensional latent space with low effective dimensionality. Instead of asking the network to predict the noise (ε) or the flow velocity (v) at each timestep, it predicts the clean latent (x₀) directly.

## How it works

**Setting.** Diffusion in the latent space of a pretrained encoder finetuned for image reconstruction. Reconstruction-finetuning collapses many latent dimensions (the encoder concentrates useful signal onto a low-dimensional manifold, leaving the rest near-zero and noise-dominated), so the effective dimensionality of the latent is much smaller than the nominal one.

**Why velocity-prediction hurts here.** Flow matching's velocity target at a noisy latent `z_t = α_t x_0 + σ_t ε` is `v = α_t' x_0 + σ_t' ε`. When most of the noise `ε` lives in directions orthogonal to the signal manifold, `v` inherits those orthogonal directions and the model wastes capacity modeling them. Optimization becomes inefficient because gradient magnitude is dominated by manifold-irrelevant components.

**Why x₀-prediction wins.** The x₀ target is entirely on the signal manifold. The network only has to learn to place samples on the manifold and doesn't spend gradient on noise geometry. Loss is computed against the clean latent; the noise appears only through the input, not the target.

At inference time the sampler is unchanged in structure — the same ODE solver runs, only the network's output is reinterpreted as x₀ and combined with the current schedule to compute the update.

## Why it matters

The community cycles between ε-, v-, and x₀-parameterization every few years without a clear conceptual reason to prefer one. This paper identifies a concrete regime — **high-dim latents from reconstruction-finetuned encoders** — where x₀ is a decisive win. Since reconstruction-finetuning is the natural direction for latent-diffusion pipelines that care about fidelity, this regime is where the field is heading.

Consistent quality improvements across multiple strong-reconstruction RAE encoders at equal training cost.

## Gotchas & tricks

- x₀-prediction can be numerically unstable at very low `t` (when the latent is nearly clean) because the loss becomes near-zero and gradients vanish. Standard fix: min-SNR weighting or clamp on `t`.
- The win depends on the encoder having concentrated signal geometry. On low-dim latents (e.g. classical 4-channel latent from Stable Diffusion) the choice matters less; v-prediction is fine.
- Consistency models and rectified flows already use x₀-adjacent parameterizations — this paper's contribution is showing *why* they work in the high-dim RAE regime.

## Sources

- Paper: *On the Diffusibility of High-Dimensional Latents* — Feng et al., 2026 — [arXiv:2609.28473](https://arxiv.org/abs/2609.28473)
