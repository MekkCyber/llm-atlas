# Flow Matching
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A regression-based training objective for continuous-time generative models. Instead of learning a denoiser (score matching / DDPM), learn a **velocity field** $v_\theta(x, t)$ that transports samples along a chosen interpolation between a simple prior and the data distribution. Simpler loss, straighter trajectories, faster sampling. Now the default backbone for large-scale text-to-image and text-to-video systems (Stable Diffusion 3, SD3.5, Sora-family, Flux).

**Prereqs:** none
**Related:** [diffusion-scaling-laws.md](diffusion-scaling-laws.md)

---

## What it is

Flow matching (FM) trains a neural velocity field that pushes a base distribution (usually a Gaussian) to a target distribution along a specified path. At inference you integrate the ODE

$$
\frac{dx_t}{dt} = v_\theta(x_t, t)
$$

from $t=0$ (noise) to $t=1$ (sample). FM subsumes many score-based/diffusion formulations as special cases of the choice of interpolation path.

## How it works

### The training objective

For each data sample $x_1$, sample noise $x_0 \sim \mathcal{N}(0, I)$ and a time $t \sim \mathcal{U}(0, 1)$. Define an interpolation $x_t = \psi_t(x_0, x_1)$ with a corresponding target velocity $u_t(x_0, x_1)$. Then

$$
\mathcal{L}_{\text{FM}} = \mathbb{E}_{t, x_0, x_1} \big\| v_\theta(x_t, t) - u_t(x_0, x_1) \big\|_2^2 .
$$

For **rectified flow** (the most common choice), $\psi_t = (1-t) x_0 + t x_1$ and $u_t = x_1 - x_0$ — a straight-line path whose target velocity is just the endpoint difference. Training is a plain regression.

### Sampling

Integrate the learned ODE with any solver (Euler, Heun, Dormand-Prince). Because rectified-flow paths are (approximately) straight, few solver steps are needed — 4–50 steps for high-quality samples, dramatically fewer than a naive DDPM sampler.

### Endpoint schedules

Standard FM keeps the endpoint $x_1$ = the fixed clean sample. Endpoint schedules (like **Energy-Guided Flow Matching**) generalise this: replace $x_1$ with a *time-dependent* smoothed target that starts blurry and sharpens over $t$. This bakes a coarse-to-fine curriculum into the loss without changing the network. The velocity field re-targets toward finer detail late in the trajectory.

$$
x_1^{(t)} = K_{\sigma(t)} * x_1
$$

for a heat kernel $K_{\sigma(t)}$ whose bandwidth $\sigma(t)$ shrinks with $t$; $\sigma(t)$ can be scheduled per-image by an *energy* signal (fraction of high-frequency content).

## Why it matters

- **Simpler and more stable than DDPM.** Regression on a straight-line velocity is easier to train than score matching, and less sensitive to noise-schedule choices.
- **Faster sampling.** Straight paths + smooth velocity fields allow 4–50 step sampling with quality parity to hundreds of DDPM steps.
- **General.** FM works cleanly on pixels, latents, and non-image modalities (audio, video, 3D fields). The velocity-field abstraction is modality-agnostic.
- **Scales predictably.** Chickering et al. (2026, Abra) show that flow-matching transformers scale in a Chinchilla-shaped way, with a 10× more data-heavy compute-optimal ratio than LLMs.

## Gotchas & tricks

- **Time-schedule choice matters.** Uniform $t \sim \mathcal{U}(0,1)$ is standard; SD3 uses a logit-normal $t$ that emphasises mid-noise levels where the model learns most.
- **CFG at inference.** Classifier-free guidance in FM is applied by combining conditional and unconditional velocity fields — different from the noise-space CFG of DDPM. Get the sign convention right.
- **Endpoint moves are the free lunch.** Moving the endpoint (heat-kernel filter, VAE-latent target, semantic-feature target) reshapes the loss landscape without touching the network — EG-FM (Tong et al., 2026) is a canonical example.
- **Rectified flow is not the only option.** Optimal-transport paths, Schrödinger-bridge paths, and stochastic interpolants all fit the same regression scaffold; rectified flow wins on simplicity, not always on quality.
- **Solver choice at inference.** Euler is safe; higher-order solvers (Heun, DPM-Solver++) can cut steps further but need the network to be well-behaved between grid points.

## Sources

- Paper: *Flow Matching for Generative Modeling* — Lipman, Chen, Ben-Hamu, Nickel, Le — Meta, 2022 — foundational FM paper.
- Paper: *Rectified Flow: A Marginal Preserving Approach to Optimal Transport* — Liu, Gong, Liu — 2022 — the straight-line FM variant.
- Paper: *Scaling Rectified Flow Transformers for High-Resolution Image Synthesis* — Esser et al. — Stability AI (SD3), 2024 — production-scale FM for T2I.
- Paper: *Energy-Guided Flow Matching* — Tong, He, Li, Ma, Fu, Chen, Chen, Huang, Cao — JD.com, 2026 — https://arxiv.org/abs/2608.05811 — moving-endpoint variant with heat-kernel schedule.
- Paper: *Abra: Scaling Diffusion Image Training* — Chickering et al., 2026 — https://arxiv.org/abs/2608.17286 — Chinchilla-style scaling laws for FM transformers.
