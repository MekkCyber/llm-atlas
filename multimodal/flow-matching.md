# Flow matching
*Depth — the continuous-time generative modeling framework behind modern image / video / audio DiTs.*

**TL;DR:** Flow matching learns a **velocity field** `v_θ(x, t)` that transports samples from a simple base distribution (Gaussian) to the data distribution along an ODE trajectory. It generalizes diffusion, is trainable with a simple regression loss on straight-line paths (rectified flow, conditional flow matching), and is now the default backbone for large-scale image / video / audio generation.

**Prereqs:** [README.md](README.md)
**Related:** [../post-training/self-opd.md](../post-training/self-opd.md)

---

## What it is

A continuous-time generative model. Given samples `x_1 ~ data`, choose an interpolant `x_t = (1−t)·x_0 + t·x_1` for `x_0 ~ N(0,I)` and `t ∈ [0,1]`. The **target velocity** is the time derivative of the interpolant: `u_t = x_1 − x_0`. Train a neural network `v_θ(x_t, t)` to regress `u_t`. Sampling is an ODE integrator (Euler / Heun / DPM-solver) that starts from noise and follows `v_θ` to a data sample.

## How it works

**Training loss:**
```
L(θ) = E_{t, x_0, x_1} ‖ v_θ(x_t, t) − (x_1 − x_0) ‖²
```
No score function, no noise schedule to design, no variational lower bound — just an L2 regression. Conditional flow matching (Lipman et al., 2023) formalized this; rectified flow (Liu et al., 2022) is the widely deployed straight-path variant.

**Sampling:** solve `dx/dt = v_θ(x, t)` from `x(0) ~ N(0, I)` to `x(1)`. Because the target is a straight line, few solver steps suffice (~20–50 for large images vs 100s for classical diffusion).

**Relationship to diffusion:** diffusion is flow matching with a curved interpolant tied to a specific SDE. Straight-line flow matching is theoretically equivalent but empirically easier to distill and easier to sample from.

**Guidance:** classifier-free guidance (CFG) transfers as-is — predict conditional and unconditional velocities, combine with a guidance scale.

## Why it matters

Flow matching is what backs most 2024–2026 large generative systems: Stable Diffusion 3, FLUX, Sora-family video, Meta's Movie Gen, and audio models. Simpler training objective and straighter sampling paths made it a Pareto improvement over prior diffusion formulations, especially at scale.

## Gotchas & tricks

- Straight-line interpolant is best paired with adaptive time-sampling — uniform `t` under-samples the informative middle range.
- SDE-flavored sampling (stochastic Euler with variance injection) improves diversity in low-step regimes; deterministic ODE integration is used at inference for speed.
- Distillation to few-step / one-step samplers (consistency models, Rectified Flow's reflow procedure) is much cleaner than for classical diffusion.
- On-policy alignment (see [Self-OPD](../post-training/self-opd.md)) benefits from the deterministic sampler as a stable self-baseline.

## Sources

- Paper: *Flow Matching for Generative Modeling* — Lipman, Chen, Ben-Hamu, Nickel, Le — 2023 — [arXiv:2210.02747](https://arxiv.org/abs/2210.02747)
- Paper: *Flow Straight and Fast: Rectified Flow* — Liu, Gong, Liu — 2022 — [arXiv:2209.03003](https://arxiv.org/abs/2209.03003)
- Recent alignment application: [Self-OPD](../post-training/self-opd.md).
