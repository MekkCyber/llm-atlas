# Physical-Time Flow (PT-Flow) for World Models
*Depth — parameterizing a world model as a continuous latent velocity field in physical time.*

**TL;DR:** Discrete-time world models (`z_{t+1} = f(z_t)`) misalign with the continuous physics they try to model. **PT-Flow** replaces the discrete transition with a continuous **latent velocity field** `dz/dt = v_θ(z, t)` integrated by an ODE solver in **physical (wall-clock) time**. The resulting world model, **ODEWorld**, predicts at arbitrary temporal resolution, is stable over long horizons, supports backward prediction, and addresses representation-collapse pathologies of prior latent world models.

**Prereqs:** [native-mesh-generation.md](./native-mesh-generation.md)
**Related:** [visuo-tactile-pretraining.md](./visuo-tactile-pretraining.md)

---

## What it is

A world-model parameterization that treats the latent trajectory as a smooth curve in physical time rather than as a sequence of discrete snapshots. The learned object is a velocity field over the latent space, indexed by physical time — the same mathematical form used in flow matching for generative modeling, applied here to *dynamics*.

## How it works

**Model form.** `dz/dt = v_θ(z(t), t, u(t))`, where `u(t)` is the (interpolated) action / control input at physical time `t`.

**Prediction.** To predict from `z(t₀)` to `z(t₁)`:

1. Interpolate action inputs into a continuous signal `u(t)` over `[t₀, t₁]`.
2. Integrate the velocity field with an ODE solver — the solver's step size is decoupled from any wall-clock cadence.

**Training.** Loss aligns the integrated trajectory against demonstrated trajectories in latent space; because the field is defined in physical time, training data at heterogeneous frame rates all fits into the same objective.

**Backward prediction** falls out of the ODE form for free: integrate backward in time to answer "what state produced this observation?"

## Why it matters

- **Arbitrary temporal resolution.** Query the world model at whatever cadence a downstream planner needs, without retraining for a new frame rate.
- **Long-horizon stability.** Continuous fields don't accumulate error the way discrete rollouts do; ODEWorld is stable at horizons that break discrete latents.
- **Backward prediction.** Useful for planning and counterfactual reasoning — no equivalent in discrete latent models.
- **Physics-aligned math.** The mathematical object matches the domain being modeled — the same trick that made flow matching work for generative modeling.

## Gotchas & tricks

- ODE solver choice matters for accuracy vs. speed — adaptive solvers are ideal for training, fixed-step solvers usually necessary at inference.
- Representation collapse is fought at *training* time; the velocity field can silently converge to zero in poorly-conditioned latent spaces. Regularization on latent kinetic energy helps.
- Interpolating actions into a continuous signal requires care — jump-discontinuous action distributions can inject numerical stiffness.

## Sources

- Paper: *ODEWorld: A Continuous Predictive Architecture via Physical-Time Flow* — Niu et al. (Tsinghua AIR · UC Berkeley BAIR), 2026 — [arXiv:2607.27924](https://arxiv.org/abs/2607.27924)
