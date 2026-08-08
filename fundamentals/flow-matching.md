# Flow Matching
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A simulation-free training objective for continuous-time generative models. Instead of learning a score / noise field over a fixed forward process (as diffusion does), directly regress a **velocity field** that transports a base distribution to the data distribution along a chosen probability path. **Rectified Flow** is the widely used straight-path instance. Now the standard trainer for latent image / video / audio diffusion (SD3, Flux, etc.) and increasingly used beyond generation (embedding adaptation, policy learning).

**Prereqs:** none (basic calculus / probability).
**Related:** none yet.

---

## What it is

A generative model that turns a base distribution $p_0$ (usually standard Gaussian) into a data distribution $p_1$ by integrating a learned time-dependent velocity field $v_\theta(x, t)$ along $t \in [0, 1]$:

$$
\frac{dx_t}{dt} = v_\theta(x_t, t) \qquad x_0 \sim p_0,\ x_1 \sim p_1
$$

Flow Matching trains $v_\theta$ to match a target velocity field defined by a **probability path** between $p_0$ and $p_1$.

## How it works

**Choose a probability path.** For each data point $x_1 \sim p_1$ and noise sample $x_0 \sim p_0$, define an interpolation:

$$
x_t = (1-t)\, x_0 + t\, x_1  \qquad \text{(straight-line, "rectified flow")}
$$

**Target velocity.** The velocity that transports $x_0 \to x_1$ along this straight line is constant in $t$:

$$
u_t(x_t \mid x_0, x_1) = x_1 - x_0
$$

**Loss.** Regress the network's velocity to this target:

$$
L_{\text{FM}} = \mathbb{E}_{t \sim U[0,1],\ x_0,\ x_1,\ x_t}\, \bigl\lVert v_\theta(x_t, t) - (x_1 - x_0)\bigr\rVert^2
$$

Simulation-free — no ODE integration during training. Just sample $t$, sample $(x_0, x_1)$, interpolate, regress.

**Sampling.** ODE-integrate $v_\theta$ from $t{=}0$ to $t{=}1$ (Euler / RK4 / Heun). Straight-line paths need few steps at inference.

## Why it matters

- **Cleaner than diffusion.** No noise schedule, no variance-preserving vs variance-exploding choice, no per-step weighting hacks. One knob: the probability path.
- **Straight paths → few-step sampling.** Rectified flow trajectories are approximately linear, so Euler with ~20 steps typically suffices; diffusion often needs 50+ or a distilled sampler.
- **Discrete-time diffusion is a special case.** Diffusion's ε-prediction can be reformulated as FM with a specific noisy path — so FM is a strict generalization.
- **Adopted by frontier open image models** (SD3, Flux) and increasingly video (Sora-lineage, Kling). Also spreading to non-generation uses: policy learning (RL), embedding adaptation ([../multimodal/README.md](../multimodal/README.md)).

## Gotchas & tricks

- **Optimal transport coupling.** The naive $(x_0, x_1)$ pairing is independent samples. Better: mini-batch OT coupling (Sinkhorn) — reduces path crossings, straightens trajectories further.
- **Time sampling matters.** Uniform $t \in [0,1]$ works, but sampling more densely at hard $t$ (small $t$ for high-frequency detail, near $t=1$ for structure) can improve quality.
- **Conditional FM.** Add conditioning $c$ (text embedding, class label) directly to $v_\theta(x_t, t, c)$; classifier-free guidance ports over from diffusion unchanged.
- **CFG scale is generally lower** for FM than for diffusion of the same model class — start at ~3.5 and tune.
- **Rectified flow can be "reflowed"** — after training, re-couple $(x_0, x_1)$ using the model's own ODE integration, retrain, iterate. Each reflow straightens paths and reduces required sampling steps (down to 1-2).
- **Not automatically better** than diffusion for every task. On highly multimodal targets where paths must cross, straight-line FM struggles unless coupled with OT.

## Sources

- Paper: *Flow Matching for Generative Modeling* — Lipman et al., Meta, 2023 — general FM framework.
- Paper: *Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow* — Liu et al., UT Austin, 2022 — rectified-flow / straight-path variant, reflow procedure.
- Paper: *Scaling Rectified Flow Transformers for High-Resolution Image Synthesis* — Esser et al., Stability AI, 2024 (SD3) — RF at scale in a DiT.
