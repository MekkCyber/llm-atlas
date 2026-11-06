# Velocity Adversarial Distillation (VAD / VA-TDM)
*Depth — adversarial distillation on velocity fields for few-step diffusion.*

**TL;DR:** Few-step diffusion distillation usually matches student and teacher outputs at a few discrete timesteps (DMD, CMD, LCM, trajectory matching). **VAD** trains a learned critic to compare *velocity fields* of student and teacher along the sampling trajectory and turns their disagreement into training feedback — an adversarial loss on velocities. Applied to intermediate steps, it becomes **VA-TDM** (Velocity-Adversarial Trajectory Distribution Matching). State-of-the-art 4-step text-to-image and text-to-video; VA-TDM-H3 closes most of the human-preference gap to the many-step H3 teacher.

**Prereqs:** [../fundamentals/attention.md](../fundamentals/attention.md)
**Related:** [../post-training/_rl.md](../post-training/_rl.md) · [../post-training/_rewards.md](../post-training/_rewards.md)

---

## What it is

A flow-matching / diffusion student is trained to produce a sample in $N \ll N_{\text{teacher}}$ steps. The distillation signal has to come from somewhere, and existing recipes supply it via different objects:

- **Output matching** (DMD, CMD) — match sample distributions at a few discrete timesteps.
- **Trajectory matching** (TDM, PGD) — match intermediate samples along the ODE.
- **Adversarial** (SDXL-GAN, Adversarial Diffusion Distillation) — a discriminator on samples.

VAD is in the *adversarial* family but discriminates **velocity fields** — the ODE's right-hand side — not pixels. A critic $D$ takes $(x_t, v_\theta(x_t, t))$ and tries to distinguish student velocities from teacher velocities; the student minimizes $D$'s ability to separate them.

## How it works

**Velocities.** For a rectified-flow / flow-matching model, $v_\theta(x_t, t)$ is the learned velocity field; the teacher has $v_\phi$. Both map a noisy sample and a time into a direction.

**Velocity discriminator.** A small neural network $D$ takes $(x_t, v, t)$ and outputs a score; it is trained with a standard GAN objective:

$$
\max_D \; \mathbb{E}[\log D(x_t, v_\phi, t)] + \mathbb{E}[\log(1 - D(x_t, v_\theta, t))]
$$

**Student loss.** The student minimizes the discriminator's signal plus a weak reconstruction term:

$$
\min_\theta \; -\mathbb{E}[\log D(x_t, v_\theta, t)] + \lambda \, \lVert v_\theta - v_\phi \rVert^2
$$

**VA-TDM.** Apply this adversarial velocity loss at *multiple* intermediate timesteps along the sampling trajectory — the "trajectory distribution matching" generalization. The adversarial signal replaces the fixed-divergence targets used by prior TDM variants.

## Why it matters

Four-step and one-step diffusion are now the frontier for both image and video generation; the field has accumulated a bestiary of distillation losses. VAD gives a **unified lens**: output matching, trajectory matching, and output-adversarial are all special cases of adversarial matching on different random variables (samples, timestep-conditioned samples, velocities). The velocity choice is a sweet spot — velocities are smoother than pixels, and matching them along the trajectory is more informative than matching endpoints.

## Gotchas & tricks

- Adversarial training inherits GAN pathologies — balance critic/student learning rates and use spectral norm or R1 regularization on the critic.
- The critic conditions on time $t$ — otherwise it collapses to matching a single velocity field.
- For video, velocity fields are higher-dimensional; share the critic across frames rather than training a per-frame critic.
- Trajectory matching with an adversarial critic needs careful step scheduling — oversampling early (noisy) or late (near-clean) timesteps biases the student.

## Sources

- Paper: *VA-TDM: A Unified View for Improving Few-Step Diffusion Models by Velocity Adversarial Distillation* — Luo, Pang, Hu, 2026 — https://arxiv.org/abs/2611.06796
- Related: *DMD / CMD* (Yin et al., 2023, 2024), *Adversarial Diffusion Distillation* (Sauer et al., 2023), *Rectified Flow* (Liu et al., 2022).
