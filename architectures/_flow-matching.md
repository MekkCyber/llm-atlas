# Flow Matching

*Taxonomy — generative models trained to predict a velocity field that transports noise to data.*

**TL;DR:** Diffusion models learn to denoise; flow-matching models learn the **vector field** $v_\theta(x_t, t)$ that flows noise into data along an ODE. Same generator family, cleaner training objective, often simpler ODE solver. The modern image / video frontier (Stable Diffusion 3, Flux, many "DiT" variants) has converged on flow-matching as the default. Variants differ along the *coupling* between source and target distributions, the *path* between them, and what extra fields (guidance, distillation, conditioning) get added in.

**Related taxonomies:** none yet (sibling: diffusion lives in the same space conceptually).
**Depth files covered here:** [../multimodal/danceopd.md](../multimodal/danceopd.md) · [../multimodal/lisa.md](../multimodal/lisa.md)

---

## The problem

A generative model has to learn a transport from a simple noise distribution to a complex data distribution. Diffusion models do this by learning a *score* (gradient of log-density) plus an SDE; sampling integrates that SDE with many steps. The score parameterization tangles the model's job with the noise schedule and complicates training stability and step reduction.

Flow matching reformulates the same transport as **learning a velocity field on a probability path**. Train $v_\theta(x_t, t)$ to match a target velocity field that, when integrated, carries any noise sample to a data sample. Simpler losses, simpler ODE integration, and more flexible paths (you choose how noise interpolates with data).

---

## The shared pattern

For any flow-matching variant you specify:

1. **A coupling** $\pi(x_0, x_1)$ between noise samples $x_0 \sim p_{\text{noise}}$ and data samples $x_1 \sim p_{\text{data}}$.
2. **A probability path** $p_t(x_t \mid x_0, x_1)$ between them parameterized by $t \in [0, 1]$.
3. **A target velocity** $u(x_t, t \mid x_0, x_1)$ — typically the time derivative of the chosen interpolation, e.g., $u = x_1 - x_0$ for straight-line paths.

Train:

$$
\mathcal{L}(\theta) = \mathbb{E}_{(x_0, x_1) \sim \pi, \, t \sim U(0,1), \, x_t \sim p_t}\Big[\| v_\theta(x_t, t) - u(x_t, t \mid x_0, x_1) \|^2\Big].
$$

Sample by ODE-integrating $\dot x_t = v_\theta(x_t, t)$ from $x_0$ to $x_1$.

Every variant below picks different $\pi$ / $p_t$ / $u$ — and that's where the design tradeoffs live.

---

## Variants

| Technique | Key idea | Main tradeoff | When it wins |
| --- | --- | --- | --- |
| Conditional Flow Matching (CFM, Lipman 2023) | Straight-line interp, independent coupling | Crooked sampling trajectories | Default, simplest |
| Rectified Flow (Liu 2023) | Iteratively "rectify" trajectories straighter | Multiple training passes | Few-step sampling |
| Stochastic Interpolants (Albergo, Vanden-Eijnden) | Generalize path to include stochastic mixing | More design knobs | When you want hybrid SDE/ODE |
| Consistency models (Song 2023) | Distill straight transport into one-step | Distillation cost | Real-time / one-step sampling |
| [danceopd](../multimodal/danceopd.md) | On-policy field distillation across multiple capability fields | Routing distribution choice | Unified T2I + edit + CFG-absorbed models |
| [lisa](../multimodal/lisa.md) | Regularize controllable-generation side branch toward likelihood score | Small training overhead | ControlNet-style dual-branch flow training |

---

## How to choose

- **Default for new generators:** straight-line CFM with independent coupling. Cheapest to set up and matches modern image generators (SD3, Flux).
- **If you need few-step inference:** rectified flow or consistency distillation. Both turn 30–50-step samplers into 4–8-step samplers.
- **If you want unified multi-capability models:** DanceOPD-style field distillation gives you capability composition + CFG absorption in one student.
- **If you're training conditional generators (depth-to-image, edge-to-image, video-to-video):** dual-branch + LISA-style likelihood-score regularization speeds convergence with negligible inference cost.

---

## Adjacent but distinct

- **Score-based diffusion models** (DDPM, EDM) — closely related; can be cast as flow matching with a particular path/velocity choice. The mental model is interchangeable in many situations.
- **Continuous Normalizing Flows (CNF, neural ODE generators)** — same ODE-integration sampling, but trained by maximum likelihood rather than velocity regression. Flow matching is the practical replacement.
- **Bridge matching / Schrödinger bridges** — closely-related transport learning between *two arbitrary distributions* (not noise → data), useful for translation tasks.

---

## Sources

- *Flow Matching for Generative Modeling* — Lipman et al., 2023 — the canonical formulation.
- *Rectified Flow: A Marginal Preserving Approach to Optimal Transport* — Liu et al., 2023.
- *Stochastic Interpolants: A Unifying Framework for Flows and Diffusions* — Albergo, Boffi, Vanden-Eijnden, 2023.
- *Consistency Models* — Song et al., 2023 — one-step distilled samplers.
- *DanceOPD: On-Policy Generative Field Distillation* — 2026 — [arXiv:2606.27377](https://arxiv.org/abs/2606.27377).
- *LISA: Likelihood Score Alignment for Visual-condition Controllable Generation* — 2026 — [arXiv:2606.27192](https://arxiv.org/abs/2606.27192).
