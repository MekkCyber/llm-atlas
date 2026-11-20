# Flow Matching

*Taxonomy — the generative-model family that learns continuous transports from noise to data by regressing a time-dependent vector field.*

**TL;DR:** Flow-matching trains a single neural vector field $v_\theta(x, t)$ to transport noise to data along a user-specified probability path. The family is characterized by three choices: the **conditional probability path** (Gaussian, rectified, interpolant-based), the **prediction target** (velocity vs x0 vs noise vs learnable), and the **sampler** (number of ODE solver steps, few-step distillation). It is the modern default for continuous generative modeling — image, video, audio, action — subsuming score-based diffusion as a special case.

**Related taxonomies:** —
**Depth files covered here:** [anisotropic-flow](anisotropic-flow.md) · [decoder-free-sampling](decoder-free-sampling.md) · [streaming-video-diffusion](streaming-video-diffusion.md)

---

## The problem

Classical diffusion trains a denoiser along a Gaussian forward process; sampling solves a reverse SDE/ODE in many steps. This is powerful but wasteful: the training path is picked once and fixed, and the number of sampling steps is high. Flow matching generalizes this to *any* probability path between noise and data, trained with a simple regression objective.

---

## The shared pattern

```
Data x ~ q(x),  Noise ε ~ N(0, I),  Time t ~ U(0, 1)
x_t = α(t) · x + σ(t) · ε
u_t  = target(x, ε, t)                     (velocity / x0 / noise / anisotropic)
L_FM = E ‖ v_θ(x_t, t) − u_t ‖²            (regression on the chosen target)
```

Every flow-matching variant specifies $\alpha(t), \sigma(t)$ (the path) and $u_t$ (the target). Everything else is the standard conditional-flow-matching loss.

---

## Variants

| Technique | Key idea | Main tradeoff | When it wins |
| --- | --- | --- | --- |
| **Gaussian flow matching** (Lipman 2022) | Straight-line path from noise to data, velocity target | Default; well-understood | Baseline for continuous generative modeling |
| **Rectified flow** (Liu 2022) | Iteratively straighten the learned path to enable few-step sampling | Needs multiple reflow rounds | Few-step image/video gen; inference-dominant workloads |
| [anisotropic-flow](anisotropic-flow.md) (2026) | Learn a per-direction prediction target instead of a fixed one | Extra parameters for the basis | When target choice has been the quality ceiling |
| [decoder-free-sampling](decoder-free-sampling.md) (2026) | Add a linear pixel head that bypasses the VAE decoder | Only applies to latent diffusion | Lower inference cost, less video warping |
| [streaming-video-diffusion](streaming-video-diffusion.md) (2026) | Four-step causal streaming model trained with progressive forcing | Needs forcing curriculum | Long-horizon interactive video |
| **Consistency models** (Song 2023) — no depth file yet | Distill a flow into a one- or few-step sampler | Distillation cost | Real-time generation |

---

## How to choose

- **Image / video generation default**: Gaussian flow matching with velocity target. Mature, stable, well-supported tooling.
- **Few-step inference needed**: rectified flow (if you can afford reflow rounds) or consistency distillation (if you can't).
- **Pixel-space generation without VAE**: decoder-free-sampling.
- **Maxing quality at fixed compute**: anisotropic-flow is the clean no-tradeoff win reported so far (ImageNet FID 1.90 → 1.55 on JiT).
- **Interactive / streaming**: streaming-video-diffusion + progressive forcing.

These can combine: e.g. rectified + anisotropic targets, or streaming + decoder-free.

---

## Adjacent but distinct

- **Score-based diffusion** — a special case of flow matching (Gaussian path, noise target). Historically separate literature; mathematically a subset.
- **Autoregressive pixel generation** — generates pixels in token order instead of a continuous flow. Different family.
- **GANs** — adversarial training on a direct noise-to-data generator. Not a flow — no continuous path.

---

## Sources

- Paper: *Flow Matching for Generative Modeling* — Lipman et al., 2022.
- Paper: *Rectified Flow* — Liu et al., 2022.
- Paper: *Learning Anisotropic Prediction Targets for Flow Models* — Chen, Wetzstein, Guibas — 2026 — [arXiv:2611.17767](https://arxiv.org/abs/2611.17767).
- Paper: *Decoding Without a Decoder* — Cai et al., 2026 — [arXiv:2611.22287](https://arxiv.org/abs/2611.22287).
- Paper: *Consistency Models* — Song et al., 2023.
