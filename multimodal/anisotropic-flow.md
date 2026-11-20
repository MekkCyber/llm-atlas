# AnisoFlow: Anisotropic Flow Models
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Replace flow matching's fixed, hand-chosen prediction target (velocity / x0 / noise) with a **learnable per-direction** target: each direction in a learned basis gets its own noise weighting, and a closed-form map recovers the velocity target at inference. ImageNet 256×256 FID drops from 1.90 to 1.55 over the JiT baseline; also reduces patch-seam artifacts under reduced-precision inference for pixel-space text-to-image.

**Prereqs:** [_flow-matching.md](_flow-matching.md)
**Related:** [decoder-free-sampling.md](decoder-free-sampling.md), [README.md](README.md)

---

## What it is

Flow-matching models are trained by regressing a neural vector field $v_\theta(x_t, t)$ to some target $u_t = \alpha'(t) x + \sigma'(t) \varepsilon$. Which linear combination of $x$ and $\varepsilon$ to regress has been treated as a design choice — v-prediction, x0-prediction, epsilon-prediction, each with known biases.

AnisoFlow treats the prediction target itself as a parameter: a direction-specific noise weighting that the model learns alongside the vector field.

---

## How it works

Let $U \in \mathbb{R}^{d \times d}$ be a learnable orthonormal basis. In the $U$ basis, decompose $x_t = U^\top \tilde x_t$ and define a **direction-specific** noise schedule $\sigma_i(t)$ per basis dimension $i$. The training target becomes
$$
u_t^{(i)} = \alpha'(t) \tilde x^{(i)} + \sigma_i'(t) \tilde \varepsilon^{(i)}
$$
with per-direction weights $\sigma_i(t)$ jointly optimized with $U$ and $v_\theta$.

A closed-form inverse recovers the standard velocity target at inference, so no change to the sampler is needed — the gain lives entirely in training.

---

## Why it matters

- **Clean no-tradeoff improvement** to flow training. Same sampler, same architecture, same compute per step; measurable FID bump on standard benchmarks.
- **Reframes an assumed-fixed choice.** The target choice (v vs x0 vs ε) had been an open research question with heuristic answers. AnisoFlow says: don't choose — learn it.
- **Helps reduced-precision inference.** Pixel-space text-to-image at lower precision shows fewer seam artifacts, because the anisotropic target happens to align with error-sensitive directions.

---

## Gotchas & tricks

- **Basis parameterization matters.** Training $U$ as an unconstrained matrix is unstable; the paper uses a Householder / Cayley parameterization to keep it orthonormal.
- **Extra parameters small but non-zero.** Per-dim $\sigma_i(t)$ is modest, $U$ matrix is $d^2$ — fine at feature-map resolution, more expensive at full pixel resolution.
- **Closed-form recovery to v-prediction** is what keeps the sampler unchanged; losing orthonormality of $U$ breaks the identity.
- **Benefit concentrated in few-step sampling regime.** Full ODE integration partially averages over target choice; gains are more pronounced at low step counts.

---

## Sources

- Paper: *Learning Anisotropic Prediction Targets for Flow Models* — Hansheng Chen, Gordon Wetzstein, Leonidas Guibas (Stanford / Caltech) — 2026 — [arXiv:2611.17767](https://arxiv.org/abs/2611.17767).
- Background: *Flow Matching for Generative Modeling* — Lipman et al., 2022.
- Baseline: *JiT* — the referenced ImageNet flow-matching baseline.
