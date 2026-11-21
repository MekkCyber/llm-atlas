# Posterior Matching Distillation (PMD)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A diffusion-distillation technique that trains a **stochastic few-step student** to recover the teacher's *per-noise-level posterior*, not just its mean. PMD matches exponentially-tilted posterior means at each noise level, so the student keeps mode coverage instead of collapsing to the teacher's mean. Reports best 4-step FID on both SDXL and FLUX.2 [klein] 4B, plus measurably higher intra-prompt diversity.

**Prereqs:** [../fundamentals/attention.md](../fundamentals/attention.md), [README.md](README.md)
**Related:** [hiuni.md](hiuni.md), [../post-training/_post-training.md](../post-training/_post-training.md)

---

## What it is

A member of the "distribution-matching" diffusion-distillation family (alongside DMD, SiD, SiDA). Those methods train a student to produce samples whose distribution matches the teacher's, but implementation-level approximations — matching posterior *means* at each noise level — turn out to collapse the student toward unimodal outputs. PMD replaces the single-posterior-mean target with a family of *tilted* posterior means that encode the full distribution shape.

## How it works

Consider the teacher's conditional posterior $p_\text{teacher}(x_0 \mid x_t, c)$ at noise level $t$, conditioning $c$. Classical DMD trains the student by matching the posterior *mean* $\mathbb{E}[x_0 \mid x_t, c]$ — a single statistic, which the student can satisfy with a Dirac on that mean.

PMD instead matches a *family* of statistics:

$$
m_\lambda(x_t, c) \;=\; \mathbb{E}_{x_0 \sim p_\text{teacher}(\cdot \mid x_t, c) \cdot e^{\lambda \phi(x_0)}}[x_0]
$$

for several tilt parameters $\lambda$ (and a chosen test function $\phi$). Each tilted mean probes a different region of the posterior; together they overdetermine the posterior shape. The student — also stochastic — is trained to reproduce the full family.

At inference the student samples in 4 steps like other DMD-family methods; the training objective is what differs.

## Why it matters

- **Fixes a known DMD failure mode.** Posterior-mean matching collapses modes. Tilted-mean matching keeps them.
- **Holds up on 4B flow-matching teachers.** FLUX.2 [klein] 4B is the first open flow-matching teacher at that scale; PMD is the first distillation method reported to work on it.
- **Best 4-step FID on SDXL and FLUX.2 [klein] 4B** plus measurable intra-prompt diversity gains vs. DMD and SiD baselines.

## Gotchas & tricks

- **Choosing $\phi$ and the $\lambda$ grid.** Too few $\lambda$ and the student can still collapse; too many and training cost blows up. The paper reports a small grid of 3–5 tilts along a learned direction.
- **Compute per step is higher than DMD** because each training step evaluates the teacher at multiple tilts. 4-step inference cost is unchanged.
- **Stochastic student.** PMD's student is explicitly a sampler, not a deterministic map. Downstream pipelines that assume deterministic distillation may need adjustment.
- **Pixel vs latent.** Validated on FLUX / SDXL latent stacks; pixel-space teachers aren't reported.

## Sources

- Paper: *Posterior Matching Distillation* — 2026 — primary source; 4-step FID on SDXL and FLUX.2 [klein] 4B, plus intra-prompt-diversity analysis against DMD/SiD baselines.
- Related: DMD / DMD-2 (Yin et al., 2023, 2024), SiD / SiDA (Zhou et al., 2024) — distribution-matching distillation family PMD generalises.
