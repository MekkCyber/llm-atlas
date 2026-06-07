# Complexity-Balanced Splitting (CBS)
*Depth — partition the diffusion timeline into equal-difficulty segments and assign a specialised sub-network to each.*

**TL;DR:** A standard continuous-time generative model uses one monolithic backbone across the entire diffusion timeline, from pure noise to data distribution. The dynamics it must approximate vary by orders of magnitude across that timeline — yet capacity is spent uniformly. **CBS** (Issachar et al., 2026) partitions the timeline into segments of *equal approximation burden* using **de Boor's equidistribution principle**, and assigns a specialised sub-network per segment. More capacity where the dynamics are harder; less where they're trivial.

**Prereqs:** [architectures/transformer-block.md](./transformer-block.md), [architectures/README.md](./README.md)
**Related:** [architectures/_moe.md](./_moe.md)

---

## What it is

For a generative process indexed by $t \in [0, T]$ (forward noise → backward data), the *score* (or velocity field, or flow) varies in regularity: near $t = T$ it's nearly isotropic Gaussian and trivial; near $t = 0$ it concentrates on the data manifold and has sharp features. Uniformly allocating capacity to the whole interval is wasteful at one end and underpowered at the other.

CBS defines a *splitting* $0 = t_0 < t_1 < \dots < t_K = T$ such that each interval $[t_{k-1}, t_k]$ carries approximately the same approximation difficulty, then trains $K$ sub-networks $f_1, \dots, f_K$ — one per segment.

## How it works

### Equidistribution-based partition

De Boor's principle (from spline theory): to minimise approximation error for a function $f$ over $[0, T]$ with $K$ pieces, place the knots so each piece carries equal $\int |f''|$ — equal *second-derivative mass*. CBS lifts this from spline approximation to the generative timeline:

1. Estimate an approximation-difficulty proxy $\rho(t)$ for the score / velocity at each $t$.
2. Place knots $t_k$ so $\int_{t_{k-1}}^{t_k} \rho(t) \, dt$ is constant across $k$.
3. Each segment gets its own specialised sub-network.

The split is *temporal*, not spatial — distinct from MoE, which routes spatially per token.

### Training

Each sub-network is trained on samples drawn with $t$ in its segment. The objective is the standard diffusion / flow-matching loss restricted to that interval. Boundary continuity between segments is handled either by a small overlap (each network is trained on a window extending slightly past its knots) or by a parameter-tying constraint at the boundaries.

### Inference

At inference, the integrator (DDIM, Euler, etc.) checks $t$ at each step and routes to the appropriate sub-network. Routing overhead is one comparison per step — negligible.

## Why it matters

- **A clean alternative to "scale the monolithic backbone."** For a fixed total parameter budget, CBS matches or beats a single uniform model by allocating capacity where it pays off.
- **Principled, not ad-hoc.** The equidistribution argument tells you *where* to split. No grid search over knot positions.
- **Generalises beyond diffusion.** The same logic applies to flow matching, rectified flow, and consistency models — anything with a continuous time index.
- **Composable with other capacity tricks.** CBS sits orthogonal to MoE-style spatial routing; you can have temporal segments, each of which is an MoE block.

## Gotchas & tricks

- **$\rho(t)$ estimation matters.** A bad difficulty proxy gives a bad partition. The paper uses score-norm-based estimators; alternatives (Lipschitz bounds, empirical loss curves from a pilot run) work too.
- **Discontinuity at knots.** Naively switching networks produces small artifacts at $t_k$. Boundary overlap or constraint terms are mandatory in practice.
- **Parameter accounting.** If $K = 4$ and each sub-network has $P$ parameters, the model has $4P$ total but only $P$ active per step. Compare against a $4P$ monolith for honest budget comparisons.
- **Not the same as multi-scale denoising.** Multi-scale architectures use different *spatial* scales per step; CBS uses different *networks* per *time* segment. The two can stack.

## Sources

- Paper: *Complexity-Balanced Diffusion Splitting* — Issachar et al., 2026 — [arXiv:2606.06477](https://arxiv.org/abs/2606.06477).
- Theory: de Boor, *A Practical Guide to Splines*, 1978 — for the equidistribution principle underlying the split.
