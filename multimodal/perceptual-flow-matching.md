# Perceptual Flow Matching (PFM)

*Depth — supervise the flow-matching velocity field in a perceptual-feature space instead of a VAE latent.*

**TL;DR:** Standard latent flow matching regresses a velocity field in a VAE latent's Euclidean space. Because the L2 regression minimizer is mean-of-modes, few-step Euler integration blurs across modes and needs 35–50 steps. PFM supervises the same velocity field in the *feature space of a pretrained perceptual encoder* — a metric where L2 aligns with human similarity. The regression minimizer shifts from mean-seeking to **mode-seeking**, so 4–8 Euler steps hit the manifold. No teacher model, no auxiliary score net, drop-in change to standard flow-matching training pipelines. Introduced by Zhao et al. (2026).

**Prereqs:** [README.md](README.md)
**Related:** [latent-foresight.md](latent-foresight.md)

---

## What it is

Flow matching trains a network $v_\theta(x_t, t)$ to predict the velocity of a probability path from noise to data. The standard loss is

$$
\mathcal{L}_{\text{FM}} = \mathbb{E}_{t, x_0, x_1} \| v_\theta(x_t, t) - (x_1 - x_0) \|_2^2
$$

usually applied in a VAE latent space $z$. The minimizer of an L2 regression under multi-modal conditional targets is the **conditional mean**. When the conditional distribution $p(x_1 | x_t, t)$ is multi-modal (many plausible clean images), the network learns to point at the *mean of the modes* — an off-manifold vector. Coarse integration (few steps) accumulates that off-manifold-ness into blur.

PFM changes only the **space** in which the L2 is measured:

$$
\mathcal{L}_{\text{PFM}} = \mathbb{E}_{t, x_0, x_1} \| \phi(v_\theta) - \phi(x_1 - x_0) \|_2^2
$$

where $\phi$ is a pretrained perceptual encoder (e.g., a DINO/CLIP feature stack).

## How it works

Two properties of $\phi$ do the work:

- **Perceptual quotient.** Points close in $\phi$-space are perceptually similar. Points close in pixel/latent space may not be.
- **Non-Euclidean geometry w.r.t. pixels.** An L2 minimizer in $\phi$-space is not an L2 minimizer in pixel space. Specifically, the regression minimizer is biased toward *modes* that project to compact regions of $\phi$-space rather than the arithmetic mean.

Consequence: the regression bias flips from mean-seeking to mode-seeking. Coarse integration steps still land near a real mode instead of an off-manifold interpolation, so 4–8 NFE gives quality that used to need 35–50.

Implementation: swap the L2 in the standard flow-matching training loop for an L2 in perceptual features. Fully differentiable, no auxiliary networks, no distillation teacher, minimal code change.

## Why it matters

- **Rare "just change the loss space" acceleration.** Nearly every prior few-step recipe (SiD, DMD, LADD, consistency distillation) requires a teacher and a two-stage pipeline. PFM is a single-stage training change.
- **Composes with existing accelerators.** Distill on top, add GAN heads, use with rectified flow — the axis is orthogonal.
- **Explains a folk result.** Practitioners have known that perceptual losses on generative models "look cleaner"; PFM ties that to a concrete claim about regression minimizers and mode-seeking.
- **Same recipe across modalities.** Reported to work on image generation, video generation, and image editing.

## Gotchas & tricks

- **Perceptual encoder matters.** DINOv2 features are the natural default; CLIP visual encoders also work but bias toward semantic modes. Encoder choice biases the mode structure.
- **Preserve early-timestep signal.** At small $t$ (near noise), the velocity target is essentially the noise direction. Applying $\phi$ can wash that out. Some implementations blend pixel-L2 + perceptual-L2 with a schedule that favors pixel-L2 at $t \approx 0$.
- **Don't fine-tune $\phi$.** The perceptual metric must be frozen. Training it end-to-end collapses toward whatever makes L2 easy.
- **Fewer artifacts vs distillation.** PFM removes the "mode collapse to teacher's favorite modes" artifact that distillation methods have — the loss doesn't reference a teacher output at all.
- **NFE floor is ~4.** Below 4 steps, mode-seeking alone isn't enough; distillation on top of PFM is required.

## Sources

- Paper: *Perceptual Flow Matching for Few-Step Generative Modeling* — Zhao, Song, Wang, Yuan, Zhang, Fu, Chen, Deng, Huang, Duan, 2026 — introduces PFM and the mode-seeking analysis.
- Related: *Flow Matching for Generative Modeling* — Lipman et al., 2023 — the flow-matching baseline PFM modifies.
- Related: *Rectified Flow* — Liu et al., 2023 — the geometry-straightening variant PFM composes with.
- Related: *Score Identity Distillation (SiD)* — Zhou et al., 2024 — teacher-based few-step baseline PFM contrasts against.
