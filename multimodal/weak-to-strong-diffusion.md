# Weak-to-Strong Diffusion (SWIFT)
*Depth — a training-time self-guidance method for diffusion models that extrapolates away from a frozen weak checkpoint.*

**TL;DR:** Diffusion models trained to the posterior mean are conservative — they underfit high-frequency details. SWIFT uses a **frozen weak denoiser** (an early checkpoint of the same model) to compute an **extrapolated regression target**: train the strong model toward `clean + α · (clean − weak)` instead of `clean`. This amplifies the components the weak model is still missing, giving a free quality bump with no inference-time guidance pass and no unconditional branch. Proven to strictly lower posterior-mean error for a non-empty α range under reasonable spectral assumptions.

**Prereqs:** [attention](../fundamentals/attention.md)
**Related:** [correction-training-diffusion](correction-training-diffusion.md)

---

## What it is

A training-objective modification for diffusion models that bakes self-guidance into the training loss. No extra parameters at inference, no classifier-free-guidance (CFG) sampling cost, no unconditional-branch training. The only extra ingredient is a snapshot of the model at an early checkpoint.

## How it works

Standard denoising loss on noisy input $x_t$ regresses toward the clean target $x_0$:

$$ \mathcal{L}_\text{standard} = \mathbb{E}\, \|f_\theta(x_t, t) - x_0\|^2. $$

SWIFT replaces the regression target with a **weak-guided extrapolation**:

$$ x_0^\text{SWIFT} = x_0 + \alpha \cdot (x_0 - f_{\theta_\text{weak}}(x_t, t)), $$

$$ \mathcal{L}_\text{SWIFT} = \mathbb{E}\, \|f_\theta(x_t, t) - x_0^\text{SWIFT}\|^2. $$

Here $f_{\theta_\text{weak}}$ is a frozen early checkpoint and $\alpha > 0$ is the extrapolation strength. There is no backprop through the weak guide. After training, the weak guide is discarded; the resulting model is used exactly like a vanilla diffusion sampler.

**Spectral intuition.** In an RKHS analysis of finite-time population gradient flow, the authors show that early-stopped training biases the solution toward low-frequency components. Extrapolating away from the weak guide amplifies the *higher-frequency* components it hasn't yet captured. A nonempty range of α yields strictly lower posterior-mean error than matched standard training despite the shifted target.

## Why it matters

- **CFG without CFG.** You get guidance-style quality improvements without doubling inference cost or training an unconditional branch.
- **Compute-efficient scaling.** On ImageNet-256, SWIFT matches the baseline's best FID with roughly half the total training compute (counting weak-guide preparation).
- **Composable.** The modification is orthogonal to architecture (DiT, SiT, JiT, RAE), sampler, and whether you still use CFG at test time.

## Gotchas & tricks

- **Choose α by validation, not analytically.** The theory guarantees a non-empty range; the exact optimum depends on model, data, and training length.
- **"Weak" means early-stopped, not small.** Using a smaller-architecture guide works but is less clean; a same-architecture early checkpoint is the recommended default.
- **Interacts with CFG at test time.** Reported FID 1.15 without inference guidance, 0.99 with guidance on RAEv2 — SWIFT and CFG stack, but with diminishing returns.
- **Not for unconditional-only models.** The target-extrapolation argument assumes a well-defined clean target; purely unconditional pre-training without data labels is fine, but tasks that don't train toward a regression target (adversarial, EBM) are out of scope.
- **Discard the guide after training.** Shipping both models defeats the point; distill-free self-guidance is the whole appeal.

## Sources

- Paper: *SWIFT: Self-guided Weak-to-strong Diffusion Models via Target Extrapolation* — Yanjin He, Cuifeng Shen, Wanzhou Lei et al.; UMich + Alibaba + Berkeley + Nanjing, 2026 — [arXiv:2610.25844](https://arxiv.org/abs/2610.25844).
