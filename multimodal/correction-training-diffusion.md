# Correction Training for Diffusion
*Depth — a training-time fix for exposure bias in diffusion samplers, with optional distillation back into the base model.*

**TL;DR:** Diffusion sampling iterates on the *model's own* intermediate predictions, which are off the training distribution; errors accumulate across steps (exposure bias). Correction Training fits a small **corrector network** to the base denoiser's actual multi-step rollouts rather than clean noisy latents, applies it iteratively at inference, and then **distills the corrected policy back into the base model** so there's no extra inference pass. Shown to lift sample quality on DiT/SiT/JiT and match classifier-free guidance with better diversity.

**Prereqs:** [attention](../fundamentals/attention.md)
**Related:** [weak-to-strong-diffusion](weak-to-strong-diffusion.md)

---

## What it is

A fine-tuning stage layered on top of a trained diffusion model. The base denoiser is frozen; a corrector module is trained to produce updates that pull the model's actual sample trajectory back onto the data manifold. Optionally, the corrector is distilled back so the resulting model has the quality of base-plus-corrector at the inference cost of the base alone.

## How it works

Standard denoising loss trains the model to map $x_t = x_0 + \epsilon_t$ (clean + fresh noise) to $x_0$. At inference, the sampler feeds $\hat{x}_{t-1}$ computed from the model's own imperfect prediction, so the inputs the model sees at step $t-1$ are not what it saw at train time.

Correction Training generates *replay trajectories*:

1. Freeze the base denoiser $f_\theta$ and run its full multi-step sampler on a batch of noise, producing a trajectory $\{\hat{x}_T, \hat{x}_{T-1}, \ldots, \hat{x}_0\}$.
2. For each intermediate $\hat{x}_t$, train a corrector $g_\phi(\hat{x}_t, t) \to \delta$ to minimize the gap between $f_\theta(\hat{x}_t) + \delta$ and the ground-truth $x_0$.
3. At inference, use $f_\theta + g_\phi$ in place of $f_\theta$.
4. (Optional) distill $f_\theta + g_\phi$ into a single network $f_{\theta'}$, which matches corrected quality at base inference cost.

## Why it matters

- **Addresses exposure bias at its root** instead of papering over it with heuristic samplers (ancestral sampling, DDIM tricks).
- **Alternative to classifier-free guidance** that keeps diversity — CFG improves quality at the cost of mode collapse; correction training doesn't trade off this way.
- **Distillation path makes it free at inference.** Papers that only add a corrector module at inference pay for the second forward; the distill-back recipe recovers the free lunch.

## Gotchas & tricks

- **Replay data is the training set.** The corrector must be trained on *the base model's actual sampler outputs*, not on clean noisy latents — otherwise it has nothing new to correct.
- **Timestep conditioning matters.** Early-step and late-step errors are structurally different; a shared corrector needs clean timestep embeddings.
- **Distillation is lossy.** A small percentage of the corrector-only quality gain is given up during distillation; whether to pay the extra inference cost is a product call.
- **Not a substitute for better base objectives.** Flow-matching bases and better score parameterizations shrink the exposure-bias gap at the root.
- **Interaction with CFG.** You can stack correction training on top of CFG, but the gains don't add linearly — most of the quality win is already captured by CFG when it's enabled.

## Sources

- Paper: *Correction Training for Diffusion Models* — Runqian Wang, Alexei A. Efros, Taesung Park, Yash Belhe; UC Berkeley + Reve, 2026 — [arXiv:2610.14136](https://arxiv.org/abs/2610.14136).
