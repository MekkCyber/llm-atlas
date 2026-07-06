# Multi-resolution sampling (diffusion / flow acceleration)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Do most of the diffusion / flow-matching sampling steps at **low resolution** (cheap), then lift to full resolution near the end. Prior multi-resolution accelerators (e.g. Bottleneck Sampling, Any-Resolution Diffusion) achieved >5× speedups but produced visible blurring / seams because they upsampled in latent space mid-sampling and edited partial regions. **MrFlow** fixes the artifact problem by staging low→high resolution transitions along the flow-matching velocity field so the transition is *consistent with the pretrained model's dynamics*.

**Prereqs:** [README](README.md), [../fundamentals/attention.md](../fundamentals/attention.md)
**Related:** [../pre-training/README.md](../pre-training/README.md)

---

## What it is

Diffusion and flow-matching generators integrate a velocity field over $T$ steps from noise to a clean sample. Wall-clock cost dominates: $T$ forward passes through a large model. Two mature acceleration axes:

- **Timestep reduction / distillation.** Learn a student that skips steps (consistency models, distilled schedulers, adversarial samplers).
- **Feature caching.** Reuse intermediate activations across nearby timesteps.

**Multi-resolution** is a third axis: run steps at $H/4 \times W/4$ or $H/2 \times W/2$ for most of the trajectory (much cheaper per step), then finish at $H \times W$. The compute savings scale with resolution ratio and dominate for high-res models.

---

## How it works

**Staged schedule.** Split the $T$ steps into $K$ resolution stages, e.g. $H/4 \to H/2 \to H$. Each stage integrates the velocity field at its own resolution.

**Stage transition.** The key design choice. Naïvely upsampling the latent between stages — bilinear or model-based interpolation — introduces artifacts: the pretrained velocity at $H$ is not the velocity at $H/2$ upsampled. MrFlow matches the velocity fields across stages using the flow-matching parametrization, so the transition is consistent with what the pretrained model would predict at each resolution.

**Training-free.** No fine-tuning; MrFlow reads out the velocity from the pretrained flow-matching model at each resolution and orchestrates the staged sampler around it.

**Composable.** Stacks with timestep distillation and feature caching — the axes multiply.

---

## Why it matters

- **>5× speedup class without training.** Comparable to prior multi-resolution methods on speed, but without the visible-artifact problem that made them hard to deploy.
- **Preserves image quality.** The velocity-consistent transition avoids the blurring, seams, and mode drift of latent-upsample approaches.
- **Zero-cost adoption.** Because it's training-free and independent of the base model, MrFlow drops into existing text-to-image pipelines with only a sampler swap.
- **Orthogonal savings axis.** Combines multiplicatively with timestep distillation (fewer steps × cheaper steps).

---

## Gotchas & tricks

- **Stage boundary placement.** Too early a lift to high resolution wastes the savings; too late leaves artifacts because low-res samples aren't yet close to the manifold. Paper's default schedule pushes the transition into the second half of the trajectory.
- **Model architecture assumptions.** Assumes the base model handles multiple resolutions gracefully (typical for DiT / U-Net models with sinusoidal or RoPE-style positional signals). Fixed-position models need adjustment.
- **Doesn't help step count.** The number of steps $T$ is unchanged; only the per-step cost decreases. If you also want fewer steps, stack with distillation.
- **Latent-space vs pixel-space.** MrFlow operates in the pretrained model's latent space; make sure the VAE tolerates the intermediate resolutions.
- **Not for models with heavy content-conditioning.** ControlNet / IP-Adapter-style conditioning fixed at $H \times W$ may need re-alignment across stages.

---

## Sources

- Paper: *Multi-Resolution Flow Matching: Training-Free Diffusion Acceleration via Staged Sampling* — Zheng, Liu, Ding, Feng, Lin, Guo, Qin, 2026 — [arXiv:2607.01642](https://arxiv.org/abs/2607.01642).
- Baselines / precursors: Bottleneck Sampling, Any-Resolution Diffusion, timestep-distillation methods (paper §2).
