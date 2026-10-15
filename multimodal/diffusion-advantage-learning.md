# Self-Referenced Advantage Learning for Diffusion (SALD)
*Depth — a teacher-free training framework for diffusion models that uses the model's own prediction at a lower noise level as the advantage baseline.*

**TL;DR:** Diffusion training normally supervises the model with a reconstruction loss at a sampled noise level. **SALD** adds a second, lower-noise evaluation of the same (image, caption) pair, computed *without* gradients, as a **self-referenced advantage baseline** — analogous to how an RL critic subtracts a value baseline. The gap between the two evaluations drives three downstream signals: sample weighting (AGD), per-example noise curriculum (TAM), and frequency-band weighting (SAD). No separate teacher network and no inference-time cost.

**Prereqs:** [../post-training/_rewards.md](../post-training/_rewards.md), [../post-training/grpo.md](../post-training/grpo.md)
**Related:** [dit-resolution-extrapolation.md](./dit-resolution-extrapolation.md) · [../fundamentals/attention.md](../fundamentals/attention.md)

---

## What it is

A training-time wrapper around any standard diffusion loss. Instead of a single noisy evaluation per (image, caption, step), SALD evaluates at two noise levels and uses the low-noise evaluation — treated as the "easier" reference — to compute an *advantage* that modulates the high-noise training signal.

## How it works

**Dual-noise pair.** For a sample `(x, c)` and training step, pick a high noise level `t_hi` and a low noise level `t_lo < t_hi`.
- Forward the model at `t_lo` with `torch.no_grad()`. Call its loss `L_lo` (the reference).
- Forward the model at `t_hi` with gradients. Call its loss `L_hi` (the training signal).
- Define the **advantage** `A = L_lo − L_hi` (positive when the model does *worse* at the hard step than at the easy one — exactly the samples we want to upweight).

**Three uses of `A`.**

1. **Advantage-Guided Diffusion (AGD).** Weight the sample's gradient by a monotone function of `A`. Samples where the model is far behind its own low-noise prediction contribute more.
2. **Temporal Advantage Memory (TAM).** Maintain a per-example running average of `A` across epochs. Samples with persistently high `A` get a shifted noise distribution — the curriculum focuses on their hard regime.
3. **Spectral Advantage Decomposition (SAD).** Decompose the per-pixel residual into frequency bands (DCT) and compute per-band advantages. Reweight the loss in each band so that bands the model is weak at (typically high-frequency texture) get more gradient.

All three ride on the same `A` signal computed from the dual-noise pair.

## Why it matters

- **No teacher**, no second model to train or serve — contrast with knowledge-distillation diffusion training, where a stronger pretrained teacher is required.
- **No inference cost** — the extra evaluation is purely training-time.
- Reported: consistent gains across diffusion architectures and datasets, with the spectral decomposition contributing most on texture-heavy data.
- Imports the RL *advantage* concept into diffusion in a surprisingly clean way; opens the door to similar self-referenced tricks for flow matching and consistency models.

## Gotchas & tricks

- **Noise-pair gap matters.** If `t_lo` and `t_hi` are too close, `A` is near zero and all three signals degenerate. Too far apart and `L_lo` is near-perfect, giving `A ≈ L_hi` which collapses to standard loss weighting.
- **TAM memory leakage.** Early-epoch advantages are noisy; smoothing with EMA stabilizes the curriculum.
- **SAD assumes a frequency decomposition.** Works clean for images; for latent-diffusion in a VAE space, pick bands in the latent domain or map back to pixel space before decomposing — the paper does the former.
- **Not an RL algorithm.** `A` is used as a weight, not fed into a policy update; don't conflate with RL fine-tuning of diffusion.

## Sources

- Paper: *SALD: Self-Referenced Advantage Learning for Diffusion Models* — Dey, Biswas, Roy, Abdar, Bhattacharya, Verma — RGIPT / IIIT-D / Tezpur / U. Queensland / IIT Kanpur, 2026 — arXiv:2610.01496.
