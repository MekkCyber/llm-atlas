# Classifier-Free Guidance (CFG)
*Depth — the guidance trick that steers diffusion samples toward a condition by extrapolating between a conditional and an unconditional model prediction.*

**TL;DR:** At sampling time, evaluate the diffusion model *twice* — once with the conditioning (text prompt, image reference), once with a null / negative condition — and produce a guided velocity as `v_guided = v_neg + w · (v_pos − v_neg)`. Setting `w > 1` sharpens conditioning; `w = 1` recovers the plain conditional model. A default component of nearly every modern text-to-image / text-to-video diffusion system.

**Prereqs:** [../multimodal/README.md](../multimodal/README.md), [../fundamentals/attention.md](../fundamentals/attention.md)
**Related:** [../post-training/_rewards.md](../post-training/_rewards.md)

---

## What it is

Classifier-based guidance steered diffusion samples toward a target by using an external classifier's gradient — one more model to train. CFG replaces that with a *single* diffusion model that is trained with random condition-dropout, so at inference it can produce both a conditional prediction `v_pos = v(x, c)` and an unconditional prediction `v_neg = v(x, ∅)`. The guided output is an extrapolation of the two.

## How it works

**Training.** During training, randomly drop the condition (replace `c` with a null token) at a rate ~10–20%. The model learns both `v(x, c)` and `v(x, ∅)` in the same weights.

**Sampling.** At each denoising step:

```
v_pos = model(x_t, c)
v_neg = model(x_t, ∅)       # or c_negative if using a negative prompt
v_guided = v_neg + w · (v_pos − v_neg)
```

The **guidance scale** `w` controls how far to extrapolate. `w = 1` is plain conditional. Typical prod values run 3–8 for image generation and 4–15 for video.

The `∅` slot can be swapped for a **negative prompt** — a condition the sampler should steer *away from* — which is what most creative tools expose to end users.

## Why it matters

CFG is what makes text-conditioned diffusion follow prompts at all. Without it, samples drift toward the prior; with it, high-w samples are visibly aligned with the condition at the cost of some diversity. It's also the substrate on which most modern diffusion-alignment work sits — reward-tilted sampling, guidance-distillation methods, and on-policy distillation objectives all express themselves in terms of the CFG-composed velocity.

## Gotchas & tricks

- Very high `w` saturates: samples become high-contrast and lose diversity ("burnt" images). Anneal `w` across timesteps rather than pinning it.
- Two model calls per step doubles inference cost. Guidance-distillation methods (few-step students) exist specifically to fold CFG into a single call.
- The negative-condition slot is a source of subtle bugs: privileged information leaking into `v_neg` that the student can't access breaks on-policy distillation (Negative Branch Asymmetry — see `positive-direction-matching.md`).
- CFG is not a Bayesian probability tilt — the extrapolation is only justified in the limit of small guidance and can degrade calibration at high scale.

## Sources

- Original paper: *Classifier-Free Diffusion Guidance* — Ho & Salimans, 2022 — [arXiv:2207.12598](https://arxiv.org/abs/2207.12598)
- Recent context: *Rethinking Classifier-Free Guidance in On-Policy Diffusion Distillation* — Li et al., 2026 — [arXiv:2607.24731](https://arxiv.org/abs/2607.24731)
