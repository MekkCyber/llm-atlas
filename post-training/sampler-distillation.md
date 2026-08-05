# Sampler Distillation
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Distil a many-step diffusion sampler into a few-step sampler by training the student to match the teacher's output distribution while using fewer denoising iterations. Originally an image-diffusion technique (consistency distillation, progressive distillation), now essential for text diffusion LMs — the ~20-tokens-per-step average of DiffusionGemma is a sampler-distillation outcome.

**Prereqs:** [../architectures/text-diffusion-lm.md](../architectures/text-diffusion-lm.md)
**Related:** [on-policy-distillation](on-policy-distillation.md), [rlvr](rlvr.md)

---

## What it is

A diffusion model's *quality-per-token* comes from the number of denoising steps it takes. A model that produces good text with 100 steps might collapse at 10. But the whole point of diffusion is throughput, so more steps = worse tokens/sec.

**Sampler distillation** trains the same (or a copy of the) model to reach comparable quality *in fewer steps* — by matching a slow teacher-sampler's output distribution with a fast student-sampler that uses shorter denoising schedules. The end result is a model where quality-per-step has been amortised into the weights.

## How it works

Two families:

**Progressive distillation** (Salimans & Ho, 2022 — originally for image diffusion). Iteratively halve the sampler length:

1. Start with a teacher that samples in $2T$ steps.
2. Train a student to reproduce the teacher's output using $T$ steps, where each student step is trained to match two teacher steps.
3. Use the student as the new teacher; repeat.
4. Chain of halving lets you go $2T \to T \to T/2 \to \dots$ down to a handful of steps.

**Consistency distillation / one-step distillation.** Train the student directly on a *consistency objective*: from any noise level, applying the student's denoiser should map to (approximately) the same clean output. Enables single-step or few-step generation but the training objective is trickier.

**Text-diffusion adaptation (DiffusionGemma-style).**

- The teacher is a many-step diffusion sampler over 256-token blocks.
- The student uses the same model backbone but with a shorter denoising schedule.
- Training objective: block-output distribution match (KL or cross-entropy over the committed positions), computed on rollouts where teacher and student receive the same initial noise.
- **Jointly trained with RL for quality.** In DiffusionGemma the RL objective (preference / verifiable reward) and the distillation objective are optimised together, so quality doesn't degrade even as the sampler shortens.

## Why it matters

- **Turns diffusion throughput from theoretical to practical.** Without sampler distillation, block diffusion needs many denoising steps per block, so parallelism gains are eaten by step count.
- **Explains DiffusionGemma's Pareto frontier.** ~20 committed tokens per forward pass is a sampler-distillation number, not an inherent block-diffusion number.
- **Preserves quality via joint RL.** Combining distillation (speed) with RL (quality) sidesteps the classical "shorter sampler = worse output" tradeoff.
- **Transfers cleanly from images.** The image-diffusion distillation literature is mature (consistency models, LCM, DMD); the recipes port over to discrete text diffusion with minor changes.

## Gotchas & tricks

- **Distillation without RL degrades quality monotonically.** Distilling from 100 steps → 10 steps without a quality signal loses ~5–15% on hard benchmarks. Joint RL is not optional at aggressive step compression.
- **Teacher-student sampler mismatch.** If the teacher uses a different noise schedule / mask pattern than the student, distillation targets are unreachable. Keep sampler families aligned.
- **Same-noise-seed rollouts** for teacher and student are essential for matching-objective distillation. Otherwise you're distilling the noise, not the model.
- **Over-aggressive step compression** collapses the model into a mode where confidence heuristics fail — the committed tokens become nearly-random.
- **Confidence calibration drifts** during distillation. Recalibrate the top-k confidence threshold after each distillation stage.
- **Not the same as speculative decoding.** Speculative decoding *keeps* the full expensive sampler but skips ahead when a cheap drafter agrees. Sampler distillation *makes* the sampler cheaper.

## Sources

- Paper: *Progressive Distillation for Fast Sampling of Diffusion Models* — Salimans & Ho, 2022 — image-diffusion origin.
- Paper: *Consistency Models* — Song et al., 2023.
- Paper: *Latent Consistency Models (LCM)* — Luo et al., 2023.
- Paper: *DiffusionGemma Technical Report* — DeepMind, arXiv:2608.00146, 2026 — joint RL + sampler distillation applied to a text-diffusion LM.
