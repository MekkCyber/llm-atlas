# On-policy multi-teacher distillation for diffusion
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Train a text-to-image diffusion student to simultaneously gain multiple capabilities (instruction following, text rendering, human preference) **and** few-step sampling in a single training stage, by distilling from frozen specialist teachers *on states the student's own rollout visits*. Introduced as StepX-OPD (StepFun, 2026). Avoids the "second stage undoes the first" failure mode of sequential distillation.

**Prereqs:** [grpo](../post-training/grpo.md)
**Related:** none

---

## What it is

Standard diffusion capability-distillation pipelines run two stages:
1. **Capability distillation.** Distil specialist teachers (e.g. a text-rendering specialist, an instruction-following specialist) into a single student at full sampling steps.
2. **Step distillation.** Reduce the sampling steps (e.g. 50 → 4) while hoping to preserve the capabilities.

Stage 2 typically degrades capabilities from stage 1 because the few-step student revisits decisions made during stage 1 at a different noise schedule.

**On-policy distillation** merges both stages: at every step of training, the student rolls out, and each teacher provides a gradient signal on *that specific rollout*.

## How it works

For each training iteration:

1. The student runs a few-step denoising rollout from noise.
2. For each capability / teacher:
   - Evaluate the teacher's score (gradient of log-likelihood) at a sample of the student's rollout states.
   - Form a teacher-wise distillation loss against the student's predicted velocity at those states.
3. Aggregate the per-teacher losses with weights (either fixed per-dataset or learned).
4. Add a **velocity-regression term on the low-noise tail** of the rollout — stabilises the student by anchoring near-clean states to the teachers' predictions.

A variant specifically targets **prompt spillover** (models writing prompt text into the image), adding a term that compares the rollout against a prompt-blanked reference.

## Why it matters

Few-step image models are the production norm; distilling multiple capabilities into them is a bottleneck. The on-policy framing has three concrete wins reported on SD3.5-Medium and FLUX.1-dev:

- **Highest normalised average** among four-step methods on standard T2I evaluation suites.
- **~37% fewer GPU-hours** on SD3.5-Medium vs the sequential multi-stage baseline.
- No "stage-2 forgetting" — because the student is always rolling out with its current parameters, there is no distribution shift between stage 1 and stage 2.

More broadly: the pattern generalises beyond diffusion. Any multi-objective fine-tune where downstream stages interfere with upstream ones can benefit from unifying the training signal on-policy, similar in spirit to on-policy LLM RL vs sequential SFT→RLHF.

## Gotchas & tricks

- **Teacher weighting** matters. Fixed weights can starve some capabilities; the paper experiments with per-step adaptive weights based on teacher losses.
- **Velocity regression window.** Too early in the trajectory and it over-constrains; too late and it does nothing. The paper uses the final ~30% of denoising steps.
- **Rollout length.** A few-step rollout (4 steps) is cheap to differentiate through; using more than ~6 steps makes gradients expensive.

## Sources

- Paper: StepX-OPD: Rethinking Multi-Teacher On-Policy Distillation in Diffusion Models — Wang et al., 2026 — [arXiv:2611.04386](https://arxiv.org/abs/2611.04386)
