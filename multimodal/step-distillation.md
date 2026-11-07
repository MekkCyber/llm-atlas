# Step Distillation for Diffusion Models
*Depth — compress a many-step teacher into a few-step student without losing quality.*

**TL;DR:** Diffusion models normally need 20–50 denoising steps per sample at serving. **Step distillation** trains a student model that produces a comparable sample in 1–8 steps, by matching the teacher's intermediate or final outputs across a much coarser step schedule. The modern frontier (**StepX-OPD**, 2026) is multi-teacher, on-policy: a single student is supervised by multiple frozen specialist teachers on the states its *own* rollouts visit, unifying "capability acquisition" and "few-step sampling" that were previously separate recipes.

**Prereqs:** [rejection-sampling](../post-training/rejection-sampling.md)
**Related:** [multi-turn-editing](./multi-turn-editing.md), [long2short](../post-training/reasoning/long2short.md)

---

## What it is

The target: a student $S$ that, given an initial noise $x_T$ and a prompt, produces a sample in $K \ll T$ steps whose distribution is close to the teacher $T$'s $T$-step distribution.

Four generations of recipe:

| Generation | Idea |
| --- | --- |
| **Progressive distillation** (Salimans & Ho, 2022) | Halve the step count iteratively; each stage matches two teacher steps with one student step. |
| **Consistency distillation** (Song et al., 2023) | Enforce that samples from different timesteps along the same trajectory land on the same final sample. |
| **Flow-matching distillation** (Rectified Flow, CM, etc.) | Match the teacher's *velocity field* under the student's parameterization. |
| **On-policy multi-teacher (StepX-OPD)** | Student rolls out; frozen specialist teachers (instruction-following, text-rendering, preference) supervise at the states the student visits. |

The last is the current state of the art and where StepX-OPD lives.

## How it works (StepX-OPD)

### On-policy rollouts

For each prompt in the training batch, the student $S$ samples a $K$-step trajectory $\tilde x_0, \tilde x_1, \ldots, \tilde x_K$. The states visited are determined by $S$ itself; this is what makes the method **on-policy**.

### Multi-teacher velocity supervision

At each visited state $\tilde x_t$, call each frozen specialist teacher $T_j$ (instruction-following teacher, text-rendering teacher, human-preference teacher) to predict its target velocity $v_j(\tilde x_t)$. The student regresses its own velocity $v_S(\tilde x_t)$ against a weighted combination $\sum_j w_j v_j(\tilde x_t)$.

### Low-noise velocity regularizer

On the low-noise segment of the rollout (late tail), add an extra pure velocity-regression term against a reference flow — this prevents the student from drifting on the final denoising region where all teachers tend to agree.

### Prompt-word de-rendering variant

An add-on loss discourages the student from rendering prompt tokens *as text in the image* when the prompt doesn't ask for text (a common failure mode when text-rendering teachers are in the mix).

## Why it matters

- **Unifies two expensive stages.** The standard diffusion pipeline runs capability-acquisition training (instruction following, text rendering, preference) first and *then* step distillation. StepX-OPD does both at once, saving ~37% GPU-hours on SD3.5-Medium vs sequential DiffusionOPD + step distillation, and beats a 40-step Flow-OPD baseline in 4 steps.
- **Specialist composition is the modern recipe.** Specialists already exist for every slice of the problem (text rendering, preference, instruction following). Multi-teacher distillation makes them reusable as training signal.
- **Decouples teacher count from student count.** You can add a new teacher without changing the student architecture, as long as all teachers emit velocity in the same parameterization.

## Gotchas & tricks

- **Teacher weights matter.** Equal weighting rarely works. Treat $w_j$ as a hyperparameter tuned on an eval set (prompt diversity + text rendering + preference).
- **On-policy collects teacher queries at inference-time rollouts.** This is expensive if the student rolls out a lot; in practice limit $K$ to 4–8 and cache teacher velocities within an epoch.
- **Teacher distribution mismatch.** If a specialist teacher was trained on a different noise schedule / parameterization than the student, direct velocity regression is biased. Reparameterize teachers before loss computation.
- **Few-step students can collapse on rare prompts.** On prompts far from the teacher's support, the student has no guidance. Mix in rejection-sampled teacher-only generations as auxiliary supervision.
- **Not a substitute for distillation of capabilities the teachers don't have.** If no teacher knows how to render Chinese characters, the student won't either.

## Sources

- Paper: *StepX-OPD: Rethinking Multi-Teacher On-Policy Distillation in Diffusion Models* — Wang et al., StepFun, 2026 — [arXiv:2611.04386](https://arxiv.org/abs/2611.04386).
- Background: *Progressive Distillation for Fast Sampling of Diffusion Models* — Salimans & Ho, 2022.
- Background: *Consistency Models* — Song et al., 2023.
- Background: *Rectified Flow* — Liu et al., 2022 — the straight-line flow-matching parameterization many distillers use.
