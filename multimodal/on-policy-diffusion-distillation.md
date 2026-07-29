# On-Policy Diffusion Distillation (OPD)
*Depth — distilling a many-step diffusion teacher into a fewer-step student by matching along the student's own generation trajectory.*

**TL;DR:** Instead of training a diffusion student on independently sampled noisy states, generate the trajectory *from the current student* and query the teacher at each visited state. This "on-policy" objective closes the distribution-shift gap between train and inference, and is the modern default for turning a slow multi-step diffusion teacher into a fast few-step student.

**Prereqs:** [../multimodal/README.md](../multimodal/README.md), [classifier-free-guidance.md](classifier-free-guidance.md)
**Related:** [../post-training/reasoning/long-cot-rl.md](../post-training/reasoning/long-cot-rl.md)

---

## What it is

Off-policy distillation trains the student on `(x_t, target = v_teacher(x_t))` pairs where `x_t` comes from adding noise to real data — but the student at inference time sees states along *its own* denoising trajectory, which drift away from that distribution. On-policy distillation samples trajectories from the student and queries the teacher on those states, so training matches inference.

## How it works

Per training step:

1. **Roll student.** Sample a start noise `x_T` and denoise using the *current student* for a few steps to obtain a state `x_t` along the student's trajectory.
2. **Query teacher.** Evaluate the teacher velocity (typically CFG-composed: `v_teacher_guided = v_neg + w(v_pos − v_neg)`) at `x_t` with the same condition.
3. **Match.** Update the student to minimize `||v_student(x_t, c) − v_teacher_guided(x_t, c)||²` — velocity matching along the on-policy trajectory.

Extensions cover **guided** matching (match the guided velocity), **branch-aware** matching (see `positive-direction-matching.md`), and multi-teacher variants.

## Why it matters

On-policy distillation is the substrate for essentially all modern few-step (1–4 step) text-to-image and text-to-video students. It closes the distribution-shift gap that off-policy methods (progressive distillation, consistency distillation on noisy real data) suffer from, and produces students that are usable at very small step counts.

## Gotchas & tricks

- **CFG composition matters.** Naively matching guided velocities can hide branch-level errors: positive- and negative-branch errors compensate under shared conditioning and diverge under privileged negative conditioning (NBA — see `positive-direction-matching.md`).
- **Trajectory noise.** Because the student defines the state distribution, early training is dominated by student-trajectory noise; a warm-up phase or an off-policy pre-training pass helps.
- **Step-count coupling.** OPD tuned for N-step inference doesn't cleanly transfer to N/2-step; retrain for the target step budget.
- **Guidance scale drift.** Students trained under one `w` often over-fit that scale and become brittle to inference-time changes.

## Sources

- Paper: *Rethinking Classifier-Free Guidance in On-Policy Diffusion Distillation* — Li et al., 2026 — [arXiv:2607.24731](https://arxiv.org/abs/2607.24731)
- Related: *On-policy distillation and consistency models* — Song et al. — foundational references cited inline in modern OPD papers.
