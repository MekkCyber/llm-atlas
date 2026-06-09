# On-policy distillation (AR → diffusion)
*Depth — distil a student model using its own rollouts, scored by a frozen teacher, eliminating train-inference distribution shift.*

**TL;DR:** Standard distillation supervises a student on teacher rollouts (off-policy) or on a fixed dataset, producing a train-vs-inference distribution mismatch. On-policy distillation has the *student* generate trajectories at training time and uses a frozen teacher's per-token logits as the supervision. Applied to converting autoregressive language models into diffusion language models (OPDLM), it cuts training tokens by 15× – 7,000× vs. standard AR-to-diffusion recipes.

**Prereqs:** [_post-training](_post-training.md), [rejection-sampling](rejection-sampling.md)
**Related:** [rlvr](rlvr.md), [grpo](grpo.md)

---

## What it is

A distillation regime in which the student's training distribution = the student's inference distribution. The student rolls out (generates trajectories) at training time; a frozen teacher evaluates them. Loss is a per-token / per-position divergence between teacher and student distributions.

Contrast with:

- **Off-policy distillation.** Student is trained on *teacher* rollouts (or a fixed dataset). At inference the student rolls out itself — distribution mismatch.
- **Behaviour cloning.** Supervises on (input, expert action) pairs. Same mismatch when the student takes its own actions at deployment.

The AR-to-diffusion conversion (OPDLM) is the headline application: a diffusion student generates noisy denoising trajectories, the AR teacher scores each predicted token, and self-on-policy distillation closes the gap between training distribution and the diffusion sampling distribution at inference.

## How it works

```
for step in steps:
    x = sample_input()
    student_trajectory = student.rollout(x)       # student-generated
    teacher_logits     = teacher.score(student_trajectory)
    loss = divergence(student.logits, teacher_logits)
    student.update(loss)
```

- **Teacher is frozen.** No teacher rollouts needed (cheap), no replay buffer needed.
- **Per-token supervision.** The teacher provides a full distribution at each position, not just an argmax. Theoretically tighter than imitation losses.
- **Diffusion-LM specifics.** The student's rollout corresponds to a denoising trajectory; supervision is at each denoising step. Eliminates the gap between training noise schedule and inference sampling.

## Why it matters

- Converts a pretrained AR model into a diffusion LM **without** rerunning pretraining.
- 15× – 7,000× fewer training tokens than standard conversion recipes at matched quality, per the OPDLM paper.
- Pattern generalises: any time the student's inference distribution differs from its training distribution, on-policy distillation is a candidate fix.

## Gotchas & tricks

- **Cold-start problem.** Early student rollouts are nearly random; some warm-up with off-policy data helps avoid divergence-loss blow-ups.
- **Teacher coverage.** If the student rolls into regions the teacher rarely sees, teacher logits are themselves uncalibrated — a per-token "teacher confidence" gate prevents trusting noisy supervision.
- **Computational asymmetry.** Diffusion students may need many denoising steps per training token; budget accordingly.

## Sources

- Paper: *Data-Efficient Autoregressive-to-Diffusion Language Models via On-Policy Distillation* — Su, Helwig, Parashar, Chagi, Jotsna, Zhi, Caverlee, Kalathil, Ji — 2026 — [arXiv:2606.06712](https://arxiv.org/abs/2606.06712)
