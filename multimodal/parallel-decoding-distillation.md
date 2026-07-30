# Parallel Decoding Distillation (PDD)
*Depth — a trajectory-based few-step distillation for diffusion and flow matching models.*

**TL;DR:** Fast image and video generation from diffusion / flow-matching models usually requires distilling a multi-step teacher into a few-step student. The dominant recipe combines **Variational Score Distillation (VSD)** with an **adversarial loss** — both hard to optimize and prone to **mode collapse** (loss of motion diversity in video). **PDD** replaces both with a simpler *trajectory-based* objective under a parallel-decoding schedule, retaining diversity and motion where prior distillations flatten them.

**Prereqs:** [README](README.md)
**Related:** [video-world-models](video-world-models.md)

---

## What it is

A distillation objective for few-step samplers of diffusion / flow-matching models:

- **Teacher.** A many-step (30–100) diffusion or flow-matching model that produces high-quality samples slowly.
- **Student.** A few-step (2–8) sampler that must match the teacher's *distribution* at each denoising step along a shared trajectory.
- **Parallel decoding.** Instead of serially predicting one denoising step at a time, the student predicts several steps of the trajectory in parallel per forward pass.

The objective is a divergence between student and teacher along sampled trajectories — trajectory-based rather than score-based (VSD) or discriminator-based (adversarial).

## How it works

```
sample x_T ~ N(0, I)
teacher_traj = teacher.sample_trajectory(x_T)          # x_T → x_{T-1} → ... → x_0
student_traj = student.parallel_decode(x_T, steps=K)   # few-step trajectory
loss = trajectory_distance(student_traj, teacher_traj) # e.g. per-step matching
```

- No VSD's variational scoring term — sidesteps its optimization issues.
- No GAN discriminator — sidesteps mode collapse.
- Parallel-decoding schedule means K < teacher steps but each student step is a wider prediction.

## Why it matters

- VSD + adversarial has been the field's dominant recipe for years despite well-known instabilities. A simpler, more stable alternative is a real methodological win.
- Video especially — mode collapse in video distillation kills motion, and motion is the point.
- Enables the interactive framerates that video world models ([video-world-models](video-world-models.md)) need.

## Gotchas & tricks

- **Trajectory alignment matters.** Student and teacher trajectories must share a starting noise; otherwise the per-step matching is noise-matching, not distribution-matching.
- **Step count K.** Too few → quality drops sharply; too many → throughput advantage disappears. Sweet spot depends on teacher and task.
- **Video vs. image.** Video has an extra temporal axis — trajectory distance must account for temporal coherence, not just per-frame matching.
- **Not all teachers distill equally.** Rectified-flow teachers with straighter trajectories distill more easily than curved-trajectory DPM++-style teachers.
- **Sampler compatibility.** The student's parallel decoding schedule fixes an inference-time sampler; you can't swap to a different schedule without redistilling.
- **Diversity check the eval.** FID / FVD can rise slightly at few-step distillation even when subjective quality holds; supplement with recall / coverage metrics to catch mode collapse.

## Sources

- Paper: *Parallel Decoding Distillation for Fast Image and Video Generation* — Shaul, Liu, Vahdat, Berner, 2026 — [arXiv:2607.26004](https://arxiv.org/abs/2607.26004).
- Related prior work: Consistency Models (Song et al.), Variational Score Distillation (Wang et al.), Adversarial Diffusion Distillation (Stability AI).
