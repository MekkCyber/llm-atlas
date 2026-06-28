# Diffusion / Flow-Matching Distillation

*Taxonomy — compress many-step diffusion / flow-matching teachers into faster, smaller, or more capable student models.*

**TL;DR:** Diffusion and flow-matching models are slow at inference (dozens to hundreds of denoising steps) and specialized per capability. Distillation produces a faster or more general student. Variants split on *what is being compressed*: **steps** (many-step → few-step), **specialists** (multiple capability teachers → one student), or **size** (big teacher → small student). The modern shift is to do this *on-policy* — query teachers at states the student visits, not at teacher-visited states.

**Related taxonomies:** [_visual-tokenizers.md](_visual-tokenizers.md), [_controllable-generation.md](_controllable-generation.md)
**Depth files covered here:** [on-policy-field-distillation.md](on-policy-field-distillation.md)

---

## The problem

A single inference of a strong diffusion model takes many denoising steps; serving them at scale is expensive. Worse, modern image pipelines need *multiple* capabilities (T2I, local edit, global edit, control) and the natural recipe — one teacher per capability — produces a model zoo that's painful to ship. Distillation is the lever for both axes (speed and capability consolidation), but naive distillation forgets base quality the moment you mix teachers or trim steps too aggressively.

## The shared pattern

Distillation = match a student's outputs to a teacher's, at queried points, under some divergence. The design choices:

1. **What outputs?** Final image (output-matching), intermediate denoising velocities (velocity/score matching), or trajectory consistency (anchor-pair matching).
2. **At what query points?** Teacher-visited states (off-policy) or student-visited states (on-policy).
3. **How many teachers?** One (step compression), or many (capability consolidation).

The on-policy variant is the newer move: student-visited states are the only ones the student will see at inference, so matching only there avoids wasted distillation pressure off the manifold.

## Variants

| Technique | What's compressed | Match target | When it wins |
| --- | --- | --- | --- |
| Progressive distillation | Steps (halve at a time) | Two-step teacher trajectory | Many-step → few-step, single capability |
| Consistency Models (CM) | Steps (to 1–4) | Teacher trajectory anchor pairs | Single-step image generation |
| Latent Consistency Model (LCM) | Steps in latent space | Latent trajectory anchors | Stable Diffusion-style latent backbones |
| Rectified Flow distillation | Steps via flow straightening | Straight ODE paths | Flow-matching backbones, very few steps |
| LoRA-distilled effects | Specialists into a single LoRA | Multi-teacher output matching | Adding new conditional effects without retraining |
| [on-policy-field-distillation](on-policy-field-distillation.md) | *Multiple heterogeneous capabilities* into one student | Velocity MSE at *student-visited* states | Capability consolidation without forgetting base quality |

## How to choose

- **Step compression on one capability:** Consistency Models (image) or rectified-flow distillation (flow-matching backbones).
- **Capability consolidation across heterogeneous teachers:** on-policy field distillation ([DanceOPD](on-policy-field-distillation.md)) is the only scheme that explicitly avoids the off-policy collapse other multi-teacher recipes hit.
- **Adding a single new conditional skill:** train a LoRA + distill on output pairs; cheaper than retraining the student.
- **Cross-paradigm distillation** (diffusion teacher → flow-matching student or vice versa): noise-schedule alignment is non-trivial; usually easier to keep the paradigm consistent end-to-end.

## Adjacent but distinct

- **Knowledge distillation for classifiers** (Hinton 2015): teacher logits → student logits at the same input. The generative analogue here matches trajectories or velocities, not classification probabilities.
- **RLHF-style preference distillation** for diffusion: optimizes student outputs against a preference reward, not against teacher outputs. Different signal type.
- **Pruning / quantization**: reduce *size* without a teacher; can stack with distillation.

## Sources

- Paper: *Progressive Distillation for Fast Sampling of Diffusion Models* — Salimans & Ho, 2022.
- Paper: *Consistency Models* — Song et al., 2023.
- Paper: *Latent Consistency Models* — Luo et al., 2023.
- Paper: *Flow Matching for Generative Modeling* — Lipman et al., 2023 — rectified flow foundations.
- Paper: *DanceOPD: On-Policy Generative Field Distillation* — Zhou et al., 2026 — multi-capability on-policy distillation.
