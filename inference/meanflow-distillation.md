# CFG-aware MeanFlow distillation
*Depth — compress a multi-step flow-matching generator into a low-step inference graph that still respects classifier-free guidance.*

**TL;DR:** Flow-matching generators (image / audio / speech) sample by integrating an ODE over many small steps. MeanFlow distillation trains a student to predict the *average* of the teacher's velocity field over a step interval, collapsing many small steps into one — and the CFG-aware variant ensures the distillation preserves the effect of classifier-free guidance at inference. Used in dots.tts to drop first-packet latency to 85 ms.

**Prereqs:** [attention](../fundamentals/attention.md)
**Related:** [dots-tts](../multimodal/dots-tts.md), [rejection-sampling](../post-training/rejection-sampling.md)

---

## What it is

A distillation recipe specifically for *flow-matching* generative heads (a sibling of diffusion). The teacher solves an ODE with a learned velocity field `v(x_t, t)` over many small `Δt`. The student predicts the *mean velocity* over a coarser step interval `[t, t+Δ]`, so a single student forward pass replaces many teacher steps.

The "CFG-aware" qualifier matters because classifier-free guidance (CFG) is a non-linear combination of conditional and unconditional velocity fields. Naively distilling against CFG-rolled samples preserves the *output* but loses the controllability — you can't dial CFG strength at inference. CFG-aware MeanFlow keeps the conditional/unconditional split inside the student, so guidance remains a knob.

## How it works

- **Teacher.** A flow-matching model with conditional velocity `v_c` and unconditional `v_u`. CFG sampling uses `v = v_u + w (v_c − v_u)`.
- **Student objective.** Predict, for an input `x_t` and interval `Δ`, the mean velocity `v̄_c = (1/Δ) ∫ v_c dt` and `v̄_u = (1/Δ) ∫ v_u dt` over the interval. At inference, recombine with the same CFG formula.
- **Loss.** Match the student's `v̄_c`, `v̄_u` to teacher integrals computed by short Runge–Kutta rollouts. Trained over a curriculum of growing `Δ`.

Result: a single (or few) step inference graph that preserves CFG.

## Why it matters

- Latency-critical generative deployments (streaming TTS, image previews, real-time agents) need few-step sampling without losing controllability.
- The CFG-aware variant preserves the controllability operators users actually tune, unlike vanilla step-distillation which bakes in a fixed `w`.
- Cleanly composes with other generator-side techniques (long-context flow heads, multi-stage decoders).

## Gotchas & tricks

- **`Δ` schedule matters.** Jumping straight to one-step distillation collapses quality; standard practice is a curriculum from many small intervals to few large ones.
- **Conditional/unconditional ratio.** The student must see both at sufficient frequency to maintain CFG fidelity; class-conditional dropout schedules carry over from teacher training.
- **Stochasticity.** Pure mean-velocity students are deterministic; some downstream tasks (TTS prosody) need a small noise injection at inference to recover diversity.

## Sources

- Paper: *dots.tts Technical Report* — Lian et al. — 2026 — [arXiv:2606.07080](https://arxiv.org/abs/2606.07080) — primary application + ablations.
- MeanFlow originates in the diffusion-/flow-distillation literature; dots.tts contributes the CFG-aware formulation.
