# Semi-OPD
*Depth — generate student rollouts once, distill against them repeatedly.*

**TL;DR:** On-policy distillation (OPD) regenerates student rollouts at every training step so the teacher-labelled distribution tracks the student's current policy. The paper shows this is often *worse* than Semi-OPD, which generates the student's rollouts once up front and then distills repeatedly against the same fixed set. Across 17 teacher–student pairs (1.5B–235B), Semi-OPD beats OPD in 14, with up to 13.6% higher accuracy and 11.4× training speedup.

**Prereqs:** [_distillation](_distillation.md), [_rl](_rl.md)
**Related:** [cgpo](cgpo.md), [rejection-sampling](rejection-sampling.md)

---

## What it is

A simple distillation recipe:

1. Freeze a copy of the current student.
2. Generate rollouts from it on the training prompts.
3. Label each rollout with the teacher (per-token KL target, or per-step reward).
4. Train the student on this fixed, teacher-labelled corpus for many steps.
5. Optionally repeat the whole cycle after a long training phase.

The "semi" refers to this halfway position between fully offline (teacher rollouts) and fully on-policy (regenerate every step).

## How it works

Why does Semi-OPD often beat OPD?

- **Alignment-dependence.** OPD only helps when teacher and student output distributions are *already close* — then the rollouts the student generates look enough like the teacher's that KL-matching makes sense.
- **Drift under misalignment.** When teacher and student are *not* well-aligned, long student rollouts drift away from anything the teacher would say — so KL targets become uninformative or actively bad.
- **Semi-OPD short-circuits the drift.** The student's rollouts are generated once; even if later-stage training leaves that student behind, the targets stay teacher-anchored. In practice this covers more teacher-preferred tokens over the trajectory than drifting OPD rollouts do.

The paper's diagnostic is **top-k output-token overlap** between teacher and student: high overlap → OPD wins; low overlap → Semi-OPD wins. The 14/17 result is explained almost entirely by this axis.

## Why it matters

Overturns a quiet consensus that OPD is strictly the better technique. If misalignment is the common case (and for most teacher–student pairs it is), Semi-OPD cuts training cost by an order of magnitude *and* produces better students. The alignment diagnostic also gives practitioners a cheap test to decide which method to use.

## Gotchas & tricks

- **Rollout coverage.** If your initial student is very weak, its single rollout pass may miss parts of the input space. Mix in some teacher-rollout coverage for the hard prompts.
- **Repeat cycles.** For long training runs, re-generate the student's rollouts periodically (e.g. once per epoch) to avoid overfitting to the fixed set.
- **Metric to pre-check.** Compute top-k teacher–student overlap on a small eval set before committing — if it's high (say ≥0.8), OPD may still be worth it despite the cost.

## Sources

- Paper: *When Do We Need On-Policy Distillation? Distilling on Offline Student Rollouts Is Often Better* — Fu, Jiang, Liu, Bian, Lee, Sreenivas, Dai, Ye, Grover, Molchanov, 2026 — [arXiv:2610.11291](https://arxiv.org/abs/2610.11291)
