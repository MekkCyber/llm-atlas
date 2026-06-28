# On-Policy Generative Field Distillation (DanceOPD)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A distillation recipe that compresses *multiple* heterogeneous flow-matching teachers (e.g. T2I, local edit, global edit) into a single student without one capability cannibalising another. Each capability stays a separate teacher "field" and is queried *only on states the student visits*, eliminating the off-policy mismatch that makes naive multi-task distillation forget base generation quality.

**Prereqs:** [_diffusion-distillation.md](_diffusion-distillation.md)
**Related:** [rejection-sampling.md](../post-training/rejection-sampling.md)

---

## What it is

Modern image stacks now want *one* generative model that does base T2I, local editing, and global editing. The natural recipe — distill several specialist teachers into one student — usually fails: the student visits a state distribution that no individual teacher matches, so velocity targets from a "wrong" teacher pull the student off the manifold and base quality collapses.

DanceOPD's framing: treat each capability as its own generative *field* (a velocity field over noisy states), route a sample to the right field, and query the field *only at states the student actually generated* — i.e. on-policy with respect to the student.

## How it works

- For each training sample, route to one of {T2I, local-edit, global-edit} teacher fields based on the sample's capability tag.
- Run the *student* forward from noise to produce a trajectory of intermediate noisy states.
- At those student-visited states, query the corresponding teacher's velocity.
- Loss is a velocity-MSE between student and teacher at the student's points.
- Different samples in the same batch are distilled against different teachers; all updates flow into one student.

The on-policy design is the load-bearing piece: classical (off-policy) distillation queries teachers at *teacher*-visited states, which the student often can't reach, so the velocity target is misleading.

## Why it matters

- Folds a "model + edit head + control head" zoo into a single student, with quality on each capability.
- The on-policy field idea cleanly transfers a habit from LLM RL (sample from the current policy, score there) to flow-matching distillation.
- Empirically: improvements on editing benchmarks while preserving base T2I quality — the failure mode the naive recipe hits.

## Gotchas & tricks

- Requires the teachers to share the same flow-matching parameterisation (e.g. all rectified flows with compatible noise schedules); mixing diffusion + flow-matching teachers needs schedule alignment.
- Capability routing assumes labelled samples — for in-the-wild data you need a router/classifier.
- Adding a new capability later means training a new teacher then a quick re-distillation pass; the student isn't continual-learning out of the box.

## Sources

- Paper: *DanceOPD: On-Policy Generative Field Distillation* — Zhou, Zhu, Xu, Dong, Gong, Liang, Chu, Qu, Kong, Liu, Chua, ByteDance Seed / NUS / UMD / HKUST, 2026 — arXiv:2606.27377.
