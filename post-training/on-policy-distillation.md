# On-Policy Distillation

*Depth — the family of distillation methods that train the student against a teacher on the student's own rollouts, not the teacher's.*

**TL;DR:** Classic knowledge distillation trains the student against teacher-generated data. That leaves the student never seeing distributional gaps where its own rollouts diverge from the teacher's — cold-start collapse, exposure bias, drift. **On-policy distillation** samples from the *student*, then aligns the student's next-token distribution to the teacher's on those same states (via KL, entropy matching, or logit matching). The student learns exactly where its current policy is wrong.

**Prereqs:** [_rl.md](./_rl.md), [grpo.md](./grpo.md), [rejection-sampling.md](./rejection-sampling.md)
**Related:** [reasoning/long2short.md](./reasoning/long2short.md), [reasoning/cast.md](./reasoning/cast.md), [dpo.md](./dpo.md), [_post-training.md](./_post-training.md)

---

## What it is

A training recipe for compressing a large teacher LLM's behavior into a smaller student:

- **Off-policy distillation:** sample from the teacher, train the student to match on teacher-generated trajectories. Cheap, but the student never sees its own failure regions.
- **On-policy distillation:** sample from the *student*, evaluate the teacher on those same prefixes, and align the student's next-token distribution to the teacher's. The student learns from its own mistakes, corrected by the teacher.

The umbrella covers per-token forward-KL, per-token reverse-KL, sequence-level KL, entropy matching, and logit-matching variants — as well as scalar-only relaxations like [CAST](./reasoning/cast.md) where only a scalar value from a solver-teacher is used.

## How it works

Per training step:

1. **Student rollout.** Sample response $o \sim \pi_\theta(\cdot | q)$.
2. **Teacher scoring.** For each token position $t$, get the teacher's distribution $\pi^{T}(\cdot | q, o_{<t})$ (or scalar value / advantage).
3. **Alignment loss.** Choose one of:
   - **Forward KL** $\text{KL}(\pi^T \| \pi_\theta)$ — mode-covering; student spreads mass to cover teacher.
   - **Reverse KL** $\text{KL}(\pi_\theta \| \pi^T)$ — mode-seeking; student collapses to teacher's high-probability modes.
   - **Convex mixture** with a per-token weight $\lambda_t$ — the CADENCE approach: state-dependent balancing between mode-seeking and mode-covering.
   - **Scalar equivalent** — for solver-teachers ([CAST](./reasoning/cast.md)) or preference-only teachers, the KL reduces to maximizing the teacher's scalar advantage on student states.
4. **Optimizer step.** Standard AdamW / policy-gradient machinery.

## Why it matters

- **Fixes exposure bias.** Off-policy distilled students diverge at inference because they've never trained under their own error distribution. On-policy closes the loop.
- **Beats matched-compute rejection sampling.** In the CADENCE report, on-policy distillation with per-token KL scheduling closed 63% of the student-teacher reasoning gap using a fraction of the traditional distillation compute.
- **Recovers the sample-efficiency of RL** without an environment. When the teacher is another LLM (rather than an environment reward), you get the density of KL training on rollouts without needing a task reward.
- **Composes with RLVR.** Many reasoning pipelines interleave on-policy distillation phases with RLVR phases — one drives coverage from the teacher, the other pushes toward the verifier.

## Gotchas & tricks

- **KL direction matters — a lot.** Reverse KL collapses coverage over training; forward KL wastes probability mass on regions the student can't reach. Per-token or per-state scheduling of the mixture (CADENCE's DRIFT) is the current best answer.
- **Cold-start collapse.** Very early in training, the student's rollouts are so bad that the teacher's KL signal on them is close to random noise. Standard remedies: warm-start with an off-policy round, curriculum from short to long generations, or a coverage-adaptive schedule.
- **Teacher-forward pass cost.** Every student-rollout token needs one teacher-forward. The teacher's inference cost scales with rollout length; keep the teacher small enough to be tractable, or shard rollout evaluation across accelerators.
- **Requires teacher logits (or a scalar surrogate).** If only teacher *samples* are available, this is just SFT on teacher data, not on-policy distillation. Scalar-only relaxations (solver values in [CAST](./reasoning/cast.md)) trade some signal quality for infrastructure simplicity.
- **Bootstrapped self-distillation** — recent recipes use the student itself, checkpointed periodically, as a teacher for the next generation. Cheap; can drift without an external anchor. See CADENCE's BSD component.
- **Not sufficient on its own for reasoning.** Distillation gives you the teacher's shape; capability gains beyond the teacher require RL against a verifier ([RLVR](./rlvr.md)).

## Sources

- Paper: *CADENCE: Closing the Reasoning Gap via Coverage-Adaptive On-Policy Distillation* — 2026 — introduces the per-token forward-KL/reverse-KL scheduling and six supporting components. See [../daily-papers/2026-07-30.md](../daily-papers/2026-07-30.md).
- Paper: *CAST: Game Solvers as Turn-Level Teachers for LLM Agents* — Wang et al., 2026 — the scalar-only relaxation with a solver as teacher. See [reasoning/cast.md](./reasoning/cast.md).
- Prior art: on-policy distillation has been used in *long2short* recipes (see [reasoning/long2short.md](./reasoning/long2short.md)) and earlier speech / translation distillation literature.
