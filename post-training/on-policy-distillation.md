# On-Policy Distillation (OPD / OPSD)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Distill a student model on trajectories **it sampled itself**, using a teacher's per-token distribution as the target. Unlike offline distillation (student trained on teacher completions), on-policy distillation aligns the student's distribution on the exact contexts the student will actually visit at inference — no distribution shift, dense token-level supervision. Modern reasoning-post-training toolkit; the family variants (u-OPSD, SPOT, Multi-teacher OPD) differ in how the target distribution is constructed.

**Prereqs:** [grpo.md](grpo.md), [_rl.md](_rl.md), [rejection-sampling.md](rejection-sampling.md)
**Related:** [u-opsd.md](u-opsd.md), [spot-distillation.md](spot-distillation.md), [reasoning/long-cot-rl.md](reasoning/long-cot-rl.md)

---

## What it is

Two axes matter for distillation from a big teacher into a smaller student: **whose completions** you supervise on (teacher's = off-policy; student's = on-policy) and **what target** you regress toward (teacher token distribution, verifier reward, ground truth, or some combination).

On-policy distillation samples completions from the *student* policy, then uses the teacher's per-token distribution on those exact contexts as the supervision target — typically via reverse KL. The student is corrected precisely where it deviates from the teacher, on the trajectories it would actually generate at inference. Off-policy distillation, by contrast, trains the student on teacher-generated completions, which the student may never produce itself.

**On-Policy Self-Distillation (OPSD)** collapses teacher and student into one model — the "teacher" is a stronger version of the student (higher temperature, more rollouts, or the pretraining base). Self-distillation is the standard framing in the reasoning-RL literature.

## How it works

Basic OPD loop:

1. **Sample.** Draw `n` student completions per prompt: `o_i ∼ π_θ(· | q)`.
2. **Score.** Run the teacher `π_T` on the same `(q, o_i)` to get the teacher's token distributions at each position.
3. **Distill.** Update the student to minimize per-token divergence to the teacher:
   ```
   L = Σ_t KL( π_T( · | q, o_{<t}) || π_θ( · | q, o_{<t}) )
   ```
   Reverse KL (mode-seeking) is standard; forward KL is used less often.
4. **Repeat.** Fresh rollouts each step so the student's evolving distribution is always the training distribution.

Variants change one or two moves in this loop:
- **u-OPSD** (Li et al., 2026) removes the external teacher entirely — the "teacher" is a self-consistency pseudo-solution built from the student's own majority-vote answer.
- **SPOT** (Qu et al., 2026) targets *which positions* to distill (acquisition), *which candidates* to consider (exploration), and *how to weight them by downstream outcome* (exploitation).
- **Multi-teacher OPD** (Motif 3, 2026) blends several RL-specialized teachers into one student.

## Why it matters

- **Denser supervision than RL.** RL sees one scalar reward per rollout; OPD sees a KL target at every token. Faster convergence per rollout, especially in reasoning where the reward is sparse.
- **No value function needed.** Like GRPO, OPD is critic-free — the supervision comes from the teacher, not from a learned baseline.
- **The right substrate for reasoning distillation.** When the teacher is a long-CoT reasoning model and the student is a generalist, per-token KL preserves the teacher's step-by-step structure in a way that outcome-only rewards can't.
- **Composes with RLVR / GRPO.** Some recipes interleave OPD steps with GRPO steps; some (SPOT) use a verifier to reshape the OPD target directly.

## Gotchas & tricks

- **Reverse KL is mode-seeking.** The student converges to the teacher's dominant continuation and can miss other plausible ones — u-OPSD's disagreement-focused updates address this.
- **Fresh rollouts every step are expensive.** In practice you sample a batch of rollouts, do a few gradient steps, then resample. Too many inner steps → off-policy drift, same as in PPO/GRPO.
- **Teacher inference cost dominates.** Every student rollout needs a teacher forward pass on the same tokens. Cheap for small teachers, prohibitive for frontier teachers unless heavily amortized.
- **Temperature matters on both sides.** Student temperature controls exploration; teacher temperature controls how sharp the target is. Mismatched temperatures make the KL loss ill-conditioned.
- **Sanity check with off-policy first.** If off-policy distillation doesn't teach the student anything, on-policy won't either — the bug is upstream.

## Sources

- Paper: *On-Policy Distillation of Language Models* — foundational OPD framing (2023–2024 literature).
- Paper: *On-Policy Self-Distillation without Any Supervision* — Li et al., 2026 — u-OPSD, removes the teacher.
- Paper: *Sparse Probing and Outcome Calibration for On-Policy Distillation* — Qu et al., 2026 — SPOT, adds acquisition-exploration-exploitation.
- Paper: *Motif 3 Technical Report* — Motif Technologies, 2026 — Multi-teacher On-Policy Distillation used in production post-training.
