# On-Policy Distillation
*Depth — student learns from teacher scores on the student's own trajectories.*

**TL;DR:** On-Policy Distillation (OPD) trains a student by sampling trajectories from the *student's* current policy, scoring them with a stronger teacher (or a privileged self-teacher), and minimizing per-token KL against the teacher's distribution at every visited prefix. Compared to off-policy distillation (fixed teacher-generated dataset), OPD keeps supervision aligned with what the student actually visits, so credit is dense and non-stale — but it needs a live teacher-forward per rollout, and its teacher-student asymmetry has to be arranged deliberately (bigger teacher, privileged inputs, or view augmentation).

**Prereqs:** [rejection-sampling](rejection-sampling.md), [grpo](grpo.md)
**Related:** [rlvr](rlvr.md), [long-cot-rl](reasoning/long-cot-rl.md)

---

## What it is

A supervision paradigm that sits between pure imitation learning and RL. The student rolls out episodes from its own policy; a teacher is queried at every visited state and provides dense per-token targets (either the full next-token distribution, top-K logits, or a chosen action). The student's loss is a KL (or NLL) against those teacher targets, summed along the trajectory. The teacher is not updated.

Contrast with (i) off-policy distillation — teacher generates the whole trajectory offline, student imitates — which underuses the student's own error modes; and (ii) RLVR/GRPO — supervision is a sparse verifier reward at the end — which is high variance and rollout-hungry.

## How it works

Given student `πθ` and teacher `πT`:

1. Sample trajectory `τ = (s0, a0, s1, a1, …)` with `a_t ~ πθ(·|s_t)`.
2. For each visited prefix `s_t`, query teacher for `πT(·|s_t)` (or a privileged variant `πT(·|s_t, c_t)`).
3. Loss `L(θ) = Σ_t KL(πT(·|s_t) || πθ(·|s_t))` — a *forward* KL puts mass where the teacher does; some variants use reverse KL or top-K cross-entropy.
4. Backprop through `πθ` only.

The **asymmetry principle**: some information must reach the teacher that the student cannot use directly, or the loss is trivially zero. Common asymmetries: bigger teacher, retrieved-context teacher (LOPD), clean-image teacher vs augmented-view student (S²VOPD), long-context teacher vs short-context student (SimpleOPD).

## Why it matters

- Cheaper than end-only RL for a fixed capability gain (dense per-token signal).
- More faithful than off-policy distillation to the student's actual failure modes.
- Composable — the teacher can be replaced with a self-teacher (OPSD), a retrieval-conditioned teacher, or an augmentation-differential teacher without changing the outer loop.
- Recent works close the historical "you need a bigger teacher" gap (S²VOPD via view augmentation; LOPD via learned latent context).

## Gotchas & tricks

- Teacher-forward cost dominates unless you cache logits or use top-K distillation.
- Reverse-KL vs forward-KL matters: reverse encourages mode-seeking (student picks one teacher mode), forward encourages coverage. Forward is the default for reasoning distillation.
- Tokenizer mismatch between teacher and student is a real blocker for cross-family OPD — SimpleOPD's semantic-token alignment is a workaround.
- Rollout budget can be reduced dramatically (LOPD reports <30% of GRPO's budget for parity) when the teacher signal is dense enough.

## Sources

- Self-Supervised Visual On-Policy Distillation (S²VOPD) — Li et al., 2026 — [arXiv:2608.14144](https://arxiv.org/abs/2608.14144)
- SimpleOPD: Tokenizer-Agnostic On-Policy Distillation for Long-Context Reasoning — He et al., 2026 — [arXiv:2608.14277](https://arxiv.org/abs/2608.14277)
- Latent On-Policy Self-Distillation (LOPD) — Zhang et al., 2026 — [arXiv:2608.13040](https://arxiv.org/abs/2608.13040)
