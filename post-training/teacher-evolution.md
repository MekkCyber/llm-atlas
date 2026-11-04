# Teacher-Evolution Self-Distillation (TESD)
*Depth — on-policy self-distillation with a co-evolving self-teacher trained by RL rewards.*

**TL;DR:** On-policy self-distillation (OPSD) trains a student to match a "self-teacher" — the same model given privileged context — on the student's own rollouts. In practice OPSD plateaus because the self-teacher's guidance is not uniformly reliable. TESD lets the self-teacher *evolve* via its own RL rollouts, rewards it for both task correctness and **context independence**, and down-weights student samples where the evolved teacher disagrees with itself across context variants. Breaks the OPSD plateau and outperforms token- and sequence-level baselines across multiple datasets and scales.

**Prereqs:** [_post-training.md](./_post-training.md) · [_rl.md](./_rl.md) · [grpo.md](./grpo.md)
**Related:** [rejection-sampling.md](./rejection-sampling.md) · [order-only-distillation.md](./order-only-distillation.md) · [rl-prompt-curation.md](./rl-prompt-curation.md)

---

## What it is

**OPSD** assumes you have a *self-teacher*: the same model given access to extra privileged information (ground-truth answer, longer context, scratchpad). The self-teacher produces a soft target distribution on the student's own rollouts, and the student is trained to match it. Compared to sequence-level self-improvement (STaR, ReST), OPSD gives dense token-level signal.

**The problem**: the self-teacher is only as good as the model is at exploiting its privileged context. On some samples the privileged context doesn't help, or the model confuses signal with noise — the teacher's distribution is then noise, and distillation makes the student worse.

**TESD** addresses this with two coupled optimizations:
- **Teacher-oriented** — the self-teacher rolls out on prompts, gets a reward mixing task correctness with a "context-invariance" bonus, and takes RL updates.
- **Student-oriented** — standard OPSD with per-sample confidence weights derived from the evolved teacher's agreement with itself across context variants.

## How it works

1. Hold a frozen copy of the base model. The student and the self-teacher are two trainable copies.
2. For each prompt:
   - The student rolls out.
   - The self-teacher rolls out with privileged context (e.g. gold answer, longer CoT).
   - Compute a reliability score `w` = agreement between teacher-with-context and teacher-without-context (e.g. KL or token-overlap).
3. **Teacher update**: standard RL step (GRPO-style) on the self-teacher. Reward = task correctness + `λ · 𝟙[w high]`. This rewards the teacher for being useful *and* for producing answers that don't depend on the privileged context (because the student will see only the non-privileged version).
4. **Student update**: OPSD cross-entropy against the teacher's distribution, but each sample weighted by `w`. Low-reliability samples barely update the student.

Steps 3 and 4 interleave: as the teacher evolves, the reliability scores change, which changes which samples count in the student update, which eventually changes the student, and so on.

## Why it matters

- **Unblocks OPSD plateaus.** The failure mode of flat training curves after a few thousand steps goes away once bad-teacher samples are down-weighted.
- **No stronger external teacher required.** The method only needs the student model itself plus a source of privileged context (which is usually already in the training data). Good for frontier models with no stronger teacher available.
- **Context-invariance is reusable.** The same "answer must not depend on privileged context" objective can be wired into any self-supervision pipeline, not just distillation.

## Gotchas & tricks

- **Teacher drift.** If the teacher RL phase overfits to its reward, it can diverge from being a useful supervisor. Keep the teacher's KL to the base model bounded.
- **Reliability weighting can collapse.** Early in training almost everything looks unreliable and the student barely moves. Use a warmup where `w` is uniform for the first few epochs.
- **Reward decomposition matters.** Correctness reward alone recovers vanilla self-RL; the context-invariance bonus is what makes the teacher a *distillation* target rather than a self-play target.
- **Compute cost.** Teacher rolls out at least as often as the student; roughly 2× OPSD compute.

## Sources

- Paper: *On-Policy Self-Distillation with Teacher Evolution* — Shen et al., ICT CAS / UCAS / Meituan, 2026 — [arXiv:2610.38637](https://arxiv.org/abs/2610.38637).
- Paper: *DeepSeekMath / GRPO* — Shao et al., 2024 — the RL algorithm the teacher update uses.
