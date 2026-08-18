# LOPD — Latent On-Policy Self-Distillation
*Depth — makes the "privileged context" fed to a self-teacher learnable end-to-end.*

**TL;DR:** LOPD is an on-policy self-distillation method where the student is its own teacher, but the *privileged context* that gives the teacher an information advantage is a **learned continuous prefix** — not a hand-picked artifact like "the correct answer" or "an oracle skill." A retrieval module composes past experience into latent tokens, and a privileged-margin loss trains those tokens so the teacher stays reliably better than the student. Beats OPSD/SDPO/Skill-SD on agent tool-use and code, and matches GRPO with under 30% of its rollout budget.

**Prereqs:** [on-policy-distillation](on-policy-distillation.md), [grpo](grpo.md)
**Related:** [rlvr](rlvr.md), [rejection-sampling](rejection-sampling.md)

---

## What it is

Prior on-policy self-distillation (OPSD) needs a hand-designed privileged input `c` — the ground-truth answer, an environment-side feedback signal, a skill label, or a demonstration trajectory — that lets the teacher `πT(·|s, c)` outperform the student `πS(·|s)` at every prefix. LOPD replaces `c` with a learned latent prefix produced by retrieving relevant experience and passing it through a projection into the model's embedding space. The latent prefix is optimized end-to-end, so the "privilege" adapts as the student improves.

## How it works

At training step `t`:

1. **Rollout.** Student samples trajectory `τ ~ πθ(·|task, history)`.
2. **Retrieve.** A retriever pulls `k` related past experiences from a memory of prior tasks/trajectories.
3. **Compose.** Retrieved items are projected into `m` continuous latent tokens `z = f_φ(retrieved)`, forming the privileged prefix.
4. **Teacher forward.** Query `πθ(·|z, s_t)` at every visited prefix. (Same weights as student — self-teacher — but with the extra prefix.)
5. **Distill.** Dense KL: `Σ_t KL(πθ(·|z, s_t) || πθ(·|s_t))`.
6. **Privileged-margin loss.** Add a regularizer that pushes teacher log-likelihood on ground-truth continuations above student log-likelihood by a margin — prevents `z` from collapsing into a no-op.

## Why it matters

- **Rollout efficiency.** <30% of GRPO's rollout budget for matched task performance. Rollouts are the dominant cost in agentic RL, so this is a big deal for post-training economics.
- **Curriculum-free.** The old OPSD recipe needed someone to pick what "privileged context" meant per task. LOPD lets the model learn what advantage-conferring context to condition on.
- **Continual self-improvement.** As the student gets stronger, `z` re-shapes to remain a useful teacher — a step toward self-directed post-training loops.

## Gotchas & tricks

- The privileged-margin loss is essential — ablations show the latent prefix degenerates into a no-op without it.
- Retrieval quality matters; a coarse retriever + strong projection can outperform a fine retriever + weak projection because the projection can compensate.
- Because the teacher shares weights with the student, teacher-forward is only the added-prefix cost — much cheaper than a separate frontier teacher.

## Sources

- Latent On-Policy Self-Distillation — Guibin Zhang et al., 2026 — [arXiv:2608.13040](https://arxiv.org/abs/2608.13040)
