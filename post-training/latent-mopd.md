# Latent-MOPD — Latent Multi-Teacher On-Policy Distillation
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Standard multi-teacher on-policy distillation (OPD) supervises the student only on teachers' output distributions. Latent-MOPD adds supervision on the *hidden states* that produce those outputs: for each on-policy sample, a routed specialist supervises the student via both channels (logits + late-layer representations), with a shared projection to bridge width mismatches and domain-grouped updates. On nine math/code/logic benchmarks the student beats token-only, representation-only, and uniform-averaging baselines. Introduced in *Latent-MOPD* (2026, arXiv:2610.02381).

**Prereqs:** [_post-training.md](_post-training.md), [rejection-sampling.md](rejection-sampling.md)
**Related:** [_rl.md](_rl.md), [grpo.md](grpo.md), [cot-reward-model.md](cot-reward-model.md)

---

## What it is

On-policy distillation (OPD) is distillation where the student generates its own rollouts and a teacher scores or rewrites them. Multi-teacher OPD routes each rollout to a specialist (math, code, logic, …). Prior work transferred only *what* each specialist predicts — output logits. Latent-MOPD adds the *hidden state* channel so representations align, not just predictions.

## How it works

Three design choices make the latent channel usable:

1. **Target layer selection.** Pick a *late* layer in each teacher as the representation target (close to the output), chosen by the teacher-student relationship (depth, width).
2. **Shared projection.** For cross-family or unequal-width teachers, project both the student and teacher hidden states through a learned shared linear map before the representation loss.
3. **Domain-grouped updates.** All samples routed to specialist A are grouped into one gradient update; domains are not interleaved within a batch. An all-layer representation-only control that *does* interleave collapses — grouping is load-bearing.

Each teacher's supervision schedule shifts the loss weight from hidden-state alignment early → token-prediction alignment later. Both channels always use the same routed specialist.

## Why it matters

OPD is becoming the standard way to fuse a zoo of RL-trained specialists into one deployment-ready student. Representation-level distillation has been studied for static (offline) distillation but was open for *on-policy* multi-teacher settings. Latent-MOPD closes the gap and shows the hidden-state signal is a free lift once width/domain coupling is handled.

Reported results: with the same parameter count as each teacher, the student surpasses the per-benchmark best teacher on a majority of nine benchmarks (same-family setting). With separately developed cross-family teachers, Latent-MOPD outperforms both single-channel baselines on all nine.

## Gotchas & tricks

- **Domain purity is non-negotiable.** Interleaving domains inside an update collapses the representation signal — the student can't align with teachers that disagree inside the same gradient step.
- **Late-layer targets only.** Early-layer alignment hurts because early features are shared across tasks; late features are specialist-shaped.
- **Shared projection is cheap but essential cross-family.** Without it, hidden-state loss is dominated by width-related noise.
- **Not a replacement for rollout-side filtering.** On-policy rollouts still benefit from rejection sampling before the student sees them — latent OPD is orthogonal to trajectory curation.

## Sources

- Paper: *Latent-MOPD: Latent Multi-Teacher On-Policy Distillation* — Fang, Hong, Yang et al., 2026 — [arXiv:2610.02381](https://arxiv.org/abs/2610.02381).
