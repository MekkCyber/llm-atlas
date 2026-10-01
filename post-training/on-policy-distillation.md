# On-Policy Distillation (OPD)
*Depth — distill a teacher into a student using the student's *own* generated trajectories.*

**TL;DR:** On-policy distillation trains a student on trajectories *the student itself* generates, with a teacher providing per-token supervision (reverse KL or direct top-token). This avoids the exposure-bias of classical offline distillation, where the student learns token-by-token on teacher trajectories it will never actually visit at inference. OPD is the default recipe for compressing reasoning-RL models (R1-style distillation, Qwen distillations, Kimi long2short, SAKI).

**Prereqs:** [_rl.md](./_rl.md), [grpo.md](./grpo.md)
**Related:** [saki.md](./saki.md), [reasoning/long2short.md](./reasoning/long2short.md), [rlvr.md](./rlvr.md), [fine-tuning/README.md](./fine-tuning/README.md)

---

## What it is

Three distillation regimes exist in LLM post-training:

1. **Offline distillation** — SFT the student on teacher-generated trajectories. Simple, but at inference the student visits states the teacher never explored (exposure bias).
2. **Hard-label distillation** — SFT on `(prompt, teacher-answer)` pairs. Same exposure-bias problem, no probability information.
3. **On-policy distillation** — sample a trajectory from the **student**, score each token with the **teacher**, update the student. The student only ever learns on states it actually visits.

OPD is close in spirit to RL (student rollouts, per-token signal) but the signal is a *teacher* rather than an environment reward. It's used when the teacher is a bigger / stronger model you want to compress into a smaller one — most 2025–2026 "R1-distill-7B" / "distill-32B" models are OPD.

## How it works

For each prompt $q$:

1. Sample a trajectory $o \sim \pi_{\text{student}}(\cdot \mid q)$.
2. For each token $o_t$, compute teacher and student probabilities $p_t = \pi_{\text{teacher}}(\cdot \mid q, o_{<t})$ and $q_t = \pi_{\text{student}}(\cdot \mid q, o_{<t})$.
3. Minimize per-token **reverse KL** $\mathrm{KL}(q_t \,\|\, p_t)$, or equivalently maximize $\log p_t(o_t)$ scaled by the sampling-correction weight.

Variants swap:
- **Which teacher is queried** (fixed teacher; teacher co-trained via a method like SCOUT; teacher-guided rollouts with [maximal coupling](./saki.md)).
- **How supervision is routed** per token (uniform reverse-KL; routed by acceptance/correction events as in [SAKI](./saki.md); gated by teacher confidence).
- **How trajectories are shared** (per-student; or heterogeneous-model exchanges as in GRAFT, 2026).

## Why it matters

- **Standard tool for compressing R1-class models.** Nearly every "distilled from" small reasoner in 2025–2026 uses OPD. Static SFT on teacher trajectories reliably underperforms once answers get long.
- **The RL/distillation boundary blurs.** OPD with reverse KL shares machinery with GRPO (sampled rollouts, per-token/per-response weighting, KL to a reference). Infrastructure built for RL-with-verifiable-rewards runs OPD cleanly with the teacher as "verifier".
- **Teacher-side design surface is open.** 2026 papers (SAKI, SCOUT, GRAFT) all target how the *teacher* participates in on-policy rollouts, not how the student is updated. This is where current gains come from.

## Gotchas & tricks

- **Off-policy for the teacher.** Trajectories sampled from the student are on-policy for the student but *off-policy for the teacher*. The teacher's token probabilities may be miscalibrated on student-dominated prefixes. SCOUT (2026) co-trains the teacher via RL on student-generated content to fix this.
- **Weak students visit teacher-misaligned prefixes.** The usual KL supervision is less representative there. SAKI's maximal-coupling trick routes supervision based on realized agree/disagree events to handle this.
- **Rollout cost dominates.** Student trajectories must be sampled every step. Engine-resident speculative verifiers (SAKI reports 4.22× rollout throughput) that preserve the exact-$q$ distribution are the practical speedup.
- **Reverse vs forward KL.** Reverse KL ($\mathrm{KL}(q \,\|\, p)$) is mode-seeking: the student focuses on where the teacher puts mass. Forward KL ($\mathrm{KL}(p \,\|\, q)$) is mode-covering: the student spreads. Reasoning distillation almost always uses reverse KL.
- **Composable with RL.** OPD-then-RLVR, and RLVR-then-OPD, are both standard pipelines; order and KL-coefficient matter.

## Sources

- Paper: *SAKI: Maximal-Coupling-Routed Teacher Supervision for On-Policy Distillation* — Wei et al., Meituan, 2026 — [arXiv:2609.36601](https://arxiv.org/abs/2609.36601).
- Paper: *Scaling Properties of Same-Family On-Policy Distillation* — Bao et al., 2026 — [arXiv:2609.32722](https://arxiv.org/abs/2609.32722).
- Paper: *On the Off-Policy Teacher in On-Policy Distillation* (SCOUT) — Huang et al., 2026 — [arXiv:2609.38360](https://arxiv.org/abs/2609.38360).
- Paper: *Learning Beyond What You Sample: Off-Policy-Aware Cross-Model Trajectory Exchange* (GRAFT) — Jang et al., 2026 — [arXiv:2609.37868](https://arxiv.org/abs/2609.37868).
