# On-Policy Distillation (OPD)
*Depth — distill a teacher into a student by supervising the student's own rollouts, not the teacher's.*

**TL;DR:** Classical distillation samples `(x, y)` from a teacher and trains a student to imitate `y`. On-policy distillation samples trajectories from the *student*, then asks the teacher to grade or correct them token-by-token. The student learns from its own mistakes under the teacher's supervision, which eliminates the train-inference distribution shift that plagues off-policy distillation and transfers teacher behavior on inputs the student actually generates.

**Prereqs:** [_post-training](../_post-training.md), [README (fine-tuning)](README.md)
**Related:** [grpo](../grpo.md), [rejection-sampling](../rejection-sampling.md), [rlvr](../rlvr.md), [_rl](../_rl.md)

---

## What it is

Three distillation regimes, distinguished by *where the training data comes from*:

| Regime | Trajectories come from | Supervision |
| --- | --- | --- |
| **Off-policy** | Teacher (or fixed dataset) | Teacher's own tokens as labels |
| **On-policy (OPD)** | Student (current policy) | Teacher scores or corrects the student's tokens |
| **Mixed / interpolated** | Blend of student and teacher | Teacher supervision on both |

OPD is the one that matters at scale because distilled students eventually have to *generate* at inference. If you only ever train them on the teacher's distribution, they're brittle on their own.

The on-policy setup looks like RL — sample from the policy, score it, update — but the scorer is the teacher rather than an environment reward. For reasoning distillation, this is essentially "RL with the teacher as the reward model, except the reward is per-token likelihood or correctness."

## How it works

### Setup

- **Student** `π_θ` — a smaller model (or the same model being self-distilled).
- **Teacher** `π_T` — larger or same-sized but conditioned on reference solutions (self-distillation case).
- Prompt set `D` (no answers required — labels come from the teacher).

### Training loop

```
for prompt x in D:
    y_s ~ π_θ(·|x)                     # sample a student trajectory
    scores = π_T.score(x, y_s)         # teacher grades token-by-token
    loss = -Σ_t weight(t) · log π_θ(y_s[t] | x, y_s[<t])
                          · (alignment signal from scores)
    update θ to minimize loss
```

Exact form of `weight(t)` and `alignment signal` varies:

- **Forward KL**: match the teacher's full next-token distribution on every student state.
- **Reverse KL**: match the student's distribution to the teacher on the student's own trajectories.
- **Token-level advantages**: use teacher-student likelihood ratios as advantages in a PPO-style update (makes OPD look structurally identical to GRPO with the teacher as the critic).

### Variants

- **Student-sampled, teacher-labeled**: default. Student generates; teacher is queried once per token.
- **Multi-teacher OPD (MOPD)**: a mixture of teachers scores the student; mixing weights can be fixed or learned (P²-MOPD: Wang et al., 2026, arXiv 2611.03953, treats teacher weights as parameter priors).
- **Self-distillation (OPSD)**: the "teacher" is the same model as the student, usually conditioned on a reference solution or extra chain-of-thought. Introduces [teacher-drift](../teacher-drift.md) risk — the teacher starts echoing the student's errors over training.

## Why it matters

- **Closes train-inference distribution shift.** The student only trains on states it actually produces — no imitation-exploit on teacher-only states it never visits.
- **Composes with existing RL infrastructure.** OPD rollouts fit into the same rollout/update pipeline as GRPO and PPO. Many open-source RL stacks (veRL, TRL, OpenRLHF) support OPD as a near-trivial alternative objective.
- **Data-efficient.** Doesn't need labeled trajectories — just prompts and a teacher. Prompt sets are much cheaper to curate than reasoning traces.
- **The backbone of modern distilled reasoning models.** Distilled R1-style reasoners (Qwen-R1-distill, Phi-reasoning, etc.) rely on OPD-style training to transfer long-CoT capability into smaller students.

## Gotchas & tricks

- **Teacher drift in self-distillation.** When the teacher is the same model conditioned on hints, it can start echoing the student's accumulated errors. [TOPSD](teacher-drift.md) diagnoses this as "Teacher Drift" and proposes teachable-prefix gating.
- **KL direction matters.** Reverse KL (mode-seeking) produces more decisive students but can collapse if the teacher's distribution is multimodal. Forward KL (mass-covering) preserves diversity but keeps lower-quality modes.
- **Reward-signal sparsity.** If the teacher only grades at the end of a trajectory (final-answer check), OPD reduces to RLVR with the teacher-check as the verifier.
- **Teacher throughput is the bottleneck.** Scoring every student token with a larger teacher costs more than the student's own forward pass. Common fixes: batch-score, cache teacher logits, or score only a subset of tokens.
- **Watch for policy collapse.** Like any on-policy RL method, OPD can mode-collapse if the KL term to a reference model is too weak. Keep a small KL penalty to a frozen SFT checkpoint.

## Sources

- Paper: *P²-MOPD: Accelerating Multi-Teacher On-Policy Distillation via Parameter Priors* — Wang et al., 2026 — arXiv 2611.03953 — multi-teacher OPD with learned per-teacher priors.
- Paper: *Knowing When to Teach: Learning from Teachable Prefixes in Multilingual Reasoning Distillation (TOPSD)* — Cho et al., 2026 — arXiv 2611.08179 — diagnoses teacher drift in on-policy self-distillation.
- Paper: *On-Policy Distillation of Language Models* — Agarwal et al., 2023 — one of the first explicit OPD formulations for LMs.
- Paper: *Teaching Small Language Models to Reason* — Hsieh et al., 2023 — adjacent distillation-for-reasoning line.
