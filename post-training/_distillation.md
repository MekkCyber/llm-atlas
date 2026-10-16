# Distillation (for LLM post-training)

*Taxonomy — using a teacher model to shape a student LLM's behavior.*

**TL;DR:** Distillation moves capability from a strong "teacher" model into a cheaper/smaller "student" model. The modern taxonomy splits by **where the supervision is applied**: on the teacher's own rollouts (offline SFT), on the student's rollouts (on-policy distillation), on the action rather than the tokens, or via a critique-then-revise loop. On-policy variants dominate for reasoning and agent post-training because they fix exposure bias; offline SFT still wins when teacher trajectories are the actual training target (e.g. R1 → small-model distillation).

**Related taxonomies:** [_post-training](./_post-training.md) · [_rl](./_rl.md) · [_rewards](./_rewards.md)
**Depth files covered here:** [on-policy-distillation](./on-policy-distillation.md) · [critique-guided-opd](./critique-guided-opd.md) · [multi-teacher-opd](./multi-teacher-opd.md) · [action-level-distillation](./action-level-distillation.md) · [rejection-sampling](./rejection-sampling.md)

---

## The problem

Direct training signals for an LLM are either expensive (frontier teachers via logits, human preferences) or sparse (binary verifier rewards). Pure SFT on teacher outputs leaks exposure bias — the student is never trained on its own mistakes. Pure RL from a verifier gets a single scalar per rollout. Distillation tries to combine the density of teacher supervision with the on-policy structure of RL.

## The shared pattern

Every distillation variant pairs:

1. A **supervisory source**: teacher logits, teacher score, teacher critique, teacher action.
2. A **training distribution**: teacher rollouts (offline) or student rollouts (on-policy).
3. A **loss shape**: cross-entropy (SFT), KL (logit-match), PPO-style clipped (RL under teacher reward).

Variants pick different combinations across these three axes.

## Variants

| Technique | Supervision source | Training distribution | When it wins |
| --- | --- | --- | --- |
| [rejection-sampling](./rejection-sampling.md) | teacher verifier passes/fails | teacher rollouts filtered → SFT | large-scale SFT distillation with cheap verifier (R1, Llama) |
| offline SFT distillation (no depth file yet) | teacher logits / responses | teacher rollouts | strong teacher, cheap student, well-matched tokenizer |
| [on-policy-distillation](./on-policy-distillation.md) | teacher logits / score | student rollouts | reasoning RL where exposure bias matters |
| [critique-guided-opd](./critique-guided-opd.md) | teacher critique → revision score | student rollout + revised suffix | long-form math where mid-trace errors dominate |
| [multi-teacher-opd](./multi-teacher-opd.md) | multiple domain-routed teachers | student rollouts, cross-teacher critics | multi-domain post-training with teacher disagreement |
| [action-level-distillation](./action-level-distillation.md) | teacher action vs student action | student rollouts in environments | black-box teachers, tool-use agents, mismatched tokenizers |

## How to choose

- **Shared tokenizer, cheap-to-run teacher, exposure bias is the pain:** logit-level [on-policy distillation](./on-policy-distillation.md).
- **Teacher is API-only or heterogeneous architecture:** [action-level distillation](./action-level-distillation.md).
- **Teacher has a verifier (math, code, unit tests):** [rejection sampling](./rejection-sampling.md) + SFT is the DeepSeek-R1 default and the strongest known distillation-to-small-model recipe.
- **Student rollouts are mostly wrong, teacher is a long-form reasoner:** [critique-guided OPD](./critique-guided-opd.md) — gives the teacher a locus to correct instead of a far-away gold answer.
- **Several teachers of different domains, one student:** [multi-teacher OPD](./multi-teacher-opd.md).

Combinations work: R1's recipe uses rejection-sampling SFT *and* subsequent GRPO — the SFT distills coverage, the RL distills correctness.

## Adjacent but distinct

- **Preference learning ([DPO](./dpo.md)):** uses pairs, not teacher outputs. A preference pair is not a distillation signal.
- **Reward modeling ([_rewards](./_rewards.md)):** trains a model *as* a reward function. A teacher-as-reward in score-level OPD overlaps but the training target is different.
- **Rejection sampling for SFT ([rejection-sampling](./rejection-sampling.md)):** a specific offline-distillation workflow — included here but the depth file covers mechanics.

## Sources

- Paper: *DeepSeek-R1* — 2025 — SFT distillation on 800k teacher rollouts; the "distill > small-model RL" result.
- Paper: *Recovering Reliable Supervision via Critique-Guided On-Policy Distillation* — Yu et al., 2026.
- Paper: *Beyond Domain-Routed Supervision: Cross-Teacher Guarding for Multi-Teacher On-Policy Distillation* — Tian et al., 2026.
- Paper: *ActionOPD: Distilling Decisions, Not Tokens* — 2026.
