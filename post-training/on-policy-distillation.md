# On-Policy Distillation
*Depth — distill a teacher into a student using rollouts the student itself produces, so supervision tracks the student's actual decoding distribution.*

**TL;DR:** Standard distillation matches teacher and student log-probs on *teacher* rollouts (or some fixed dataset). On-policy distillation samples from the *student* and supervises against the teacher's responses to those same prefixes. This fixes the train/test-distribution mismatch that hurts naive distillation on autoregressive models. Two flavors: **output-space** (match log-probs at the LM head — the historical default) and **representation-space** (match hidden states at chosen layers — OPRD, 2026, which avoids LM-head variance and exposes more signal).

**Prereqs:** [_rl](_rl.md), [grpo](grpo.md)
**Related:** [rlvr](rlvr.md), [reasoning/long-cot-rl](reasoning/long-cot-rl.md)

---

## What it is

Distillation: train a student to match a teacher. The student-vs-teacher gap is dominated by *where* you supervise:

| Supervision data | Issue |
| --- | --- |
| Teacher rollouts only | Student sees prefixes it would never generate; train/test mismatch |
| Student rollouts, no teacher | No teacher signal at all — just self-training |
| **Student rollouts, teacher relabels** | On-policy distillation: teacher's distribution evaluated at student-visited states |

The third is the modern default for reasoning post-training: it's the recipe behind the DeepSeek-R1 → smaller-model distillations and the Qwen reasoning-distill family.

## How it works

The basic loop:

```
for step:
    prompts ← sample batch
    student_rollouts ← π_student.generate(prompts)
    for each prefix in student_rollouts:
        target ← π_teacher(prefix)        # teacher distribution at the student's actual state
        loss += supervision(student(prefix), target)
    backprop, step
```

The choice of `supervision(·, target)` is the axis where output-space and representation-space split:

- **Output-space (canonical).** KL divergence between student and teacher next-token distributions at every position. Standard distillation loss. Limit: variance from sampling-token mismatch dominates late in training; the LM head compresses high-dimensional hidden state down to a vocab-size logit, throwing away structural information.
- **Representation-space (OPRD, 2026).** L2 or cosine on hidden states at chosen layers, *before* the LM head. Deterministic supervision per state (no sampling variance in the target); exposes the full hidden-state geometry, not just the projection onto vocab space.

Both share the on-policy backbone — student rollouts, teacher relabels. They differ only in *what* the teacher provides and *where* the loss is computed.

## Why it matters

- **Bridges SFT and RL.** On-policy distillation is "SFT on student-visited states with a teacher's soft targets" — it inherits SFT's stability and RL's distribution-matching property.
- **Smaller-model reasoning.** Distilling a strong reasoner (R1-32B, etc.) into a 1.5B–7B model with on-policy distillation routinely beats running RL directly on the small model. This is the empirical lesson of the R1-distill line.
- **OPRD-specific wins.** 1.44× faster training and 54% less memory on math benchmarks vs. output-space on-policy distillation, while closing the student-teacher gap.

## Gotchas & tricks

- **Teacher must be available during training.** Unlike pure SFT, you can't precompute a fixed dataset — the teacher is queried at student-visited states. Co-located teacher serving is the standard infra.
- **Pick supervision layers carefully (representation-space).** Late layers carry more task-specific signal; early layers may not need matching. OPRD selects a subset rather than all.
- **LM head compression is real.** A vocab-size softmax discards hidden-state structure. Representation-space loss recovers it but introduces a new question: how to align student and teacher hidden dims when they differ (linear adapter typically).
- **KL on output-space is high-variance late in training.** As the student converges, near-zero log-probs dominate the KL estimator. Representation-space sidesteps this entirely.

## Sources

- Paper: *OPRD: On-Policy Representation Distillation* — Yang et al., 2026 — [arXiv:2606.06021](https://arxiv.org/abs/2606.06021) — primary source for the representation-space variant.
- Paper: *DeepSeek-R1* — DeepSeek, 2025 — output-space on-policy distillation from R1 to smaller models is the canonical recipe.
- Paper: *MiniLLM: Knowledge Distillation of Large Language Models* — Gu et al., 2024 — earlier on-policy KL distillation for LLMs.
