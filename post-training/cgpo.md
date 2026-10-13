# CGPO — Competence-Guided Policy Optimization
*Depth — adaptive per-sample mixing of verifier-RL and on-policy distillation.*

**TL;DR:** GRPO works well when the policy already produces positive samples; on-policy distillation works well when a stronger teacher is available but the policy is weak. CGPO estimates a per-task *competence* and uses it to mix the two losses on each training sample: distill where the student is weak, RL where it is already competent. Introduced in the PolyOCR-Venus OCR foundation-model recipe but formulated generally.

**Prereqs:** [grpo](grpo.md), [rlvr](rlvr.md)
**Related:** [_rl](_rl.md), [rejection-sampling](rejection-sampling.md), [rl-prompt-curation](rl-prompt-curation.md)

---

## What it is

A post-training algorithm that routes each (prompt, sample) pair toward one of two loss modes — verifier-based GRPO or on-policy distillation from a teacher — based on an online estimate of how competent the current policy is on that prompt's task. Where many prior methods use *phases* (first SFT/distill, then RL), CGPO uses *per-sample* routing so both signals contribute throughout training.

## How it works

For each training sample:

1. **Competence estimate.** Compute a cheap proxy — rollout success rate over a small group, or teacher–student top-k overlap, depending on the task.
2. **Route.** If competence is low, apply an on-policy distillation loss (KL to teacher logits on student rollouts). If competence is high, apply a GRPO-style policy-gradient update using the verifier reward with a group baseline.
3. **Blend near the boundary.** A soft gate (sigmoid of competence) allows a weighted combination in the ambiguous region, so routing decisions don't thrash.

The verifier and teacher can be the same model used in different modes (teacher logits for distillation, scalar verdict for RL), which keeps infrastructure simple.

## Why it matters

The usual failure mode of pure GRPO on sparse-reward tasks is wasted training on prompts where the policy never samples a positive. The usual failure of pure distillation is capped performance at the teacher's level. CGPO addresses both: distillation unblocks sparse-reward prompts; RL lifts above the teacher where positives exist. The idea is likely to transfer well beyond OCR, which is why it is worth a standalone page.

## Gotchas & tricks

- **Competence proxy choice matters.** Rollout success rate is noisy at low N; teacher-overlap requires teacher logits for every sample.
- **Teacher drift.** If the teacher is also being updated, distillation targets move; freeze the teacher or use an EMA copy.
- **Does not fix bad verifiers.** CGPO assumes the verifier reward is clean on at least a subset of prompts; if the verifier is noisy everywhere, both branches degrade.

## Sources

- Paper: *PolyOCR-Venus: Unified OCR Foundation Models for Text-Centric Visual Intelligence* — GuangJian Team (Ant Group), 2026 — [arXiv:2609.37712](https://arxiv.org/abs/2609.37712)
- Code: [github.com/inclusionAI/PolyOCR-Venus](https://github.com/inclusionAI/PolyOCR-Venus)
