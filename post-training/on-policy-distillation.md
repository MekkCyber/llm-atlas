# On-Policy Distillation (OPD)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Hybrid post-training recipe: the **student** generates rollouts (on-policy, like RL); the **teacher** scores them with *dense* per-token supervision (like distillation). Combines exploration with rich training signal. An empirical analysis on several LM/VLM pairs shows OPD updates are **coordinate-sparse and FFN-heavy** (training only the discovered sparse subnetwork ≈ full OPD) and **spectrally orthogonal** to the source weights' principal singular directions (updates land on coordinates where the source is near-zero). Implies a cheaper sparse-OPD variant and a clean fingerprint for OPD-trained checkpoints.

**Prereqs:** [_post-training.md](_post-training.md), [_rl.md](_rl.md)
**Related:** [rejection-sampling.md](rejection-sampling.md), [grpo.md](grpo.md), [appo.md](appo.md)

---

## What it is

Three default post-training recipes:

| Recipe | Source of trajectories | Source of supervision |
| --- | --- | --- |
| SFT | Teacher / dataset | Teacher tokens (next-token) |
| RL (GRPO/PPO) | Student rollouts | Scalar reward |
| **OPD** | **Student rollouts** | **Teacher per-token distribution** |

OPD inherits SFT's dense gradient signal (no variance from sparse rewards) and RL's coverage of states the student actually visits (no train/deploy distribution shift). The cost is teacher inference on every rollout.

This depth file focuses on the empirical analysis of *what OPD changes in the student's weights* — itself the contribution of the 2026 paper.

---

## How it works

### Recipe

1. Student samples rollouts under its current policy.
2. Teacher evaluates each rollout, producing per-token logits or distributions.
3. Loss: KL between student and teacher at each token (or cross-entropy if teacher targets are token argmax).
4. Backprop into student; teacher is frozen.

The KL term is dense — every token contributes — unlike RL's outcome-only reward. The trajectories are on-policy — unlike SFT.

### Sparsity finding

Across several student/teacher pairs (Qwen 7B/72B, LLaVA 7B/34B, others), OPD's parameter updates are **sparse in coordinate space**: a small fraction of parameters do the work. Training only the discovered sparse subnetwork recovers near-full OPD performance.

The sparsity is **FFN-heavy**: feed-forward layer parameters dominate the updated subnetwork; attention parameters move less.

### Geometry finding

OPD updates are *not* low-rank. They are numerically full-rank but **spectrally concentrated**: they lie disproportionately on coordinates where the source weight matrix is near zero, away from its principal singular subspaces.

This is qualitatively different from SFT (which moves principal directions, since teacher-token gradients align with strong-signal directions) and from pure RL (which produces noisier, less structured updates).

### Optimizer interaction

Naive sparsity-inducing SGD underperforms AdamW for OPD. The dense teacher supervision has heterogeneous coordinate-wise gradient scales, and AdamW's adaptive scaling preserves the gradient geometry that SGD averages away.

---

## Why it matters

- **Cheap sparse-OPD variant.** Train only the discovered subnetwork after a short warmup; major compute savings.
- **OPD checkpoints have a fingerprint.** The "updates land on near-zero coordinates" property is a candidate signature for detecting OPD-trained models — relevant for provenance and IP enforcement.
- **Reframes the role of dense teacher signal.** It's not just SFT-with-on-policy-data; it produces qualitatively different weight changes.
- **Connects to interpretability.** Sparse, FFN-localized, spectrally-distinct updates are an ideal substrate for mech-interp probes.

---

## Gotchas & tricks

- **Teacher inference dominates cost.** Each rollout needs a teacher forward pass at every token; for long rollouts and a large teacher, this is the bottleneck.
- **KL vs. cross-entropy.** Soft KL preserves the teacher's distribution; hard cross-entropy (argmax targets) loses information but is cheaper to store. Soft is usually worth it.
- **AdamW > SGD.** Don't try to exploit sparsity with sparsity-inducing optimizers — empirically loses to standard AdamW.
- **Coordinate-sparse ≠ low-rank.** Methods that assume low-rank updates (LoRA-style adaptation) don't capture OPD's structure; the sparse subnetwork is the right approximation.
- **Cold start.** If the student's initial policy diverges sharply from the teacher's, KL gradients explode. Warm with a brief SFT phase or clip the per-token KL.
- **Domain shift between teacher and student rollouts.** If the teacher is strong on the rollout distribution, OPD works well; if not, the teacher's supervision is itself noisy and the dense signal advantage shrinks.

---

## Sources

- Paper: *Dense Supervision, Sparse Updates: On the Sparsity and Geometry of On-Policy Distillation* — Yu, Ma, Jiang, Ye, Liu, Hu — Nanjing U. · Alibaba AMAP, 2026 — [arXiv 2606.13657](https://arxiv.org/abs/2606.13657).
- Background: hybrid SFT+RL recipes (DAgger, Tülu-3 stages, R1 distillation) for the broader family OPD sits in.
