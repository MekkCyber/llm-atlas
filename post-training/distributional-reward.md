# Distributional Reward Modeling (Z-Reward)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A reward-modeling framework that **predicts a distribution over rubric scores** instead of a single scalar. The motivation: subjective preferences (image quality, helpfulness) are inherently distributional — different raters score the same response differently, and a scalar collapses that signal. **Z-Reward** (Alibaba Z-Image, 2026) uses a heavyweight 27B teacher (**GDSO**) that generates a CoT + score distribution per rubric, and distills it into a 9B student (**RISD**) that emits the same distribution forward-pass-cheaply. The student keeps **88.6%** of the teacher's 89.6% human-preference accuracy at one-third the parameters; used as the online RL reward, it yields +41.3% net human-preference improvement over the SFT baseline.

**Prereqs:** [_rewards.md](_rewards.md), [cot-reward-model.md](cot-reward-model.md)
**Related:** [dpo.md](dpo.md) · [_post-training.md](_post-training.md)

---

## What it is

Standard reward models output a scalar — "this response is a 7/10." That scalar collapses two distinct sources of variance:

1. **Inherent ambiguity** in the rubric (Is "creativity" 6 or 9? Defensible both ways.)
2. **Aleatoric noise** in raters (one rater says 7, another 8.)

A learned scalar RM averages these away; the RL signal becomes "shift toward the consensus" rather than "shift toward responses that the population would like." Z-Reward's claim: predict the *full distribution* over rubric scores instead, and use the distribution as the RL signal.

In practice this looks like outputting a probability over a discrete score bucket per rubric ($\{1,2,...,10\}$ for each of $k$ rubrics), trained on labels that are themselves distributions (multiple rater scores per (response, rubric) pair).

---

## How it works

### Teacher: GDSO (27B)

The teacher is a heavyweight reasoning-style model: given a response and rubric, it generates a **chain-of-thought** explaining its judgment, then emits a **categorical distribution** over the rubric's score buckets. Training labels are rater distributions; the CoT is a learned scaffold, not directly supervised.

Conceptually GDSO is a CoT-Reward-Model whose output is distributional rather than scalar. The CoT lets the teacher *reason about* the response before scoring — captures nuanced cases that a non-CoT model collapses.

### Student: RISD (9B)

A smaller model distilled from the teacher. RISD does **not** generate CoT at inference; it directly emits the rubric-score distribution in one forward pass. The teacher's CoT is *internalized* during distillation — the student learns to compute the distribution without explicitly reasoning.

This is the deployment-cheap path: the teacher reasons heavily (slow), the student outputs the same distribution cheaply (fast enough for RL rollouts).

### Distillation recipe

The student's loss is KL divergence between its predicted distribution and the teacher's distribution. The teacher's CoT is not used as a separate input to the student — it's purely a teacher-side scaffold that shapes the teacher's distribution, which the student learns to match.

### Using it as an RL reward

The distributional output is converted to a usable scalar at RL time. The paper uses the **expected score** under the predicted distribution as the policy-gradient signal — but the distribution itself is preserved for adaptive uses (e.g. variance-aware GRPO normalization, uncertainty-gated updates).

---

## Why it matters

- **Closes the teacher-student gap** at one-third the parameters — 88.6% vs 89.6% human-preference accuracy. The CoT really is distillable into a forward-pass head.
- **+41.3% net human-preference improvement** over the SFT baseline when used as the online T2I RL reward — large enough to register as a frontier-quality jump.
- **Transferable beyond T2I.** The teacher/student distillation pattern and the distributional reward target both apply directly to LLM RL: replace CoT-Reward-Model's expensive CoT-per-query with a distilled distributional head.
- **Cleaner uncertainty signal.** Variance of the predicted distribution is a usable uncertainty signal — high-variance predictions can be flagged for human review or downweighted in policy updates.

---

## Gotchas & tricks

- **Labels need to be distributions, not majority votes.** If you collapse multiple rater scores to a single "consensus" label, the distributional target degenerates to a near-delta and you've lost the benefit. Keep individual rater scores during data collection.
- **Bucket choice is load-bearing.** Coarse buckets (1–10) are easier to predict but miss fine distinctions; fine buckets (1–100) are sparser. The paper uses rubric-specific bucket counts; tune per rubric.
- **Student forgets CoT structure.** The teacher's CoT contains intermediate reasoning that the student approximates with attention patterns. Edge cases the teacher handled via explicit reasoning may degrade in the student — periodic teacher re-distillation helps.
- **Don't confuse with quantile regression rewards.** Distributional reward predicts a categorical over discrete score buckets; quantile regression predicts quantiles of a continuous reward distribution. Same family, different parameterization.
- **Borderline for LLM RL.** The original paper is text-to-image. The mechanism transfers but hasn't been validated on LLM-RL benchmarks at the time of writing — treat the LLM-RL claims as plausible-but-unverified.

---

## Sources

- Paper: *Beyond Scalar Rewards by Internalizing Reasoning into Score Distributions* (Z-Reward) — Jin, Cai, Li, Zhan, et al. (Alibaba Z-Image / Nankai), 2026 — [arXiv 2606.09076](https://arxiv.org/abs/2606.09076).
- Concept: CoT Reward Model — see [cot-reward-model.md](cot-reward-model.md) for the scalar predecessor.
