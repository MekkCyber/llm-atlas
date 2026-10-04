# Distillation for LLM Post-Training
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Teach a *student* model to match a stronger *teacher*'s distribution by minimizing a token-level KL between them. The two live knobs are **KL direction** (forward or reverse) and **rollout policy** (on-policy: generate with the student; off-policy: use teacher traces). The 2026 study by Piskorz et al. isolates these knobs on Llama3 and Qwen2.5 and finds that *KL direction*, not *rollout policy*, is what actually drives task performance and output coverage.

**Prereqs:** [../post-training/_post-training.md](./_post-training.md), [../post-training/_rl.md](./_rl.md)
**Related:** [dpo.md](./dpo.md) · [rlvr.md](./rlvr.md) · [rejection-sampling.md](./rejection-sampling.md) · [fine-tuning/README.md](./fine-tuning/README.md)

---

## What it is

Classical knowledge distillation (Hinton 2015) trains a student to match a teacher's softmax at every position with a KL loss. For LLMs, two implementation choices open up:

- **KL direction.** *Forward* KL ≈ `KL(teacher ‖ student)` — mode-covering, mass on all teacher-likely tokens. *Reverse* KL ≈ `KL(student ‖ teacher)` — mode-seeking, mass on tokens the student actually samples.
- **Rollout policy.** *Off-policy*: compute KL on teacher-generated sequences (standard SFT-style distillation). *On-policy*: generate with the student, score with the teacher — closer to an RL step.

These two choices define a 2×2 recipe grid that most "distillation" and "on-policy distillation" papers occupy.

## How it works

For each prompt $q$ and token position $t$:

$$
L_{\text{fwd}} = \sum_v p_T(v\mid q, o_{<t}) \log \frac{p_T(v\mid \cdot)}{p_S(v\mid \cdot)} \qquad L_{\text{rev}} = \sum_v p_S(v\mid \cdot) \log \frac{p_S(v\mid \cdot)}{p_T(v\mid \cdot)}
$$

- Off-policy forward KL ≈ standard SFT on teacher logits.
- On-policy forward KL ≈ sampled from the student, graded by the teacher.
- Reverse KL is the DPO/RL-style objective and is sharper against teacher failure modes.

Piskorz et al. (2026) sweep the mixture of student-vs-teacher rollouts continuously and look at loss gradients to explain the sensitivities below.

## Why it matters

The field had internalized "on-policy > off-policy" as a universal rule for post-training. In the controlled distillation setting:

- **Forward KL is nearly invariant to rollout policy** — off-policy distillation gives the same accuracy as on-policy, at a fraction of the compute.
- **Reverse KL is sharply sensitive** — reverse KL needs on-policy or near-on-policy rollouts; off-policy reverse-KL hurts.
- **On-policy improves hard-variant generalization** (Countdown), but *the advantage washes out after subsequent RLVR*.
- **Learning rate** controls catastrophic forgetting and update sparsity much more than rollout policy does.

Operationally: for forward-KL distillation into a smaller model, pre-generate teacher completions once and reuse them — the on-policy premium is zero. For reverse-KL / policy-matching, budget for the student rollouts.

## Gotchas & tricks

- **"On-policy" overloaded.** Prior work mixes rollout policy with loss form; isolate them before comparing.
- **KL estimator.** The sampled KL estimator (log-ratio of a single sampled token) biases differently from the full-vocab sum; the paper confirms the direction effect holds under both.
- **Gradient clipping** masks some of these sensitivities. Remove it when diagnosing which knob is driving your runs.
- **RLVR dominates.** If a distilled student will undergo RLVR, don't over-invest in on-policy distillation first — RLVR erases the generalization gain.
- **Teacher-student capacity gap.** Reverse KL at a large gap collapses the student to high-confidence narrow modes; prefer forward KL when the student is much smaller.

## Sources

- Paper: *On-Policy or Off-Policy Learning? A Systematic Study of Distillation Dynamics* — Piskorz, Berthon, van der Schaar, 2026 — [arXiv:2609.35259](https://arxiv.org/abs/2609.35259).
- Paper: *Distilling the Knowledge in a Neural Network* — Hinton, Vinyals, Dean, 2015 — the original KL-matching recipe.
- Paper: *On-Policy Distillation of Language Models (f-Distill / GKD)* — Agarwal et al., 2023 — introduces the on-policy student-generated rollout framing.
