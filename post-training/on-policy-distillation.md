# On-Policy Distillation (OPD)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A post-training regime that trains a **student model on its own rollouts**, supervised by a **teacher's distribution over the student's trajectory**. Unlike SFT (which fits to teacher trajectories) and RLVR (which uses scalar reward on student trajectories), OPD uses **dense per-token teacher logits on student-sampled tokens**. The 2026 geometric analysis (Shen et al.) shows OPD is not an intermediate between SFT and RLVR but its own update mode: a *relaxed off-principal* regime, with cumulative updates **locked into a narrow low-dimensional subspace** within a few hundred steps.

**Prereqs:** [_post-training.md](_post-training.md), [grpo.md](grpo.md)
**Related:** [rlvr.md](rlvr.md) · [cot-reward-model.md](cot-reward-model.md) · [reasoning/long-cot-rl.md](reasoning/long-cot-rl.md)

---

## What it is

Three post-training data sources differ in *whose tokens carry the loss* and *who supervises*:

| Regime | Tokens come from | Supervision signal |
| --- | --- | --- |
| SFT | Teacher / human | Token-level cross-entropy to teacher tokens |
| RLVR | Student rollout | Scalar reward (verifier) on whole response |
| **OPD** | Student rollout | Teacher's per-token distribution over student tokens |

OPD takes the best of both: the **distribution shift problem** that SFT has (student is fit to data it never generates) is gone — supervision comes on student-distribution tokens. The **sparse reward** problem that RLVR has is gone — every token gets a dense KL/cross-entropy signal.

## How it works

Per step:
1. Sample a rollout $o = (o_1, \ldots, o_T)$ from the student policy $\pi_\theta$ given prompt $q$.
2. For each token position $t$, query the teacher $\pi_T$ for its distribution over the vocabulary at that position: $\pi_T(\cdot \mid q, o_{<t})$.
3. Loss: per-token KL or forward-cross-entropy from $\pi_T$ to $\pi_\theta$:
$$
\mathcal{L}_{\text{OPD}} = \frac{1}{T} \sum_t \mathrm{KL}\big(\pi_T(\cdot \mid q, o_{<t})\,\|\,\pi_\theta(\cdot \mid q, o_{<t})\big)
$$
4. Standard backprop on the student. No reward model, no advantage estimation, no value head.

The teacher is queried *on the student's actual rollout*, which is the "on-policy" part. The student's update direction is **the teacher's gradient of choice at every position the student would have generated**.

## Geometry of OPD updates

The Shen et al. 2026 analysis runs parameter-space diagnostics across the three regimes and finds:

- **SFT:** updates affect many weights, align strongly with the top principal directions of pretraining gradients.
- **RLVR:** updates are tightly constrained — sparse, off-principal.
- **OPD:** *fewer weights touched than SFT* (so it's like RLVR there) but *less tightly constrained than RLVR* (so it's not just RLVR with extra signal). Distinct **relaxed off-principal** regime.

Cumulative OPD updates exhibit **subspace locking**: within a few hundred steps, the running update lives in a narrow low-dim channel. Constraining subsequent training to that early-locked subspace **preserves OPD performance** but tanks SFT — evidence the locked subspace is functionally sufficient *for OPD*.

## Why it matters

- **Better than SFT for reasoning.** Dense supervision on student-distribution tokens fixes SFT's distribution-shift problem without the variance of scalar RL.
- **Cheaper than RLVR.** No rollout grading, no verifier infra, no value/advantage machinery. Just teacher forward passes.
- **Subspace structure suggests engineering wins.** If updates live in a low-dim subspace, OPD-friendly LoRA, principled early stopping via subspace saturation, and checkpoint merging via subspace alignment are all natural.
- **Reframes the post-training ladder.** SFT → OPD → RLVR isn't a smooth gradient — they're three distinct geometric regimes, each with its own merits.

## Gotchas & tricks

- **Teacher quality is the ceiling.** OPD is bounded by the teacher's distribution; if the teacher is wrong at a token, the student learns to be wrong. RLVR's verifier reward can in principle exceed the teacher.
- **KL direction matters.** Forward KL (teacher → student) is mode-covering — fine for capability transfer. Reverse KL is mode-seeking — more aggressive but riskier.
- **Top-K teacher logits suffice.** Storing the full vocab distribution per token is expensive; in practice the top-K logits + a residual mass capture nearly all the signal.
- **Mixing with RLVR breaks the geometry.** Shen et al.'s ablations show that adding an RLVR term changes the rank dynamics — OPD's subspace locking disappears. Use OPD alone, or as a stage before RLVR, not blended.
- **Rollout off-policy drift.** If you use stale rollouts (sampled from an older policy snapshot), you lose some of the "on-policy" benefit. Refresh rollouts at least per epoch.

## Sources

- Paper: *On the Geometry of On-Policy Distillation* — Shen, Li, Yin, Leong, Wang et al., HKUST + 7 partners, 2026 — arXiv 2606.07082 — parameter-space diagnostics, subspace locking.
- Paper: *Distilling step-by-step* — Hsieh et al., 2023 — early OPD-style recipe for reasoning.
