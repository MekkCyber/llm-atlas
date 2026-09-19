# On-Policy Distillation (OPD)
*Depth — token-level distillation where the student's own rollouts drive the loss.*

**TL;DR:** On-policy distillation trains a student to match a teacher's *token-level distribution* on the student's own rollouts, giving dense supervision that outcome-only RL doesn't. It's the modern post-training recipe for compressing reasoning models (DeepSeek-R1, Kimi-K1.5) and the substrate for several 2026 variants — self-OPD, self-retiring (RetireOPD), privileged-reference OPD. The recipe is quietly failure-prone: EOS-token mismatch inflates response length, and a fixed distillation schedule can hurt late-stage RL.

**Prereqs:** [_rl](_rl.md), [rlvr](rlvr.md), [grpo](grpo.md)
**Related:** [rejection-sampling](rejection-sampling.md), [reasoning/long-cot-rl](reasoning/long-cot-rl.md), [eos-alignment](eos-alignment.md), [adaptive-retirement](adaptive-retirement.md)

---

## What it is

Given a teacher policy $\pi_T$ (a stronger model, or the student itself with privileged info), OPD minimizes a token-level divergence between teacher and student on *rollouts sampled from the student*:

$$
\mathcal{L}_{\text{OPD}} = \mathbb{E}_{y \sim \pi_S(\cdot|x)}\left[\, \sum_t \mathrm{KL}\big(\pi_T(\cdot | x, y_{<t}) \,\|\, \pi_S(\cdot | x, y_{<t})\big)\,\right]
$$

Contrast:

- **Off-policy distillation:** rollouts from $\pi_T$; student never sees its own mistakes.
- **RL (outcome-only):** one scalar reward per trajectory; sparse.
- **OPD:** rollouts from $\pi_S$ (like RL), dense token-level supervision (like distillation).

**Self-OPD (OPSD)** is the case $\pi_T = \pi_S$-with-extra-context (e.g. the same model given the answer or a worked solution). **RetireOPD** turns OPD off adaptively once the student catches up.

---

## How it works

Each step:

1. Sample $y \sim \pi_S(\cdot | x)$ — the student generates.
2. Score both $\pi_T(y_t | x, y_{<t})$ and $\pi_S(y_t | x, y_{<t})$ token-by-token.
3. Compute the KL (forward or reverse, sometimes reverse-KL for mode-seeking).
4. Backprop into the student. Teacher is frozen (or a stop-gradient copy of the student in self-OPD).

Often combined with an RL objective at the same step (joint RL + OPD), then anneal or retire OPD as the student closes the gap.

---

## Why it matters

- **Dense signal.** Every token gets a gradient; hard tasks that RL flounders on (long-horizon, rare-success) train faster.
- **Compression recipe.** OPD is how R1-style models produce their distilled ("R1-Zero → R1", "K1.5 → K1.5-Slim") variants.
- **General substrate.** Adaptive retirement, privileged references, self-OPD — the 2026 wave of post-training variants all layer on this same loss.

---

## Gotchas & tricks

- **EOS-token mismatch inflates length.** Base students and post-trained teachers can place stopping probability on *different* EOS tokens even with identical declared stop sets. The student's preferred termination gets suppressed without the teacher's transferring in, and responses grow unboundedly. Treat functionally equivalent EOS tokens as one semantic stop action ([eos-alignment](eos-alignment.md)).
- **Late-stage OPD hurts agentic RL.** After the student closes the gap, continued distillation drags it toward teacher artifacts. Adaptive retirement drops OPD when discrepancy stops shrinking ([adaptive-retirement](adaptive-retirement.md)).
- **Privileged references add less than they look.** Reference-free OPSD delivers most of the gain on math (Qwen3-1.7B); adding worked solutions is a modest bump, not a step change (Zhang et al. 2026).
- **Reverse KL for mode-seeking.** Standard forward KL makes the student cover teacher's distribution; reverse KL concentrates on the teacher's mode, often more sample-efficient for reasoning tasks.
- **Joint RL + OPD needs coefficient tuning.** Too much OPD suppresses exploration; too little wastes the dense signal. Simple linear anneal is a common baseline.

---

## Sources

- Paper: *When EOS Tokens Disagree: Understanding Length Inflation in On-Policy Distillation* — Yang et al., 2026 — [arXiv:2609.20511](https://arxiv.org/abs/2609.20511).
- Paper: *RetireOPD: Self-Retiring On-Policy Distillation for Agentic Reinforcement Learning* — Yu et al., 2026 — [arXiv:2609.20784](https://arxiv.org/abs/2609.20784).
- Paper: *What Does Privileged Information Add to On-Policy Self-Distillation?* — Zhang et al., 2026 — [arXiv:2609.20612](https://arxiv.org/abs/2609.20612).
- Precedent: *DeepSeek-R1* — DeepSeek-AI, 2025 — distillation stage.
