# On-Policy Distillation (OPD)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Reverse-KL distillation of a teacher into a student, where distillation targets are generated *on-policy* by the student and re-scored by the teacher. The alternative to RL in **open-ended domains** where verifiable rewards don't exist. **Flux-OPD** (Peking U. / Kling, 2026) improves it by using **contexts that evolve with the student** as an additional supervision signal, correcting the anchor teacher toward context-conditioned preferences and downweighting conflicting contexts.

**Prereqs:** [_rl](_rl.md), [grpo](grpo.md)
**Related:** [rlvr](rlvr.md), [rejection-sampling](rejection-sampling.md)

---

## What it is

On-policy distillation replaces the standard "teacher generates, student imitates" pipeline: the *student* generates on its own current distribution, the teacher scores its outputs, and the student minimizes reverse KL to the teacher. Because rollouts come from the student, gradients are aligned with the distribution being deployed — no distribution mismatch between train and inference.

OPD is used where RL fails: **open-ended tasks** (creative writing, dialog, code style) where no verifier can score a rollout. The teacher becomes the reward.

## How it works

**Vanilla OPD.** Student samples $y \sim \pi_\theta(y \mid x)$; teacher provides token-level distributions $\pi_T(y_t \mid x, y_{<t})$; loss is reverse KL:
$$
L_{\text{OPD}} = \mathbb{E}_{x, y \sim \pi_\theta}\!\left[ \sum_t \mathrm{KL}(\pi_\theta \,\|\, \pi_T) \right]
$$

**Flux-OPD's contribution.** A theoretical decomposition of the reverse-KL objective under multiple context-conditioned teachers reveals two facts:

1. The student is distilled toward the **geometric mean** of context-conditioned teacher distributions.
2. The objective contains an explicit **conflict term** measuring teacher-context disagreement.

Flux-OPD operationalizes both. It treats $(\pi_T \mid \text{context}_i) - (\pi_T \mid \emptyset)$ as a *contextual correction* vector, injects it into a context-free teacher anchor, and weights the correction strength using the conflict term as an indicator. Contexts evolve with student performance — once the student masters a level, the context escalates to keep providing signal.

## Why it matters

- **Fills the "no verifier" gap.** Verifiable-reward RL (GRPO, RLVR) has dominated 2024–2025 post-training. For everything without a verifier, OPD is now the tool of choice.
- **Aligns train/test distribution** — same reason on-policy RL beats off-policy imitation. Student rollouts, not teacher rollouts, drive the gradient.
- **Evolving contexts** solve the OPD saturation problem: fixed context distills once, then the student stops benefiting.

## Gotchas & tricks

- Reverse KL is mode-seeking — student may collapse to a subset of teacher modes. Balance with occasional forward-KL updates or with entropy regularization.
- On-policy rollouts are expensive at scale — student must run inference every batch. Use partial rollouts and mini-batched updates.
- Context evolution needs a **student-progress signal** (a proxy score, teacher-agreement rate, or held-out task performance) — a fixed context schedule wastes the mechanism.
- The conflict term is essential when contexts conflict — without downweighting, you get worse-than-vanilla-OPD.

## Sources

- Paper: *Flux-OPD: On-Policy Distillation with Evolving Contexts* — Wang et al., Peking U. / Kling Team, 2026 — [arXiv:2607.28022](https://arxiv.org/abs/2607.28022).
- Precursor: on-policy distillation as an alternative to RL — sometimes called "distillation with on-policy sampling" or "student-generated distillation."
