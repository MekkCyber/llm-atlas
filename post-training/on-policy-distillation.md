# On-Policy Distillation
*Depth — distill a teacher into a student using trajectories sampled from the student itself.*

**TL;DR:** Classical knowledge distillation feeds the student teacher-generated outputs and minimizes a token-level KL. *On-policy distillation* (OPD) flips the data source: the student rolls out, and the teacher provides per-token probabilities along the student's own trajectory. The student is corrected exactly where it tends to go wrong. Sample-efficient compared to RL (no reward model) and more on-distribution than off-policy distillation; the dominant recipe for shrinking reasoning models in 2025–2026.

**Prereqs:** [_rl](_rl.md), [_post-training](_post-training.md)
**Related:** [grpo](grpo.md), [rejection-sampling](rejection-sampling.md), [rlvr](rlvr.md), [reasoning/long-cot-rl](reasoning/long-cot-rl.md)

---

## What it is

Given a strong teacher $\pi_T$ and a smaller student $\pi_\theta$, on-policy distillation trains $\theta$ by:

1. Sampling a trajectory $o = (o_1, \ldots, o_L) \sim \pi_\theta(\cdot \mid q)$ — the *student's own* generation.
2. Computing the teacher's per-token distribution $\pi_T(\cdot \mid q, o_{<t})$ along that trajectory.
3. Minimizing a per-token loss between $\pi_\theta$ and $\pi_T$ (forward KL, reverse KL, or a mixed objective) on the realized tokens $o_t$.

In short: **the student picks the path; the teacher labels it.** The teacher is run in inference-only mode (frozen, no gradients).

## How it works

The basic loss is the per-token forward KL:

$$
L_{\text{OPD}} = \mathbb{E}_{o \sim \pi_\theta} \!\left[\, \sum_t \mathrm{KL}\big(\, \pi_T(\cdot \mid q, o_{<t}) \;\|\; \pi_\theta(\cdot \mid q, o_{<t}) \,\big) \,\right]
$$

Practically simplified to:

$$
L \approx -\,\mathbb{E}_{o \sim \pi_\theta}\!\left[\,\sum_t \log \pi_\theta(o_t \mid q, o_{<t}) \cdot \pi_T(o_t \mid q, o_{<t}) \,\right]
$$

Three implementation knobs:

- **Trajectory source.** Always the student. Some variants mix a small fraction of teacher rollouts (off-policy correction).
- **Stop-gradient through the teacher.** Always — the teacher is frozen. This is what *V-Zero* (2026) calls "negative-free stop-gradient alignment."
- **Per-trajectory weighting.** Vanilla OPD weights every trajectory equally. *ReNIO* (2026) shows that **incorrect** trajectories carry more signal than correct ones and proposes a per-sample reweighting based on student-vs-teacher divergence at "pivotal" tokens.

### Two recent variants

- **V-Zero** (Sun et al., 2026): gates OPD on *visual-contrast* signal — a question-relevant image crop vs a negative crop are used to evaluate the trajectory and decide whether to fire the distillation loss. Label-free and >5× faster than supervised baselines on fine-grained visual reasoning.
- **ReNIO** (Lin et al., 2026): identifies "pivotal" tokens in incorrect trajectories (largest student-to-teacher log-prob ratio gap) and uses their aggregated importance as a per-sample weight. +8.9–10.0 % on math reasoning over uniform OPD.

## Why it matters

- **No reward model.** OPD trades RL's reward-modeling step for a teacher forward pass. Cheaper to set up and harder to hack (no learned RM to game).
- **On-distribution corrections.** Off-policy distillation moves the student toward the teacher *on the teacher's data*. OPD moves it toward the teacher *on the student's data* — exactly the trajectories the student would actually emit at inference.
- **Beats RL on small models.** DeepSeek-R1's own ablations show distillation SFT from a stronger reasoner beats running RL directly on small students. OPD is the on-policy refinement of that finding.
- **Composes with rejection.** Drop incorrect-but-uninformative trajectories; weight or focus on pivotal-token ones (ReNIO); gate by an external signal (V-Zero).

## Gotchas & tricks

- **Teacher serving is the bottleneck.** You're running a forward pass through the *larger* model per training step. Co-locate teacher and student or batch heavily.
- **Trajectory length.** OPD's variance scales with trajectory length; for long-CoT reasoning, cap at a sensible max-tokens to avoid the loss being dominated by tail tokens.
- **Don't mix in off-policy data without reweighting.** Mixing teacher-rollouts into the batch without importance correction biases the gradient.
- **Reverse vs forward KL.** Forward KL is mode-covering (student tries to match all teacher modes); reverse KL is mode-seeking (student picks the teacher's most-likely mode). For reasoning tasks where there's usually one correct answer, reverse KL is often more stable.
- **Pairs naturally with verifiers.** If you have a verifier, dropping incorrect trajectories before OPD is essentially free quality.

## Sources

- Paper: *V-Zero: Answer-Label-Free On-Policy Distillation with Contrastive Evidence Gating* — Sun et al., 2026 — [arXiv 2606.25319](https://arxiv.org/abs/2606.25319).
- Paper: *ReNIO: Reweighting Negative Trajectory Importance for LLM On-Policy Distillation* — Lin, Chen, Zhang, 2026 — [arXiv 2606.23104](https://arxiv.org/abs/2606.23104).
- Background: *DeepSeek-R1 — Distillation from Reasoners* — DeepSeek, 2025 — establishes that distillation beats RL on small models.
- Earlier: *On-Policy Distillation of Language Models* — Agarwal et al. (Generalized Knowledge Distillation, GKD), 2024 — the foundational OPD formulation outside reasoning.
