# Multi-Expert On-Policy Distillation
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A post-training recipe that trains several **specialized experts** (each fine-tuned or RL-optimized for a narrow capability — aesthetics, text rendering, editing, etc.) then consolidates them into a **single generalist** via on-policy distillation from each expert. The generalist inherits each expert's strength without adding inference cost, and — because distillation is on the generalist's own rollouts — it also preserves base-model instruction following. Used by **SenseNova-U1.5** (2026) to merge four visual-generation experts into one 8B unified model.

**Prereqs:** [_post-training.md](./_post-training.md)
**Related:** [../architectures/_moe.md](../architectures/_moe.md), [dpo.md](./dpo.md)

---

## What it is

Two common approaches to "multiple capabilities in one model":

- **Mixture-of-Experts** — different experts activate per input; parameter count grows, and routing adds inference overhead.
- **Sequential fine-tuning** — fine-tune the base on task A, then B, then C; each stage risks catastrophic forgetting of the previous.

**Multi-expert on-policy distillation** is a third route: train each specialist *separately* off the same base, then have a shared student model *imitate them on its own rollouts*, one specialist per rollout batch. The student ends up as a single dense model with no routing overhead, holding the joint capability envelope.

## How it works

Setup:
- Base model $\theta_0$.
- $N$ experts $\theta_1, \ldots, \theta_N$, each fine-tuned/RL-optimized from $\theta_0$ for a distinct capability (SenseNova: visual aesthetics, bilingual text rendering, infographic generation, image editing).
- Student $\theta_s$, initialized from $\theta_0$.

Per training step:

1. Sample a prompt appropriate to expert $i$.
2. Roll out from the **student** $\theta_s$ (this is what "on-policy" means — the trajectories are the student's, not the expert's).
3. Score each token position with the expert $\theta_i$'s token distribution.
4. Update the student to minimize $\mathrm{KL}(\pi_{\theta_s} \| \pi_{\theta_i})$ token-wise along the rollout.
5. Cycle experts across batches so the student sees every specialty.

Regularizer: a small KL to $\theta_0$ keeps the student from drifting off the base's general capabilities.

## Why it matters

- **Generalist without routing.** Downstream deployments get one dense model with the joint capability envelope, no MoE routing overhead, no per-request expert selection.
- **On-policy vs off-policy is the trick.** Distilling on the *student*'s rollouts (not the expert's) means the student's own distribution is corrected step-by-step where it diverges from the expert; off-policy distillation on the expert's rollouts creates covariate shift and slower convergence.
- **Composable with more experts later.** Adding a new capability is: fine-tune another expert on the base, add it to the distillation cycle. No re-tuning of previous experts.
- **Modular capability engineering.** Teams can iterate on experts independently and merge them centrally, which fits large-org production pipelines.

## Gotchas & tricks

- **Cycle order and share matter.** If one expert dominates the training mix, its capability crowds out the others. Balanced expert-sampling schedules are essential.
- **The base-KL regularizer is non-negotiable.** Without it, the student's general capabilities degrade as it specializes in the experts' narrow objectives.
- **Not the same as MoT / MoE.** Mixture-of-Transformers routes tokens between experts at inference; multi-expert distillation merges them into one dense model at training time. Both are valid; the deployment vs training tradeoff differs.
- **Expert-vs-student capability gap.** If the expert is much stronger than the student on a task, KL distillation can overshoot — the student's outputs become confidently wrong in a way that regularizes badly. Temperature-scaled distillation helps.
- **Not distillation of preferences.** This is teacher-forced KL distillation over the specialist's output distribution, not DPO-style preference-pair training.
- **Verify per-expert capability retention.** The generalist should be evaluated on each expert's original benchmark; a small drop is expected, a large drop means the cycle is out of balance.

## Sources

- Paper: *SenseNova-U1.5: Towards Native Unified Visual Intelligence* — Diao, Wang, Ding, ..., Lin (SenseTime, with S-Lab NTU + MMLab CUHK), 2026 — arXiv:2609.11929. Section on post-training expert consolidation.
