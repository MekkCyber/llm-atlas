# β-OPSD (Beta On-Policy Self-Distillation)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** On-policy self-distillation (OPSD) is exactly the $\beta=1$ member of a broader KL-regularized policy-optimization family, where $\beta$ weights the KL penalty anchoring the student to a reference policy. Making $\beta$ a knob interpolates the optimal policy between "stay near the reference" and "trust the privileged teacher" — and the exact optimum has a closed form. β-OPSD implements that optimum as a **cheap logit-mixture distillation target**, turning an expensive on-policy RL objective into a stable supervised loss.

**Prereqs:** [_post-training](_post-training.md), [dpo](dpo.md)
**Related:** [ppo](ppo.md), [grpo](grpo.md), [reasoning/long2short](reasoning/long2short.md), [fine-tuning/on-policy-distillation](fine-tuning/on-policy-distillation.md)

---

## What it is

Vanilla OPSD trains the student to match a privileged (stronger) teacher on the *student's own* rollouts. It works, but is famously brittle: sensitive to KL coefficients, teacher–student capacity gaps, and initialization. The insight of β-OPSD: OPSD is not a standalone algorithm, it's the $\beta = 1$ case of

$$
\max_{\pi_\theta}\ \mathbb{E}_{y \sim \pi_\theta}\bigl[\log \pi_\text{teach}(y) - \beta \cdot \mathrm{KL}(\pi_\theta \,\|\, \pi_\text{ref})\bigr]
$$

Solving this KL-regularized objective yields a **closed-form optimal policy** that is a geometric interpolation between reference and teacher — think DPO-style derivation, but for the reference-vs-teacher axis instead of preferred-vs-rejected.

## How it works

- **Derivation.** For each $\beta$, the optimal policy has the form (schematically) $\pi^*_\beta \propto \pi_\text{ref}^{1/\beta} \cdot \pi_\text{teach}^{1 - 1/\beta}$. This is the path traced from the reference ($\beta \to \infty$) to the teacher ($\beta \to 0$).
- **Implementation.** Directly optimizing the RL objective is costly and high-variance. Instead, use the closed form as a **distillation target**: mix the reference logits and the teacher logits at the appropriate ratio to construct $\log \pi^*_\beta(\cdot)$ token-wise, then train the student to match via standard distillation loss.
- **Credit assignment.** Add return-to-go weighting so per-token updates track the *sequence-level* objective (a familiar idea from policy gradient), while keeping the outer form of a cheap distillation.
- No PPO clipping, no importance sampling, no value network. Same throughput profile as offline distillation with a stronger fixed point.

## Why it matters

- **Explains OPSD's brittleness.** $\beta = 1$ is one arbitrary point on the reference-to-teacher path. A different $\beta$ can be *strictly better* for a given capacity gap, and now you have the knob.
- **Bridges distillation and RL.** The same objective is realized as (a) an RL update with a KL penalty or (b) a logit-mixture distillation target. That equivalence is the same shape as [DPO](dpo.md) discovered for preference optimization — expensive RL replaced by supervised training against a closed-form target.
- **Consistently beats vanilla OPSD** on math-reasoning benchmarks in the paper, in both optimization stability and downstream performance.

## Gotchas & tricks

- **Pick $\beta$ per capacity gap.** Large teacher–student gaps → smaller $\beta$ (trust the teacher more). Small gaps → larger $\beta$ (stay near the reference).
- **Logit mixing must be token-level.** Interpolating in *probability space* and in *log-probability space* give different targets; the paper's derivation uses log-space mixing.
- **Return-to-go weighting is important** at long horizons; without it, per-token updates optimize a proxy that decouples from sequence-level rewards.

## Sources

- Paper: *β-OPSD: Deriving with Policy Optimization, Training with Self-Distillation* — Liu, Zhang, Goldstein, Huang, 2026 — [arXiv:2607.28582](https://arxiv.org/abs/2607.28582)
- Adjacent: *Flux-OPD: On-Policy Distillation with Evolving Contexts* — Wang et al., 2026 — [arXiv:2607.28022](https://arxiv.org/abs/2607.28022) — empirical companion for open-ended domains.
