# Negative Self-Distillation (NSD)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** On-Policy Self-Distillation (OPSD) — teaching a model to imitate its own privileged-info-conditioned reasoning traces — has been the default late-stage self-improvement loop, but it degrades reasoning: forced imitation of confident traces suppresses exploration and self-correction. NSD flips the sign: the model is trained to **diverge** from its own flawed rollouts. Wrong rollouts become negative teachers, and the policy is pushed away from their token distributions, preserving calibrated uncertainty and self-corrective behavior.

**Prereqs:** [dpo](dpo.md), [rejection-sampling](rejection-sampling.md)
**Related:** [grpo](grpo.md), [reasoning/long-cot-rl](reasoning/long-cot-rl.md)

---

## What it is

Standard self-distillation minimizes $\mathrm{KL}\!\left(\pi_\theta \,\|\, \pi_{\text{teacher}}\right)$ where the teacher is the same model conditioned on privileged information (the correct answer, a hint, or a rubric). OPSD's failure mode: those "teacher" traces read as confidently correct even when the reasoning was skipped, so imitation collapses uncertainty. NSD introduces an **anti-teacher**: the model's *own* wrong rollouts. The loss pushes the policy away from those distributions rather than toward them.

## How it works

For each problem $p$:

1. **Sample rollouts.** Draw $n$ trajectories $y_i \sim \pi_\theta(\cdot \mid p)$ without privileged info.
2. **Partition.** Mark $y_i$ as *correct* / *flawed* via an outcome check (verifier, gold answer, unit tests).
3. **Anti-target set.** Keep the flawed $y_i$ as anti-teachers $\pi^-_i$ (their empirical token distributions).
4. **NSD loss.** Minimize
   $$L_{\text{NSD}} = -\alpha\, \mathbb{E}_{y\sim\pi_\theta}[\log \pi_\theta(y\mid p)]_{\text{correct}} \;+\; \beta \sum_{i \in \text{flawed}} \mathrm{KL}\!\left(\pi^-_i \,\|\, \pi_\theta\right)$$
   The first term is a standard likelihood term on correct trajectories; the second *maximizes* KL from the wrong ones.
5. **Regularize.** A KL trust region to a reference policy prevents drift, same as in DPO/GRPO.

Equivalent view: NSD is a one-sided contrastive update where only *negative* pairs contribute, unlike DPO which requires paired positive+negative preferences.

## Why it matters

- **Reasoning-preserving self-improvement.** OPSD kills exploration by rewarding overconfidence; NSD preserves the model's tendency to hedge and self-correct, which is exactly what long-CoT reasoning needs.
- **Cheap negatives, no pair construction.** Every wrong on-policy rollout is a negative — no human preferences, no paired setup, no reward model.
- **Composes with GRPO.** The negative-KL term slots into a GRPO loop as an extra gradient direction on failed group members, complementing the outcome reward.

## Gotchas & tricks

- **Anti-teacher entropy.** Very confident wrong rollouts push hard on the KL — clip per-token to prevent one bad trace from dominating the update.
- **Balance $\alpha, \beta$.** Too much negative pressure = the policy walks away from *anything* resembling the wrong trace, including the correct steps inside it. In practice $\beta / \alpha \sim 0.1$–$0.3$.
- **Rollout diversity is the fuel.** Low-diversity samples give a narrow anti-target — sample with higher temperature during the rollout phase than during evaluation.
- **Not a replacement for verifiers.** You still need to know which rollouts are wrong. NSD is orthogonal to reward modeling / verification, not a substitute for it.
- **Watch for mode collapse away from the anti-target.** If the anti-targets share a mode with the true distribution (e.g. common English phrasing), the model can be pushed into unnatural phrasing; the reference-KL regularizer damps this.

## Sources

- Paper: *Negative Self-Distillation: Learning to Reason by Avoiding Flaws* — Pei, Wei, Xu, Zhu, Chen, Meng (UVA), 2026 — [arXiv:2609.11699](https://arxiv.org/abs/2609.11699)
- Related: DPO — pairs positive and negative preferences; NSD is the negative-only, on-policy sibling.
- Related: OPSD — the "imitate your privileged teacher" baseline NSD reverses.
