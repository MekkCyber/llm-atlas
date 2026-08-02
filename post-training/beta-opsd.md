# β-OPSD
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** On-Policy Self-Distillation (OPSD) is brittle in practice because it's the **β=1** corner of a broader KL-regularized policy-optimization family. Making β adjustable turns OPSD's implicit "no reference anchor" into a tunable KL penalty, and yields a closed-form optimal policy that's a **geometric interpolation** of the reference and privileged teacher. Implemented by mixing token-level logits — cheap approximation of expensive KL-regularized RL for reasoning models.

**Prereqs:** [on-policy-distillation](on-policy-distillation.md), [ppo](ppo.md), [grpo](grpo.md)
**Related:** [rlvr](rlvr.md), [long-cot-rl](reasoning/long-cot-rl.md)

---

## What it is

OPSD (on-policy self-distillation) trains a student on its own rollouts by matching logits from a *privileged teacher* — often the same model given extra information (e.g. reference solutions, tools, or a stronger prompt). It's used heavily for reasoning models because it captures teacher-style CoT structure without paying for full RL.

β-OPSD (Liu, Zhang, Goldstein, Huang, UMD, 2026) shows that OPSD implicitly sits inside the KL-regularized policy-optimization family:

$$
\pi^\star \propto \pi_{\mathrm{ref}}^{\,\beta} \cdot \pi_{\mathrm{teacher}}^{\,1-\beta}
$$

The vanilla OPSD recipe is this optimum at $\beta = 0$ (no anchor to the reference policy). β = 1 recovers "stay at reference." Intermediate β trades exploration against teacher fidelity.

## How it works

Solve the KL-regularized RL objective analytically → the optimum is the geometric interpolation above → **implement it by mixing token-level logits** of the reference and teacher policies:

$$
\log \pi^\star(x_t | \cdot) = (1-\beta) \log \pi_{\mathrm{teacher}}(x_t | \cdot) + \beta \log \pi_{\mathrm{ref}}(x_t | \cdot) + \text{const}
$$

Distill the student against this mixed target. Adds essentially zero compute over vanilla OPSD — just an extra logit-space addition. **Return-to-go credit assignment** aligns per-token updates with sequence-level rewards, keeping the OPSD-style efficiency while capturing PPO-like credit shaping.

## Why it matters

- **Explains fragility.** Vanilla OPSD's engineering pain isn't inherent to self-distillation; it's the unregularized corner of a well-behaved family. β > 0 restores stability with one knob.
- **Cheap approximation of KL-regularized RL.** Full PPO/GRPO with a KL penalty requires the rollout loop, advantage estimator, and clipping. β-OPSD gets the same closed-form target for free.
- **Practical default for reasoning distillation.** Improves optimization stability *and* final accuracy on math-reasoning benchmarks over vanilla OPSD across seeds.

## Gotchas & tricks

- **Tokenizer/vocab must match.** Logit mixing requires reference and teacher to share vocabulary — otherwise per-token addition is ill-defined.
- **β sweep is fast.** Since implementation is a logit mix, β can be swept cheaply; a small validation set is enough to pick a value.
- **Reference policy choice matters.** The reference is typically the SFT-initialized model. Using a stronger reference collapses β-OPSD toward pure teacher-following; using a weaker one increases variance.
- **Not a replacement for RL when the reward is verifiable.** For RLVR-friendly tasks, GRPO/PPO still exploit the verifier signal directly; β-OPSD shines when the teacher's soft logits carry information the verifier can't.

## Sources

- Paper: *Deriving with Policy Optimization, Training with Self-Distillation* — Liu, Zhang, Goldstein, Huang, 2026 — [arXiv:2607.28582](https://arxiv.org/abs/2607.28582).
- Related: [on-policy-distillation](on-policy-distillation.md), [ppo](ppo.md), [grpo](grpo.md).
