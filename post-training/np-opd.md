# NP-OPD: On-Policy Distillation with Negative-Policy Rollouts

*Depth — adds anti-signal from a weaker policy to standard on-policy distillation.*

**TL;DR:** Vanilla on-policy distillation (OPD) only has a "move toward the teacher" signal; it has no explicit "move away from bad" signal and tends to collapse into long, repetitive outputs. NP-OPD samples a second stream of rollouts from a **negative policy** — a lower-capability model (older checkpoint, smaller sibling, degraded prompt) — and tells the student to suppress tokens the negative policy preferred over the teacher. The student still receives teacher supervision for positive signal; the negative stream supplies anti-signal.

**Prereqs:** [on-policy-distillation](on-policy-distillation.md), [_rl](_rl.md)
**Related:** [_distillation](_distillation.md) · [grpo](grpo.md) · [dpo](dpo.md) · [self-retrospection-distillation](self-retrospection-distillation.md)

---

## What it is

Three streams instead of two:

- **Student** $\pi_\theta$ — generates on-policy rollouts (positive).
- **Teacher** $\pi_T$ — frozen, strong; supervises both streams.
- **Negative policy** $\pi_N$ — cheap, weak; generates its own rollouts.

The composite loss combines OPD on the student's rollouts with a **contrastive term** on the negative-policy rollouts:

$$
\mathcal{L}_{\text{NP-OPD}} = \mathcal{L}_{\text{OPD}}(\pi_\theta, \pi_T) + \lambda \cdot \mathcal{L}_{\text{neg}}(\pi_\theta, \pi_T, \pi_N)
$$

The negative term is designed so that **tokens where $\pi_N$ scores higher than $\pi_T$** are explicitly suppressed in $\pi_\theta$. The weight $\lambda$ trades off the two.

## How it works

The negative-policy rollouts are the trick's cheap input: they share the input distribution with the student (same prompts) but exhibit the failure modes you want to avoid (short, degenerate, off-task). Averaging the teacher's logits over those rollouts gives a token-level "what good looks like under the bad distribution" signal; the contrastive loss pushes $\pi_\theta$ to assign lower probability to the tokens where $\pi_N > \pi_T$.

Framed under the [_distillation](_distillation.md) taxonomy: the negative policy supplies an **anti-reward**. This upgrades OPD from a one-sided IL algorithm into a two-sided contrastive one, in the spirit of DPO's chosen/rejected pairs but at the policy level, not the preference level.

## Why it matters

- Fixes OPD's **collapse mode** (long, repetitive outputs) without any auxiliary verifier or human preference data.
- The negative policy is nearly free: an older student checkpoint, a smaller sibling model, or a bad-prompt variant works.
- Consistent gains across model scales, reasoning domains, and OPD variants in NAVER's results.
- Composes with [self-retrospection-distillation](self-retrospection-distillation.md) and other dense-signal methods.

## Gotchas & tricks

- **Negative policy must share input distribution.** An unrelated model on different data breaks the contrast; the simplest reliable choice is an older checkpoint of the student itself.
- **Negative must be weaker, not pathological.** Too weak (random tokens) → no useful anti-signal. Too strong (close to the student) → nothing to push away from.
- **$\lambda$ sensitivity.** Too high → the student overcorrects and loses teacher mass. Analyses in the paper suggest keeping the positive-OPD term dominant.
- **Monitor the per-token suppression.** The paper shows the method mainly suppresses tokens the negative policy prefers over the teacher — a cheap diagnostic.

## Sources

- *On-Policy Distillation with Negative-Policy Rollouts* — Han, Yun, Heo et al. — NAVER AI Lab — 2026 — [arXiv:2610.07874](https://arxiv.org/abs/2610.07874)
