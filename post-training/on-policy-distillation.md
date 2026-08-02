# On-Policy Distillation (OPD)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Distill a student from a teacher (or from context-conditioned teachers) using **the student's own on-policy samples** as the input distribution, matching teacher logits on those samples. Sidesteps RL's high variance while providing preference-like supervision for open-ended tasks where no verifiable reward exists. Flux-OPD (2026) extends this by letting the teacher-conditioning **context** evolve with the student.

**Prereqs:** [_post-training](_post-training.md), [ppo](ppo.md)
**Related:** [dpo](dpo.md), [grpo](grpo.md), [rejection-sampling](rejection-sampling.md), [beta-opsd](beta-opsd.md)

---

## What it is

Post-training splits into three families by *what supervises the student*:

- **RLHF / PPO** — a learned reward model rates on-policy rollouts (high variance, needs a good RM).
- **RLVR** — a verifier judges rollouts (only works when rewards are verifiable).
- **On-policy distillation** — a teacher LM's logits on the student's own rollouts are the supervision (works for open-ended tasks, no reward needed).

OPD keeps RL's benefit — the student is trained on **its own** distribution, not the teacher's — while replacing the reward signal with dense, low-variance logit-matching. When multiple teachers are used, each conditioned on a different **context prompt**, the student is effectively distilled toward a preference over how it should behave.

## How it works

At each step:

1. Sample a rollout $y$ from the current student $\pi_\theta$ given prompt $x$ — this is the on-policy part.
2. Compute the teacher's token-level distribution $\pi_T(\cdot | x, c, y_{<t})$, optionally conditioned on a context $c$ (system prompt, style guide, few-shot exemplars).
3. Minimize the reverse KL $D_{\mathrm{KL}}\!\left(\pi_\theta(\cdot|x, y_{<t}) \,\|\, \pi_T(\cdot|x, c, y_{<t})\right)$ across positions.

Flux-OPD (2026) decomposes the reverse-KL objective under **multiple** context-conditioned teachers and shows two things:

- The student is distilled toward the **geometric mean** of the context-conditioned teachers.
- A **conflict term** measures disagreement between teachers; downweighting it stabilizes training when contexts evolve with the student.

Evolving contexts — prompts that adapt to the student's current failures — provide fresh supervision each step but create a moving target. The conflict-downweighting is what makes evolving contexts usable as in-training supervision.

## Why it matters

OPD covers the gap between RLHF (needs a reward model) and RLVR (needs verifiability): a large family of *open-ended* tasks — style transfer, tone shaping, persona alignment, creative generation — have no verifiable reward and expensive human preferences. OPD trades those away for a teacher LM and a set of context prompts, both cheap. Flux-OPD makes contexts adaptive rather than static, so supervision keeps improving with the student.

## Gotchas & tricks

- **Teacher must dominate the student.** If the teacher is only marginally better, OPD gains are small and can even regress on things the teacher does worse.
- **On-policy is load-bearing.** Distilling from teacher-sampled data (off-policy) suffers exposure bias and can drift the student off its own manifold. Always sample from the student.
- **Evolving contexts need stabilization.** Directly using an evolving context as supervision creates a non-stationary target; the reverse-KL conflict term is what keeps training stable (see [beta-opsd](beta-opsd.md) for a related β-controlled variant).
- **Overlaps with rejection-sampling SFT.** Both use on-policy rollouts. OPD supervises with soft logits; rejection-sampling SFT supervises with a hard filter. OPD gives more signal per token; RS-SFT is simpler.

## Sources

- Paper: *Flux-OPD: On-Policy Distillation with Evolving Contexts* — Wang, Wang, Zeng et al., 2026 — [arXiv:2607.28022](https://arxiv.org/abs/2607.28022).
- Paper: *On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes* — Agarwal et al., 2024 — earlier formulation of OPD.
- Related: [beta-opsd](beta-opsd.md) — on-policy *self*-distillation with a KL knob.
