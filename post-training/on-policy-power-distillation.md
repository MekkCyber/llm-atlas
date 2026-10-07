# On-Policy Power Distillation (OPPD)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Sampling a model's **power distribution** — raising each full-answer probability to an exponent $\alpha>1$ and renormalizing — concentrates mass on the model's most-likely answers (sharpening) and improves reasoning accuracy, but needs many scored candidates per query. OPPD **distills that sharpened distribution into the student's one-shot sampling** via a sequential Monte Carlo (SMC) loop in which the student generates candidates and a frozen teacher's power distribution reweights them for a maximum-likelihood update. Up to **+23.0 pp MATH500 / +27.3 pp GSM8K** at the same temperature, with one-shot output beating 64-candidate power sampling by 2–3 pp.

**Prereqs:** [ppo.md](ppo.md), [rejection-sampling.md](rejection-sampling.md), [_post-training.md](_post-training.md)
**Related:** [grpo.md](grpo.md) · [reasoning/mcts.md](reasoning/mcts.md) · [reasoning/long-cot-rl.md](reasoning/long-cot-rl.md)

---

## What it is

A model that gives the correct answer *more* probability than any single wrong answer may still mostly sample a wrong one, because the wrong answers *together* hold more mass. The power distribution $p^\alpha / Z$ (with $\alpha>1$) attacks exactly that: it widens the gap between the top answer and the long tail. Test-time power sampling (sample $N$, score, renormalize-by-power, pick) raises reasoning accuracy — but $N$ scored candidates per query is expensive.

**OPPD** is the training-time analogue: let the model inherit the sharpened distribution so one sample at eval time is as good as $N$ candidates reweighted. The result is a reasoning-tuning recipe that doesn't need a verifier or human preferences — only a frozen teacher (often the model itself at $t=0$).

## How it works

- **Sampler.** Run an SMC pass per prompt: the **student** $\pi_\theta$ generates candidate answers; the **frozen teacher** $\pi_T$ assigns probabilities; weights are $w_i \propto \pi_T(o_i)^\alpha$.
- **Objective.** Treat $\{(o_i, w_i)\}$ as weighted targets for a maximum-likelihood update of $\pi_\theta$: maximize $\sum_i w_i \log \pi_\theta(o_i \mid q)$. This matches the student to the power-sharpened distribution of the teacher.
- **On-policy.** Candidates come from the current student, not an off-policy buffer; this keeps the surrogate aligned with the model's actual sampling behavior.
- **Hyperparameter.** $\alpha$ controls sharpening; small SMC population (16 is enough) is fine because the per-step update is cheap.

## Why it matters

- Replaces N-candidate test-time search with a one-shot trained policy. **Same quality, $1/N$ the inference cost.**
- Generalizes the recipe "distill a search algorithm into the policy" — the same template works for any reweighting rule you can evaluate on candidates (power, verifier scores, teacher-aggregated preferences).
- Doesn't need external supervision: the teacher can be the untrained copy of the student.

## Gotchas & tricks

- **$\alpha$ too high → mode collapse.** Very large exponents push mass onto a single answer and the student loses sampling diversity. Pick $\alpha$ by held-out accuracy, not just training loss.
- **SMC weights have variance.** Normalize weights per-prompt and clip very large $w_i$ to control gradient variance.
- **Needs a usable base.** If the teacher gives near-uniform probability over wrong answers, power sharpening has nothing to sharpen onto. Base-model competence sets the ceiling (just like RL-for-reasoning).
- **Composes with RL.** Use OPPD first to recover the sharpening effect in one shot, then run GRPO/RLVR on top for additional gains — no conflict.

## Sources

- Paper: *Sharpen Without Search: On-Policy Distillation of Sequence-Level Power Distribution* — Azizi et al., USC / Intel AI, 2026 — [arXiv:2610.06804](https://arxiv.org/abs/2610.06804).
