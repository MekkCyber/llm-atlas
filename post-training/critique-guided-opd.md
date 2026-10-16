# Critique-Guided On-Policy Distillation (CriticOPD)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** On-policy distillation where the teacher writes a *critique* of the student's incorrect rollout, the student rewrites the trace from the last error, and the teacher scores the revision **with the critique in context** while the student is trained on the revision **without** the critique. Beats vanilla OPD on competition math.

**Prereqs:** [on-policy-distillation.md](./on-policy-distillation.md), [grpo.md](./grpo.md)
**Related:** [cot-reward-model.md](./cot-reward-model.md) · [reasoning/prm.md](./reasoning/prm.md) · [reasoning/long-cot-rl.md](./reasoning/long-cot-rl.md) · [_distillation.md](./_distillation.md)

---

## What it is

Standard OPD assigns dense token-level supervision across the entire student rollout, including the correct prefix. When the student's rollout is mostly wrong, the signal collapses to "do everything differently" — which gives no useful gradient. CriticOPD narrows the correction to **the specific point where the student's reasoning first went off-track**, by routing supervision through a teacher-written critique rather than a from-scratch teacher trace.

## How it works

Per incorrect student rollout $o = (o_1, \ldots, o_T)$:

1. **Critique generation.** Teacher $\pi_T$ reads $(q, o)$ and emits a critique $c$ that identifies the **last-error position** $t^\star$ and explains the mistake.
2. **Student revision.** The student rewrites the suffix from $t^\star$ onward, conditioned on $(q, o_{<t^\star}, c)$ — the student *sees* the critique while revising:
   $$o'_{>t^\star} \sim \pi_\theta(\cdot \mid q, o_{<t^\star}, c).$$
3. **Teacher scoring.** The teacher scores the full revised trajectory $(q, o_{<t^\star}, o'_{>t^\star})$ **with the critique in context**, giving a reliable grade of the revision.
4. **Student training.** The student is trained on $(q, o_{<t^\star}, o'_{>t^\star})$ **without** the critique — forward loss / PPO-style update under the teacher-provided score as advantage. The asymmetric context (teacher sees $c$, student doesn't) forces the student to *internalize* the correction rather than copy the critique.

## Why it matters

- **Localized correction.** The critique points to one spot; supervision applies after that spot. Avoids wasted gradient on correct prefixes and avoids the implicit "rewrite everything" signal.
- **Reachable revisions.** Because the revision shares the prefix the student already produced, it lies inside the student's rollout distribution — the training update isn't fighting a large distribution shift.
- **Context asymmetry does the work.** Scoring with critique in context gives a dense, grounded reward. Training without critique forces internalization. Together they close the loop.

## Gotchas & tricks

- **Last-error identification is fragile.** If the critique mis-locates the error, the suffix revision inherits the mistake and the training signal is wrong. Validate critique accuracy before scaling.
- **Teacher must be strong enough to critique and score.** Weak teachers can't produce reliable critiques; the method degenerates toward random suffix sampling.
- **Costs two teacher calls per update.** One for critique, one for scoring. Cheaper than full teacher rollouts, more expensive than one-shot OPD.
- **Correct rollouts need separate handling.** Papers typically train only on incorrect rollouts (where critiques apply); correct rollouts bypass CriticOPD or go through standard GRPO.
- **Works best on long-form reasoning.** Short-answer tasks (classification, extraction) have no "last error" structure.

## Sources

- Paper: *Recovering Reliable Supervision via Critique-Guided On-Policy Distillation* — Yu, Li, Yang, Hughes, Zhang, Killian, 2026 — [arXiv:2610.20572](https://arxiv.org/abs/2610.20572). Tested on two Qwen3 student sizes and six math-reasoning benchmarks; largest gains on competition problems.
