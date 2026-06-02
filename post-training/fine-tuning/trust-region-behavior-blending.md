# Trust-Region Behavior Blending (TRB)

*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A warmup recipe for [on-policy distillation](on-policy-distillation.md). Early in training the student's rollouts are too poor for the teacher's supervision to do useful work. TRB replaces the rollout-generating *behavior policy* with the closest-to-teacher policy that stays inside a student-centered KL trust region, gradually shrinks the trust region to zero, and then training reverts to pure student rollouts. The on-policy distillation loss itself (per-prefix reverse KL) is untouched.

**Prereqs:** [on-policy-distillation](on-policy-distillation.md), [../grpo](../grpo.md), [../_rl](../_rl.md)
**Related:** [../rejection-sampling](../rejection-sampling.md) · [../reasoning/long-cot-rl](../reasoning/long-cot-rl.md)

---

## What it is

On-policy distillation supervises the student on its *own* rollouts. The structural problem is that the rollouts at step 0 are essentially random — the teacher's per-prefix targets on those prefixes are degenerate, and the student burns budget on prefixes far outside any reasonable policy's support.

TRB sidesteps the problem during warmup only. Instead of sampling rollouts from the raw student, it samples from a **behavior policy** π_b chosen to be:

1. As close to the teacher as possible (so prefixes are coherent and the teacher's supervision is meaningful),
2. Inside a KL ball around the *student* (so the supervision the student sees is still relevant to its actual policy).

The KL budget β shrinks toward zero over warmup. By the end, π_b ≈ π_student and training returns to vanilla on-policy distillation.

---

## How it works

### The behavior-policy construction

Define a per-step trust-region constraint:

```
KL( π_b(· | q, o_<t)  ||  π_student(· | q, o_<t) )  ≤  β_t
```

Among all distributions satisfying this constraint, pick the one closest to the teacher (minimum KL to π_teacher). This has a closed-form-ish solution: a temperature-mixed blend between student and teacher logits, with the mixing weight chosen to saturate the trust-region bound.

In practice, this is implemented as **logit blending**: sample tokens from a softmax over an interpolation `α · logit_teacher + (1 - α) · logit_student`, with α set per step (or per layer of the policy) so the KL to the student equals β_t.

### Annealing the KL budget

β_t starts at some moderate value (large enough that π_b is meaningfully different from π_student — i.e. teacher-leaning prefixes survive) and is linearly annealed to 0 over warmup. When β_t = 0, the trust region forces π_b = π_student, and rollouts are pure student again.

### The distillation loss is unchanged

Critically, the **OPD loss is computed exactly as before** — per-prefix reverse KL between student and teacher on the rollout. Only the *rollout-generating policy* changes during warmup. This makes TRB additive: it sits as a warmup wrapper around any OPD pipeline without modifying the inner loss.

---

## Why it matters

- **Fixes the OPD cold-start problem cheaply.** No reward function, no separate offline-distillation phase, no extra loss term.
- **Drop-in for any OPD recipe.** Because the loss is unchanged, TRB combines with any OPD variant — reverse-KL, JSD, or specialized teacher-targets.
- **Strongest average across two math-reasoning distillation settings** in the source paper, beating prior warmup tricks.

---

## Gotchas & tricks

- **The trust-region radius β is the only real knob.** Too small and π_b ≈ π_student from the start (no warmup benefit). Too large and rollouts are essentially teacher rollouts — collapsing to offline distillation, with the same prefix-mismatch problem you started with. Reasonable values are in the "noticeably teacher-leaning but still student-recognizable" range; the paper's exact schedule is in the appendix.
- **Logit blending requires both policies' logits.** This is cheap during warmup (one extra forward pass per rollout) but does add a teacher forward pass per step you wouldn't otherwise need.
- **Anneal to exactly zero.** If β is annealed only partway, the rollout distribution stays teacher-leaning forever — you're effectively doing soft offline distillation at steady state, losing the on-policy guarantee.
- **Not a substitute for a strong cold-start SFT.** TRB compresses the warmup phase rather than eliminating it; pairing TRB with a tiny offline-distillation seed is fine and often complementary.

---

## Sources

- Paper: *Trust-Region Behavior Blending for On-Policy Distillation* — Plyusov, Gorbatovski, Malakhov, Balagansky, Shaposhnikov, Korotyshova, Gavrilov, 2026 — T-Tech. Introduces TRB and validates on math-reasoning distillation.
