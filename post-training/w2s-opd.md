# W2S-OPD — Weak-to-Strong On-Policy Distillation
*Depth — on-policy distillation into a frontier student using only weaker models.*

**TL;DR:** On-policy distillation (OPD) usually requires a teacher at least as capable as the student, which breaks at the frontier. W2S-OPD builds a **proxy teacher in logit space** from a contrast pair of two *smaller* models (positive − negative), adds that direction to the student's own base-model logits, and distills the student against the result with per-token reverse-KL on its own rollouts. The student improves even when every supervisor is weaker than it.

**Prereqs:** [grpo.md](./grpo.md), [rejection-sampling.md](./rejection-sampling.md)
**Related:** [saf-opd.md](./saf-opd.md), [rlvr.md](./rlvr.md)

---

## What it is

An OPD variant that removes the "stronger teacher required" assumption. Given two related smaller models — one "good" at some capability and one that isn't — their logit difference isolates the *direction* of that capability. Adding that direction to the student's own base-model logits produces a proxy teacher that (a) points in the improvement direction and (b) stays distributionally close to the student, keeping the reverse-KL well-behaved.

## How it works

For each token during an on-policy rollout of the student `π_S`:

1. Compute logits from a positive smaller model `π_+` and a negative smaller model `π_−`. Take the difference `Δ = logit(π_+) − logit(π_−)`.
2. Compute the student's own base-model logits `logit(π_S^base)` (its pre-fine-tune checkpoint).
3. Form the proxy teacher: `logit(π_T) = logit(π_S^base) + α · Δ`.
4. Distill: `L = KL(π_S ∥ π_T)` per token, on the student's own rollouts.

Three concrete contrast pairs:
- **`(π_+, π_−) = (post-RL expert, its pre-RL init)`** — isolates the skill RL taught.
- **`(π_+, π_−) = (larger base, smaller base)`** — isolates the capability that came from scale.
- **`(π_+, π_−) = (base w/ correct hint, base w/ wrong hint)`** — instance-level direction toward the answer.

## Why it matters

- Cracks the frontier ceiling: no larger model exists → still improve via weaker contrasts.
- Cheap. The contrast models are order-of-magnitude smaller than the student; only inference is required.
- Different contrasts induce different improvements: post-RL and hint contrasts push *reasoning frameworks*; scale contrast pushes *solving procedure*. Composable.

## Gotchas & tricks

- Adding Δ to `π_S^base`, not to `π_S` itself, is what keeps the proxy distributionally adjacent — critical for reverse-KL stability.
- α is a scalar, but effectively also a per-domain choice; scale contrast wants a smaller α than a hint contrast.
- Works on the student's *own* rollouts (on-policy) — off-policy distillation from proxy-teacher rollouts loses the alignment benefit.

## Sources

- Paper: *Weak-to-Strong On-Policy Distillation* — Yu et al., 2026 — [arXiv:2607.26246](https://arxiv.org/abs/2607.26246)
- Code: https://github.com/Yu-Fangxu/W2S-OPD
