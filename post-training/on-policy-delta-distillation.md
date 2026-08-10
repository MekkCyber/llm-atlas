# On-policy delta distillation (OPD 2)
*Depth — distill the *effect* of post-training, not the teacher's raw distribution.*

**TL;DR:** On-Policy Distillation (OPD) has a student roll out its own responses and match the teacher's token distribution on them. **On-Policy Delta Distillation (OPD 2)** replaces the target with the *log-prob gap* between a post-trained teacher and its own base model — `Δ = log π_teacher − log π_base`. The student learns what post-training *added*, not what the base already knew. Practical alternative to RLVR when a strong post-trained teacher exists.

**Prereqs:** [../post-training/_post-training.md](../post-training/_post-training.md), [../post-training/rlvr.md](../post-training/rlvr.md)
**Related:** [../post-training/rejection-sampling.md](../post-training/rejection-sampling.md), [../post-training/dpo.md](../post-training/dpo.md)

---

## What it is

A variant of on-policy distillation for post-training. The student generates rollouts on-policy (sampled from its own current distribution), then updates against a target derived from a paired (teacher, teacher's base model) — not from the teacher alone. In the multilingual math setting, the delta signal isolates *reasoning ability introduced by post-training* from *general language ability already present in the base*.

## How it works

**Setup.** Given:
- `π_θ` — the student to train.
- `π_teacher` — a post-trained reference model (e.g., a strong Qwen3 reasoning checkpoint).
- `π_base` — the base model *from which* `π_teacher` was post-trained.

**Sampling.** Roll out a response `y` from the student `π_θ` (on-policy).

**Target.** For each token `t` in `y`, define the delta signal:

```
Δ_t(v) = log π_teacher(v | x, y_<t) − log π_base(v | x, y_<t)
```

The training loss pulls the student's log-probabilities toward `Δ_t` (up to a normalization and a KL to prevent drift), typically over the top-k vocabulary tokens per position.

**Contrast with plain OPD.** OPD's target is `log π_teacher` — the full teacher distribution. OPD 2's target is what the teacher added *on top of* its base, isolating the post-training-attributable behavior.

## Why it matters

- **Cross-lingual transfer with less English drift.** English-only OPD helps Korean/Japanese scores but pulls responses toward English. OPD 2 with a multilingual teacher/base pair narrows the English–Korean gap while preserving target-language responses.
- **RL-adjacent, no rollouts-with-reward.** No value network, no verifiable reward, no learned RM. If you have a strong post-trained teacher and a compatible base, OPD 2 gives you a scalable post-training route.
- **Isolates what post-training changed.** Since the delta signal cancels the base's contribution, the student inherits post-training capability more efficiently than by matching the teacher's absolute distribution.

## Gotchas & tricks

- Requires a *matched pair* `(teacher, base)` — the teacher must be a direct post-training descendant of the base for the delta to be meaningful.
- Numerical stability: for tokens where `π_base` is near zero, the delta can blow up. Standard fix is a floor on `log π_base` or top-k truncation.
- Multilingual settings need target-language data in the student's on-policy rollouts, or responses drift toward the language the teacher's post-training data was in.

## Sources

- Paper: *On-Policy Delta Distillation for Multilingual Math Reasoning* — 2026 — [arXiv:2608.05802](https://arxiv.org/abs/2608.05802)
- Prior: OPD 2 methodology attributed to Heo et al., 2026 (referenced but distinct from this paper).
