# Order-Only On-Policy Distillation
*Depth — a probability-free on-policy distillation loss that matches the teacher's token ranking, not its probabilities.*

**TL;DR:** Standard on-policy distillation matches the teacher's full softmax distribution at each step of the student's rollout. Order-only distillation matches only the **ranking** of the teacher's top-K tokens with a pairwise rank loss, and ignores the actual probabilities. This makes the loss invariant to teacher temperature, softens the signal where the teacher is overconfident, and empirically beats KL-based on-policy distillation on math reasoning across six teacher-student pairs.

**Prereqs:** [_post-training.md](./_post-training.md) · [_rl.md](./_rl.md)
**Related:** [rejection-sampling.md](./rejection-sampling.md) · [reasoning/long-cot-rl.md](./reasoning/long-cot-rl.md) · [../case-studies/deepseek-r1.md](../case-studies/deepseek-r1.md)

---

## What it is

A distillation objective designed for the **on-policy** regime — the student generates its own rollouts and the teacher scores them. The usual target is the teacher's full softmax (KL divergence from teacher to student). Order-only throws away the probability values and keeps only *which token the teacher ranked higher than which*.

The intuition is that over-long reasoning rollouts, the teacher's exact probabilities are high-variance and sometimes mis-calibrated (sharply peaked or flat), while the ordering is more robust and carries the useful supervision signal: "prefer these tokens, in this order."

## How it works

At each student-sampled token position, read the teacher's top-K logits (K = 8 to 64 typical). Convert to a ranking `t_{(1)} > t_{(2)} > … > t_{(K)}`. For every pair `(i, j)` with `i < j` in that ranking, add a pairwise rank term pushing the student to score the higher-ranked token above the lower-ranked one:

```
L_pair(i, j) = -log σ( s_student(t_{(i)}) − s_student(t_{(j)}) )
```

Sum over all pairs in the top-K; average over rollout positions. There is no KL term, no temperature, and no cross-entropy on a soft label.

Because the loss depends only on the student's logit *differences* over K specific tokens, it imposes no penalty on how the student distributes mass over the other `vocab − K` tokens, nor on the sharpness of the student's distribution overall. The student's distribution is free to find its own shape as long as token order on the top-K agrees with the teacher.

## Why it matters

- **Robust to teacher calibration.** A teacher that over-concentrates on one token still produces a usable ranking; its KL signal becomes degenerate (everything collapses to one token).
- **Temperature-invariant.** Scaling teacher logits by any positive constant preserves the ranking. The student is never implicitly tuned to one teacher temperature.
- **Strong where it matters.** Distilling long-CoT reasoners (DeepSeek-R1-style, Kimi) is a dominant 2025–2026 post-training recipe. Order-only wins on HMMT25 / AIME24 / AIME25 across six pairs, with the biggest gains where the student is much smaller than the teacher — exactly the regime where exact probability imitation is poorly conditioned.
- **Cheap to swap in.** Same data pipeline, same teacher, different loss head.

## Gotchas & tricks

- **Choose K carefully.** Too small (K < 4) and the signal is too coarse to differentiate students. Too large (K > 64) and pairs involving low-mass teacher tokens add noise.
- **Ties in teacher ranking.** Break arbitrary ties (float equality is rare but happens with sampled teacher distributions) rather than including both pairings.
- **Not a replacement everywhere.** In token-rich, short-answer tasks where the teacher's probability mass itself is accurate information (translation, straightforward QA), KL distillation may still be competitive.
- **Combine with rejection sampling.** The paper reports gains stack with a rejection step at the trajectory level.

## Sources

- Paper: *Order-Only On-Policy Distillation* — Han et al., University of Memphis, 2026 — [arXiv:2611.02112](https://arxiv.org/abs/2611.02112).
- Paper: *DeepSeek-R1* — DeepSeek, 2025 — the baseline KL distillation recipe this paper replaces.
