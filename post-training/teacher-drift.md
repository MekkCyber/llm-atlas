# Teacher Drift
*Depth — the failure mode where an on-policy self-distillation teacher starts following the student's errors.*

**TL;DR:** In on-policy self-distillation (OPSD) the "teacher" is often the same model as the student, conditioned on a reference solution or extra prompt. As the student learns, the distribution the teacher is conditioned on shifts: the teacher is now asked to continue trajectories that reflect the student's accumulated mistakes. It starts producing confident but *wrong* supervision. The TOPSD paper names this failure Teacher Drift, measures it, and proposes a teachable-prefix gating fix.

**Prereqs:** [on-policy-distillation](fine-tuning/on-policy-distillation.md), [_post-training](_post-training.md)
**Related:** [grpo](grpo.md), [rlvr](rlvr.md), [long-cot-rl](reasoning/long-cot-rl.md), [cot-reward-model](cot-reward-model.md)

---

## What it is

**On-policy self-distillation setup.** One model `π` plays both student and teacher:

- **Student:** samples a trajectory `y_s ~ π(·|x)`.
- **Teacher:** given a hint `h` (reference solution, partial answer, or auxiliary conditioning), computes `π(y_s | x, h)` and uses it to supervise the student.

If `π` is strong and `h` is informative, the teacher's token-by-token guidance pulls the student toward correct behavior. The implicit assumption is that `π(· | x, h)` has roughly the same shape across training — a strong, oracle-adjacent distribution.

**What breaks.** As the student updates, the kinds of trajectories it generates change. Partway through training, the teacher is being asked to grade or continue trajectories that have *already* diverged from what a competent policy would produce. Because the teacher shares weights with the student, the drift in the student's prior pulls the teacher's conditional distribution along. The teacher now produces smooth continuations of broken reasoning — which the student then imitates.

This failure is particularly acute when:

- The reference signal `h` is weak (short hint, sparse feedback).
- The student starts further from competence (low-resource language, hard reasoning domain).
- Training runs long enough for the teacher distribution to walk away from its starting point.

## How it works

TOPSD's diagnostic measures teacher-student likelihood divergence along a sampled trajectory. Normally, the teacher's probability of the correct continuation should stay high throughout; under drift, it collapses earlier and earlier in training.

The fix has three parts:

1. **Find the teachable prefix.** Walk forward through the student's trajectory. Record the earliest token at which the teacher-student likelihood ratio drops below a threshold — that's where the teacher's signal becomes unreliable. Call this prefix length `k`.
2. **Recovery check.** Condition the teacher on the student's prefix of length `k` and ask whether it can complete to the correct answer. If yes, this example is usable; if no, skip it (the student's state is too far gone for the teacher to rescue).
3. **Prefix-gated supervision.** Apply the distillation loss only on the continuation from position `k` onward, never on tokens inside the teachable prefix where the student already agrees with the teacher.

This drops two kinds of wasted updates: supervision on tokens that are already correct (prefix), and supervision on tokens the teacher itself can't recover from (bad examples).

## Why it matters

- **Explains why OPSD underperforms in low-resource settings.** Drift happens earliest where the student is weakest. Multilingual math, rare domains, and underrepresented languages all see outsized teacher-drift effects; TOPSD's gains concentrate there.
- **Generalizes beyond self-distillation.** Anywhere the teacher's conditioning depends on the student's state — self-rewarding models, cross-distillation between two policies being updated — drift can occur. The teachable-prefix diagnostic is a generic test.
- **Clarifies the OPSD vs plain-SFT tradeoff.** Classical SFT doesn't drift because the teacher is a fixed dataset. OPSD can beat SFT when the teacher signal is strong, but fails worse when drift sets in. TOPSD-style gating is what makes OPSD robustly dominate SFT.

## Gotchas & tricks

- **"Teacher" doesn't have to share weights for drift to happen.** Any teacher whose supervision depends on the student's current trajectory distribution is vulnerable. A frozen larger model conditioned on the student's rollout is less vulnerable but not immune, because its conditional distribution may be miscalibrated on distributions it never saw.
- **Drift is not the same as reward hacking.** Reward hacking is the *student* exploiting the reward signal. Teacher drift is the *teacher* losing calibration; the student's loss function silently stops pointing at the right place.
- **Monitoring.** Track the teacher's probability of the correct token at the start, middle, and end of trajectories across training. Collapse at the tail is normal; collapse creeping earlier into the sequence is drift.
- **Prefix threshold tuning.** Too strict (short teachable prefix, skip everything hard) and you train on nothing; too loose (long prefix, include broken trajectories) and drift sneaks back in. TOPSD picks the threshold via held-out teacher-student likelihood statistics.

## Sources

- Paper: *Knowing When to Teach: Learning from Teachable Prefixes in Multilingual Reasoning Distillation (TOPSD)* — Cho, Yoon, Park, Seo, Chu, Oh, Kim, 2026 — arXiv 2611.08179 — the primary source for Teacher Drift as a named phenomenon and the teachable-prefix fix.
- Paper: *On-Policy Distillation of Language Models* — Agarwal et al., 2023 — the OPSD setup teacher drift occurs inside.
