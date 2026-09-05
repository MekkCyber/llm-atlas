# On-policy distillation
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** On-policy distillation (OPD) combines student-generated rollouts with dense **token-level teacher supervision** on those rollouts. The student explores; the teacher grades every token. Modern reanalysis (Fu et al., 2026) shows OPD's data-efficiency ceiling is dominated by *rollout state coverage*, not prompt-set diversity: training on a **single query** recovers most of a full-data run's gains across tasks and model families.

**Prereqs:** [_post-training](_post-training.md), [_rl](_rl.md)
**Related:** [rlvr](rlvr.md), [rejection-sampling](rejection-sampling.md), [rl-prompt-curation](rl-prompt-curation.md), [grpo](grpo.md)

---

## What it is

Two distillation regimes for LLMs sit on a spectrum:

- **Off-policy (SFT) distillation.** Sample completions from the *teacher*, train the student to imitate them (next-token loss). Student never generates during training.
- **On-policy distillation (OPD).** Sample completions from the *student*, then have the teacher score every token — usually via KL between teacher and student token distributions along the student's own trajectory.

OPD sits between SFT and RL: the rollout is on-policy (like RL) and the supervision is dense per-token (like SFT). Compared to RL, the reward isn't a scalar over the whole trajectory but a *teacher distribution* at every step; compared to SFT, the student sees its own failure modes rather than a teacher's polished trace.

## How it works

Per training step:

```
1. Sample a prompt q from the training set.
2. Student rolls out a completion y ~ π_student(·|q).
3. Teacher runs a forward pass on (q, y), producing p_teacher(·|q, y_<t) at every position.
4. Loss = KL(p_teacher || p_student) averaged over positions in y.
5. Update the student with standard optimizer.
```

The rollout is on-policy; the loss is a soft-label imitation on each token. No reward model, no advantage estimation, no PPO clipping.

## Why it matters

**Data-minimal training.** Fu et al. (2026) train OPD on a **single query**, iterate for hundreds of steps, and recover the majority of full-data OPD's gains across math, code, and open-ended tasks. They introduce a **state coverage** metric that quantifies the distinct trajectories the student generates as it drifts under training — one prompt turns out to cover a wide state manifold under on-policy sampling.

The lesson: what OPD needs is *rollout diversity around the current policy*, not *prompt-set diversity*. That reframes OPD as compute-bound rather than data-bound, and reopens the practicality of teacher distillation at scale — the teacher forward pass is the expensive step, and the finding says you get most of the gain even without curating a corpus.

## Gotchas & tricks

- **Teacher-student capability gap.** Wider gaps make OPD more expensive (more teacher passes per state) but improve final quality. Same-family teacher-student pairs are the easiest wins.
- **Choice of query matters (a little).** One-shot OPD's *ceiling* varies with the query: harder queries buy broader state coverage. Practical recipe: start with a query near the student's current failure region.
- **This is not RL.** No advantage, no KL-to-reference constraint. Combining OPD with RL (KL to reference on the same rollouts) is an underexplored middle ground.
- **Data efficiency ≠ compute efficiency.** Fewer prompts, but teacher-per-token supervision is still the bottleneck. If teacher forwards are cheap, prefer OPD over sparse-reward RL.

## Sources

- Paper: *Rethinking On-Policy Distillation of Large Language Models II: One Training Example* — Fu, He, Zuo, Huang, Zhang, Xiao, Qian, Luo, Gao, Wang, Liu, Ding, Xiao, 2026 — [arXiv:2609.04172](https://arxiv.org/abs/2609.04172).
- Related earlier work: on-policy KD (Agarwal et al., 2023); teacher-guided rollouts (various).
