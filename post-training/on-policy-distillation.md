# On-Policy Distillation
*Depth — distilling a teacher into a student where the student samples from its own policy at training time.*

**TL;DR:** Standard (off-policy) distillation trains the student to imitate a fixed dataset of teacher samples. **On-policy distillation** lets the student sample from *its own* policy at each step, then asks the teacher what it would do on those student samples and trains the student toward that target. The distribution under which supervision is provided then *matches* the student's deployed distribution — robust to aggressive compression (e.g. step-count reductions in diffusion or capacity reductions in LLMs).

**Prereqs:** [_rl](../post-training/_rl.md), [grpo](../post-training/grpo.md)
**Related:** [_rewards](../post-training/_rewards.md), [rejection-sampling](../post-training/rejection-sampling.md), [diffusion-grpo](../post-training/diffusion-grpo.md)

---

## What it is

A distillation regime in which the *generator of supervision states* is the student, not the teacher. The teacher remains the supervision *signal* (its output on those states is the target), but the student decides where in the input/state space to ask for supervision.

For an LLM student, that means: the student samples a (partial) response; the teacher's distribution over the next token (or the teacher's full continuation) is the target.

For a diffusion student, that means: the student runs its own sampling trajectory; the teacher's predicted velocity at each step along that trajectory is the target.

## How it works

**The training loop.**

1. The student samples from its current policy to produce a state (a partial generation, or a noisy image at some `t`).
2. The teacher is evaluated on that state: produces the supervision target (logits, velocity, or full continuation).
3. The student's parameters are updated to bring its prediction closer to the teacher's at that state.
4. Repeat — the state distribution evolves with the student.

**Optional reward filter.** A scalar reward (e.g. the composite reward used for RL in the same pipeline) can act as a verification signal: down-weight student samples whose reward collapses, focusing distillation effort on viable trajectories.

**Difference from off-policy distillation.** Off-policy distillation freezes a corpus of teacher samples and trains the student to imitate them. The student is supervised on the *teacher's* distribution, which may not overlap with where the student actually operates after compression — the classic exposure-bias problem.

## Why it matters

- **Robust under aggressive compression.** Whether the compression is fewer diffusion steps, fewer parameters, or fewer attention heads, the student visits states the teacher never did. On-policy targets put supervision *exactly* where the student needs it.
- **Closes the teacher→student gap** more reliably than off-policy distillation in the regimes where it matters (low-step diffusion, small student LMs).
- **Composes with RL.** Same infra as RL — sample from the student, evaluate something on those samples, update — so a team running RL already has 90% of the on-policy distillation stack.
- **Empirical use in Qwen-Image-2.0-RL** to compress the diffusion-GRPO teacher into a fast deployable student without losing the RL gains. (See [qwen-image-2 case study](../case-studies/qwen-image-2.md).)

## Gotchas & tricks

- **Cost.** Teacher evaluation on student samples means a teacher forward per training step — much more expensive than reading a precomputed off-policy corpus. Cache where you can (e.g. cache teacher KV state under fixed prefixes).
- **Cold-start problem.** Early in training the student samples can be arbitrarily bad; the teacher's targets on those samples are also low-information. A short warmup of off-policy distillation, or a reward-filter cutoff, helps avoid wasted compute.
- **Reward over-reliance.** Using the reward as a hard filter rather than a soft weight collapses diversity; treat the reward as a downweighting signal, not a gate.
- **Not the same as policy distillation in RL.** RL policy distillation typically targets *actions* (or action distributions); generic on-policy distillation can target any teacher output — logits, velocities, intermediate features.

## Sources

- Paper: *Qwen-Image-2.0-RL Technical Report* — arXiv:2606.27608 — https://arxiv.org/abs/2606.27608
- See also: [qwen-image-2 case study](../case-studies/qwen-image-2.md), [diffusion-grpo](../post-training/diffusion-grpo.md).
