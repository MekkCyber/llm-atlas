# MOPD — Multi-teacher On-Policy Distillation
*Depth — distilling several RL-trained specialist teachers into one student via on-policy rollouts and clipped log-ratio advantages.*

**TL;DR:** MOPD trains a single student policy to inherit the capabilities of several frozen RL-trained specialist teachers. The student rolls out on-policy; each task is routed to the matching domain teacher; the teacher scores the student's tokens; the advantage is a clipped log-ratio `log(q_teacher / p_student)`; the loss is a length-normalized reverse-KL-motivated surrogate over student-generated tokens. Tool observations and instructions are masked from the loss and repetitive rollouts are filtered. The pattern replaces a multi-objective RL run with a distillation step whose objectives compose linearly in output space.

**Prereqs:** [grpo](./grpo.md), [rlvr](./rlvr.md)
**Related:** [dpo](./dpo.md), [rejection-sampling](./rejection-sampling.md)

---

## What it is

A post-training recipe for merging the capabilities of several specialist models into one. Each specialist is produced by its own domain-specialized RL run (e.g. agentic UX, multi-harness, long-horizon, general agentic). MOPD then distills all specialists into one student via on-policy rollouts routed by task domain. The student does not run RL during MOPD; it runs supervised distillation against the teacher that is most expert in each task.

The method applied in IQuest-Q1 follows the general multi-teacher on-policy distillation pattern (see Sources); IQuest-Q1 contributes the specific advantage, loss, and masking recipe used at frontier scale.

## How it works

1. **Teacher training.** Produce `K` specialist teachers by running domain-specialized RL from a common SFT checkpoint. Each teacher is expert in one capability axis.
2. **Task routing.** For each training task, route to the matching domain teacher. The student knows nothing about the routing; it just generates.
3. **Student rollouts.** The student samples `B` rollouts per task on-policy.
4. **Teacher scoring.** For each student-generated token, the chosen teacher computes `q_teacher(token | context)`. The student's own probability `p_student(token | context)` is already known.
5. **Advantage.** `A_t = clip(log(q_teacher) - log(p_student), -c, c)`. The clip prevents a single catastrophic log-ratio from dominating the batch.
6. **Loss.** Length-normalized reverse-KL-motivated surrogate over student-generated tokens, weighted by `A_t`. Think of it as a soft-advantage distillation loss.
7. **Masking.** Tool observations and instructions are excluded from the loss (they're "input" the student didn't generate). Rollouts that loop or repeat get masked out.
8. **Student update.** Standard supervised-style update (not RL); no reward model, no value function.

## Why it matters

- **Composes capabilities linearly.** Multi-objective RL has reward balancing problems; distilling from several already-trained specialists shifts composition into output space where it reduces to task routing + weighted supervision.
- **Decouples training.** Specialists train independently and can be swapped out individually without re-running the full stack.
- **On-policy correction.** Student rollouts prevent the exposure-bias failures of offline distillation; the student is scored on its own distribution.
- **Negative transfer avoided.** Routing per task means a teacher only sees inputs it is expert in; the student doesn't inherit a weak teacher's mistakes outside its specialty.
- Deployed by IQuest-Q1 across four specialist domains to produce one agentic foundation model.

## Gotchas & tricks

- **Clip `c` matters.** Unclipped log-ratios explode on low-probability student tokens and destabilize training. The IQuest-Q1 recipe uses a modest clip.
- **Repetitive rollouts leak.** Student rollouts that loop (common early in training) must be filtered or the loss collapses to repeating the loop under the teacher's blessing.
- **Masking tool outputs is non-optional.** If tool observations are included in the loss, the student learns to predict tool outputs — a wasted capacity sink.
- **Teacher coverage gaps.** If some task domains have no specialist, those tasks must be routed to a fallback (e.g. the original SFT checkpoint) or dropped entirely. Treating a general-purpose teacher as a fallback risks pulling the student away from its specialist strengths.
- **Reward-model-free, which is a feature.** No reward model to maintain, which also means no reward-hacking surface.

## Sources

- Paper: *IQuest-Q1: Advancing Foundation Capabilities for Agentic CLI Systems* — 2026 — [arXiv:2610.30949](https://arxiv.org/abs/2610.30949) (worked example at 320B/15B scale, specific advantage + loss + masking recipe).
- The multi-teacher on-policy distillation pattern follows prior work cited by that paper as reference 40; MOPD here is the IQuest-Q1 instantiation.
