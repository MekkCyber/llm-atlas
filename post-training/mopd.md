# MOPD (Multi-teacher On-Policy Distillation)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Reasoning-distillation (R1-style) uses one specialist teacher's traces as SFT targets for a general student. **MOPD** generalises this to a **committee of specialist RL-trained teachers** that steer a single student *on-policy* — i.e. the student rolls out its own trajectory and the teachers supply the distillation signal against the student's own tokens, not against teacher-generated SFT data. Used in IQuest-Q1's post-training stage to combine complementary agentic-coding specialists into one model (IQuest, 2026).

**Prereqs:** [_rl.md](_rl.md), [grpo.md](grpo.md)
**Related:** [rejection-sampling.md](rejection-sampling.md), [_rewards.md](_rewards.md), [../case-studies/iquest-q1.md](../case-studies/iquest-q1.md)

---

## What it is

A post-training step where several specialist models (each already RL-trained on its own capability — coding, tool use, reasoning, …) jointly distill into one general student. "On-policy" means the student samples its own completions during distillation; teachers score or regenerate against the student's own trajectories, so distribution shift between teacher and student is accounted for directly in the loss.

## How it works

- **Student rollout.** For each prompt, the student samples a response using its current policy. No teacher-generated SFT data is used as the direct target.
- **Teacher scoring.** Each specialist teacher scores (or re-samples against) the student's trajectory. Scoring may be token-level logit comparison, trajectory-level reward, or a per-step preference depending on the capability — exact scoring policy is per-teacher.
- **Committee aggregation.** The teachers' signals are aggregated into a distillation loss on the student. Aggregation can be a weighted sum (per-prompt weights chosen by which capability the prompt tests), max/argmax (pick the teacher that most prefers the student's trajectory), or a learned mixture.
- **Student update.** Combine the distillation loss with the regular RL loss (so the student still learns from environment reward, not only from teachers) and update the student. Teachers are frozen.

## Why it matters

- **Beyond one-teacher distillation.** R1-style long-CoT distillation has shown that one strong teacher can transfer reasoning cheaply via SFT. MOPD extends the pattern to multi-capability students without having to pick one teacher.
- **On-policy avoids data drift.** SFT distillation trains on teacher-sampled trajectories, which may be far from where the student actually samples. On-policy keeps the student's own distribution in the loss.
- **Composable with RL.** Doesn't replace the RL signal; it adds a committee gradient on top. The student still learns from environment reward.
- **Natural fit for agentic systems.** Agentic tasks fan out into multiple capabilities (code edit, shell use, retrieval, reasoning) that are hard to RL-train together end-to-end. Specialists first, MOPD second is a cleaner factorisation.

## Gotchas & tricks

- **Teacher quality dominates.** A weak specialist drags the committee; MOPD gain is bounded above by the best teacher per capability.
- **Aggregation is a design choice.** Naive averaging smears signal; capability-weighted or argmax aggregation preserves it. The paper's exact aggregation is not in the abstract.
- **Compute cost scales with teachers.** Each teacher runs on every student rollout. Infrastructure must support efficient fan-out.
- **Reward and distillation can disagree.** If a teacher strongly prefers a trajectory that scores poorly on environment reward, the student is pulled in two directions. The combined loss weight matters.
- **Not the same as model souping.** Souping merges weights of trained teachers directly; MOPD distills them into a *separately-trained* student. The two are complementary; IQuest-Q1 uses both.

## Sources

- Paper: *IQuest-Q1: Advancing Foundation Capabilities for Agentic CLI Systems* — IQuest, 2026 — [arXiv:2610.30949](https://arxiv.org/abs/2610.30949) — introduces MOPD in its multi-scaffold multi-version post-training stack.
