# Rubric-Guided Reward Model (RewardVerse / RGPO)
*Depth — two-stage reward models that emit a per-prompt rubric before scoring, and the co-training procedure.*

**TL;DR:** Scalar reward models for open-ended generation (especially video) suffer from **scalar drift** — the scoring scale collapses or shifts across prompts because the RM has to fold complex, subjective quality into a single number without explicit criteria. A **rubric-guided RM** decomposes the task: first generate an explicit rubric of evaluation criteria for the current prompt, then score against that rubric. RewardVerse's **Rubric-Guided Policy Optimization (RGPO)** co-trains the rubric generator and the scorer so the rubric stays useful as the scorer improves. State-of-the-art on pointwise and pairwise video-RM evaluation and stable enough to drive RL post-training of video generators.

**Prereqs:** [_rewards](_rewards.md), [cot-reward-model](cot-reward-model.md)
**Related:** [grpo](grpo.md), [rlvr](rlvr.md)

---

## What it is

A reward-model architecture and training procedure for tasks where scalar-only regression from human ratings is unreliable. The RM is decomposed into two stages:

- **Rubric generator.** Given the prompt, emits a set of evaluation criteria (e.g. "motion realism," "audio-visual sync," "prompt fidelity") each with a scoring scale and importance weight.
- **Scorer.** Given the prompt, the candidate output, and the rubric, emits per-criterion scores and (optionally) a scalar aggregate.

Both stages are LLM/VLM heads sharing a backbone, trained jointly.

## How it works

**RGPO training** is two-stage:

1. **Scorer warm-up.** Generate seed rubrics via self-evolving prompts (the RM proposes rubrics for a batch of prompts, filters them by consistency, and uses them as bootstrap training data). Freeze the rubric generator; train the scorer on human pairwise/pointwise preferences conditioned on the seed rubrics until it aligns well with humans.
2. **Joint optimization.** Unfreeze the rubric generator. On each batch:
   - Rubric generator emits a query-adaptive rubric.
   - Scorer scores candidates against it.
   - Both are updated with a coupled objective that (a) drives the scorer toward human preferences and (b) drives the rubric generator toward rubrics that make the scorer easier to align (rubrics that lead to lower scoring loss).

At inference time in RL post-training of a generator: sample the rubric once per prompt, then score all rollouts against the same rubric to keep the reward stable within a prompt's rollout group.

## Why it matters

Standard video RMs produce a scalar that drifts across prompts, making RL rollouts unstable. Rubric conditioning gives the reward a **stable semantic anchor** — the same rubric text pins the scoring scale within a prompt — which is exactly what GRPO's group-relative baseline needs. RewardVerse is the first working recipe for video-generation RL reward at pointwise + pairwise SOTA on EvalVerse's 16-dimensional benchmark.

Generalizes beyond video: any subjective, high-dimensional evaluation surface (creative writing, dialogue quality, long-form reasoning trace evaluation) has the same scalar-drift problem.

## Gotchas & tricks

- The rubric generator can collapse to trivial or uniform rubrics if the joint loss rewards ease-of-scoring too heavily. Mitigation: a diversity/coverage regularizer on rubric criteria.
- Rubric caching per prompt is a practical necessity when using this RM as an RL reward — sampling a fresh rubric per rollout would destroy the group-baseline stability GRPO relies on.
- The rubric is *not* directly interpretable to humans as a full spec, but per-criterion scores are more auditable than a scalar, which helps reward debugging.

## Sources

- Paper: *RewardVerse: Rubric-Guided Policy Optimization for Video Reward Modeling* — Tang et al., 2026 — [arXiv:2609.22947](https://arxiv.org/abs/2609.22947)
