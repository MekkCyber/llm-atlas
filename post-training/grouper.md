# GrouPER (Group-wise SimPER)

*Depth — a group-wise preference-optimization objective used by K-EXAONE 2.0 across reasoning, agentic, chat, and safety phases.*

**TL;DR:** GrouPER is K-EXAONE 2.0's post-training preference-optimization algorithm — a **group-wise** variant of SimPER (Simple Preference Optimization). Instead of DPO's paired chosen-vs-rejected loss, GrouPER operates on **groups of ranked responses per prompt** (similar in spirit to GRPO's group structure, but on preference data rather than RL rollouts). Applied in two sequential stages: (1) multi-task preference optimization across reasoning / agentic / chat, then (2) safety-aware preference optimization with domain-specific reward criteria.

**Prereqs:** [dpo.md](dpo.md), [grpo.md](grpo.md), [_rewards.md](_rewards.md)
**Related:** [rlvr.md](rlvr.md) · [_post-training.md](_post-training.md) · [../case-studies/k-exaone-2.md](../case-studies/k-exaone-2.md)

---

## What it is

Reference-free preference optimization at group scale. Standard DPO takes a pair `(chosen, rejected)` per prompt and pushes the policy toward the chosen response. SimPER dropped DPO's KL-to-reference term for simplicity and stability. GrouPER extends this to a *group* of ranked responses per prompt — the whole group contributes to the update, not just a hand-picked pair — while keeping SimPER's reference-free structure.

The K-EXAONE 2.0 paper introduces GrouPER as its preference-optimization workhorse; it replaces both DPO and GRPO in the pipeline stages that follow SFT.

## How it works

For each prompt in a preference batch:

1. **Sample a group** of `G` responses (typically from the SFT-stage policy, or from earlier post-training checkpoints).
2. **Score / rank** the group using the appropriate reward source for the stage:
   - Math/code: verifiable signal (unit tests, answer match) + LLM-as-a-judge
   - Chat: instance-specific rubrics
   - Agentic: correctness of actions + response quality/depth/comprehensiveness
   - Safety: domain-specific safety reward criteria (K-EXAONE 2.0's stage-2 use)
3. **Compute a group-wise preference loss.** The objective aggregates preferences across all pairs (or all rankings) in the group — every response contributes signal, not just the top-1 vs bottom-1 pair.
4. **No KL-to-reference term** (SimPER's reference-free property is inherited). Regularization comes from mini-batching and group-relative scaling rather than an explicit KL anchor.

Two sequential passes on K-EXAONE 2.0:

- **Stage 1 (multi-task):** groups mix reasoning, agentic, and chat prompts; rewards are stage-specific per prompt.
- **Stage 2 (safety):** groups are safety-oriented prompts; reward criteria draw from the K-AUT-V2 taxonomy.

## Why it matters

- **Uses the whole group.** Pairwise DPO wastes most of a group's information — GrouPER keeps it, similar to how GRPO extracts more signal per prompt than PPO with a single sampled response.
- **Reference-free.** No SFT-checkpoint anchor to maintain, no `π_ref` forward pass per step — cheaper than DPO on large models.
- **Stage-compatible.** Works with heterogeneous reward sources (verifiable + rubric + judge) in the same batch; the group ranking abstracts over reward-source differences.
- **Safety-stage bolt-on.** A dedicated Stage-2 pass on safety-labeled groups gives explicit steering without touching the general-capability policy.

## Gotchas & tricks

- **Group size is a compute lever.** Larger `G` gives lower-variance updates but multiplies scoring cost; K-EXAONE 2.0 doesn't publish specific `G` values.
- **Reward heterogeneity requires calibration** across group members — verifiable rewards, judge scores, and rubric scores have different scales. Some normalization (per-prompt or per-stage) is essential.
- **Reference-free means no KL guardrail.** If the reward source is biased, GrouPER will chase it without the DPO-style KL anchor to hold it back — safety-stage separation is one way to compartmentalize that risk.
- **Stackable.** Multi-stage GrouPER (multi-task → safety) is how K-EXAONE 2.0 keeps stage-specific reward criteria from cross-contaminating.
- **Doesn't replace SFT.** GrouPER is a post-SFT refinement; the paper trains 350B SFT tokens with frozen router before entering the preference stages.

## Sources

- Paper: *K-EXAONE 2.0 Technical Report* — LG AI Research, 2026 — [arXiv 2608.04505](https://arxiv.org/abs/2608.04505). Introduces GrouPER as the group-wise SimPER variant.
- Prior: *SimPER: A Simple Preference Optimization Objective* — 2024/2025 line that GrouPER extends to groups.
