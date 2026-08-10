# Reward model evaluation
*Depth — how you evaluate the evaluator, grounded in OSReward for computer-use agents.*

**TL;DR:** In RL for LLMs and agents, the reward model is upstream of every policy update, but there is no shared protocol for measuring reward-model quality. OSReward proposes a cross-platform benchmark of computer-use trajectories with ground-truth outcome labels, and scores reward models along three axes: **pairwise trajectory ranking**, **absolute pass/fail classification**, and **per-step usefulness**. The gap between trajectory-level and step-level accuracy is the field's current weak spot.

**Prereqs:** [../post-training/_rewards.md](../post-training/_rewards.md), [../post-training/reasoning/orm.md](../post-training/reasoning/orm.md)
**Related:** [../post-training/reasoning/prm.md](../post-training/reasoning/prm.md), [../evaluation/README.md](../evaluation/README.md), [../agents/README.md](../agents/README.md)

---

## What it is

An evaluation protocol — and an OSReward-specific instantiation — for scoring reward models rather than policies. Fills the "reward model evaluation" gap flagged in the atlas's evaluation reading list. The claim is that reward-model rankings in RLHF/RLVR pipelines are currently opaque: teams pick RMs by end-to-end policy score, which conflates RM quality with the policy optimizer.

## How it works

Three evaluation axes, applied to the same trajectory pool:

| Axis | Query the RM answers | Metric |
| --- | --- | --- |
| **Pairwise ranking** | Which of these two trajectories is better? | Accuracy vs. ground-truth outcome label |
| **Absolute classification** | Did this trajectory succeed? | AUROC / F1 vs. binary label |
| **Step-level scoring** | How useful is this step? | Correlation with an independently derived step-level label; localization accuracy on the harmful step |

OSReward's specific setup: cross-platform trajectories (web, desktop, mobile), independently verified outcome labels, and a held-out test split. Each RM is scored under the same pool, so the results compare RMs, not tasks.

## Why it matters

Two shifts:

1. **Debugging.** A policy that plateaus under RL can be traced to either the optimizer or the RM. Without an RM benchmark, teams tune the optimizer first and re-discover reward-hacking failures later. With one, you can diagnose "the RM cannot even rank pairs of trajectories" before spending compute.
2. **Cross-platform generalization.** OSReward shows RMs tuned on web trajectories drop sharply on mobile and vice versa — a fact hidden when you only look at one-platform end-to-end scores.

Step-level scoring is the biggest gap: current RMs can classify whole trajectories reasonably but poorly localize the harmful step, which limits best-of-N and PRM-as-reward strategies.

## Gotchas & tricks

- Absolute-classification scores can be gamed by an RM that memorizes trajectory-level heuristics (length, format) — the pairwise axis is the discriminator.
- Best-of-N gains from RM ranking saturate quickly (typical N=8–16 plateau); a strong pairwise-accuracy RM helps, but pathological ordering across the top few still limits gains.
- If your target platform is not in the eval, cross-platform transfer is worse than intra-platform gains — cheaper to add target-platform trajectories than to hope generalization holds.

## Sources

- Paper: *OSReward: Instituting Standardized Evaluation for Cross-Platform Computer-Use Reward Models* — Cheng et al., 2026 — [arXiv:2607.28609](https://arxiv.org/abs/2607.28609)
